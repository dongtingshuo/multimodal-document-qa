from PIL import Image

from docqa.config import Settings
from docqa.evidence_quality import page_continuations, support_issues
from docqa.generation import RAGPipeline, evidence_context
from docqa.numeric_evidence import percentage_issues, table_notes
from docqa.page_links import table_neighbors
from docqa.retrieval import HybridRetriever
from docqa.schemas import ChatRequest, Element, Evidence, RetrieveRequest, Source
from docqa.store import Store


def element(eid, page, kind='table', text='表8 能耗', box=(.1, .04, .9, .94)):
    return Element(id=eid, document_id='d', kind=kind, text=text, title_path=['能耗统计'],
                   sources=[Source(page=page, bbox=box)],
                   table=[['项目', '2023年', '2024年'], [eid, '10%', '15%']] if kind == 'table' else None)


def ev(items):
    return [Evidence(element=e, matched_text=e.text, document_name='source', page_label='页码',
                     rank=i, channels=[], scores={}) for i, e in enumerate(items, 1)]


def test_repeated_headers_link_three_pages_keep_original_sources(tmp_path):
    items = [element('a', 1), element('b', 2), element('c', 3)]
    store = Store(tmp_path)
    store.put_document({'id':'d', 'name':'d', 'sha256':'d', 'status':'ready'})
    store.replace_elements('d', items)
    result, trace = HybridRetriever(store, Settings(data_dir=tmp_path)).retrieve(
        RetrieveRequest(question='能耗', top_k=3, document_ids=['d']))
    assert {e.element.id for e in result} == {'a', 'b', 'c'}
    assert len(trace['page_continuations']) == 2
    assert {e.element.sources[0].page for e in result} == {1, 2, 3}
    _, context, _ = evidence_context(result, question='能耗')
    assert all('重复表头不计为数据' in c['text'] for c in context)
    assert table_neighbors(items[2], items)[0].id == 'b'


def test_similar_tables_do_not_cross_number_unit_section_or_document():
    a, b = element('a', 1), element('b', 2)
    for change in [{'text': '表9 能耗'}, {'document_id': 'other'}, {'title_path': ['财务']},
                   {'text': '没有明确表号'}, {'table': [['项目', '地区'], ['a', '10']]}]:
        assert not table_neighbors(a, [b.model_copy(update=change)])
    a.text += '\n单位：万元'
    b.text += '\n单位：亿元'
    assert not table_neighbors(a, [b])
    a.text = b.text = '表8 能耗'
    heading = element('h', 2, kind='text', text='二、其他指标', box=(.1, .01, .9, .02))
    assert not table_neighbors(a, [b, heading])
    assert not table_neighbors(a, [b, b.model_copy(update={'id':'ambiguous'})])


def test_same_section_sentence_continuation_and_new_heading_guard():
    a = element('a', 1, 'text', '年度变化主要受到', (.1, .85, .9, .95))
    b = element('b', 2, 'text', '新增产能与生产效率共同影响。', (.1, .05, .9, .1))
    assert page_continuations(a, [b]) == [b]
    b.text = '第二章其他结果'
    assert not page_continuations(a, [b])


def test_percent_points_relative_growth_and_zero_base():
    assert percentage_issues('57.7% - 54.8% = 3.9%', '提高几个百分点')
    assert percentage_issues('提高2.9%。', '提高几个百分点')
    assert not percentage_issues('57.7% - 54.8% = 2.9个百分点。[E1]', '提高几个百分点')
    notes = table_notes([['指标','2021年','2024年'],['合格率','20%','30%']], '合格率2021到2024相对增长率')
    assert '50.0000%' in notes and '10个百分点' in notes
    notes = table_notes([['指标','2021年','2024年'],['合格率','0%','30%']], '合格率2021到2024相对增长率')
    assert '基期为零' in notes


def test_wrong_figure_id_cannot_borrow_another_image(tmp_path):
    items = [element('a', 1, 'image', '图1 总量'), element('b', 1, 'image', '图2 比例')]
    for item in items:
        path = tmp_path / (item.id + '.png')
        Image.new('RGB', (200, 200), 'white').save(path)
        item.image_path = str(path)
    evidence = ev(items)
    _, context, _ = evidence_context(evidence)
    assert any('图号' in i for i in support_issues('比例为57.7%。[E1]', context, evidence, '图2比例多少？'))
    assert not support_issues('比例为57.7%。[E2]', context, evidence, '图2比例多少？')


def test_missing_or_corrupt_image_degrades_to_text_with_location(tmp_path):
    image = element('a', 1, 'image', '图3 样本')
    image.image_path = str(tmp_path / 'missing.png')
    evidence = ev([image])
    class Generator:
        def complete(self, messages):
            return {'choices':[{'message':{'content':'无法确认图3的具体读数，请查看原图。[E1]'}}]}
    settings = Settings(data_dir=tmp_path/'runtime', api_key='test')
    answer = RAGPipeline(None, Generator(), Store(settings.data_dir), settings).run(
        ChatRequest(question='图3读数？'), retrieval_result=(evidence, {'timings':{}}))
    assert answer.image_count == 0
    assert answer.uncertainty[0]['sources'][0]['page'] == 1
    assert any('source' in w and '无法读取' in w for w in answer.warnings)
    assert '无法确认' in answer.text
    (tmp_path / 'missing.png').write_bytes(b'broken')
    _, context, count = evidence_context(evidence)
    assert count == 0 and context[0]['image_issue']


def test_unreadable_explanation_is_not_mistaken_for_a_reading():
    image = element('x', 1, 'image', '图9 产量')
    evidence = ev([image])
    _, context, _ = evidence_context(evidence)
    answer = '当前文档中没有足够依据。问题要求图9中2024年的产量，但证据仅有图题，未提供具体数值[E1]。因此无法确认2024年产量。'
    assert not support_issues(answer, context, evidence, '图9中2024年产量？')
    assert support_issues('无法确认原图，但2024年产量为12.3万吨。[E1]', context, evidence, '图9中2024年产量？')


def test_both_requested_table_rows_must_be_addressed():
    a, b = element('a', 1), element('b', 2)
    a.table = [['项目', '2024年'], ['甲线', '92.4%']]
    b.table = [['项目', '2024年'], ['乙线', '96.8%']]
    evidence = ev([a, b])
    question = '甲线和乙线分别多少？'
    _, context, _ = evidence_context(evidence, question=question)
    assert any('甲线' in i for i in support_issues('乙线96.8%。[E2]', context, evidence, question))
    assert not support_issues('甲线92.4%。[E1]乙线96.8%。[E2]', context, evidence, question)
