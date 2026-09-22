from pathlib import Path

import pymupdf as fitz
import pytest
from PIL import Image

from docqa.config import Settings
from docqa.evidence_quality import page_continuations, source_image, support_issues
from docqa.generation import RAGPipeline, evidence_context
from docqa.retrieval import HybridRetriever
from docqa.schemas import ChatRequest, Element, Evidence, RetrieveRequest, Source
from docqa.store import Store


def text_element(eid, text, page, box, doc='d'):
    return Element(id=eid,document_id=doc,kind='text',text=text,sources=[Source(page=page,bbox=box)])


def test_cross_page_retrieval_preserves_ids_and_scope(tmp_path):
    store=Store(tmp_path)
    a=text_element('a','初中专任教师414.88万人，生师比',2,(.15,.85,.85,.9))
    b=text_element('b','12.98:1。教师本科以上学历比例94.19%。',3,(.15,.08,.85,.14))
    foreign=text_element('foreign','其他文档生师比99.99:1。',3,(.15,.08,.85,.14),doc='other')
    for doc,items in [('d',[a,b]),('other',[foreign])]:
        store.put_document({'id':doc,'name':doc,'sha256':doc,'status':'ready'})
        store.replace_elements(doc,items)
    retriever=HybridRetriever(store,Settings(data_dir=tmp_path))
    ev,trace=retriever.retrieve(RetrieveRequest(question='初中专任教师生师比',document_ids=['d'],top_k=2))
    assert [e.element.id for e in ev]==['a','b']
    assert ev[1].element.sources[0].page==3
    assert trace['page_continuations']==[{'anchor':'a','element_id':'b'}]
    assert page_continuations(a,[foreign])==[]
    b.text='四、特殊教育'
    assert page_continuations(a,[b])==[]
    a.text+='。'
    assert page_continuations(a,[b])==[]


@pytest.mark.parametrize('rotation',[0,90,180,270])
def test_source_crop_is_sharp_bounded_and_rotation_correct(tmp_path,rotation):
    pdf_path=tmp_path/'source.pdf'
    with fitz.open() as pdf:
        page=pdf.new_page(width=600,height=800)
        rect=fitz.Rect(100,200,300,300)
        page.draw_rect(rect,fill=(1,0,0),color=(1,0,0))
        page.set_rotation(rotation)
        rotated=rect*page.rotation_matrix
        box=(rotated.x0/page.rect.width,rotated.y0/page.rect.height,
             rotated.x1/page.rect.width,rotated.y1/page.rect.height)
        pdf.save(pdf_path)
    old=tmp_path/'low.png'
    Image.new('RGB',(40,20),'white').save(old)
    e=text_element('e','图1',1,box)
    e.kind='image'
    e.image_path=str(old)
    item=Evidence(element=e,matched_text=e.text,document_name='source',page_label='页码',
                  rank=1,channels=[],scores={})
    store=Store(tmp_path/'store')
    store.put_document({'id':'d','sha256':'s','name':'source','status':'ready','original_path':str(pdf_path)})
    path=source_image(item,store,tmp_path/'crops')
    assert Path(path)!=old
    with Image.open(path) as img:
        assert max(img.size)>40 and max(img.size)<=1600
        assert img.getpixel((img.width//2,img.height//2))[0]>240
        assert img.getpixel((img.width//2,img.height//2))[1]<15
    assert source_image(item,store,tmp_path/'crops')==path


def test_caption_reference_is_repaired_and_unresolved_problem_is_flagged(tmp_path):
    caption=text_element('caption','图4 2021—2025年城镇新增就业人数',1,(0,0,.5,.1))
    chart=caption.model_copy(deep=True)
    chart.id='chart'
    chart.kind='image'
    img=tmp_path/'chart.png'
    Image.new('RGB',(50,50),'white').save(img)
    chart.image_path=str(img)
    ev=[Evidence(element=e,matched_text=e.text,document_name='d',page_label='页码',rank=i,
                 channels=[],scores={}) for i,e in enumerate([caption,chart],1)]
    _,ctx,_=evidence_context(ev)
    assert support_issues('2021年最高，1269万人。[E1]',ctx,ev)
    assert '[E2]' in ' '.join(support_issues('2021年最高，1269万人。[E1]',ctx,ev))
    assert not support_issues('2021年最高，1269万人。[E2]',ctx,ev)
    settings=Settings(data_dir=tmp_path/'runtime',api_key='test')
    store=Store(settings.data_dir)
    class Generator:
        def __init__(self,repair):
            self.calls=[]
            self.repair=repair
        def complete(self,messages):
            self.calls.append(messages)
            label='E2' if len(self.calls)>1 and self.repair else 'E1'
            return {'choices':[{'message':{'content':f'2021年最高，1269万人。[{label}]'}}]}
    for repair,expected in [(True,'answered'),(False,'citation_error')]:
        generator=Generator(repair)
        stages = []
        answer=RAGPipeline(None,generator,store,settings).run(
            ChatRequest(question='图4哪年最高？'),retrieval_result=(ev,{'timings':{}}),
            progress=stages.append)
        assert stages[-5:] == ['preparing', 'generating', 'validating', 'repairing', 'validating']
        assert len(generator.calls)==2
        assert answer.status==expected
        assert answer.support_checks['semantic_verification'] is False


def test_absent_textual_ratio_is_not_accepted(tmp_path):
    e=text_element('a','教师414.88万人，生师比',1,(0,0,1,1))
    item=Evidence(element=e,matched_text=e.text,document_name='d',page_label='页码',rank=1,channels=[],scores={})
    _,ctx,_=evidence_context([item])
    assert support_issues('教师414.88万人，生师比16.85:1。[E1]',ctx,[item])


def test_cross_page_repair_points_to_available_number_source():
    items = [text_element('a','教师414.88万人，生师比',2,(0,.8,1,.9)),
             text_element('b','12.98:1。',3,(0,.1,1,.2))]
    evidence=[Evidence(element=e,matched_text=e.text,document_name='d',page_label='页码',
                       rank=i,channels=[],scores={}) for i,e in enumerate(items,1)]
    _,context,_=evidence_context(evidence)
    assert any('12.98出现在[E2]' in s for s in support_issues('生师比12.98:1。[E1]',context,evidence))


def test_delete_document_removes_only_its_high_resolution_cache(tmp_path):
    from docqa.service import Service
    service=Service(Settings(data_dir=tmp_path,parser='pymupdf',api_key=''))
    try:
        service.store.put_document({'id':'d','sha256':'d','status':'ready','name':'d'})
        owned=tmp_path/'evidence-images/d/crop.png'
        other=tmp_path/'evidence-images/other/crop.png'
        for p in [owned,other]:
            p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes(b'fixture')
        service.delete('d')
        assert not owned.exists() and other.exists()
    finally:
        service.close()


def test_retrieval_heading_cannot_supply_a_fact_outside_highlight():
    body = text_element('body', 'Right column document', 1, (.5,.3,.9,.4))
    heading = text_element('heading', 'Metadata: SQLite', 1, (.1,.1,.4,.2))
    items = [Evidence(element=body, matched_text='文档：sample\n章节：Metadata: SQLite\nRight column document',
                      document_name='sample', page_label='页码', rank=1, channels=[], scores={}),
             Evidence(element=heading, matched_text='文档：sample\nMetadata: SQLite',
                      document_name='sample', page_label='页码', rank=2, channels=[], scores={})]
    _, context, _ = evidence_context(items)
    assert 'SQLite' not in context[0]['text']
    assert 'SQLite' in context[1]['text']
    assert items[0].element.sources[0].bbox == (.5,.3,.9,.4)


@pytest.mark.parametrize('reply,expected,calls', [
    ('当前文档中没有足够依据。', 'no_evidence', 1),
    ('当前文档中没有足够依据。[E1]', 'no_evidence', 1),
    ('当前文档中没有足够依据。[E9]', 'citation_error', 2),
    ('没有足够依据，但数据库是 MySQL。', 'citation_error', 2),
    ('数据库是 SQLite。[E2] 其他配置没有足够依据。', 'answered', 1),
    ('数据库是 SQLite。[E2]', 'answered', 1),
])
def test_refusal_status_and_uncited_claims(tmp_path, reply, expected, calls):
    items = [text_element('body', 'Right column document', 1, (.5,.3,.9,.4)),
             text_element('heading', 'Metadata: SQLite', 1, (.1,.1,.4,.2))]
    ev = [Evidence(element=e, matched_text=e.text, document_name='sample', page_label='页码',
                   rank=i, channels=[], scores={}) for i, e in enumerate(items, 1)]
    settings = Settings(data_dir=tmp_path, api_key='test')
    store = Store(tmp_path)

    class Generator:
        count = 0

        def complete(self, messages):
            self.count += 1
            return {'choices': [{'message': {'content': reply}}]}

    generator = Generator()
    answer = RAGPipeline(None, generator, store, settings).run(
        ChatRequest(question='Metadata使用什么数据库？'), retrieval_result=(ev, {'timings': {}}))
    assert answer.status == expected
    assert generator.count == calls
    assert store.get('answers', answer.run_id)['status'] == expected
    if expected == 'no_evidence':
        assert any('不代表文档中一定没有答案' in w for w in answer.warnings)
    assert answer.context[1]['text'] == 'Metadata: SQLite'
    assert not answer.context[1]['truncated']


def test_short_text_messages_keep_sources_and_complex_rules():
    from docqa.generation import (
        MIXED_LOOKUP_PROMPT,
        SHORT_TEXT_PROMPT,
        SYSTEM_PROMPT,
        generation_messages,
        generation_prompt,
    )
    items = [text_element('body', 'Right column document', 1, (.5,.3,.9,.4)),
             text_element('heading', 'Metadata: SQLite Upload limit: 50 MB', 1, (.1,.1,.4,.2))]
    ev = [Evidence(element=e, matched_text='文档：sample\n' + e.text,
                   document_name='sample', page_label='页码', rank=i, channels=[], scores={})
          for i, e in enumerate(items, 1)]
    content, ctx, _ = evidence_context(ev)
    question = '样本中 Metadata 使用什么数据库？'
    messages = generation_messages(ev, ctx, content, question)
    assert messages[0]['content'] == SHORT_TEXT_PROMPT
    assert '[E1] Right column document\n[E2] Metadata: SQLite' in messages[1]['content']
    assert messages[1]['content'].endswith(question)
    assert '原文字段定位' not in messages[1]['content']
    assert generation_prompt(ev, ctx, '跨页表格增长多少个百分点？') == SYSTEM_PROMPT
    ctx[1]['truncated'] = True
    assert generation_prompt(ev, ctx, question) == SYSTEM_PROMPT
    ctx[1]['truncated'] = False
    ev[1].element.kind = 'image'
    assert generation_prompt(ev, ctx, question) == MIXED_LOOKUP_PROMPT
    assert generation_prompt(ev, ctx, '图中最大值是多少？') == SYSTEM_PROMPT
    full = generation_messages(ev, ctx, content, question)
    assert full[1]['content'][:-1] == content


def test_mixed_lookup_preserves_image_binding_and_scope():
    from docqa.generation import MIXED_LOOKUP_PROMPT, SYSTEM_PROMPT, generation_messages
    item = text_element('image', '图题', 1, (0,0,1,1))
    item.kind = 'image'
    ev = [Evidence(element=item, matched_text=item.text, document_name='文档甲',
                   page_label='页码', rank=1, channels=[], scores={})]
    ctx = [{'label':'E1', 'text':'图题', 'truncated':False, 'image_sent':True}]
    content = [{'type':'text','text':'[E1] 文档甲，页码1；下一张图片属于[E1]'},
               {'type':'image_url','image_url':{'url':'data:image/png;base64,dGVzdA=='}}]
    result = generation_messages(ev, ctx, content, '系统使用什么数据库？')
    assert result[0]['content'] == MIXED_LOOKUP_PROMPT
    assert result[1]['content'][:-1] == content
    assert generation_messages(ev, ctx, content, '两年增长多少个百分点？')[0]['content'] == SYSTEM_PROMPT
    ctx[0]['truncated'] = True
    assert generation_messages(ev, ctx, content, '系统使用什么数据库？')[0]['content'] == SYSTEM_PROMPT


def test_picture_caption_does_not_leak_to_header_or_other_column():
    from docqa.parsers import local_picture_text
    lines = [('图1 国内生产总值', (.03,.48,.27,.5)),
             ('图2', (.03,.60,.06,.62)),
             ('产业比重', (.07,.60,.27,.62)),
             ('图3 劳动生产率', (.03,.72,.27,.73)),
             ('图4 其他栏', (.30,.60,.54,.62)),
             ('57.7', (.1,.65,.15,.67))]
    assert local_picture_text((.74,.03,.8,.07), lines) == ''
    result = local_picture_text((.03,.623,.27,.71), lines)
    assert result.startswith('图2 产业比重') and '57.7' in result
    assert all(t not in result for t in ['图1', '图3', '图4'])


def test_cross_page_prompt_keeps_distinct_citation_requirements():
    from docqa.generation import generation_messages
    items = [text_element('a','教师414.88万人，生师比',2,(.1,.85,.8,.9)),
             text_element('b','12.98:1。',3,(.1,.1,.8,.15))]
    ev = [Evidence(element=e, matched_text=e.text, document_name='d', page_label='页码',
                   rank=i, channels=[], scores={}) for i,e in enumerate(items,1)]
    content, ctx, _ = evidence_context(ev)
    messages = generation_messages(ev, ctx, content, '专任教师有多少万人，生师比是多少？')
    assert '[G1]' in messages[1]['content']
    assert '第2页原文：教师414.88万人，生师比' in messages[1]['content']
    assert '第3页原文：12.98:1。' in messages[1]['content']
    assert all(c['citation_group'] == 'G1' for c in ctx)



def test_latex_percent_formula_is_recognized_but_wrong_unit_is_rejected():
    e = text_element('chart', '图2 产业比重', 1, (0,0,1,1))
    e.kind = 'image'
    ev = [Evidence(element=e, matched_text=e.text, document_name='d', page_label='页码',
                   rank=1, channels=[], scores={})]
    ctx = [{'label':'E1','text':e.text,'truncated':False,'image_sent':True}]
    question = '根据图2，比重提高多少个百分点？'
    wrong = support_issues(r'\[57.7\% - 54.8\% = 2.9\%\]。[E1]', ctx, ev, question)
    assert any('百分点' in issue for issue in wrong)
    assert not any('缺少原始数值和减法' in issue for issue in wrong)
    assert not support_issues(r'\[57.7\% - 54.8\% = 2.9个百分点\]。[E1]', ctx, ev, question)
    assert any('缺少就近引用' in issue for issue in support_issues(
        r'\[57.7\% - 54.8\% = 2.9个百分点\]', ctx, ev, question))


def test_group_citation_expansion_requires_explicit_group():
    from docqa.generation import expand_group_citations, validate_citations
    groups = {'G1': ['E2', 'E3']}
    assert expand_group_citations('原文。[G1]', groups) == '原文。[E2][E3]'
    assert expand_group_citations('原文。[E2]', groups) == '原文。[E2]'
    assert validate_citations('原文。[G9][E2]', ['E2', 'E3'])[1] == {'G9'}


def test_continuation_group_respects_budget_scope_and_page_order():
    from docqa.generation import continuation_groups, generation_messages
    a = text_element('a', '教师414.88万人，生师比', 2, (.1,.85,.8,.9))
    b = text_element('b', '12.98:1。后续未入选内容', 3, (.1,.1,.8,.15))
    ev = [Evidence(element=e, matched_text=e.text, document_name='d', page_label='页码',
                   rank=i, channels=[], scores={}) for i, e in enumerate([b,a],1)]
    content, ctx, _ = evidence_context(ev)
    ctx[0]['text'] = '12.98:1。'
    result = generation_messages(ev, ctx, content, '生师比是什么？')[1]['content']
    assert result.index('第2页原文') < result.index('第3页原文')
    assert '后续未入选内容' not in result
    ctx[0]['truncated'] = True
    assert not continuation_groups(ev, ctx)
    ctx[0]['truncated'] = False
    b.document_id = 'other'
    # Evidence objects copy the model reference; explicitly change the stored element.
    ev[0].element.document_id = 'other'
    assert not continuation_groups(ev, ctx)
