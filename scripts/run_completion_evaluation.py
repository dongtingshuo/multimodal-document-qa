"""Frozen external-corpus paired generation and archived synthetic-corpus follow-up."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(ROOT / 'models/huggingface'))
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
sys.path.insert(0, str(ROOT))
from docqa.config import Settings
from docqa.evaluation import retrieval_metrics
from docqa.generation import RAGPipeline, create_generator
from docqa.retrieval import HybridRetriever
from docqa.schemas import ChatRequest, Element, Evidence, RetrieveRequest
from docqa.store import Store

OUT = ROOT / 'data/completion-20260911'


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)


def prepare():
    out = OUT / 'external'
    frozen = out / 'frozen.json'
    previous = json.loads(frozen.read_text()) if frozen.exists() else None
    if previous and len(previous['rows']) != 8:
        raise ValueError('Frozen evaluation already exists; do not silently replace it')
    if previous:
        assert not list(out.glob('*-answers.json')), 'Cannot expand after generation'
        (out/'frozen-initial8.json').write_bytes(frozen.read_bytes())
    settings = Settings(data_dir=out/'runtime', parser='pymupdf', enable_ocr=False,
                        enable_clip=False, local_models_only=True, api_key='')
    store = Store(settings.data_dir)
    sources = json.loads((out/'sources.json').read_text())
    for source in sources:
        name = source['id']
        result = json.loads((out/(name+'-parsed.json')).read_text())
        store.put_document({'id':name, 'name':name+'.pdf', 'sha256':source['sha256'],
                            'status':'ready','page_count':result['page_count'], 'converted':False,
                            'parser_version':result['parser_version'],
                            'original_path':str(out/(name+'.pdf'))})
        store.replace_elements(name,[Element.model_validate(e) for e in result['elements']])
    elements = store.elements()

    def anchors(doc, *needles):
        return [e.id for e in elements if e.document_id==doc and e.kind=='text'
                and all(n in e.text for n in needles)]

    questions = [
        ('ext-01','text','2024年全国教育事业发展统计公报列出的各级各类学校总数是多少？','47.00万所',anchors('education','47.00')),
        ('ext-02','text','2024年全国初中阶段专任教师有多少万人，生师比是多少？','414.88万人；12.98:1',anchors('education','414.88')+anchors('education','12.98:1')),
        ('ext-03','table','2025年年末人口构成表中，女性人口数和占全国人口比重各是多少？','68804万人；49.0%',['economy-p0001-e0001']),
        ('ext-04','table','按2025年年末人口构成表，男性比女性多多少万人？列出计算。','71685-68804=2881万人',['economy-p0001-e0001']),
        ('ext-05','chart','根据图4的2021—2025年城镇新增就业人数柱状图，哪年最高，数值多少？','2021年；1269万人',['economy-p0001-e0086']),
        ('ext-06','chart','根据图1的2021—2025年国内生产总值及其增长速度图，哪年的增长速度最低，是多少？','2022年；3.1%',['economy-p0001-e0083']),
        ('ext-07','unanswerable','根据所提供的教育公报，2026年全国教育经费预算总额是多少？','资料未提供2026年教育经费预算。',[]),
        ('ext-08','unanswerable','根据所提供的统计公报版面，2026年GDP的预测值是多少？','资料未提供2026年GDP预测。',[]),
        ('ext-09','text','2024年全国高等教育毛入学率是多少？','60.80%',anchors('education','60.80')),
        ('ext-10','text','2024年义务教育阶段进城务工人员随迁子女共有多少万人？','1308.83万人',anchors('education','1308.83')),
        ('ext-11','table','2025年人口构成表中，60周岁及以上人口及其比重是多少？','32338万人；23.0%',['economy-p0001-e0001']),
        ('ext-12','table','2025年人口构成表中，乡村人口数是多少？','45109万人',['economy-p0001-e0001']),
        ('ext-13','table','2025年居民消费价格涨跌幅度表里，交通通信全国和农村分别变化多少？','全国-2.6%；农村-2.5%',['economy-p0001-e0000']),
        ('ext-14','table','2025年居民消费价格涨跌幅度表里，居民消费价格全国、城市、农村三项分别是多少？','全国0.0%；城市0.1%；农村-0.2%',['economy-p0001-e0000']),
        ('ext-15','chart','根据图7，2025年年末国家外汇储备比2024年增加多少亿美元？','33579-32024=1555亿美元',['economy-p0001-e0089']),
        ('ext-16','chart','根据图5，2025年全国城镇调查失业率最高的月份和数值是多少？','2月；5.4%',['economy-p0001-e0087']),
        ('ext-17','chart','图3所示2021—2025年全员劳动生产率是否逐年上升？2025年是多少？','逐年上升；184413元/人',['economy-p0001-e0085']),
        ('ext-18','chart','根据图2，第三产业增加值占GDP比重从2021年到2025年提高多少个百分点？','57.7%-54.8%=2.9个百分点',['economy-p0001-e0084']),
        ('ext-19','unanswerable','请根据教育公报列出全国每所普通初中的校长姓名。','资料未提供逐校校长名单。',[]),
        ('ext-20','unanswerable','根据教育公报，2027年高等教育毛入学率预计达到多少？','资料未提供2027年预测。',[]),
    ]
    rows=[]
    for qid,category,question,reference,gold in questions:
        assert gold or category=='unanswerable'
        rows.append({'id':qid,'category':category,'question':question,
                     'reference_answer':reference,'relevant_element_ids':gold,
                     'reference_sources':[e.model_dump(mode='json') for e in elements if e.id in gold]})
    # Freeze labels before retrieving or generating any answer.
    save(out/'questions.json',rows)
    retriever = HybridRetriever(store,settings)
    frozen_rows=[]
    for q in rows:
        prior = next((r for r in previous['rows'] if r['id']==q['id']),None) if previous else None
        if prior:
            frozen_rows.append(prior)
            continue
        ev,trace = retriever.retrieve(RetrieveRequest(question=q['question'],strategy='hybrid_reranker',
                                                      document_ids=['economy','education'],top_k=10))
        metrics = retrieval_metrics([e.element.id for e in ev], q['relevant_element_ids'])
        frozen_rows.append({**q,'metrics':metrics,'evidence':[e.model_dump(mode='json') for e in ev[:6]],'trace':trace})
        print(q['id'],metrics,flush=True)
    hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((ROOT/'docqa').glob('*.py'))}
    save(frozen,{'scope':'external-new-documents-first-evaluation; lightweight parser; E generation',
                 'code_sha256':hashes,'source_manifest':sources,'rows':frozen_rows})


def generate(provider, synthetic=False):
    out = OUT / ('synthetic' if synthetic else 'external')
    out.mkdir(parents=True,exist_ok=True)
    settings = Settings(data_dir=out/(provider+'-runtime'), generation_provider=provider,
                        local_models_only=True, max_api_calls=1000)
    store = Store(settings.data_dir)
    # Cloud calls always use the existing project ledger and its original limit.
    ledger = Store(Settings().data_dir) if provider=='bailian' else store
    generator = create_generator(settings,ledger)
    if synthetic and not (out/'frozen.json').exists():
        source = ROOT/'data/generalization-20260911'
        dataset=json.loads((source/'questions.json').read_text())
        rsettings=Settings(data_dir=source/'snapshot',local_models_only=True,enable_clip=False)
        retriever=HybridRetriever(Store(rsettings.data_dir),rsettings)
        rows=[]
        for q in dataset['questions']:
            evidence,trace=retriever.retrieve(RetrieveRequest(question=q['question'],strategy='hybrid_reranker',
                                                               document_ids=dataset['document_ids'],top_k=6))
            rows.append({**q,'evidence':[e.model_dump(mode='json') for e in evidence], 'trace':trace})
        save(out/'frozen.json',{'rows':rows,'scope':'archived synthetic corpus, E-style image-enabled follow-up'})
    frozen_bytes=(out/'frozen.json').read_bytes()
    frozen=json.loads(frozen_bytes)
    for name,digest in frozen.get('code_sha256',{}).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest, 'Code changed after freeze'
    path=out/(provider+'-answers.json')
    result=json.loads(path.read_text()) if path.exists() else {
        'provider':provider,'model':settings.active_generation_model,
        'input_sha256':hashlib.sha256(frozen_bytes).hexdigest(),
        'cloud_calls_before':ledger.call_count() if provider=='bailian' else 0,'rows':[]}
    assert result['input_sha256']==hashlib.sha256(frozen_bytes).hexdigest()
    done={r['id'] for r in result['rows']}
    pipeline=RAGPipeline(None,generator,store,settings)
    for q in frozen['rows']:
        if q['id'] in done:
            continue
        if provider=='bailian' and ledger.call_count()>=settings.max_api_calls:
            result['blocked']='Project API budget reached; limit not changed'
            save(path,result)
            break
        evidence=[Evidence.model_validate(e) for e in q['evidence']]
        req=ChatRequest(question=q['question'],strategy='hybrid_reranker',include_images=True)
        answer=pipeline.run(req,retrieval_result=(evidence,q['trace']))
        result['rows'].append({'id':q['id'],'category':q['category'],
                              'reference_answer':q['reference_answer'],'answer':answer.model_dump(mode='json')})
        result['cloud_calls_after']=ledger.call_count() if provider=='bailian' else 0
        save(path,result)
        print(provider,q['id'],answer.status,flush=True)
    result['completed']=len(result['rows'])==len(frozen['rows'])
    save(path,result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['prepare','ollama','bailian','synthetic'])
    args=parser.parse_args()
    if args.phase=='prepare':
        prepare()
    else:
        generate('ollama' if args.phase=='synthetic' else args.phase,args.phase=='synthetic')
