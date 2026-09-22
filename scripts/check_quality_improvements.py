"""Known-case regression for page continuation, source images and citation checks."""
import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME',str(ROOT/'models/huggingface'))
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
sys.path.insert(0,str(ROOT))
from docqa.config import Settings
from docqa.evaluation import retrieval_metrics
from docqa.generation import RAGPipeline, create_generator
from docqa.retrieval import HybridRetriever
from docqa.schemas import ChatRequest, Evidence, RetrieveRequest
from docqa.store import Store

OUT=ROOT/'data/quality-improvement-20260911'
BASE=ROOT/'data/completion-20260911/external'
IDS=['ext-01','ext-02','ext-05','ext-13','ext-15','ext-17','ext-18','ext-19']


def save(path,value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2))
    tmp.replace(path)


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'frozen.json').exists(), 'Do not overwrite frozen results'
    target=OUT/'runtime'
    target.mkdir(exist_ok=True)
    with sqlite3.connect((BASE/'runtime/metadata.sqlite3').as_uri()+'?mode=ro',uri=True) as src:
        with sqlite3.connect(target/'metadata.sqlite3') as dst:
            src.backup(dst)
    if (BASE/'runtime/indexes').exists():
        shutil.copytree(BASE/'runtime/indexes',target/'indexes',dirs_exist_ok=True)
    settings=Settings(data_dir=target,local_models_only=True,enable_clip=False)
    retriever=HybridRetriever(Store(target),settings)
    previous=json.loads((BASE/'frozen.json').read_text())
    rows=[]
    for q in previous['rows']:
        if q['id'] not in IDS:
            continue
        evidence,trace=retriever.retrieve(RetrieveRequest(question=q['question'],strategy='hybrid_reranker',
                                                          document_ids=['economy','education'],top_k=10))
        rows.append({**q,'baseline_metrics':q['metrics'],
                     'metrics':retrieval_metrics([e.element.id for e in evidence],q['relevant_element_ids']),
                     'evidence':[e.model_dump(mode='json') for e in evidence[:6]],'trace':trace})
        print(q['id'],rows[-1]['metrics'],trace['page_continuations'],flush=True)
    save(OUT/'frozen.json',{'baseline_sha256':hashlib.sha256((BASE/'frozen.json').read_bytes()).hexdigest(),
                           'scope':'known-failure regression plus two controls; not a new blind test',
                           'code_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in sorted((ROOT/'docqa').glob('*.py'))},'rows':rows})


def run(provider):
    frozen_bytes=(OUT/'frozen.json').read_bytes()
    frozen=json.loads(frozen_bytes)
    for path,digest in frozen['code_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    settings=Settings(data_dir=OUT/'runtime',generation_provider=provider,local_models_only=True)
    store=Store(settings.data_dir)
    ledger=Store(Settings().data_dir) if provider=='bailian' else store
    pipeline=RAGPipeline(None,create_generator(settings,ledger),store,settings)
    path=OUT/(provider+'.json')
    result=json.loads(path.read_text()) if path.exists() else {
        'provider':provider,'input_sha256':hashlib.sha256(frozen_bytes).hexdigest(),
        'calls_before':ledger.call_count(),'rows':[]}
    assert result['input_sha256']==hashlib.sha256(frozen_bytes).hexdigest()
    for q in frozen['rows']:
        if q['id'] in {r['id'] for r in result['rows']}:
            continue
        ev=[Evidence.model_validate(e) for e in q['evidence']]
        answer=pipeline.run(ChatRequest(question=q['question'],strategy='hybrid_reranker'),
                            retrieval_result=(ev,q['trace']))
        result['rows'].append({'id':q['id'],'reference_answer':q['reference_answer'],
                              'answer':answer.model_dump(mode='json')})
        result['calls_after']=ledger.call_count()
        save(path,result)
        print(provider,q['id'],answer.status,flush=True)
    result['completed']=len(result['rows'])==len(frozen['rows'])
    save(path,result)


def prepare_refined():
    target=OUT/'refined'
    target.mkdir(parents=True,exist_ok=True)
    assert not (target/'frozen.json').exists()
    (target/'runtime').mkdir(exist_ok=True)
    with sqlite3.connect((OUT/'runtime/metadata.sqlite3').as_uri()+'?mode=ro',uri=True) as src:
        with sqlite3.connect(target/'runtime/metadata.sqlite3') as dst:
            src.backup(dst)
    frozen=json.loads((OUT/'frozen.json').read_text())
    frozen['rows']=[r for r in frozen['rows'] if r['id'] in ['ext-02','ext-05']]
    frozen['scope']='Two citation repair prompt regressions; first-pass results retained'
    frozen['code_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted((ROOT/'docqa').glob('*.py'))}
    save(target/'frozen.json',frozen)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['prepare','bailian','ollama','prepare-refined','refined-ollama'])
    args=parser.parse_args()
    if args.phase=='prepare':
        prepare()
    elif args.phase=='prepare-refined':
        prepare_refined()
    elif args.phase=='refined-ollama':
        OUT=OUT/'refined'
        run('ollama')
    else:
        run(args.phase)
