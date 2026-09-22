"""Preserved dual-model regression: two known failures plus four fresh fixtures."""
import argparse
import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('OMP_NUM_THREADS', '1')
from docqa.config import Settings
from docqa.generation import RAGPipeline, create_generator
from docqa.schemas import ChatRequest, Element, Evidence, Source
from docqa.store import Store

OUT = ROOT / 'data/numeric-quality-20260912'
BASE = ROOT / 'data/quality-improvement-20260911'


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def prepare():
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'frozen.json').exists()
    (OUT / 'runtime').mkdir(exist_ok=True)
    with sqlite3.connect((BASE / 'runtime/metadata.sqlite3').as_uri() + '?mode=ro', uri=True) as src:
        with sqlite3.connect(OUT / 'runtime/metadata.sqlite3') as dst:
            src.backup(dst)
    rows = [r for r in json.loads((BASE / 'frozen.json').read_text())['rows'] if r['id'] in ['ext-13', 'ext-18']]
    fixtures = [
        ('fresh-01', '配送费在沿海和内陆的变化分别是多少？', '沿海-3.4%；内陆+1.7%',
         [['指标', '沿海', '内陆'], ['配送费', '-3.4%', '1.7%']], ''),
        ('fresh-02', '清洁能源占比从2020年到2023年提高几个百分点？', '63.45%-48.20%=15.25个百分点',
         [['指标', '2020年', '2023年'], ['清洁能源占比', '48.20%', '63.45%']], ''),
        ('fresh-03', '甲区和乙区的合格率分别是多少？请分别引用。', '甲区91.35%；乙区97.62%',
         None, '甲区合格率91.35%。'),
        ('fresh-04', '配送费表能说明2028年全国平均工资是多少吗？', '没有足够依据',
         [['指标', '沿海', '内陆'], ['配送费', '-3.4%', '1.7%']], ''),
    ]
    for eid, question, reference, table, text in fixtures:
        text = text or '\n'.join(' | '.join(row) for row in table)
        element = Element(id=eid, document_id=eid, kind='table' if table else 'text', text=text,
                          table=table, sources=[Source(page=1, bbox=(0, 0, 1, 1))])
        ev = [Evidence(element=element, matched_text=text, document_name='新合成验收样本', page_label='页码',
                       rank=1, channels=[], scores={}).model_dump(mode='json')]
        if eid == 'fresh-03':
            other = element.model_copy(deep=True)
            other.id += '-b'
            other.text = '乙区合格率97.62%。'
            other.sources[0].page = 2
            ev.append(Evidence(element=other, matched_text=other.text, document_name='新合成验收样本',
                               page_label='页码', rank=2, channels=[], scores={}).model_dump(mode='json'))
        rows.append({'id': eid, 'question': question, 'reference_answer': reference, 'evidence': ev,
                     'trace': {'timings': {}}, 'scope': 'new synthetic generation fixture; not retrieval or real-world generalization'})
    save(OUT / 'frozen.json', {'rows': rows, 'code_sha256': {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT / 'docqa').glob('*.py'))}})


def run(provider):
    raw = (OUT / 'frozen.json').read_bytes()
    frozen = json.loads(raw)
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in frozen['code_sha256'].items())
    settings = Settings(data_dir=OUT / 'runtime', generation_provider=provider)
    store = Store(settings.data_dir)
    ledger = Store(Settings().data_dir) if provider == 'bailian' else store
    pipeline = RAGPipeline(None, create_generator(settings, ledger), store, settings)
    path = OUT / (provider + '.json')
    result = json.loads(path.read_text()) if path.exists() else {'provider': provider, 'input_sha256': hashlib.sha256(raw).hexdigest(), 'rows': []}
    assert result['input_sha256'] == hashlib.sha256(raw).hexdigest()
    for row in frozen['rows']:
        if row['id'] in {r['id'] for r in result['rows']}:
            continue
        answer = pipeline.run(ChatRequest(question=row['question'], strategy='hybrid_reranker'),
                              retrieval_result=([Evidence.model_validate(e) for e in row['evidence']], row['trace']))
        result['rows'].append({'id': row['id'], 'answer': answer.model_dump(mode='json')})
        save(path, result)
        print(provider, row['id'], answer.status, flush=True)
    result['completed'] = True
    save(path, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'bailian', 'ollama', 'prepare-refined', 'refined-ollama'])
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare()
    elif args.phase == 'prepare-refined':
        target = OUT / 'refined'
        target.mkdir(exist_ok=True)
        assert not (target / 'frozen.json').exists()
        (target / 'runtime').mkdir(exist_ok=True)
        with sqlite3.connect((OUT / 'runtime/metadata.sqlite3').as_uri() + '?mode=ro', uri=True) as src:
            with sqlite3.connect(target / 'runtime/metadata.sqlite3') as dst:
                src.backup(dst)
        frozen = json.loads((OUT / 'frozen.json').read_text())
        frozen['rows'] = [r for r in frozen['rows'] if r['id'] in ['ext-13', 'ext-18']]
        frozen['code_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted((ROOT / 'docqa').glob('*.py'))}
        frozen['scope'] = 'Final table-title repair and missing graph calculation guard; two known cases'
        save(target / 'frozen.json', frozen)
    elif args.phase == 'refined-ollama':
        OUT = OUT / 'refined'
        run('ollama')
    else:
        run(args.phase)
