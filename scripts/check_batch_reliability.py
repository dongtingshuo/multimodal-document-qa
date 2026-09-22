"""One frozen batch for graph references, units, adjacent tables and unreadable input."""
import argparse
import hashlib
import json
import sqlite3

import check_numeric_quality as runner

ROOT = runner.ROOT
OUT = ROOT / 'data/batch-reliability-20260912'


def prepare():
    from docqa.schemas import Element, Evidence, Source
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'frozen.json').exists()
    (OUT / 'runtime').mkdir(exist_ok=True)
    previous = ROOT / 'data/numeric-quality-20260912'
    with sqlite3.connect((previous / 'runtime/metadata.sqlite3').as_uri() + '?mode=ro', uri=True) as src:
        with sqlite3.connect(OUT / 'runtime/metadata.sqlite3') as dst:
            src.backup(dst)
    rows = [r for r in json.loads((previous / 'frozen.json').read_text())['rows']
            if r['id'] in ['ext-18', 'fresh-02']]
    elements = [Element(id='batch-a', document_id='batch', kind='table', text='表6 合格率',
                        table=[['项目', '2024年'], ['甲线', '92.4%']], title_path=['质量'],
                        sources=[Source(page=1, bbox=(.1, .7, .9, .95))]),
                Element(id='batch-b', document_id='batch', kind='table', text='表6 合格率（续表）',
                        table=[['项目', '2024年'], ['乙线', '96.8%']], title_path=['质量'],
                        sources=[Source(page=2, bbox=(.1, .05, .9, .2))])]
    evidence = [Evidence(element=e, matched_text=e.text + '\n' + '\n'.join(' | '.join(r) for r in e.table),
                         document_name='合成跨页表格', page_label='页码', rank=i, channels=[], scores={}).model_dump(mode='json')
                for i, e in enumerate(elements, 1)]
    rows.append({'id':'batch-table', 'question':'表6中甲线和乙线2024年合格率分别多少？请分别引用各页。',
                 'reference_answer':'甲线92.4%[E1]；乙线96.8%[E2]', 'evidence':evidence, 'trace':{'timings':{}}})
    missing = Element(id='missing', document_id='missing', kind='image', text='图9 产量',
                      image_path=str(OUT / 'intentionally-unavailable.png'),
                      sources=[Source(page=3, bbox=(.1, .1, .9, .9))])
    item = Evidence(element=missing, matched_text=missing.text, document_name='缺失图像样本',
                    page_label='页码', rank=1, channels=[], scores={})
    rows.append({'id':'batch-unclear', 'question':'图9里2024年的产量具体是多少？',
                 'reference_answer':'无法确认读数；图像不可读且文字没有数值，不能猜测。',
                 'evidence':[item.model_dump(mode='json')], 'trace':{'timings':{}}})
    runner.save(OUT / 'frozen.json', {'scope':'One known chart case, one prior numeric fixture and two new synthetic generation cases; not a blind real-document test',
                                    'rows':rows, 'code_sha256':{
                                        str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in sorted((ROOT / 'docqa').glob('*.py'))}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'bailian', 'ollama'])
    args = parser.parse_args()
    runner.OUT = OUT
    prepare() if args.phase == 'prepare' else runner.run(args.phase)
