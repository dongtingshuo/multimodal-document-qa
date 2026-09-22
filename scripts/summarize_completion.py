"""Summarize completed supplemental checks, requiring explicit per-answer review."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/completion-20260911'


def read(path):
    return json.loads(path.read_text())


def main():
    frozen=read(OUT/'external/frozen.json')
    left=read(OUT/'external/ollama-answers.json')
    right=read(OUT/'external/bailian-answers.json')
    reviews=read(OUT/'external/reviews.json')
    expected=[r['id'] for r in frozen['rows']]
    assert len(expected)==20 and len(set(expected))==20
    digest=hashlib.sha256((OUT/'external/frozen.json').read_bytes()).hexdigest()
    for run in [left,right]:
        assert run['completed'] and [r['id'] for r in run['rows']]==expected
        assert run['input_sha256']==digest
    review_map={(r['provider'],r['id']):r for r in reviews}
    assert len(review_map)==40
    pairs=[]
    for left_row,right_row in zip(left['rows'],right['rows'],strict=True):
        la,ra=left_row['answer'],right_row['answer']
        same=(la['context']==ra['context'] and la['image_count']==ra['image_count']
              and la['evidence']==ra['evidence'])
        assert same
        pairs.append({'id':left_row['id'],'category':left_row['category'],'same_input':same,
                      'ollama_status':la['status'],'bailian_status':ra['status'],
                      'ollama_score':review_map[('ollama',left_row['id'])]['score'],
                      'bailian_score':review_map[('bailian',left_row['id'])]['score'],
                      'ollama_seconds':la['timings'].get('generation_seconds',0),
                      'bailian_seconds':ra['timings'].get('generation_seconds',0)})
    (OUT/'external/comparison.json').write_text(json.dumps(pairs,ensure_ascii=False,indent=2))
    with (OUT/'external/comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(pairs[0]))
        w.writeheader()
        w.writerows(pairs)
    retrieval=[r['metrics'] for r in frozen['rows'] if r['metrics']['recall@5'] is not None]
    metrics={k:sum(r[k] for r in retrieval)/len(retrieval) for k in retrieval[0]}
    summary={'retrieval':metrics,'models':{},'pairs':len(pairs),'identical_inputs':len(pairs)}
    for provider,run in [('ollama',left),('bailian',right)]:
        scores=[review_map[(provider,r['id'])]['score'] for r in run['rows']]
        summary['models'][provider]={
            'score_sum':sum(scores),'denominator':len(scores),'score_mean':sum(scores)/len(scores),
            'answered':sum(r['answer']['status']=='answered' for r in run['rows']),
            'generation_seconds_mean':sum(p[provider+'_seconds'] for p in pairs)/len(pairs)}
    summary['cloud_requests']=right['cloud_calls_after']-right['cloud_calls_before']
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    lines=['# 外部真实语料双模型补测','',
           '固定两份原题库之外的PDF，共11页、20题：文本4、表格6、图表6、无答案4。统计材料为一版报纸上的公报节选与评读，不是完整公报。',
           '采用轻量解析、BGE混合检索与重排序，前10位计算检索指标，前6项证据及最多3张原图用于生成。模型为Ollama qwen2.5vl:7b与百炼qwen3-vl-plus-2025-12-19。',
           '',f"16道可回答题的Recall@5为{metrics['recall@5']:.4f}，MRR@10为{metrics['mrr@10']:.4f}。20对输入的文字上下文、证据结构、图片数量全部一致。",'',
           '|模型|辅助评分（含全部失败）|引用格式通过|平均生成秒数|','|---|---:|---:|---:|']
    for provider in ['ollama','bailian']:
        m=summary['models'][provider]
        lines.append(f"|{provider}|{m['score_sum']}/20（{m['score_mean']:.1%}）|{m['answered']}/20|{m['generation_seconds_mean']:.2f}|")
    lines += ['',f"百炼本轮实际请求{summary['cloud_requests']}次（包括可能的重试）；沿用原项目调用账本。用户已授权解除原100次限制，本地项目上限调整至1000。未查询云端账户余额或费用。",'',
              '## 逐题对照','','|题号|题型|Ollama得分|百炼得分|','|---|---|---:|---:|']
    for pair in pairs:
        lines.append(f"|{pair['id']}|{pair['category']}|{pair['ollama_score']}|{pair['bailian_score']}|")
    lines += ['','## 未得满分的回答','']
    for review in reviews:
        if review['score']<1:
            lines.append(f"- {review['provider']} / {review['id']}：{review['note']}")
    lines += ['','## 解释与限制','',
              '每题分数依据问题必要事实、数值、单位、范围和拒答规则逐题辅助评阅；不是独立人工盲评。引用编号合法不代表语义支持，所有错误回答原样保留。',
              '例如Ollama ext-05数值正确，答案得1分，但引用E1仅为图题，图表证据实际是E2。这里的答案分数不能冒充引用语义支持率。',
              '这两份新文档在本轮用于发现并修复图题关联问题，生成前冻结代码与证据，未依据回答改题；应称为外部新语料回归，不冒充完全未参与开发的盲测。样本只有20题，不能推断所有领域的总体质量。',
              '性能是本机单次运行观察值：本地模型已加载，云端含网络时间。未做统一冷启动与多次重复，不作严格速度排名。报纸小字图表与跨页段落碎片是已知难点。',
              '原始资料、来源URL及摘要见sources.json；题目与金标准见questions.json及DATASET.md；固定输入见frozen.json；原始回答、逐题评分、并排CSV均在本目录。']
    (OUT/'external/REPORT.md').write_text('\n'.join(lines)+'\n')
    highlights=read(OUT/'highlights/cases.json')
    citation=read(OUT/'citation/live.json')
    synthetic=read(OUT/'synthetic/ollama-answers.json')
    synthetic_reviews=read(OUT/'synthetic/reviews.json')
    assert len(highlights)==20 and all(r['visual_review']=='passed' for r in highlights)
    assert len(citation['rows'])==10 and citation['all_answered']
    assert synthetic['completed'] and len(synthetic['rows'])==12 and len(synthetic_reviews)==12
    local=summary['models']['ollama']
    cloud=summary['models']['bailian']
    overview=['# 尚缺验收项目补齐记录','',
              '检查日期：2026-09-11。本轮补齐上次列出的四类程序验收与实验材料；课程报告正文按用户要求暂缓。验收完成不代表任意文档或模型回答都正确。','',
              '|项目|结果|材料|','|---|---|---|',
              '|20例高亮定位|8文本、6表格、6图表，逐项视觉核查通过|[核查记录](highlights/REPORT.md)|',
              '|引用问题真实复测|7原失败、2疑似空回答、1图表，共10题有效返回；辅助核查10/10|[复测记录](citation/REPORT.md)|',
              '|合成语料真实生成|3文档12题，引用格式12/12，答案评分11/12|[生成记录](synthetic/REPORT.md)|',
              f"|外部真实语料与双模型|2份PDF、11页、20题×2模型；Ollama {local['score_sum']}/20，百炼 {cloud['score_sum']}/20；20对输入一致|[对照记录](external/REPORT.md)|",'',
              '本轮发现并修复轻量解析器图表复用页首无关文字的问题：按同栏空间距离关联文本行，避免同一PDF文字块合并左右两栏图题。新增双栏图题回归测试；全量76项通过，静态检查通过。默认Docling已解析语料和原有实验没有被重建覆盖。','',
              '本轮的外部语料是实际公开报纸版面与教育统计公报PDF，不是改写原题库或新造事实；新语料也参与了本轮解析器问题诊断，所以不称作严格盲测。原图小字、跨页信息遗漏和错误拒答均作为真实失败保留。20题不能证明通用智能或所有复杂版式能力。','',
              f"本轮百炼请求{summary['cloud_requests']}次，账本继续累积；用户授权不再受原100次项目限制，配置上限改为1000。密钥未写入材料，实际云端余额及账单未查询。",'',
              '原来的80题回答、7条历史失败、空白教师评分表和所有旧实验均保留；本轮定向成功不回填或篡改历史准确率。结果不出现在应用界面，供后续课程报告引用。','',
              '复核入口：external/questions.json、external/frozen.json、external/reviews.json、external/comparison.csv、summary.json；具体原文URL和文件摘要见external/sources.json。']
    (OUT/'REPORT.md').write_text('\n'.join(overview)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
