"""Audit paired visual ablations without assigning answer-quality scores."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean


def audit(result):
    rows = result["rows"]
    if result.get("status") != "ready":
        raise ValueError("实验尚未完成")
    by_key = {(r["group"], r["question_id"]): r for r in rows}
    questions = {r["question_id"] for r in rows}
    expected = {(g, q) for g in "DEF" for q in questions}
    if len(by_key) != len(rows) or set(by_key) != expected:
        raise ValueError("三组结果缺失、重复或包含额外组")
    def text_context(row):
        return [(c["label"], c["element_id"], c["text"], c["truncated"])
                for c in row["answer"]["context"]]

    checks = []
    for qid in sorted(questions):
        d, e, f = [by_key[g, qid] for g in "DEF"]
        check = {
            "question_id": qid,
            "de_same_retrieved": d["retrieved"] == e["retrieved"],
            "de_same_text": text_context(d) == text_context(e),
            "d_no_images": d["answer"]["image_count"] == 0,
            "same_model": len({r["answer"]["model"] for r in [d, e, f]}) == 1,
            "ef_same_retrieved": e["retrieved"] == f["retrieved"],
            "e_image_count": e["answer"]["image_count"],
            "f_image_count": f["answer"]["image_count"],
        }
        if not all(check[k] for k in ["de_same_retrieved", "de_same_text", "d_no_images", "same_model"]):
            raise ValueError(f"对照变量检查失败：{check}")
        checks.append(check)
    summaries = []
    for group in "DEF":
        selected = [r for r in rows if r["group"] == group]
        answerable = [r for r in selected if r["answerable"]]
        summaries.append({
            "group": group, "rows": len(selected),
            "status_counts": dict(Counter(r["answer"]["status"] for r in selected)),
            "recall_at_5": mean(r["recall@5"] for r in answerable),
            "mrr_at_10": mean(r["mrr@10"] for r in answerable),
            "rows_with_images": sum(r["answer"]["image_count"] > 0 for r in selected),
            "mean_seconds": mean(r["seconds"] for r in selected),
        })
    return {"checks": checks, "summary": summaries,
            "interpretation": "对照完整性和调用状态统计；不自动推断事实正确或引用语义支持。"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.result.read_bytes()
    output = {"result_sha256": hashlib.sha256(raw).hexdigest(), **audit(json.loads(raw))}
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))
