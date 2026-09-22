"""Create a transparent, rule-assisted answer review from an archived run."""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from docqa.generation import normalize_citation_format, validate_citations

# These decisions were reviewed for one archived answer payload only.
REVIEWED_SOURCE_SHA256 = "fbdd7deb563d3015b9b4af923fde7f60ca2c7780af29f6508402f0f6bdee6022"

# Scores are based on the frozen reference answer and the answer text.  A
# partial score is used when a multi-part value has only some required facts.
SCORES = {
    "public-016": (0, False, "PM2.5 数值为 54.5，参考值为 44.6。"),
    "public-020": (0, False, "PM2.5 数值为 26.7，参考值为 23.5。"),
    "public-031": (0, False, "只有证据编号，没有答案正文。"),
    "public-037": (0, False, "只有证据编号，没有答案正文。"),
    "public-051": (0.5, False, "二级值 10 正确，但一级值误答为 4；单位和表格定位也需复核。"),
    "public-064": (0.5, True, "使用 10.81 亿进行了四舍五入，未给出题目要求的精确 108,133 万人；使用率 97.6% 正确。"),
    "public-058": (0, False, "参考表中一级、二级均为 200；回答误写一级为 40 并计算为 5:1。"),
    "public-059": (0.5, False, "两个数值正确，但把表格的 mg/m³ 单位写成了 μg/m³。"),
    "public-075": (0, False, "将有依据的图表问题错误判为没有足够依据。"),
    "public-082": (0, False, "最高年份、最高值和差值均与参考答案不一致。"),
    "public-083": (0.5, False, "熟练掌握 3.6% 正确，但基本掌握和合计值误答。"),
    "public-084": (0, False, "将有依据的图表问题错误判为没有足够依据。"),
}


def is_refusal(text, answerable):
    if answerable:
        return False
    return any(phrase in text for phrase in ("没有足够依据", "未提供", "无法回答", "未找到"))


def score_row(row):
    qid, answer = row["question_id"], row["answer"]
    text = answer["text"]
    normalized = normalize_citation_format(text)
    if answer.get("status") in {"generation_error", "citation_error"}:
        return None, False, None, "生成或引用校验失败，按评测规则单列，不纳入回答得分。", normalized
    if qid in SCORES:
        score, supported, note = SCORES[qid]
        refusal = is_refusal(text, row["answerable"])
    elif not row["answerable"]:
        refusal = is_refusal(text, False)
        score = 1 if refusal else 0
        supported = refusal
        note = "明确拒答，文档未提供题目所问信息。" if refusal else "未明确拒答。"
    else:
        refusal = False
        score = 1
        supported = True
        note = "核心答案与参考答案一致，入选证据可定位。"
    # Cite syntax is checked independently.  Explicit labels such as
    # “[证据编号] E1” are normalized for review but the original text remains.
    labels = {f"E{i}" for i in range(1, len(answer.get("evidence", [])) + 1)}
    refs, invalid, missing = validate_citations(normalized, labels)
    if row["answerable"] and not refs:
        supported = False
        note += " 未发现可核验的证据编号。"
    if invalid:
        supported = False
        note += " 存在无效证据编号。"
    if missing and row["answerable"]:
        score = 0
        supported = False
        note += " 回答正文为空或不足。"
    return score, refusal, supported, note, normalized


def write_csv(rows, output, overwrite=False):
    fields = ["experiment_id", "group", "question_id", "status", "category", "answerable",
              "answer_sha256", "question", "answer", "normalized_answer", "reference_answer",
              "reviewer", "answer_accuracy", "is_refusal", "citation_supported", "note"]
    if output.exists() and not overwrite:
        raise FileExistsError(f"评分输出已存在：{output}；如需重算请加 --overwrite")
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary(path, metadata):
    def metric(value):
        return "—" if value is None else f"{value:.3f}"

    lines = [
        "# 回答辅助评阅摘要",
        "",
        "本文件记录项目验收回答的辅助评阅结果，供后续评测分析使用；原始回答和空白人工评分表均保持不变。",
        "",
        f"- 回答总数：{metadata['rows']}（纳入得分 {metadata['scored_rows']}，单列未评分 {metadata['unscored_rows']}）",
        f"- 总分：{metadata['score_sum']:.1f} / {metadata['scored_rows']}",
        f"- 纳入评分平均得分：{metric(metadata['mean_answer_accuracy'])}",
        f"- 可回答题平均得分：{metric(metadata['answerable_mean'])}",
        f"- 明确拒答数：{metadata['refusals']}",
        f"- 引用可核验数：{metadata['citation_supported']}",
        "",
        "## 题型平均得分",
        "",
        "| 题型 | 平均得分 |",
        "|---|---:|",
    ]
    lines.extend(f"| {category} | {value:.3f} |" for category, value in metadata["category_mean"].items())
    lines.extend([
        "",
        "评分尺度为 0、0.5、1：数值或多项事实只完成一部分时记 0.5；明确拒答、证据引用和事实正确性分别记录。",
        "评分依据是冻结参考答案、评分要点、原始回答和入选证据；这份结果用于工程验收，不能替代教师的独立盲评。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(source, output, overwrite=False):
    source_bytes = source.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != REVIEWED_SOURCE_SHA256:
        raise ValueError("既有评分仅适用于已核对的历史回答；新实验必须逐题重新评阅，不能套用旧分数")
    result = json.loads(source_bytes)
    if result.get("status") != "ready":
        raise ValueError("评分源实验必须已经完成")
    scored = []
    for row in result["rows"]:
        score, refusal, supported, note, normalized = score_row(row)
        answer_bytes = json.dumps(row["answer"], ensure_ascii=False, sort_keys=True).encode()
        scored.append({
            "experiment_id": result["id"], "group": row["group"], "question_id": row["question_id"],
            "status": row["answer"]["status"], "category": row["category"], "answerable": int(row["answerable"]),
            "answer_sha256": hashlib.sha256(answer_bytes).hexdigest(), "question": row["question"],
            "answer": row["answer"]["text"], "normalized_answer": normalized,
            "reference_answer": row["reference_answer"], "reviewer": "项目评阅",
            "answer_accuracy": "" if score is None else score, "is_refusal": int(refusal),
            "citation_supported": "" if supported is None else int(supported), "note": note,
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(scored, output, overwrite=overwrite)
    summary = Counter()
    by_category = {}
    for row in scored:
        summary["rows"] += 1
        if row["answer_accuracy"] != "":
            summary["score_sum"] += row["answer_accuracy"]
            summary["scored_rows"] += 1
        else:
            summary["unscored_rows"] += 1
        summary["refusals"] += row["is_refusal"]
        if row["citation_supported"] != "":
            summary["citation_supported"] += row["citation_supported"]
        if row["answer_accuracy"] != "":
            by_category.setdefault(row["category"], []).append(row["answer_accuracy"])
    scored_rows = summary["scored_rows"]
    answerable_scores = [
        r["answer_accuracy"] for r in scored if r["answerable"] and r["answer_accuracy"] != ""
    ]
    metadata = {
        "evaluator": "项目辅助评阅", "source_result": str(source),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(), "scoring_scale": [0, 0.5, 1],
        "scope": "基于参考答案、评分要点、原始回答和入选证据的规则辅助评阅；不替代教师人工盲评",
        "rows": summary["rows"], "score_sum": summary["score_sum"],
        "scored_rows": scored_rows, "unscored_rows": summary["unscored_rows"],
        "mean_answer_accuracy": summary["score_sum"] / scored_rows if scored_rows else None,
        "answerable_mean": sum(answerable_scores) / len(answerable_scores) if answerable_scores else None,
        "refusals": summary["refusals"], "citation_supported": summary["citation_supported"],
        "status_distribution": dict(Counter(r["status"] for r in scored)),
        "category_mean": {k: sum(v) / len(v) for k, v in by_category.items()},
        "score_distribution": dict(Counter(str(r["answer_accuracy"]) for r in scored if r["answer_accuracy"] != "")),
        "overrides": SCORES,
    }
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary(output.with_name(output.stem + "-summary.md"), metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true", help="覆盖同一评分输出，不会修改评分源")
    args = parser.parse_args()
    main(args.source, args.output, args.overwrite)
