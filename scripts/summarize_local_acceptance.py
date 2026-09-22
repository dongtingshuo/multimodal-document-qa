"""Verify archived regression records and export report material without model calls."""

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from docqa.evaluation import retrieval_metrics, summarize


def verify_run(run, questions, split, groups, generate):
    expected = {(g, q["id"]) for g in groups for q in questions if q["split"] == split}
    rows = run["rows"]
    if run["status"] != "ready" or len(rows) != len(expected):
        raise ValueError("实验未完成或记录数量错误")
    if run.get("completed") != len(rows) or run.get("total") != len(expected):
        raise ValueError("进度计数与实际记录不一致")
    if {(r["group"], r["question_id"]) for r in rows} != expected:
        raise ValueError("题目与组别缺失或重复")
    if run["config"]["generate"] != generate or run["config"]["split"] != split:
        raise ValueError("实验类型或划分错误")
    for row in rows:
        for name, value in retrieval_metrics(row["retrieved"], row["gold"]).items():
            if row[name] != value:
                raise ValueError("逐题召回列表与指标不一致")
        if generate and row.get("answer", {}).get("model") != "qwen2.5vl:7b":
            # A no-evidence response can bypass the generator.
            if row.get("answer", {}).get("status") != "no_evidence":
                raise ValueError("存在非指定本地模型的回答")
    if run["summary"] != summarize(rows):
        raise ValueError("汇总与逐题记录不一致")
    if run["category_summary"] != summarize(rows, by_category=True):
        raise ValueError("分类汇总与逐题记录不一致")
    if "D" in groups and "E" in groups:
        left = {r["question_id"]: r["retrieved"] for r in rows if r["group"] == "D"}
        right = {r["question_id"]: r["retrieved"] for r in rows if r["group"] == "E"}
        if left != right:
            raise ValueError("D/E 固定检索对照不一致")


def plot_retrieval(directory, baseline, current):
    os.environ.setdefault("MPLCONFIGDIR", str(directory.resolve() / ".matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(directory.resolve() / ".cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    old = {r["group"]: r for r in baseline["summary"]}
    new = {r["group"]: r for r in current["summary"]}
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2), layout="constrained")
    for ax, metric in zip(axes, ["recall@5", "mrr@10"], strict=True):
        for offset, records, label, color in [(-0.18, old, "Original baseline", "#a2a9b4"),
                                               (0.18, new, "Current regression", "#265c97")]:
            bars = ax.barh([i + offset for i in range(6)], [records[g][metric] for g in "ABCDEF"],
                           height=0.31, color=color, label=label)
            ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=3)
        ax.set(yticks=list(range(6)), yticklabels=list("ABCDEF"), xlim=(0, 1.13),
               xticks=[0, 0.25, 0.5, 0.75, 1], xlabel=metric, ylabel="Group")
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.15)
        ax.set_axisbelow(True)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.55, 1.12), ncols=2, fontsize=9, frameon=False)
    fig.suptitle("Fixed-corpus retrieval regression | 67 answerable questions", fontsize=12)
    fig.supxlabel("Known questions; retrieval scores do not measure answer accuracy.", fontsize=9)
    for extension in ["png", "pdf"]:
        fig.savefig(directory / f"retrieval-regression.{extension}", dpi=240)
    plt.close(fig)


def build_report(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "run.json").read_text())
    questions = json.loads((directory / "questions.json").read_text())["questions"]
    runs = {}
    for label, split, groups, generate in [
        ("retrieval-dev", "dev", list("ABCDEF"), False),
        ("retrieval-test", "test", list("ABCDEF"), False),
        ("ollama-test", "test", ["E"], True),
    ]:
        path = directory / label / (manifest["runs"][label] + ".json")
        runs[label] = json.loads(path.read_text())
        verify_run(runs[label], questions, split, groups, generate)
    for label in ["retrieval-test", "ollama-test"]:
        for key in ["code_sha256", "elements_sha256", "models", "prompt_sha256", "documents", "packages"]:
            if runs[label]["provenance"][key] != runs["retrieval-dev"]["provenance"][key]:
                raise ValueError(f"不同阶段配置发生变化：{label} / {key}")

    lines = [
        "# 当前版本完整回归与 Ollama 回答验收",
        "",
        "本报告由逐题结果重新核算生成。包含开发集 20 题 × 六组、测试划分 80 题 × 六组，及 Ollama E 组 80 条回答记录。",
        "",
        "本轮使用既有固定题集，部分测试题曾用于调试，因此是完整回归，不是全新独立测试。人工评分与模型调用成功分开统计。",
        "",
        "## 本轮检索结果",
        "",
        "| 划分 | 组别 | 可回答题 | Recall@5 | Recall@10 | MRR@10 | 平均秒/题 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for label in ["retrieval-dev", "retrieval-test"]:
        for row in runs[label]["summary"]:
            lines.append(f"| {label} | {row['group']} | {row['answerable_questions']} | "
                         f"{row['recall@5']:.3f} | {row['recall@10']:.3f} | {row['mrr@10']:.3f} | {row['seconds']:.2f} |")
    lines += ["", "耗时包含模型加载、索引和独立检索进程开销；开发集曾暂停续跑，不能作为严格速度排名。", "",
              "## 与原始测试基线比较", "",
              "| 组别 | 原 Recall@5 | 本轮 Recall@5 | 原 MRR@10 | 本轮 MRR@10 |", "|---|---:|---:|---:|---:|"]
    baseline = json.loads((ROOT / "data/public-evaluation/test/d170e5e1291a457280a16323de7131f6.json").read_text())
    if baseline["config"]["questions"] != runs["retrieval-test"]["config"]["questions"]:
        raise ValueError("基线题集与本轮题集不同，不能直接列为同题比较")
    if baseline["provenance"]["elements_sha256"] != runs["retrieval-test"]["provenance"]["elements_sha256"]:
        raise ValueError("基线解析元素与本轮不同，请单独说明后再比较")
    old = {r["group"]: r for r in baseline["summary"]}
    for row in runs["retrieval-test"]["summary"]:
        before = old[row["group"]]
        lines.append(f"| {row['group']} | {before['recall@5']:.3f} | {row['recall@5']:.3f} | "
                     f"{before['mrr@10']:.3f} | {row['mrr@10']:.3f} |")
    plot_retrieval(directory, baseline, runs["retrieval-test"])
    lines += ["", "![固定题集检索回归](retrieval-regression.png)", "",
              "图中对比的是原始基线与当前版本在相同题集上的检索表现，D/E 的差别不在检索阶段。"]
    lines += ["", "基线代码与当前代码不同，题集包含已知失败题。差异用于回归分析，不能据此宣称对未见文档的泛化提升。原基线文件未改写。", "",
              "## 检索失败案例", "",
              "| 组别 | 题目 | Recall@5 | Recall@10 |", "|---|---|---:|---:|"]
    misses = [{key: row[key] for key in ["group", "question_id", "question", "reference_answer", "gold", "retrieved", "recall@5", "recall@10"]}
              for row in runs["retrieval-test"]["rows"] if row["answerable"] and row["recall@5"] < 1]
    (directory / "retrieval-misses.json").write_text(json.dumps(misses, ensure_ascii=False, indent=2))
    for row in misses:
        if row["group"] in {"D", "F"}:
            lines.append(f"| {row['group']} | {row['question_id']}：{row['question']} | {row['recall@5']:.3f} | {row['recall@10']:.3f} |")
    assisted_path = directory / "reviews" / "assisted-review.csv"
    assisted = []
    scored_by_category = Counter()
    if assisted_path.is_file():
        import csv

        with assisted_path.open(encoding="utf-8-sig", newline="") as handle:
            assisted = list(csv.DictReader(handle))
        scored_by_category.update(
            row["category"] for row in assisted if row.get("answer_accuracy", "") != ""
        )
    lines += ["", "完整六组未命中列表见 retrieval-misses.json。保留所有失败题，不据此修改本轮题集或冻结的代码。", "",
              "## 本地回答验收", "", "| 题型 | 回答记录 | 状态分布 | 实际图片总数 | 辅助评阅数 |",
              "|---|---:|---|---:|---:|"]
    generated = runs["ollama-test"]
    for category in ["text", "table", "chart", "unanswerable"]:
        rows = [r for r in generated["rows"] if r["category"] == category]
        status = dict(Counter(r["answer"]["status"] for r in rows))
        lines.append(f"| {category} | {len(rows)} | {status} | "
                     f"{sum(r['answer'].get('image_count', 0) for r in rows)} | {scored_by_category[category]} |")
    failures = [{"question_id": r["question_id"], "status": r["answer"]["status"],
                 "text": r["answer"]["text"], "warnings": r["answer"].get("warnings", [])}
                for r in generated["rows"] if r["answer"]["status"] in {"generation_error", "citation_error"}]
    (directory / "generation-failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2))
    empty_answers = [r["question_id"] for r in generated["rows"]
                     if not re.sub(r"\[(?:E\d+|证据编号|证据内容)\]|[\s，。,:：;；]", "", r["answer"]["text"])]
    attempts = sum(sum(key.startswith("generation_") for key in r["answer"].get("usage", {}))
                   for r in generated["rows"])
    times = [r["seconds"] for r in generated["rows"]]
    lines += ["", f"全题型状态合计：{dict(Counter(r['answer']['status'] for r in generated['rows']))}。",
              f"记录了 {attempts} 次有用量响应的生成请求（包含引用修正）；没有响应的底层重试不包含在此计数中。",
              f"逐题总耗时均值 {statistics.mean(times):.2f} 秒，中位数 {statistics.median(times):.2f} 秒，最大 {max(times):.2f} 秒。",
              "", "去除引用编号、空白及常见字段标记后没有正文的疑似空回答：" + (", ".join(empty_answers) or "无") + "。",
              "此文本检查仅用于定位问题，不是自动答案评分；原始状态与回答保持不变。"]
    if assisted:
        scored_assisted = [row for row in assisted if row.get("answer_accuracy", "") != ""]
        lines += ["", f"项目辅助评阅已记录 {len(scored_assisted)} 条可评分回答，总分 "
                  f"{sum(float(row['answer_accuracy']) for row in scored_assisted):.1f}/{len(scored_assisted)}；"
                  f"{len(assisted) - len(scored_assisted)} 条生成或引用校验失败单列。",
                  "原空白 human-review.csv 和盲评材料继续保留，辅助评阅结果不覆盖原始回答。"]
    lines += ["", "状态 answered 只表示请求完成且引用编号检查通过，不代表事实正确；生成失败也不算正确拒答。", "",
              "正式教师盲评仍可使用 reviews/human-review.csv；该表含参考答案，blind.json 不含答案键。", "",
              f"此前已有生成记录的测试题共 {len(manifest['prior_generated_test_ids'])} 道：" + ", ".join(manifest["prior_generated_test_ids"]),
              "", "此数量仅依据归档生成记录，未覆盖对话中所有人工检查；其余题目也不能自动称为严格未见题。", "",
              "## 复现与文件完整性", "",
              f"检索版本：`{generated['provenance']['retrieval_version']}`。仅调用本地 Ollama，验收进程未配置云端密钥，云端预算为零。", "",
              "逐题指标重算、题目覆盖、D/E 相同检索、阶段间代码/模型/证据一致性检查均通过。"]
    lines += ["引用格式修复是在本地回答实验冻结后加入的；完整回答 JSON 代表冻结前的可复现实验，修复后的代码摘要另存于 `runtime_manifest-postfix.json`。", ""]
    hashes = {}
    for label in runs:
        relative = Path(label) / (manifest["runs"][label] + ".json")
        hashes[str(relative)] = hashlib.sha256((directory / relative).read_bytes()).hexdigest()
        lines.append(f"\n- [{label}]({relative.as_posix()})；SHA256：`{hashes[str(relative)]}`。")
    (directory / "verified-files.json").write_text(json.dumps(hashes, indent=2))
    report = directory / "REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    review_lines = ["# 逐题人工核查材料", "",
                    "含参考答案，不属于盲评文件。正式评分填写 human-review.csv，不要修改原始回答。",
                    "有答案题：错误或拒答 0，部分正确 0.5，正确 1；另核查拒答与引用是否支持。",
                    "调用失败单列，不评分为正确拒答。所有分数等待人工填写。"]
    for row in generated["rows"]:
        answer = row["answer"]
        review_lines += ["", f"## {row['question_id']} · {row['category']}", "", row["question"], "",
                         f"**状态**：{answer['status']}；**原图输入数**：{answer.get('image_count', 0)}", "",
                         "**模型原始回答**", "", answer["text"], "", "**参考答案**", "",
                         row.get("reference_answer", ""), "", "**评分要点**", ""]
        review_lines.extend("- " + point for point in row.get("scoring_points", []))
        review_lines += ["", "**实际入选证据**", ""]
        for evidence in answer.get("evidence", []):
            element = evidence["element"]
            locations = "; ".join(f"第 {s['page']} 页，bbox={s['bbox']}" for s in element["sources"])
            review_lines += [f"- {evidence['document_name']}；{locations}；元素 {element['id']}"]
            if element.get("image_path"):
                review_lines.append(f"  [查看证据原图](<{element['image_path']}>)")
        review_lines += ["", "完整输入文字与引用映射请核对本实验 JSON 中的 context/citations；不得仅凭编号存在判断支持性。"]
    (directory / "reviews" / "READING.md").write_text("\n".join(review_lines), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(build_report(parser.parse_args().directory))
