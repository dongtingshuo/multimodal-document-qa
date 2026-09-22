"""Export a retrieval report from two completed, frozen experiment snapshots."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
METRICS = ["recall@1", "recall@3", "recall@5", "recall@10", "mrr@10", "seconds"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(rows, category=False):
    headings = (
        ["组别"]
        + (["题型"] if category else [])
        + ["可回答题", "R@1", "R@3", "R@5", "R@10", "MRR@10", "秒/题"]
    )
    lines = ["| " + " | ".join(headings) + " |", "|" + "---|" * len(headings)]
    for row in rows:
        if not row["answerable_questions"]:
            continue
        cells = [row["group"]] + ([row["category"]] if category else []) + [str(row["answerable_questions"])]
        cells += [f"{row[m]:.3f}" if row[m] is not None else "—" for m in METRICS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    args = parser.parse_args()
    runs = {s: json.loads(p.read_text()) for s, p in [("dev", args.dev), ("test", args.test)]}
    for split, run in runs.items():
        assert run["status"] == "ready" and run["config"]["split"] == split
        assert run["config"]["generate"] is False, "本报告仅适用于本地检索实验"
        expected = 20 if split == "dev" else 80
        assert len(run["rows"]) == expected * 6
        assert Counter(r["group"] for r in run["rows"]) == dict.fromkeys("ABCDEF", expected)
        assert len({(r["group"], r["question_id"]) for r in run["rows"]}) == expected * 6
        assert all(r["answer_accuracy"] is None and "answer" not in r for r in run["rows"])
    for key in ["elements_sha256", "models", "context_char_budget", "prompt_sha256", "documents"]:
        assert runs["dev"]["provenance"][key] == runs["test"]["provenance"][key], key
    dev_ids = {r["question_id"] for r in runs["dev"]["rows"]}
    assert not dev_ids.intersection(r["question_id"] for r in runs["test"]["rows"])

    out = ROOT / "data/public-evaluation"
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for split, run in runs.items():
        records.extend({"split": split, **row} for row in run["summary"])
    with (out / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    misses = [
        {
            "split": split,
            "group": r["group"],
            "question_id": r["question_id"],
            "category": r["category"],
            "question": r["question"],
            "reference_answer": r["reference_answer"],
            "gold": json.dumps(r["gold"]),
            "retrieved": json.dumps(r["retrieved"]),
        }
        for split, run in runs.items()
        for r in run["rows"]
        if r["answerable"] and r["recall@10"] < 1
    ]
    (out / "misses.json").write_text(json.dumps(misses, ensure_ascii=False, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    for ax, (split, run) in zip(axes, runs.items(), strict=True):
        xs = list(range(6))
        for offset, metric, color in [(-0.18, "recall@5", "#2369a0"), (0.18, "mrr@10", "#de8c3a")]:
            bars = ax.bar(
                [x + offset for x in xs],
                [r[metric] for r in run["summary"]],
                width=0.34,
                color=color,
                label=metric,
            )
            ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=3)
        ax.set(
            xticks=xs,
            xticklabels=list("ABCDEF"),
            ylim=(0, 1.12),
            ylabel="Score",
            title=f"{split.upper()} ({run['summary'][0]['answerable_questions']} answerable questions)",
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", ncols=2, fontsize=8)
    fig.savefig(out / "retrieval-results.png", dpi=180)
    fig.savefig(out / "retrieval-results.pdf")
    plt.close(fig)

    lines = [
        "# 公开报告检索实验结果",
        "\n本报告从完成的逐题 JSON 自动汇总。共 20 份公开报告、285 页、100 道问题，六组各运行一次，共 600 条检索记录。未调用云端生成；答案准确率、拒答率和引用语义支持率均未验收。",
        "\n## 数据与方法",
        "\n文本 40、表格 25、图表 20、无答案 15。开发集 20 题（18 题可回答）；测试集 80 题（67 题可回答）。按元素 ID 评价证据命中，无答案题排除在 Recall/MRR 分母之外。划分不是分层随机抽样，两个集合的总分不适合直接比较难度或进步。",
        "\nA 为 BM25，B 为 Dense，C 为 RRF 混合，D 为混合加重排序，E 复用 D 检索并预留原图生成，F 增加 CLIP 图像召回后融合重排序。关闭生成时 D/E 是重复检索对照，不能证明图像输入对回答的作用。所有组使用相同文档范围与文本切分；参数在测试集运行前冻结。",
        "\n实现参数：400 token 内容切块、60 token 重叠，另加最多 80 token 文档名与章节元数据；RRF 常数 60；检索前 10 个原始元素评测。生成参数虽已记录，本轮未使用：前 6 条证据、最多 3 张原图、6,000 字符证据预算、温度 0。",
        "\n## 总体指标",
        "\n所有分值取值 0–1，保留三位小数。秒/题为该组全部问题的平均耗时，包括无答案题；它包含加载、索引和进程开销，不能作为严格的稳态性能基准。开发集先运行，测试集复用模型和索引缓存。",
    ]
    for split, run in runs.items():
        lines += [f"\n### {split}\n", table(run["summary"])]
    lines += [
        "\n![检索对比](../data/public-evaluation/retrieval-results.png)",
        "\n## 测试集分题型指标\n",
        table(runs["test"]["category_summary"], True),
    ]
    lines += [
        "\n## 解释边界与错误记录",
        "\n19 份月度空气质量报告结构高度相似，日期和报告名容易混淆；标准表在不同月报重复。单个目标元素未命中并不总等于事实缺失，某些等价表格可能位于另一份报告，本轮采用严格来源匹配。图表题全部来自 1 份 CNNIC 报告，且图题可被文本召回，因此不能据此评价所有视觉问题或证明 CLIP 普遍有效。",
        "\n所有 Recall@10 未完全命中的记录保留在 [misses.json](../data/public-evaluation/misses.json)，含问题、参考答案、目标元素和实际召回列表，未删除失败题或按测试结果修改标注。后续改善需先用开发集设计，再使用新的独立测试题验证。",
        "\n答案准确率、正确拒答率、错误拒答率、引用语义支持率目前为空。模型 API 的真实鉴权、图像理解、费用及多模态回答效果，必须在用户配置并确认云端调用后单独测试；模拟接口功能测试不能代替这些结论。",
        "\n## 原始证据与复现",
        "\n题集与流程见 [evaluation.md](evaluation.md)。模型权重、配置和依赖摘要见 [runtime_manifest.json](../datasets/runtime_manifest.json)。解析权重尚未完全锁定，已归档当前解析元素及其摘要。",
    ]
    for split, path in [("dev", args.dev), ("test", args.test)]:
        relative = path.resolve().relative_to(ROOT)
        lines.append(f"\n- {split}：[{runs[split]['id']}](../{relative})；文件 SHA256：`{digest(path)}`。")
    lines += [
        f"\n元素集合摘要：`{runs['test']['provenance']['elements_sha256']}`。",
        "\n汇总数据：[metrics.csv](../data/public-evaluation/metrics.csv)。矢量图：[retrieval-results.pdf](../data/public-evaluation/retrieval-results.pdf)。",
    ]
    (ROOT / "docs/EXPERIMENT_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"rows": 600, "misses": len(misses), "report": "docs/EXPERIMENT_RESULTS.md"}))


if __name__ == "__main__":
    main()
