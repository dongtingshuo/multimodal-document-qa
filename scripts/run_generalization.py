"""Build and evaluate an unseen local document pack without cloud generation."""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import pymupdf as fitz

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "models/huggingface"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(ROOT))

from docqa.config import Settings
from docqa.evaluation import ExperimentRequest, run_experiment
from docqa.generation import RAGPipeline, create_generator
from docqa.parsers import LocalParser
from docqa.retrieval import HybridRetriever
from docqa.store import Store


def create_pdf(path, pages):
    """Create deterministic, text/table/chart fixtures in a separate corpus."""
    with fitz.open() as document:
        for page_spec in pages:
            page = document.new_page(width=720, height=960)
            page.insert_text((48, 58), page_spec["title"], fontsize=20, fontname="helv")
            y = 105
            for paragraph in page_spec.get("paragraphs", []):
                page.insert_textbox((48, y, 670, y + 48), paragraph, fontsize=12, fontname="helv")
                y += 62
            if page_spec.get("table"):
                table = page_spec["table"]
                x0, y0, cell_w, cell_h = 48, y + 15, 190, 30
                # Keep the table geometry simple so PyMuPDF's table detector can recover it.
                rows, cols = len(table), len(table[0])
                for row in range(rows + 1):
                    page.draw_line((x0, y0 + row * cell_h), (x0 + cols * cell_w, y0 + row * cell_h))
                for col in range(cols + 1):
                    page.draw_line((x0 + col * cell_w, y0), (x0 + col * cell_w, y0 + rows * cell_h))
                for row, values in enumerate(table):
                    for col, value in enumerate(values):
                        page.insert_text(
                            (x0 + col * cell_w + 8, y0 + row * cell_h + 20),
                            value,
                            fontsize=11,
                            fontname="helv",
                        )
                y = y0 + rows * cell_h + 28
            chart = page_spec.get("chart")
            if chart:
                chart_x, chart_y = 70, max(y, 450)
                page.insert_text((48, chart_y - 20), chart["caption"], fontsize=12, fontname="helv")
                page.draw_line((chart_x, chart_y + 180), (chart_x, chart_y))
                page.draw_line((chart_x, chart_y + 180), (chart_x + 520, chart_y + 180))
                max_value = max(item[1] for item in chart["bars"])
                for index, (label, value, color) in enumerate(chart["bars"]):
                    left = chart_x + 45 + index * 150
                    height = 145 * value / max_value
                    page.draw_rect(
                        (left, chart_y + 180 - height, left + 70, chart_y + 180),
                        fill=color,
                        color=color,
                    )
                    page.insert_text((left, chart_y + 205), label, fontsize=11, fontname="helv")
                    page.insert_text((left, chart_y + 165 - height), str(value), fontsize=11, fontname="helv")
        document.save(path)


def build_corpus(root):
    source_dir = root / "documents"
    source_dir.mkdir(parents=True, exist_ok=True)
    specs = {
        "library-2025": {
            "name": "Campus Library 2025 Brief.pdf",
            "pages": [
                {
                    "title": "Campus Library Annual Brief 2025",
                    "paragraphs": [
                        "The library recorded 128400 visits in 2025. Digital borrowing represented 46.5 percent of all borrowing.",
                        "The service desk operated from 08:00 to 22:00 on weekdays.",
                    ],
                    "table": [
                        ["Metric", "Value", "Unit"],
                        ["New books", "12600", "items"],
                        ["Study seats", "480", "seats"],
                        ["Satisfaction", "94.2", "percent"],
                    ],
                }
            ],
        },
        "energy-q2": {
            "name": "Community Energy Monitoring Q2 2026.pdf",
            "pages": [
                {
                    "title": "Community Energy Monitoring Q2 2026",
                    "paragraphs": [
                        "The report tracks rooftop solar generation and grid consumption for the second quarter.",
                        "All energy values in the table and chart are measured in megawatt-hours (MWh).",
                    ],
                    "table": [
                        ["Month", "Solar generation", "Grid consumption"],
                        ["April", "420", "610"],
                        ["May", "465", "590"],
                        ["June", "510", "560"],
                    ],
                    "chart": {
                        "caption": "Figure 1. Solar generation by month (MWh)",
                        "bars": [("April", 420, (0.20, 0.45, 0.85)), ("May", 465, (0.20, 0.65, 0.35)),
                                 ("June", 510, (0.85, 0.45, 0.20))],
                    },
                }
            ],
        },
        "lab-safety": {
            "name": "Laboratory Safety Review May 2026.pdf",
            "pages": [
                {
                    "title": "Laboratory Safety Review May 2026",
                    "paragraphs": [
                        "The emergency evacuation drill was completed on 12 May 2026.",
                        "Inspection coverage reached 98 percent across the laboratory rooms.",
                        "Two corrective actions were closed before the review was signed off.",
                    ],
                }
            ],
        },
    }
    documents = []
    for doc_id, spec in specs.items():
        path = source_dir / spec["name"]
        create_pdf(path, spec["pages"])
        content = path.read_bytes()
        documents.append({"id": doc_id, "name": spec["name"], "path": path, "sha256": hashlib.sha256(content).hexdigest()})
    return documents


def choose_element(elements, *needles, kind=None):
    for element in elements:
        if kind is not None and element.kind != kind:
            continue
        body = element.text.casefold()
        if all(needle.casefold() in body for needle in needles):
            return element.id
    raise RuntimeError(f"未找到留出语料金标准元素：{needles!r} kind={kind!r}")


def make_questions(parsed):
    by_doc = {doc_id: result["elements"] for doc_id, result in parsed.items()}
    library = by_doc["library-2025"]
    energy = by_doc["energy-q2"]
    lab = by_doc["lab-safety"]

    def q(qid, question, doc_id, anchor, category, reference, points):
        return {
            "id": qid,
            "question": question,
            "document_ids": [doc_id],
            "relevant_element_ids": [anchor] if anchor else [],
            "category": category,
            "split": "test",
            "answerable": anchor is not None,
            "reference_answer": reference,
            "fact_group": qid,
            "scoring_points": points,
        }

    return [
        q("gen-001", "How many visits did the library record in 2025?", "library-2025",
          choose_element(library, "128400"), "text", "128400 visits", ["128400"]),
        q("gen-002", "What percentage of borrowing was digital?", "library-2025",
          choose_element(library, "46.5"), "text", "46.5 percent", ["46.5"]),
        q("gen-003", "How many study seats are listed in the library table?", "library-2025",
          choose_element(library, "Study seats", "480", kind="table"), "table", "480 seats", ["480"]),
        q("gen-004", "What satisfaction percentage is shown in the library table?", "library-2025",
          choose_element(library, "Satisfaction", "94.2", kind="table"), "table", "94.2 percent", ["94.2"]),
        q("gen-005", "Which month had the highest solar generation?", "energy-q2",
          choose_element(energy, "June", "510", kind="table"), "table", "June, with 510 MWh", ["June", "510"]),
        q("gen-006", "What was the grid consumption in June?", "energy-q2",
          choose_element(energy, "June", "560", kind="table"), "table", "560 MWh", ["560"]),
        q("gen-007", "How much did solar generation increase from April to June?", "energy-q2",
          choose_element(energy, "April", "420", kind="table"), "table", "90 MWh", ["510 - 420 = 90"]),
        q("gen-008", "In the solar generation chart, which bar is tallest?", "energy-q2",
          choose_element(energy, "Figure 1", kind="image"), "chart", "June is the tallest bar (510 MWh)", ["June", "tallest"]),
        q("gen-009", "When was the emergency evacuation drill completed?", "lab-safety",
          choose_element(lab, "12 May 2026"), "text", "12 May 2026", ["date"]),
        q("gen-010", "What inspection coverage did the laboratory reach?", "lab-safety",
          choose_element(lab, "98 percent"), "text", "98 percent", ["98"]),
        q("gen-011", "How many corrective actions were closed?", "lab-safety",
          choose_element(lab, "Two corrective actions"), "text", "Two", ["two"]),
        q("gen-012", "What was the laboratory's total budget in 2025?", "lab-safety", None,
          "unanswerable", "文档未提供实验室 2025 年预算。", []),
    ]


def write_report(path, result):
    def metric(value):
        return "—" if value is None else f"{value:.3f}"

    summary = result["summary"][0]
    category_rows = result["category_summary"]
    lines = [
        "# 新语料泛化回归",
        "",
        "本目录使用 3 份未进入原公开题库的本地构造文档，覆盖文本、表格、图形和不可回答问题。结果用于工程回归，不等同于外部真实世界泛化准确率。",
        "",
        f"- 题目数：{summary['questions']}（可回答 {summary['answerable_questions']}）",
        f"- Recall@1：{metric(summary['recall@1'])}",
        f"- Recall@3：{metric(summary['recall@3'])}",
        f"- Recall@5：{metric(summary['recall@5'])}",
        f"- MRR@10：{metric(summary['mrr@10'])}",
        "",
        "## 按题型",
        "",
        "| 题型 | 题数 | Recall@1 | Recall@5 | MRR@10 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in category_rows:
        lines.append(
            f"| {row['category']} | {row['questions']} | {metric(row['recall@1'])} | "
            f"{metric(row['recall@5'])} | {metric(row['mrr@10'])} |"
        )
    lines.extend([
        "",
        "## 解释",
        "",
        "D 组使用 BM25、向量检索、RRF 和重排器。图形题保留图形元素作为证据，表格题保留表头和数值行。不可回答题不计入 Recall/MRR。",
        "",
        f"生成测试：{'已请求本地 Ollama' if result['provenance'].get('generation_requested') else '未请求；本次为纯检索回归'}。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(output, generate_ollama=False):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / "snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    documents = build_corpus(output)
    settings = Settings(
        data_dir=snapshot,
        parser="pymupdf",
        enable_ocr=False,
        enable_clip=False,
        local_models_only=True,
        generation_provider="ollama",
        api_key="",
        max_api_calls=0,
    )
    store = Store(snapshot)
    parser = LocalParser(settings)
    parsed = {}
    for doc in documents:
        parsed_result = parser.parse(doc["path"], doc["id"], snapshot / "parsed" / doc["id"])
        parsed[doc["id"]] = parsed_result
        store.put_document({
            "id": doc["id"], "name": doc["name"], "sha256": doc["sha256"], "status": "ready",
            "original_path": str(doc["path"]), "parser_version": parsed_result["parser_version"],
            "page_count": parsed_result["page_count"], "converted": parsed_result["converted"],
            "element_count": len(parsed_result["elements"]), "warnings": parsed_result["warnings"],
        })
        store.replace_elements(doc["id"], parsed_result["elements"])
    questions = make_questions(parsed)
    dataset = {"scope": "new-local-heldout", "document_ids": [doc["id"] for doc in documents], "questions": questions}
    dataset_path = output / "questions.json"
    dataset_path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    retriever = HybridRetriever(store, settings)
    pipeline = None
    if generate_ollama:
        import httpx

        response = httpx.get(settings.ollama_base.rstrip("/") + "/tags", timeout=10)
        response.raise_for_status()
        names = {model["name"] for model in response.json().get("models", [])}
        if settings.ollama_model not in names:
            raise RuntimeError(f"Ollama 未找到 {settings.ollama_model}，只完成检索泛化测试")
        pipeline = RAGPipeline(retriever, create_generator(settings, store), store, settings)
    request = ExperimentRequest(
        groups=["D"], questions=questions, document_ids=dataset["document_ids"], split="test",
        generate=generate_ollama, expected_provider="ollama" if generate_ollama else None,
    )
    started = time.monotonic()
    result = run_experiment(request, retriever, store, pipeline)
    result["provenance"]["scope"] = "新构造留出语料；文档、事实和题目均未进入原公开题库"
    result["provenance"]["generation_requested"] = generate_ollama
    result["provenance"]["source_hashes"] = {doc["id"]: doc["sha256"] for doc in documents}
    result["notes"].extend([
        "这是本地新构造文档的工程泛化回归，验证未见内容、表格、图形和不可回答问题；不等同于外部真实世界泛化准确率。",
        f"运行耗时 {time.monotonic() - started:.1f} 秒；仅使用本地索引" + ("和 Ollama 生成。" if generate_ollama else "。"),
    ])
    result_path = output / ("ollama-result.json" if generate_ollama else "retrieval-result.json")
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output / ("OLLAMA-REPORT.md" if generate_ollama else "REPORT.md"), result)
    print(json.dumps({"result": str(result_path), "questions": len(questions), "summary": result["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument("--output", type=Path, required=True)
    arguments.add_argument("--generate-ollama", action="store_true")
    options = arguments.parse_args()
    main(options.output, options.generate_ollama)
