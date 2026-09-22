"""Replay citation syntax and optionally rerun failures plus empty-answer/chart controls."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from docqa.config import Settings
from docqa.generation import RAGPipeline, create_generator, normalize_citation_format, validate_citations
from docqa.schemas import ChatRequest, Evidence
from docqa.store import Store


def check(source_path, output, execute=False):
    source_bytes = source_path.read_bytes()
    source = json.loads(source_bytes)
    if source["status"] != "ready":
        raise ValueError("只能核查已经完成并归档的实验")
    output.mkdir(parents=True, exist_ok=False)
    report = {"scope": "引用格式离线重放；不是新生成或答案准确率，原始实验保持不变",
              "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
              "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted((ROOT / "docqa").glob("*.py"))}, "rows": []}
    for row in source["rows"]:
        answer = row["answer"]
        text = normalize_citation_format(answer["text"])
        labels = {f"E{i}" for i in range(1, len(answer["evidence"]) + 1)}
        refs, invalid, missing = validate_citations(text, labels)
        report["rows"].append({"question_id": row["question_id"], "old_status": answer["status"],
                               "original_text": answer["text"], "normalized_text": text,
                               "changed": text != answer["text"], "refs": sorted(refs),
                               "invalid_refs": sorted(invalid), "missing_content_or_citation": missing,
                               "format_check_passed": not invalid and not missing})
    (output / "replay.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    if execute:
        settings = Settings(data_dir=output / "runtime", generation_provider="ollama",
                            api_key="", max_api_calls=0, local_models_only=True)
        store = Store(settings.data_dir)
        pipeline = RAGPipeline(None, create_generator(settings, store), store, settings)
        live = {"scope": "已知题的本地定向回归；不是独立测试或全量重跑", "source_sha256": report["source_sha256"],
                "code_sha256": report["code_sha256"], "cloud_calls": 0, "rows": []}
        by_id = {r["question_id"]: r for r in source["rows"]}
        selected = [r["question_id"] for r in source["rows"]
                    if r["answer"]["status"] in {"citation_error", "generation_error"}]
        selected = list(dict.fromkeys([*selected, "public-031", "public-037", "public-074"]))
        live["selected_ids"] = selected
        for qid in selected:
            previous = by_id[qid]
            evidence = [Evidence.model_validate(e) for e in previous["answer"]["evidence"]]
            request = ChatRequest(question=previous["question"], document_ids=source["config"]["document_ids"],
                                  strategy="hybrid_reranker", include_images=True, generate=True)
            answer = pipeline.run(request, retrieval_result=(evidence, {"timings": {}}))
            same_context = answer.context == previous["answer"]["context"]
            same_images = answer.image_count == previous["answer"]["image_count"]
            live["rows"].append({"question_id": qid, "reference_answer": previous["reference_answer"],
                                 "same_context": same_context, "same_images": same_images,
                                 "answer": answer.model_dump(mode="json")})
            (output / "live.json").write_text(json.dumps(live, ensure_ascii=False, indent=2))
            print(qid, answer.status, "same input:", same_context and same_images, flush=True)
            # A budget fix can change serialized context while keeping the frozen retrieval.
            # Record the difference rather than discarding the remaining failure cases.
        live["all_answered"] = all(r["answer"]["status"] == "answered" for r in live["rows"])
        (output / "live.json").write_text(json.dumps(live, ensure_ascii=False, indent=2))
    if source_path.read_bytes() != source_bytes:
        raise ValueError("核查期间原始实验文件发生变化")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="调用本地 Ollama 验证原失败题及空回答、图表回归题，不调用云端")
    args = parser.parse_args()
    print(check(args.source, args.output, args.execute))
