"""Run full retrieval regression and Ollama answers in an isolated, resumable snapshot."""

import argparse
import fcntl
import hashlib
import json
import os
import signal
import sqlite3
import sys
import uuid
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "models/huggingface"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(ROOT))

from benchmark import export
from export_blind_review import export as export_blind

from docqa.config import Settings
from docqa.evaluation import ExperimentPaused, ExperimentRequest, run_experiment
from docqa.generation import RAGPipeline, create_generator
from docqa.retrieval import HybridRetriever
from docqa.review_io import export_reviews
from docqa.store import Store


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "run.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    dataset_path = out / "questions.json"
    source_path = ROOT / "datasets/public_questions.json"
    if not dataset_path.exists():
        dataset_path.write_bytes(source_path.read_bytes())
    dataset = json.loads(dataset_path.read_text())
    snapshot = out / "snapshot"
    snapshot.mkdir(exist_ok=True)
    database = snapshot / "metadata.sqlite3"
    if not database.exists():
        temporary = snapshot / "snapshot.tmp"
        source = sqlite3.connect((ROOT / "data/metadata.sqlite3").as_uri() + "?mode=ro", uri=True)
        destination = sqlite3.connect(temporary)
        try:
            source.backup(destination)
        finally:
            destination.close()
            source.close()
        temporary.replace(database)

    settings = Settings(data_dir=snapshot, enable_clip=True, local_models_only=True,
                        generation_provider="ollama", api_key="", max_api_calls=0)
    store = Store(snapshot)
    retriever = HybridRetriever(store, settings)
    pipeline = RAGPipeline(retriever, create_generator(settings, store), store, settings)
    manifest_path = out / "run.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        # Only past generated answers indicate prior answer-level exposure.
        previous = set()
        for record in store.records("experiments"):
            previous.update(row["question_id"] for row in record.get("rows", []) if "answer" in row)
        manifest = {
            "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
            "prior_generated_test_ids": sorted(previous & {
                q["id"] for q in dataset["questions"] if q["split"] == "test"}),
            "interpretation": "当前版本固定题集完整回归；包含曾参与调试的题目，不称为全新独立测试。人工评分保持空值。",
            "runs": {},
        }
        write_json(manifest_path, manifest)
    for label, split, groups, generate in [
        ("retrieval-dev", "dev", list("ABCDEF"), False),
        ("retrieval-test", "test", list("ABCDEF"), False),
        ("ollama-test", "test", ["E"], True),
    ]:
        eid = manifest["runs"].setdefault(label, uuid.uuid4().hex)
        write_json(manifest_path, manifest)
        directory = out / label
        result_path = directory / (eid + ".json")
        previous = json.loads(result_path.read_text()) if result_path.exists() else None
        if previous and previous["status"] == "ready":
            print(label, "already complete", flush=True)
            continue
        if stopped:
            return
        if generate:
            response = httpx.get(settings.ollama_base.rstrip("/") + "/tags", timeout=10)
            response.raise_for_status()
            names = {model["name"] for model in response.json().get("models", [])}
            if settings.ollama_model not in names:
                raise RuntimeError("Ollama 模型未下载，保留检索结果并停止生成验收")
        request = ExperimentRequest(questions=dataset["questions"], document_ids=dataset["document_ids"],
                                    split=split, groups=groups, generate=generate, expected_provider="ollama")

        def progress(partial, eid=eid, directory=directory, label=label):
            export({"id": eid, "status": "running", **partial}, directory)
            print(label, partial["completed"], "/", partial["total"], flush=True)

        try:
            result = run_experiment(request, retriever, store, pipeline, progress,
                                    resume=previous, should_stop=lambda: stopped)
        except ExperimentPaused as exc:
            export({"id": eid, "status": "paused", **exc.result}, directory)
            print("Paused. Repeat this command to resume.", flush=True)
            return
        except Exception as exc:
            partial = json.loads(result_path.read_text()) if result_path.exists() else {"id": eid}
            export({**partial, "status": "failed", "error": str(exc)[:400]}, directory)
            raise
        final = {"id": eid, "status": "ready", **result}
        export(final, directory)
        if generate:
            export_blind(result_path, out / "reviews")
            (out / "reviews" / "human-review.csv").write_bytes(export_reviews(final))
        print(label, "COMPLETE", flush=True)
    print("ALL EXPERIMENTS COMPLETE; human review is pending.", flush=True)


if __name__ == "__main__":
    main()
