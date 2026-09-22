"""Replay known failures without changing questions, gold labels, or corpus scope."""

import argparse
import hashlib
import json
from datetime import datetime, timezone

import httpx
from api_address import ROOT, api_address

parser = argparse.ArgumentParser()
parser.add_argument("--execute", action="store_true", help="同时调用云端生成，会消耗额度")
args = parser.parse_args()
dataset = json.loads((ROOT / "datasets/public_questions.json").read_text())
questions = {q["id"]: q for q in dataset["questions"]}
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
output = ROOT / "data/public-evaluation/optimization-checks" / f"{stamp}.json"
output.parent.mkdir(parents=True, exist_ok=True)
report = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "scope": "已知失败的两题回归，含已查看过的测试题；不是独立测试集评测",
    "code_sha256": {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT / "docqa").glob("*.py"))
    },
    "checks": [],
}
with httpx.Client(base_url=api_address(), timeout=300) as client:
    health = client.get("/health")
    health.raise_for_status()
    report["health_before"] = health.json()
    for qid in ["public-002", "public-041"]:
        q = questions[qid]
        payload = {
            "question": q["question"],
            "document_ids": dataset["document_ids"],
            "strategy": "hybrid_reranker",
            "top_k": 6,
        }
        response = client.post("/retrieve", json=payload)
        response.raise_for_status()
        check = {
            "question_id": qid,
            "request": payload,
            "reference_answer": q["reference_answer"],
            "reference_sources": q["reference_sources"],
            "retrieval": response.json(),
        }
        report["checks"].append(check)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(qid, "retrieved", [e["element"]["id"] for e in response.json()["evidence"]], flush=True)
        if args.execute:
            answer = client.post("/chat", json={**payload, "generate": True, "include_images": True})
            answer.raise_for_status()
            check["answer"] = answer.json()
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
            print(
                json.dumps(
                    {k: answer.json()[k] for k in ["status", "text", "usage", "image_count"]},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if answer.json()["status"] in {"generation_error", "citation_error"}:
                raise SystemExit("生成验证失败，已保存并停止后续调用")
    health = client.get("/health")
    health.raise_for_status()
    report["health_after"] = health.json()
output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
print("Report:", output)
