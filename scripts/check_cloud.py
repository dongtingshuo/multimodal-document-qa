"""Small real-cloud acceptance run; explicitly opts into five user-visible answers."""

import argparse
import json

import httpx
from api_address import ROOT, api_address

parser = argparse.ArgumentParser()
parser.add_argument("--execute", action="store_true", help="实际调用已配置的云端模型，会消耗额度")
args = parser.parse_args()
if not args.execute:
    parser.error("需要 --execute 才会调用云端")
dataset = json.loads((ROOT / "datasets/public_questions.json").read_text())
questions = {q["id"]: q for q in dataset["questions"]}
output = ROOT / "data/cloud-check/report.json"
output.parent.mkdir(parents=True, exist_ok=True)
report = {"scope": "五次小样本功能验证，不作为测试集效果结论", "answers": []}
with httpx.Client(base_url=api_address(), timeout=300) as client:
    health = client.get("/health")
    health.raise_for_status()
    if not health.json()["generation_configured"]:
        raise SystemExit("服务未加载密钥，请先重启")
    if health.json().get("generation_provider") == "ollama":
        raise SystemExit("当前生成后端是 Ollama；如需运行云端验收，请将 DOCQA_GENERATION_PROVIDER 改为 bailian 后重启")
    session = None
    for qid in ["public-001", "follow-up", "public-041", "public-066", "public-086"]:
        q = questions["public-001" if qid == "follow-up" else qid]
        scope = list(dict.fromkeys(s["document_id"] for s in q.get("reference_sources", [])))
        payload = {
            "question": "同一份报告中，当月PM2.5平均浓度是多少？请注明单位。"
            if qid == "follow-up"
            else q["question"],
            "document_ids": scope or dataset["document_ids"],
            "strategy": "hybrid_reranker",
            "generate": True,
            "include_images": True,
        }
        if qid == "follow-up":
            payload["session_id"] = session
        response = client.post("/chat", json=payload)
        response.raise_for_status()
        answer = response.json()
        if qid == "public-001":
            session = answer["session_id"]
        report["answers"].append(
            {
                "question_id": qid,
                "request": payload,
                "reference_answer": "43.1 μg/m³" if qid == "follow-up" else q["reference_answer"],
                "answer": answer,
            }
        )
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(
            json.dumps(
                {
                    "question_id": qid,
                    "status": answer["status"],
                    "text": answer["text"],
                    "image_count": answer["image_count"],
                    "usage": answer["usage"],
                    "warnings": answer["warnings"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if answer["status"] in {"generation_error", "citation_error"}:
            raise SystemExit("验证未通过，已停止后续调用并保留结果")
print("Report:", output)
