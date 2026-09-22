"""Load the synthetic sample through the same upload API used by the UI."""

import json
import time
from pathlib import Path

import httpx
from api_address import api_address

root = Path(__file__).resolve().parents[1]
sample = root / "data" / "smoke" / "合成演示.pdf"
with httpx.Client(base_url=api_address(), timeout=300) as client:
    response = client.post(
        "/documents", files={"file": (sample.name, sample.read_bytes(), "application/pdf")}
    )
    response.raise_for_status()
    doc = response.json()
    deadline = time.monotonic() + 300
    while True:
        response = client.get("/jobs/" + doc["job_id"])
        response.raise_for_status()
        job = response.json()
        if job["status"] in {"ready", "failed"}:
            break
        if time.monotonic() >= deadline:
            raise SystemExit("解析仍在进行，请在文档管理页查看任务状态")
        time.sleep(1)
    assert job["status"] == "ready", job
    elements = client.get(f"/documents/{doc['id']}/elements").json()
    questions = []
    for i, (question, anchor, answer) in enumerate(
        [
            ("系统用什么数据库保存文档元数据？", "SQLite", "SQLite"),
            ("上传文件的大小限制是多少？", "50 MB", "50 MB"),
            ("混合检索如何融合排名？", "RRF", "RRF 排名融合"),
        ],
        1,
    ):
        normalized_anchor = anchor.replace(" ", "").replace("\xa0", "")
        element = next(
            e
            for e in elements
            if e["kind"] == "text" and normalized_anchor in e["text"].replace(" ", "").replace("\xa0", "")
        )
        questions.append(
            {
                "id": f"demo-{i}",
                "question": question,
                "relevant_element_ids": [element["id"]],
                "reference_answer": answer,
                "split": "dev",
                "category": "text",
            }
        )
    questions.append(
        {
            "id": "demo-4",
            "question": "系统开发者的身份证号码是多少？",
            "answerable": False,
            "relevant_element_ids": [],
            "split": "dev",
            "category": "unanswerable",
        }
    )
    path = root / "data" / "demo-questions.json"
    path.write_text(
        json.dumps(
            {
                "note": "合成演示题集，不能当作正式性能评估",
                "document_ids": [doc["id"]],
                "questions": questions,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(doc["id"], path)
