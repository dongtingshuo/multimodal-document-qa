"""Import and verify a source manifest through the application's public upload API."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import httpx
from api_address import ROOT, api_address

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", type=Path, default=ROOT / "datasets/public_sources.json")
parser.add_argument("--api")
args = parser.parse_args()
manifest = json.loads(args.manifest.read_text())
report = {"documents": [], "failed": []}
out = ROOT / "data/corpus/import-report.json"
with httpx.Client(base_url=api_address(args.api), timeout=900) as client:
    for source in manifest["documents"]:
        path = ROOT / "data/corpus" / source["local_path"]
        content = path.read_bytes()
        assert hashlib.sha256(content).hexdigest() == source["sha256"], path
        response = client.post(
            "/documents", files={"file": (source["title"] + ".pdf", content, "application/pdf")}
        )
        response.raise_for_status()
        doc = response.json()
        deadline = time.monotonic() + 1800
        while True:
            response = client.get("/jobs/" + doc["job_id"])
            response.raise_for_status()
            job = response.json()
            if job["status"] in {"ready", "failed"}:
                break
            if time.monotonic() > deadline:
                raise SystemExit("处理超时，原任务仍可在工作台查看：" + source["title"])
            time.sleep(2)
        row = {**source, "document_id": doc["id"], "status": job["status"], "error": job.get("error")}
        if job["status"] == "ready":
            response = client.get(f"/documents/{doc['id']}/elements")
            response.raise_for_status()
            elements = response.json()
            row["element_count"] = len(elements)
            row["kinds"] = {
                k: sum(e["kind"] == k for e in elements) for k in ["text", "table", "image", "page"]
            }
            row["parser_version"] = next(
                d["parser_version"] for d in client.get("/documents").json() if d["id"] == doc["id"]
            )
            (ROOT / "data/corpus" / (doc["id"] + "-elements.json")).write_text(
                json.dumps(elements, ensure_ascii=False, indent=2)
            )
            report["documents"].append(row)
        else:
            report["failed"].append(row)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(source["title"], job["status"], row.get("kinds", job.get("error")), flush=True)
print("Ready", len(report["documents"]), "Failed", len(report["failed"]))
