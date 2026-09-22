"""Reproduce the frozen public corpus from its source manifest with SHA256 checks."""

import hashlib
import json
from pathlib import Path

import httpx
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/corpus"


def main():
    manifest = json.loads((ROOT / "datasets/public_sources.json").read_text())
    failures = []
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        for source in manifest["documents"]:
            path = OUT / source["local_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
                    failures.append({"title": source["title"], "error": "已有文件摘要不一致，未覆盖"})
                continue
            try:
                response = client.get(source["download_url"])
                response.raise_for_status()
                content = response.content
                if hashlib.sha256(content).hexdigest() != source["sha256"]:
                    raise ValueError("下载内容与冻结版本不同，请核对原站更新，不能沿用旧标注")
                with pymupdf.open(stream=content, filetype="pdf") as pdf:
                    if len(pdf) != source["pages"]:
                        raise ValueError("PDF页数与清单不同")
                path.write_bytes(content)
                print(source["title"], "verified", flush=True)
            except (httpx.HTTPError, ValueError) as exc:
                failures.append({"title": source["title"], "error": type(exc).__name__})
    report = {
        "expected_documents": len(manifest["documents"]),
        "expected_pages": sum(s["pages"] for s in manifest["documents"]),
        "failures": failures,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "download-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
