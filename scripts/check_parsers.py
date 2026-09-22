"""Real parser checks on synthetic input only. May download public OCR/layout weights."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "huggingface"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(ROOT))

import pymupdf as fitz

from docqa.config import Settings
from docqa.parsers import DoclingParser, LocalParser

directory = ROOT / "data" / "parser-check"
directory.mkdir(parents=True, exist_ok=True)
settings = Settings(data_dir=directory)
report = {}
sample = ROOT / "data" / "smoke" / "合成演示.pdf"
with fitz.open(sample) as pdf:
    pdf[0].get_pixmap(matrix=fitz.Matrix(2, 2)).save(directory / "scan.png")
for label, parser, path in [
    ("ocr_image", LocalParser(settings), directory / "scan.png"),
    ("docling_pdf", DoclingParser(settings), sample),
]:
    try:
        result = parser.parse(path, label, directory / label)
        text = "\n".join(e.text for e in result["elements"])
        report[label] = {
            "element_count": len(result["elements"]),
            "kinds": sorted({e.kind for e in result["elements"]}),
            "sqlite_found": "SQLite" in text.replace(" ", ""),
            "warnings": result["warnings"],
        }
        assert "SQLite" in text.replace(" ", ""), text[:300]
        assert not result["warnings"], result["warnings"]
        print(label, report[label], flush=True)
    except Exception as exc:
        report[label] = {"error": str(exc), "type": type(exc).__name__}
        print(label, report[label], flush=True)
(directory / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
if any("error" in r for r in report.values()):
    sys.exit(1)
