"""Exercise real format conversion, OCR and page coordinates on labelled synthetic files."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(ROOT))

import pymupdf as fitz
from PIL import Image

from docqa.config import Settings
from docqa.parsers import LocalParser


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--output", type=Path, required=True)
    args.add_argument("--chinese", action="store_true", help="PDF 和图片样本增加中文关键文字检查")
    options = args.parse_args()
    out = options.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    inputs = out / "inputs"
    inputs.mkdir(exist_ok=True)
    cases = []
    for angle in [0, 90, 180, 270]:
        path = inputs / f"rotation-{angle}.pdf"
        with fitz.open() as doc:
            page = doc.new_page(width=600, height=800)
            page.insert_text((45, 90), "Metadata: SQLite", fontsize=20)
            page.insert_text((330, 90), "Upload limit: 50 MB", fontsize=18)
            page.insert_text((45, 160), "Left column document", fontsize=16)
            page.insert_text((330, 160), "Right column document", fontsize=16)
            if options.chinese:
                page.insert_text((45, 210), "检索增强生成：智能文档问答系统", fontname="china-s", fontsize=20)
            page.draw_rect((50, 260, 220, 350), fill=(0.2, 0.6, 0.8))
            page.set_rotation(angle)
            doc.save(path)
        cases.append((f"pdf-{angle}", path, ["SQLite", "50 MB"] + (["检索增强生成"] if options.chinese else [])))
    with fitz.open(inputs / "rotation-0.pdf") as doc:
        doc[0].get_pixmap(matrix=fitz.Matrix(2, 2)).save(inputs / "scan.png")
    with Image.open(inputs / "scan.png") as img:
        img.convert("RGB").save(inputs / "scan.jpg", quality=92)
        img.convert("RGB").save(inputs / "scan.pdf", "PDF", resolution=144)
        # EXIF tells image readers to rotate this stored pixel arrangement back.
        rotated = img.transpose(Image.Transpose.ROTATE_90).convert("RGB")
        exif = Image.Exif()
        exif[274] = 6
        rotated.save(inputs / "exif.jpg", quality=92, exif=exif)
    for name in ["scan.png", "scan.jpg", "scan.pdf", "exif.jpg"]:
        cases.append((name.replace(".", "-"), inputs / name, ["SQLite", "50 MB"] + (["检索增强生成"] if options.chinese else [])))
    word = ROOT / "data/word-check/synthetic.docx"
    cases.append(("docx", word, ["SQLite", "50 MB"]))
    conversion_error = None
    try:
        with tempfile.TemporaryDirectory(prefix="docqa-format-lo-") as profile:
            result = subprocess.run([
                shutil.which("soffice") or "soffice", f"-env:UserInstallation={Path(profile).as_uri()}",
                "--headless", "--convert-to", "doc:MS Word 97", "--outdir", str(inputs), str(word),
            ], capture_output=True, text=True, timeout=120, check=True)
        converted = inputs / "synthetic.doc"
        if not converted.exists():
            raise RuntimeError("未生成 DOC 样本：" + result.stderr[-200:])
        cases.append(("doc", converted, ["SQLite", "50 MB"]))
    except Exception as exc:
        conversion_error = str(exc)

    parser = LocalParser(Settings(data_dir=ROOT / "data", enable_ocr=True, api_key=""))
    records = []
    for name, source, anchors in cases:
        started = time.monotonic()
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        try:
            parsed = parser.parse(source, name, out / name)
            text = " ".join(e.text for e in parsed["elements"])
            compact = "".join(text.split()).casefold()
            found = {anchor: "".join(anchor.split()).casefold() in compact for anchor in anchors}
            boxes_valid = all(0 <= s.bbox[0] < s.bbox[2] <= 1 and 0 <= s.bbox[1] < s.bbox[3] <= 1
                              for e in parsed["elements"] for s in e.sources)
            original_unchanged = hashlib.sha256(source.read_bytes()).hexdigest() == before
            ok = all(found.values()) and boxes_valid and original_unchanged and not parsed["warnings"]
            record = {"case": name, "ok": ok, "anchors": found, "page_count": parsed["page_count"],
                      "elements": len(parsed["elements"]), "converted": parsed["converted"],
                      "boxes_valid": boxes_valid, "original_unchanged": original_unchanged,
                      "warnings": parsed["warnings"], "text": text}
        except Exception as exc:
            record = {"case": name, "ok": False, "error": str(exc)}
        record["seconds"] = round(time.monotonic() - started, 3)
        records.append(record)
        print(name, record["ok"], record.get("error", record.get("warnings")), flush=True)
    if conversion_error:
        records.append({"case": "doc", "ok": False, "error": conversion_error})
    report = {"scope": "合成文件格式/OCR/坐标回归，使用轻量 LocalParser；不代表复杂真实文档识别准确率。",
              "passed": sum(r["ok"] for r in records), "total": len(records), "records": records}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if report["passed"] != report["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
