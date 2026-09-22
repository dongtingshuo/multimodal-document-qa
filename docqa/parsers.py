"""Local parsing; original files are never modified. Coordinates refer to rendered pages."""

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pymupdf as fitz
from PIL import Image, ImageOps

from .schemas import Element, Source


def normalized(page, rect):
    rect = fitz.Rect(rect) * page.rotation_matrix
    w, h = page.rect.width, page.rect.height
    return tuple(max(0.0, min(1.0, v)) for v in (rect.x0 / w, rect.y0 / h, rect.x1 / w, rect.y1 / h))


def crop(page_path: Path, box, output: Path):
    with Image.open(page_path) as img:
        w, h = img.size
        rect = (
            int(box[0] * w),
            int(box[1] * h),
            max(int(box[2] * w), int(box[0] * w) + 1),
            max(int(box[3] * h), int(box[1] * h) + 1),
        )
        img.crop(rect).save(output)


def standardize(path: Path, output: Path) -> tuple[Path, bool]:
    if path.suffix.lower() in {".doc", ".docx"}:
        soffice = shutil.which("soffice")
        if not soffice:
            raise RuntimeError("Word 转换需要 LibreOffice，请安装后重试")
        with tempfile.TemporaryDirectory(prefix="docqa-lo-") as profile:
            result = subprocess.run(
                [
                    soffice,
                    f"-env:UserInstallation={Path(profile).as_uri()}",
                    "--headless",
                    "--convert-to",
                    "pdf",
                    "--outdir",
                    str(output),
                    str(path),
                ],
                capture_output=True,
                timeout=120,
                check=False,
            )
        target = output / (path.stem + ".pdf")
        if result.returncode or not target.is_file():
            raise RuntimeError("LibreOffice 转换失败，文件可能已损坏或受保护")
        return target, True
    if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            if img.width * img.height > 40_000_000:
                raise ValueError("图片超过 4000 万像素")
            target = output / "normalized.pdf"
            img.save(target, "PDF", resolution=144)
        return target, False
    return path, False


class LocalParser:
    version = "pymupdf-rapidocr-v1"

    def __init__(self, settings):
        self.settings = settings
        self._ocr = None

    def ocr(self, page_path):
        if self._ocr is None:
            from rapidocr import RapidOCR

            self._ocr = RapidOCR(params={"Global.model_root_dir": str(self.settings.data_dir / "ocr_models")})
        result = self._ocr(str(page_path))
        if result.txts is None or result.boxes is None:
            return []
        with Image.open(page_path) as img:
            w, h = img.size
        output = []
        for points, text in zip(result.boxes, result.txts, strict=False):
            xs, ys = [float(p[0]) for p in points], [float(p[1]) for p in points]
            box = tuple(max(0, min(1, v)) for v in (min(xs) / w, min(ys) / h, max(xs) / w, max(ys) / h))
            output.append((text, box))
        return output

    def parse(self, path: Path, document_id: str, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        pdf_path, converted = standardize(path, output_dir)
        elements, warnings, raw = [], [], []
        with fitz.open(pdf_path) as pdf:
            if pdf.needs_pass:
                raise ValueError("暂不支持加密 PDF")
            if not 1 <= len(pdf) <= self.settings.max_pages:
                raise ValueError(f"页数须为 1–{self.settings.max_pages}")
            for number, page in enumerate(pdf, 1):
                page_path = output_dir / f"page-{number}.png"
                scale = min(2.0, 2400 / max(page.rect.width, page.rect.height))
                page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False).save(page_path)
                blocks = page.get_text("blocks", sort=True)
                raw.append({"page": number, "rotation": page.rotation, "blocks": blocks})
                page_elements = []

                def add(
                    kind,
                    text,
                    box,
                    table=None,
                    image=False,
                    number=number,
                    page_elements=page_elements,
                    page_path=page_path,
                ):
                    eid = f"{document_id}-p{number:04d}-e{len(page_elements):04d}"
                    image_path = None
                    if image:
                        image_path = str(output_dir / f"{eid}.png")
                        crop(page_path, box, Path(image_path))
                    element = Element(
                        id=eid,
                        document_id=document_id,
                        kind=kind,
                        text=text.strip(),
                        sources=[Source(page=number, bbox=box)],
                        image_path=image_path,
                        table=table,
                    )
                    page_elements.append(element)

                tables = []
                try:
                    for table in page.find_tables().tables:
                        rows = [[str(c or "").replace("\n", " ") for c in row] for row in table.extract()]
                        if rows:
                            text = "\n".join(" | ".join(row) for row in rows)
                            add("table", text, normalized(page, table.bbox), rows, True)
                            tables.append(fitz.Rect(table.bbox))
                except Exception as exc:
                    warnings.append(f"第 {number} 页表格识别失败：{type(exc).__name__}")
                text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]
                for block in text_blocks:
                    if any(t.contains(fitz.Rect(block[:4])) for t in tables):
                        continue
                    add("text", block[4], normalized(page, block[:4]))
                if sum(len(b[4].strip()) for b in text_blocks) < 20:
                    if self.settings.enable_ocr:
                        try:
                            for text, box in self.ocr(page_path):
                                add("text", text, box)
                        except Exception as exc:
                            warnings.append(f"第 {number} 页 OCR 失败：{type(exc).__name__}；已保留原页")
                    else:
                        warnings.append(f"第 {number} 页文本较少，OCR 未开启；已保留原页")
                context = "\n".join(e.text for e in page_elements)[:1600]
                visual_lines = [
                    ("".join(span["text"] for span in line["spans"]), normalized(page, line["bbox"]))
                    for block in page.get_text("dict")["blocks"] if block["type"] == 0
                    for line in block["lines"]
                ]
                boxes = [fitz.Rect(info["bbox"]) for info in page.get_image_info()]
                # Drawing clusters preserve vector charts even when there is no embedded bitmap.
                boxes += list(page.cluster_drawings())
                seen = []
                for box in boxes:
                    if box.width < 35 or box.height < 35 or any(t.contains(box) for t in tables):
                        continue
                    norm = normalized(page, box)
                    if norm in seen:
                        continue
                    seen.append(norm)
                    # Prefer nearby text in the same column. Reusing the first
                    # page paragraph gives every chart the same unrelated caption.
                    x0, y0, x1, y1 = norm
                    def proximity(line, x0=x0, y0=y0, x1=x1, y1=y1):
                        a, b, c, d = line[1]
                        overlap = max(0, min(c, x1) - max(a, x0))
                        aligned = overlap >= 0.3 * max(1e-6, min(c - a, x1 - x0))
                        return (not aligned, max(0, y0 - d, b - y1))
                    neighbors = sorted(visual_lines, key=proximity)
                    nearby = "\n".join(line[0] for line in neighbors[:20])[:1600]
                    add("image", nearby, norm, image=True)
                if not page_elements or not text_blocks:
                    add("page", context, (0, 0, 1, 1), image=True)
                elements.extend(page_elements)
            page_count = len(pdf)
        (output_dir / "raw.json").write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        return {
            "elements": elements,
            "page_count": page_count,
            "converted": converted,
            "warnings": warnings,
            "parser_version": self.version,
            "pdf_path": str(pdf_path),
        }


def local_picture_text(box, lines, fallback=""):
    """Attach only a nearby same-column caption and text inside the picture."""
    x0, y0, x1, y1 = box
    inside, captions = [], []
    for text, (a, b, c, d) in lines:
        overlap = max(0, min(c, x1) - max(a, x0))
        if overlap < .5 * max(1e-6, min(c-a, x1-x0)):
            continue
        gap = max(0, y0-d, b-y1)
        if re.match(r"^\s*图\s*\d+", text) and gap <= .035:
            captions.append((gap, text, b, d))
        if y0 <= (b+d)/2 <= y1 and x0 <= (a+c)/2 <= x1:
            inside.append(text)
    chosen = min(captions, default=None)
    caption = ""
    if chosen:
        _, _, top, bottom = chosen
        caption = " ".join(text for text, (a, b, c, d) in sorted(lines, key=lambda line: line[1][0])
                           if abs((b+d)/2 - (top+bottom)/2) < .004
                           and max(a, x0) < min(c, x1))
    return "\n".join(dict.fromkeys([t for t in [caption, *inside] if t])) or fallback


def remove_repeated_margin_artwork(elements):
    """Exclude small recurring header/footer graphics; retain unique margin evidence."""
    groups = {}
    for element in elements:
        if element.kind != "image":
            continue
        source = element.sources[0]
        x0, y0, x1, y1 = source.bbox
        if y1 - y0 < 0.05 and (y1 < 0.1 or y0 > 0.9):
            key = ("top" if y1 < 0.1 else "bottom", round(x0, 1), round(x1, 1))
            groups.setdefault(key, []).append(element)
    repeated = {
        element.id
        for items in groups.values()
        if len({e.sources[0].page for e in items}) >= 3
        for element in items
    }
    return [e for e in elements if e.id not in repeated]


class DoclingParser(LocalParser):
    version = "docling-rapidocr-v4"

    def parse(self, path, document_id, output_dir):
        from docling.datamodel.accelerator_options import AcceleratorOptions
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling_core.types.doc import PictureItem, TableItem

        output_dir.mkdir(parents=True, exist_ok=True)
        pdf_path, converted = standardize(path, output_dir)
        with fitz.open(pdf_path) as pdf:
            if pdf.needs_pass or not 1 <= len(pdf) <= self.settings.max_pages:
                raise ValueError("PDF 加密或页数超过限制")
        options = PdfPipelineOptions()
        options.accelerator_options = AcceleratorOptions(device=self.settings.device, num_threads=1)
        options.do_ocr = self.settings.enable_ocr
        options.ocr_options = RapidOcrOptions(
            rapidocr_params={"Global.model_root_dir": str(self.settings.data_dir / "ocr_models")}
        )
        options.generate_page_images = True
        options.generate_picture_images = True
        options.images_scale = 2.0
        if not hasattr(self, "converter"):
            self.converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
            )
        result = self.converter.convert(pdf_path)
        if result.status.value != "success":
            raise RuntimeError(f"Docling 未完整解析：{result.status.value}")
        doc = result.document
        doc.save_as_json(output_dir / "raw-docling.json")
        for number, page in doc.pages.items():
            page.image.pil_image.save(output_dir / f"page-{number}.png")
        elements = []
        headings = []
        for item, _ in doc.iterate_items():
            label = getattr(getattr(item, "label", None), "value", "")
            if label in {"page_header", "page_footer"}:
                continue
            if label in {"title", "section_header"}:
                level = 1 if label == "title" else max(1, getattr(item, "level", 1))
                headings = headings[: level - 1] + [getattr(item, "text", "")]
            sources = []
            for prov in getattr(item, "prov", []):
                size = doc.pages[prov.page_no].size
                box = prov.bbox.to_top_left_origin(page_height=size.height)
                norm = tuple(
                    max(0, min(1, v))
                    for v in (
                        box.l / size.width,
                        box.t / size.height,
                        box.r / size.width,
                        box.b / size.height,
                    )
                )
                sources.append(Source(page=prov.page_no, bbox=norm))
            if not sources:
                continue
            kind, text, rows = "text", getattr(item, "text", ""), None
            if isinstance(item, TableItem):
                kind, text = "table", item.export_to_markdown(doc=doc)
                frame = item.export_to_dataframe(doc=doc)
                rows = [list(map(str, frame.columns))] + frame.fillna("").astype(str).values.tolist()
            elif isinstance(item, PictureItem):
                kind, text = "image", item.caption_text(doc)
            if not text and kind == "text":
                continue
            eid = f"{document_id}-e{len(elements):06d}"
            image_path = None
            if kind != "text":
                image_path = str(output_dir / f"{eid}.png")
                crop(output_dir / f"page-{sources[0].page}.png", sources[0].bbox, Path(image_path))
            elements.append(
                Element(
                    id=eid,
                    document_id=document_id,
                    kind=kind,
                    text=text,
                    sources=sources,
                    image_path=image_path,
                    table=rows,
                    title_path=list(headings),
                )
            )
        elements = remove_repeated_margin_artwork(elements)
        with fitz.open(pdf_path) as original:
            for element in elements:
                if element.kind == "image":
                    page = original[element.sources[0].page - 1]
                    lines = [("".join(span["text"] for span in line["spans"]),
                              normalized(page, line["bbox"]))
                             for block in page.get_text("dict")["blocks"] if block["type"] == 0
                             for line in block["lines"]]
                    element.text = local_picture_text(element.sources[0].bbox, lines, element.text)
        return {
            "elements": elements,
            "page_count": len(doc.pages),
            "converted": converted,
            "warnings": [],
            "parser_version": self.version,
            "pdf_path": str(pdf_path),
        }
