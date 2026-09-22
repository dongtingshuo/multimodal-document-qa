"""Exercise real models against synthetic, explicitly labelled demonstration data."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "huggingface"))
sys.path.insert(0, str(ROOT))

import pymupdf as fitz

from docqa.config import Settings
from docqa.schemas import ChatRequest, RetrieveRequest
from docqa.service import Service


def make_sample(path):
    with fitz.open() as pdf:
        page = pdf.new_page(width=595, height=842)
        page.insert_text((48, 60), "文档问答系统：合成演示材料", fontname="china-s", fontsize=18)
        lines = [
            "本文仅用于功能测试，不是真实实验结果。",
            "系统使用 SQLite 保存文档元数据和问答会话。",
            "文件上传大小限制为 50 MB，支持 PDF、Word 和图片。",
            "关键词检索使用 BM25，语义检索采用 BGE 中文向量模型。",
            "混合检索利用 RRF 融合排名，重排序模型进一步评价相关性。",
            "生成回答必须引用本轮证据，并提供原始页码及高亮区域。",
        ]
        for i, text in enumerate(lines):
            page.insert_text((48, 105 + i * 32), text, fontname="china-s", fontsize=12)
        page.insert_text((48, 340), "下表数据为人工设定的测试值", fontname="china-s", fontsize=12)
        for x in [48, 240, 450]:
            page.draw_line((x, 360), (x, 480))
        for y in [360, 400, 440, 480]:
            page.draw_line((48, y), (450, y))
        for y, row in [(386, ["检索策略", "演示数值"]), (426, ["BM25", "0.60"]), (466, ["Hybrid", "0.85"])]:
            for x, text in zip([60, 250], row, strict=False):
                page.insert_text((x, y), text, fontname="china-s", fontsize=12)
        page.draw_rect((80, 550, 155, 670), fill=(0.3, 0.5, 0.8))
        page.draw_rect((200, 500, 275, 670), fill=(0.2, 0.65, 0.5))
        page.insert_text((60, 710), "蓝色和绿色矩形是矢量绘图测试区域。", fontname="china-s", fontsize=12)
        pdf.save(path)


if __name__ == "__main__":
    output = ROOT / "data" / "smoke"
    output.mkdir(parents=True, exist_ok=True)
    sample = output / "合成演示.pdf"
    make_sample(sample)
    settings = Settings(data_dir=output / "runtime", enable_ocr=False, local_models_only=True, api_key="")
    service = Service(settings)
    try:
        doc = service.import_document(sample.name, sample.read_bytes())
        service.worker.submit(lambda: None).result(timeout=120)
        ready = service.store.document(doc["id"])
        assert ready["status"] == "ready", ready
        results = {"document": ready, "strategies": {}}
        for strategy in ["bm25", "dense", "hybrid", "hybrid_reranker"]:
            evidence, trace = service.retriever.retrieve(
                RetrieveRequest(question="系统用什么数据库保存元数据？", strategy=strategy, top_k=3)
            )
            assert any("SQLite" in e.matched_text for e in evidence), (strategy, evidence)
            results["strategies"][strategy] = {"evidence": [e.model_dump() for e in evidence], "trace": trace}
            print(strategy, "OK", trace["timings"], flush=True)
        answer = service.pipeline.run(ChatRequest(question="上传文件的大小上限是多少？", generate=False))
        assert answer.status == "evidence_only"
        results["answer"] = answer.model_dump()
        (output / "result.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("Report:", output / "result.json", flush=True)
    finally:
        service.close()
