"""Fresh image/text encoding, FAISS, BGE, reranker and six ablations in one process."""

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw

from docqa.config import Settings
from docqa.evaluation import ExperimentRequest, run_experiment
from docqa.retrieval import HybridRetriever
from docqa.schemas import Element, Source
from docqa.store import Store


def main():
    directory = ROOT / "data" / "vision-check"
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fresh-", dir=directory) as tmp:
        root = Path(tmp)
        settings = Settings(data_dir=root, enable_clip=True, local_models_only=True, api_key="")
        store = Store(root)
        store.put_document(
            {"id": "synthetic", "sha256": "synthetic", "name": "合成图文测试", "status": "ready"}
        )
        elements = []
        for i, (color, caption) in enumerate([("blue", "蓝色矩形"), ("green", "绿色矩形")]):
            path = root / f"{color}.png"
            image = Image.new("RGB", (300, 300), "white")
            ImageDraw.Draw(image).rectangle((40, 50, 240, 230), fill=color)
            image.save(path)
            elements.append(
                Element(
                    id=f"e{i}",
                    document_id="synthetic",
                    kind="image",
                    text=caption,
                    image_path=str(path),
                    sources=[Source(page=1, bbox=(0, 0, 1, 1))],
                )
            )
        store.replace_elements("synthetic", elements)
        retriever = HybridRetriever(store, settings)
        result = run_experiment(
            ExperimentRequest(
                groups=["A", "B", "C", "D", "E", "F"],
                questions=[
                    {
                        "id": "visual-smoke",
                        "question": "蓝色矩形",
                        "category": "chart",
                        "relevant_element_ids": ["e0"],
                    }
                ],
            ),
            retriever,
            store,
        )
        assert len(result["rows"]) == 6
        visual = result["rows"][-1]
        assert visual["trace"]["rankings"]["clip"] and visual["retrieved"]
        assert any(root.glob("indexes/clip-*.faiss")), "未生成真实视觉索引"
        result["notes"].append("合成两张新图片，索引从零建立；仅证明模型共存与流程，不代表真实准确率。")
        (directory / "report.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("PASS: fresh Chinese-CLIP + BGE + reranker + isolated FAISS; six groups, no cloud calls")
        print("Report:", directory / "report.json")


if __name__ == "__main__":
    main()
