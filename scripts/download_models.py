"""Download only inference files to project-local storage; never executes remote model code."""

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "huggingface"))

from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser()
parser.add_argument("--reranker", action="store_true")
args = parser.parse_args()
repos = ["BAAI/bge-small-zh-v1.5"]
if args.reranker:
    repos.append("BAAI/bge-reranker-base")
for repo in repos:
    location = snapshot_download(
        repo,
        allow_patterns=["*.json", "*.txt", "*.safetensors", "pytorch_model.bin", "1_Pooling/*"],
        ignore_patterns=["onnx/*", "openvino/*"],
    )
    print(repo, location)
