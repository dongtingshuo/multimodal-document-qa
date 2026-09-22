"""Multimodal document QA. Independent components are reusable as future Agent tools."""

import os
from pathlib import Path

# Apple ARM: the default OpenMP thread pool crashed during combined Torch/FAISS inference.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / "models" / "huggingface"))
