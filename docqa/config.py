import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

GENERATION_MODEL_OPTIONS = {
    "bailian": ["qwen3-vl-plus-2025-12-19"],
    "ollama": ["qwen2.5vl:7b"],
}


def flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes"}


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DOCQA_DATA_DIR", "data")).resolve())
    parser: str = field(default_factory=lambda: os.getenv("DOCQA_PARSER", "docling"))
    embedding_model: str = field(
        default_factory=lambda: os.getenv("DOCQA_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    )
    reranker_model: str = field(
        default_factory=lambda: os.getenv("DOCQA_RERANKER_MODEL", "BAAI/bge-reranker-base")
    )
    device: str = field(default_factory=lambda: os.getenv("DOCQA_DEVICE", "cpu"))
    local_models_only: bool = field(default_factory=lambda: flag("DOCQA_LOCAL_MODELS_ONLY"))
    enable_ocr: bool = field(default_factory=lambda: flag("DOCQA_ENABLE_OCR", "true"))
    enable_clip: bool = field(default_factory=lambda: flag("DOCQA_ENABLE_CLIP"))
    clip_model: str = field(default_factory=lambda: os.getenv("DOCQA_CLIP_MODEL", "ViT-B-16"))
    clip_model_dir: Path = field(
        default_factory=lambda: Path(os.getenv("DOCQA_CLIP_MODEL_DIR", "models/chinese-clip")).resolve()
    )
    api_base: str = field(
        default_factory=lambda: os.getenv(
            "DOCQA_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    )
    api_key: str = field(default_factory=lambda: os.getenv("DOCQA_API_KEY", ""), repr=False)
    generation_model: str = field(
        default_factory=lambda: os.getenv("DOCQA_GENERATION_MODEL", "qwen3-vl-plus-2025-12-19")
    )
    generation_provider: str = field(
        default_factory=lambda: os.getenv("DOCQA_GENERATION_PROVIDER", "bailian").strip().lower()
    )
    ollama_base: str = field(
        default_factory=lambda: os.getenv("DOCQA_OLLAMA_BASE", "http://127.0.0.1:11434/api")
    )
    ollama_model: str = field(
        default_factory=lambda: os.getenv("DOCQA_OLLAMA_MODEL", "qwen2.5vl:7b")
    )
    ollama_keep_alive: str = field(
        default_factory=lambda: os.getenv("DOCQA_OLLAMA_KEEP_ALIVE", "10m")
    )
    max_api_calls: int = field(default_factory=lambda: int(os.getenv("DOCQA_MAX_API_CALLS", "100")))
    api_timeout: float = field(default_factory=lambda: float(os.getenv("DOCQA_API_TIMEOUT", "60")))
    ollama_timeout: float = field(default_factory=lambda: float(os.getenv("DOCQA_OLLAMA_TIMEOUT", "300")))
    ollama_context_length: int = field(default_factory=lambda: int(os.getenv("DOCQA_OLLAMA_CONTEXT_LENGTH", "8192")))
    max_upload_bytes: int = 50 * 1024 * 1024
    max_pages: int = 300

    @property
    def uses_ollama(self) -> bool:
        return self.generation_provider in {"ollama", "local", "local_ollama"}

    @property
    def generation_configured(self) -> bool:
        return self.uses_ollama or bool(self.api_key)

    @property
    def active_generation_model(self) -> str:
        return self.ollama_model if self.uses_ollama else self.generation_model
