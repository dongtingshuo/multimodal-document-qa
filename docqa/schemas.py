from typing import Literal, Protocol

from pydantic import BaseModel, Field, field_validator

Strategy = Literal["bm25", "dense", "hybrid", "hybrid_reranker"]


class Source(BaseModel):
    page: int = Field(ge=1)
    bbox: tuple[float, float, float, float]

    @field_validator("bbox")
    @classmethod
    def valid_bbox(cls, box):
        x0, y0, x1, y1 = box
        if not (0 <= x0 <= x1 <= 1 and 0 <= y0 <= y1 <= 1):
            raise ValueError("bbox 必须是左上角原点的归一化区域")
        return box


class Element(BaseModel):
    id: str
    document_id: str
    kind: Literal["text", "table", "image", "page"]
    text: str
    sources: list[Source] = Field(min_length=1)
    image_path: str | None = None
    table: list[list[str]] | None = None
    title_path: list[str] = Field(default_factory=list)


class Chunk(BaseModel):
    id: str
    element_id: str
    document_id: str
    text: str


class Evidence(BaseModel):
    element: Element
    document_name: str
    page_label: str
    matched_text: str
    rank: int
    channels: list[str]
    scores: dict[str, float]


class RetrieveRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    document_ids: list[str] | None = None
    strategy: Strategy = "bm25"
    top_k: int = Field(default=6, ge=1, le=30)
    multimodal: bool = False
    kinds: list[Literal["text", "table", "image", "page"]] | None = None

    @field_validator("question")
    @classmethod
    def nonempty(cls, value):
        if not value.strip():
            raise ValueError("问题不能为空")
        return value.strip()


class ChatRequest(RetrieveRequest):
    session_id: str | None = None
    include_images: bool = True
    generate: bool = True


class GenerationConfigRequest(BaseModel):
    provider: Literal["bailian", "ollama"]
    model: str = Field(min_length=1, max_length=160)

    @field_validator("model")
    @classmethod
    def nonempty_model(cls, value):
        return value.strip()


class Answer(BaseModel):
    run_id: str
    session_id: str
    question: str
    retrieval_question: str
    text: str
    status: str
    citations: dict[str, Evidence]
    evidence: list[Evidence]
    strategy: str
    model: str | None = None
    usage: dict = Field(default_factory=dict)
    timings: dict[str, float] = Field(default_factory=dict)
    image_count: int = 0
    warnings: list[str] = Field(default_factory=list)
    context: list[dict] = Field(default_factory=list)
    support_checks: dict = Field(default_factory=dict)
    uncertainty: list[dict] = Field(default_factory=list)


class Parser(Protocol):
    def parse(self, path, document_id: str, output_dir) -> dict: ...


class Retriever(Protocol):
    def retrieve(self, request: RetrieveRequest) -> tuple[list[Evidence], dict]: ...


class Reranker(Protocol):
    def predict(self, pairs: list[tuple[str, str]]): ...


class Generator(Protocol):
    def complete(self, messages: list[dict]) -> dict: ...
