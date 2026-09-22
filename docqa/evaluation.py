"""Controlled ablations, element-level retrieval metrics and explicit human review."""

import copy
import hashlib
import importlib.metadata
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .generation import SYSTEM_PROMPT
from .retrieval import RETRIEVAL_VERSION
from .schemas import ChatRequest, RetrieveRequest, Source, Strategy

Group = Literal["A", "B", "C", "D", "E", "F"]
GROUPS = {
    "A": ("bm25", False, False),
    "B": ("dense", False, False),
    "C": ("hybrid", False, False),
    "D": ("hybrid_reranker", False, False),
    "E": ("hybrid_reranker", False, True),
    "F": ("hybrid_reranker", True, True),
}


class GoldSource(Source):
    document_id: str
    document_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class EvalQuestion(BaseModel):
    id: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=4000)
    relevant_element_ids: list[str] = Field(default_factory=list)
    category: Literal["text", "table", "chart", "unanswerable"] = "text"
    split: Literal["dev", "test"] = "dev"
    answerable: bool = True
    reference_answer: str = ""
    fact_group: str = ""
    reference_sources: list[GoldSource] = Field(default_factory=list)
    scoring_points: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_gold(self):
        if not self.question.strip():
            raise ValueError("问题不可为空白")
        if self.answerable and not self.relevant_element_ids:
            raise ValueError("有答案问题必须提供原始元素金标准")
        if not self.answerable and self.relevant_element_ids:
            raise ValueError("无答案问题不应包含相关证据")
        if self.category == "unanswerable" and self.answerable:
            raise ValueError("无答案类别必须设置 answerable=false")
        if not self.answerable:
            self.category = "unanswerable"
        return self


class ExperimentRequest(BaseModel):
    questions: list[EvalQuestion] = Field(min_length=1, max_length=1000)
    strategies: list[Strategy] = Field(
        default_factory=lambda: ["bm25", "dense", "hybrid", "hybrid_reranker"], min_length=1
    )
    groups: list[Group] | None = Field(default=None, min_length=1)
    document_ids: list[str] | None = None
    split: Literal["dev", "test"] = "dev"
    generate: bool = False
    expected_provider: Literal["bailian", "ollama"] | None = None

    @model_validator(mode="after")
    def validate_dataset(self):
        if len({q.id for q in self.questions}) != len(self.questions):
            raise ValueError("题目 ID 不可重复")
        if not any(q.split == self.split for q in self.questions):
            raise ValueError("所选划分没有题目")
        facts, wording = {}, {}
        for q in self.questions:
            for key, seen in [(q.fact_group.strip(), facts), (q.question.strip(), wording)]:
                if key and seen.setdefault(key, q.split) != q.split:
                    raise ValueError("同一事实组或相同问题不能跨 dev/test，避免数据泄漏")
            if self.generate and q.split == self.split and q.answerable and not q.reference_answer.strip():
                raise ValueError("生成评测的有答案问题需要 reference_answer 供人工评分")
        return self

    def variants(self):
        if self.groups is not None:
            return [(g, *GROUPS[g]) for g in dict.fromkeys(self.groups)]
        return [(s, s, False, False) for s in dict.fromkeys(self.strategies)]


class Review(BaseModel):
    question_id: str
    group: str
    answer_accuracy: Literal[0, 0.5, 1] | None = None
    is_refusal: bool
    citation_supported: bool | None = None
    reviewer: str = Field(min_length=1, max_length=100)
    note: str = Field(default="", max_length=2000)

    @field_validator("reviewer")
    @classmethod
    def reviewer_required(cls, value):
        if not value.strip():
            raise ValueError("评分人不可为空白")
        return value.strip()


def retrieval_metrics(retrieved, relevant):
    unique = list(dict.fromkeys(retrieved))
    gold = set(relevant)
    if not gold:
        return {**{f"recall@{k}": None for k in [1, 3, 5, 10]}, "mrr@10": None}
    metrics = {f"recall@{k}": len(set(unique[:k]) & gold) / len(gold) for k in [1, 3, 5, 10]}
    metrics["mrr@10"] = next((1 / rank for rank, eid in enumerate(unique[:10], 1) if eid in gold), 0.0)
    return metrics


def summarize(rows, by_category=False):
    result = []
    keys = list(dict.fromkeys((r["group"], r["category"] if by_category else "all") for r in rows))
    for group, category in keys:
        subset = [r for r in rows if r["group"] == group and (category == "all" or r["category"] == category)]
        answerable = [r for r in subset if r["answerable"]]
        unanswerable = [r for r in subset if not r["answerable"]]
        reviewed = [r for r in subset if r.get("review")]
        scored = [r for r in answerable if r.get("answer_accuracy") is not None]
        refused = [r for r in unanswerable if r.get("review")]
        false_refusals = [r for r in answerable if r.get("review")]
        citation_checked = [
            r for r in reviewed if r.get("review", {}).get("citation_supported") is not None
        ]
        item = {
            "group": group,
            "strategy": subset[0]["strategy"],
            "category": category,
            "questions": len(subset),
            "answerable_questions": len(answerable),
            "reviewed_questions": len(reviewed),
            "scored_answers": len(scored),
            "review_coverage": len(reviewed) / len(subset),
            "answer_accuracy": sum(r["answer_accuracy"] for r in scored) / len(scored) if scored else None,
            "refusal_reviewed": len(refused),
            "correct_refusal_rate": sum(r["review"]["is_refusal"] for r in refused) / len(refused)
            if refused
            else None,
            "false_refusal_rate": sum(r["review"]["is_refusal"] for r in false_refusals) / len(false_refusals)
            if false_refusals
            else None,
            "citation_reviewed": len(citation_checked),
            "citation_supported_rate": sum(
                r["review"]["citation_supported"] for r in citation_checked
            )
            / len(citation_checked)
            if citation_checked
            else None,
            "generation_failures": sum(
                r.get("answer", {}).get("status") in {"generation_error", "citation_error"} for r in subset
            ),
        }
        for metric in ["recall@1", "recall@3", "recall@5", "recall@10", "mrr@10", "seconds"]:
            values = [r[metric] for r in subset if r[metric] is not None]
            item[metric] = sum(values) / len(values) if values else None
        result.append(item)
    return result


def apply_review(result, review):
    row = next(
        (r for r in result["rows"] if r["question_id"] == review.question_id and r["group"] == review.group),
        None,
    )
    if row is None:
        raise ValueError("逐题结果不存在")
    if row.get("answer", {}).get("status") not in {"answered", "no_evidence"}:
        raise ValueError("只可评分正常生成或没有证据的回答；失败调用不能当作拒答")
    if row["answerable"] and review.answer_accuracy is None:
        raise ValueError("有答案问题须给出 0、0.5 或 1 分")
    if not row["answerable"] and review.answer_accuracy is not None:
        raise ValueError("无答案问题按拒答统计，不混入答案准确率")
    if row["answerable"] and review.is_refusal and review.answer_accuracy != 0:
        raise ValueError("有答案问题被拒答时应记 0 分")
    stamped = {**review.model_dump(), "reviewed_at": datetime.now(timezone.utc).isoformat()}
    row.setdefault("review_history", []).append(stamped)
    row["review"] = stamped
    row["answer_accuracy"] = review.answer_accuracy
    result["summary"] = summarize(result["rows"])
    result["category_summary"] = summarize(result["rows"], by_category=True)
    return result


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()


class ExperimentPaused(Exception):
    def __init__(self, result):
        super().__init__("实验已暂停，已完成结果已保留")
        self.result = result


def run_experiment(request, retriever, store, pipeline=None, progress=None, resume=None, should_stop=None):
    elements = store.elements(request.document_ids)
    allowed = {e.id for e in elements}
    selected = [q for q in request.questions if q.split == request.split]
    for q in selected:
        if not set(q.relevant_element_ids) <= allowed:
            raise ValueError(f"问题 {q.id} 的金标准不在当前文档范围内，请更新标注")
        for source in q.reference_sources:
            doc = store.document(source.document_id)
            if doc is None or doc["sha256"] != source.document_sha256:
                raise ValueError(f"问题 {q.id} 的原文摘要不匹配，请核对文档版本")
    if request.generate and (pipeline is None or not pipeline.settings.generation_configured):
        raise ValueError(
            "生成评测需要配置 DOCQA_API_KEY，或将 DOCQA_GENERATION_PROVIDER=ollama 后启动本地模型；可先关闭生成，仅运行本地检索"
        )
    variants = request.variants()
    if any(v[2] for v in variants) and not retriever.settings.enable_clip:
        raise ValueError("F 组需要开启 DOCQA_ENABLE_CLIP 并准备模型")
    settings = pipeline.settings if pipeline else retriever.settings
    doc_ids = {e.document_id for e in elements}
    documents = [
        {k: d.get(k) for k in ["id", "name", "sha256", "parser_version", "page_count", "element_count"]}
        for d in store.documents()
        if d["id"] in doc_ids
    ]
    versions = {}
    for package in ["docling", "PyMuPDF", "sentence-transformers", "faiss-cpu", "cn-clip"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    result = {
        "config": request.model_dump(mode="json"),
        "rows": [],
        "summary": [],
        "category_summary": [],
        "provenance": {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "packages": versions,
            "documents": documents,
            "dataset_sha256": hashlib.sha256(canonical(request.model_dump()["questions"])).hexdigest(),
            "elements_sha256": hashlib.sha256(canonical([e.model_dump() for e in elements])).hexdigest(),
            "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
            "retrieval_version": RETRIEVAL_VERSION,
            "code_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(Path(__file__).parent.glob("*.py"))
            },
            "models": {
                "embedding": settings.embedding_model,
                "reranker": settings.reranker_model,
                "clip": settings.clip_model,
                "generation": settings.active_generation_model,
            },
            "generation_provider": "ollama" if settings.uses_ollama else "bailian",
            "generation_endpoint_sha256": hashlib.sha256(
                (settings.ollama_base if settings.uses_ollama else settings.api_base).encode()
            ).hexdigest(),
            "generation_timeout": settings.ollama_timeout if settings.uses_ollama else settings.api_timeout,
            "generation_context_length": settings.ollama_context_length if settings.uses_ollama else None,
            "device": settings.device,
            "generation_top_k": 6,
            "max_images": 3,
            "context_char_budget": 6000,
            "temperature": 0,
            "timing_policy": "顺序运行；包含首次模型加载与索引构建，逐题保留耗时，不视作稳态吞吐量",
        },
        "notes": [
            "无答案问题不计入 Recall/MRR；未评分答案为 null。",
            "准确率与拒答率仅统计已人工评分的对应子集；结合覆盖率解读。",
            "引用编号合法不等于语义支持正确；后者由人工检查。",
            "E/F 的生成差异仅在启用生成时可评估；离线运行不代表视觉回答质量。",
        ],
    }
    total = len(variants) * len(selected)
    if resume:
        if resume.get("config") != result["config"]:
            raise ValueError("续跑配置与原实验不一致，请新建实验")
        for key, value in result["provenance"].items():
            if key in {"started_at"}:
                continue
            if resume.get("provenance", {}).get(key) != value:
                raise ValueError(f"续跑验证失败：{key} 已改变，请新建实验以保留对比有效性")
        result["rows"] = copy.deepcopy(resume.get("rows", []))
        result["provenance"] = copy.deepcopy(resume["provenance"])
        result["provenance"].setdefault("resumed_at", []).append(datetime.now(timezone.utc).isoformat())
    completed_keys = {(r["group"], r["question_id"]) for r in result["rows"]}
    expected_keys = {(group, q.id) for group, *_ in variants for q in selected}
    if len(completed_keys) != len(result["rows"]) or not completed_keys <= expected_keys:
        raise ValueError("已保存结果存在重复或范围外题目，不能续跑")
    result.update(completed=len(result["rows"]), total=total)
    if progress:
        progress(result)
    for group, strategy, multimodal, include_images in variants:
        for q in selected:
            if (group, q.id) in completed_keys:
                continue
            if should_stop and should_stop():
                result["summary"] = summarize(result["rows"])
                result["category_summary"] = summarize(result["rows"], by_category=True)
                result["provenance"]["paused_at"] = datetime.now(timezone.utc).isoformat()
                raise ExperimentPaused(result)
            started = time.perf_counter()
            evidence, trace = retriever.retrieve(
                RetrieveRequest(
                    question=q.question,
                    document_ids=request.document_ids,
                    strategy=strategy,
                    top_k=10,
                    multimodal=multimodal,
                )
            )
            ids = [e.element.id for e in evidence]
            row = {
                "question_id": q.id,
                "question": q.question,
                "reference_answer": q.reference_answer,
                "scoring_points": q.scoring_points,
                "gold_evidence": [e.model_dump() for e in elements if e.id in q.relevant_element_ids],
                "group": group,
                "strategy": strategy,
                "multimodal": multimodal,
                "include_images": include_images,
                "category": q.category,
                "answerable": q.answerable,
                "retrieved": ids,
                "gold": q.relevant_element_ids,
                **retrieval_metrics(ids, q.relevant_element_ids),
                "answer_accuracy": None,
                "trace": trace,
            }
            if request.generate:
                answer = pipeline.run(
                    ChatRequest(
                        question=q.question,
                        document_ids=request.document_ids,
                        strategy=strategy,
                        top_k=6,
                        multimodal=multimodal,
                        include_images=include_images,
                        generate=True,
                    ),
                    retrieval_result=(evidence[:6], trace),
                )
                row["answer"] = answer.model_dump()
            row["seconds"] = time.perf_counter() - started
            result["rows"].append(row)
            result["completed"] = len(result["rows"])
            result["total"] = total
            if progress:
                progress(result)
    result["summary"] = summarize(result["rows"])
    result["category_summary"] = summarize(result["rows"], by_category=True)
    result["provenance"]["finished_at"] = datetime.now(timezone.utc).isoformat()
    return result
