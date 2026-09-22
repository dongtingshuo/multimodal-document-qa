import copy
import fcntl
import hashlib
import os
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from .config import GENERATION_MODEL_OPTIONS
from .evaluation import ExperimentPaused, ExperimentRequest, run_experiment
from .generation import RAGPipeline, create_generator
from .parsers import DoclingParser, LocalParser
from .retrieval import HybridRetriever
from .store import Store


class Service:
    def __init__(self, settings):
        self.settings = settings
        self.store = Store(settings.data_dir)
        self.instance_lock = (settings.data_dir / "backend.lock").open("a+")
        try:
            fcntl.flock(self.instance_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.instance_lock.close()
            raise RuntimeError("该数据目录已有后端运行，请使用现有实例") from None
        self.instance_lock.seek(0)
        self.instance_lock.truncate()
        self.instance_lock.write(str(os.getpid()))
        self.instance_lock.flush()
        self.lock = threading.RLock()
        self.chat_lock = threading.Lock()
        self.submission_lock = threading.Lock()
        self.experiment_controls = threading.Lock()
        self.experiment_events = {}
        self.chat_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="docqa-chat")
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="docqa")
        self.parser = DoclingParser(settings) if settings.parser == "docling" else LocalParser(settings)
        self.retriever = HybridRetriever(self.store, settings)
        self._restore_generation_config()
        self.pipeline = RAGPipeline(
            self.retriever, create_generator(settings, self.store), self.store, settings
        )
        for doc in self.store.documents():
            if doc["status"] in {"queued", "parsing", "indexing"}:
                doc.update(status="failed", error="上次处理被中断，可重试")
                self.store.put_document(doc)
                self.store.put(
                    "jobs",
                    doc["job_id"],
                    {
                        "id": doc["job_id"],
                        "document_id": doc["id"],
                        "status": "failed",
                        "error": doc["error"],
                    },
                )
        for experiment in self.store.records("experiments"):
            if experiment["status"] in {"queued", "running"}:
                experiment.update(status="paused" if experiment.get("provenance") else "failed",
                                  error="上次实验被中断，已有结果已保留；可在配置一致时继续实验")
                self.store.put("experiments", experiment["id"], experiment)

    def generation_config(self):
        return {
            "provider": "ollama" if self.settings.uses_ollama else "bailian",
            "model": self.settings.active_generation_model,
            "configured": self.settings.generation_configured,
            "api_key_configured": bool(self.settings.api_key),
            "options": GENERATION_MODEL_OPTIONS,
        }

    def _restore_generation_config(self):
        saved = self.store.get("settings", "generation")
        if not isinstance(saved, dict):
            return
        provider, model = saved.get("provider"), saved.get("model")
        if provider not in GENERATION_MODEL_OPTIONS:
            return
        # Migrate earlier saved choices to the two models exposed by the UI.
        if model not in GENERATION_MODEL_OPTIONS[provider]:
            model = GENERATION_MODEL_OPTIONS[provider][0]
        self.settings.generation_provider = provider
        if provider == "ollama":
            self.settings.ollama_model = model.strip()
        else:
            self.settings.generation_model = model.strip()

    def configure_generation(self, provider, model):
        provider = str(provider).strip().lower()
        model = str(model).strip()
        if provider not in GENERATION_MODEL_OPTIONS:
            raise ValueError("生成后端只能选择 bailian 或 ollama")
        if not model:
            raise ValueError("生成模型不能为空")
        if model not in GENERATION_MODEL_OPTIONS[provider]:
            raise ValueError("该模型不属于所选后端，请使用界面提供的模型选项")
        with self.lock:
            self.settings.generation_provider = provider
            if provider == "ollama":
                self.settings.ollama_model = model
            else:
                self.settings.generation_model = model
            self.pipeline.generator = create_generator(self.settings, self.store)
            self.store.put("settings", "generation", {"provider": provider, "model": model})
            return self.generation_config()

    def import_document(self, filename, content):
        name = Path(filename.replace("\\", "/")).name
        suffix = Path(name).suffix.lower()
        if suffix not in {".pdf", ".docx", ".doc", ".png", ".jpg", ".jpeg"}:
            raise ValueError("支持 PDF、DOCX、DOC、PNG、JPG")
        if not content or len(content) > self.settings.max_upload_bytes:
            raise ValueError("文件为空或超过 50 MB")
        if suffix == ".pdf" and b"%PDF-" not in content[:1024]:
            raise ValueError("文件内容不是 PDF")
        digest = hashlib.sha256(content).hexdigest()
        with self.lock:
            for doc in self.store.documents():
                if doc["sha256"] == digest:
                    return {**doc, "duplicate": True}
            doc_id, job_id = digest[:24], uuid.uuid4().hex
            root = self.settings.data_dir / "documents" / doc_id
            root.mkdir(parents=True, exist_ok=True)
            original = root / ("original" + suffix)
            original.write_bytes(content)
            doc = {
                "id": doc_id,
                "name": name,
                "sha256": digest,
                "original_path": str(original),
                "status": "queued",
                "job_id": job_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "warnings": [],
                "size_bytes": len(content),
                "error": None,
            }
            self.store.put_document(doc)
            self.store.put("jobs", job_id, {"id": job_id, "document_id": doc_id, "status": "queued"})
            self.worker.submit(self.process, doc_id)
            return doc

    def process(self, doc_id):
        started = time.perf_counter()
        with self.lock:
            doc = self.store.document(doc_id)
            if not doc:
                return
            try:
                self.status(doc, "parsing")
                output = self.settings.data_dir / "documents" / doc_id / "parsed"
                parsed = self.parser.parse(Path(doc["original_path"]), doc_id, output)
                elements = parsed.pop("elements")
                if not elements:
                    raise ValueError("文档未提取出可用元素")
                self.status(doc, "indexing")
                self.store.replace_elements(doc_id, elements)
                chunks = self.retriever.chunks(elements)
                (output / "chunks.json").write_text(
                    __import__("json").dumps([c.model_dump() for c in chunks], ensure_ascii=False),
                    encoding="utf-8",
                )
                self.retriever.invalidate()
                doc.update(
                    parsed,
                    element_count=len(elements),
                    chunk_count=len(chunks),
                    parse_seconds=time.perf_counter() - started,
                    index_status={
                        "bm25": "ready",
                        "dense": "lazy",
                        "clip": "lazy" if self.settings.enable_clip else "disabled",
                    },
                )
                self.status(doc, "ready")
            except Exception as exc:
                # No model keys or HTTP response bodies are included in persisted errors.
                doc["error"] = (
                    str(exc)[:400] if isinstance(exc, (ValueError, RuntimeError)) else type(exc).__name__
                )
                self.status(doc, "failed")

    def status(self, doc, value):
        doc["status"] = value
        self.store.put_document(doc)
        self.store.put(
            "jobs",
            doc["job_id"],
            {"id": doc["job_id"], "document_id": doc["id"], "status": value, "error": doc.get("error")},
        )

    def retry(self, doc_id):
        # Queue submission must not wait for a long-running parse or chat.
        with self.submission_lock:
            doc = self.store.document(doc_id)
            if doc is None:
                raise KeyError(doc_id)
            if doc["status"] in {"queued", "parsing", "indexing"}:
                return doc
            doc.update(job_id=uuid.uuid4().hex, error=None)
            self.status(doc, "queued")
            self.worker.submit(self.process, doc_id)
            return doc

    def delete(self, doc_id):
        with self.lock, self.submission_lock:
            if self.store.document(doc_id) is None:
                raise KeyError(doc_id)
            self.store.purge_document_history(doc_id)
            self.store.delete(doc_id)
            self.retriever.invalidate()
            shutil.rmtree(self.settings.data_dir / "documents" / doc_id, ignore_errors=True)
            shutil.rmtree(self.settings.data_dir / "evidence-images" / doc_id, ignore_errors=True)

    def experiment(self, request, previous=None):
        # Freeze provider/model at submission; a queued task must not silently
        # change provider (and billing) when the user switches the sidebar.
        with self.lock:
            frozen = replace(self.settings)
            generator = copy.copy(self.pipeline.generator)
            generator.settings = frozen
        provider = "ollama" if frozen.uses_ollama else "bailian"
        if request.generate and request.expected_provider and request.expected_provider != provider:
            raise ValueError("生成后端已改变，请刷新页面并重新确认模型")
        if request.generate and not frozen.generation_configured:
            raise ValueError("尚未配置生成服务；请配置 DOCQA_API_KEY，或选择 Ollama 本地模型")
        if any(v[2] for v in request.variants()) and not frozen.enable_clip:
            raise ValueError("F 组需要开启 DOCQA_ENABLE_CLIP")
        if request.generate and not frozen.uses_ollama:
            needed = sum(q.split == request.split for q in request.questions) * len(request.variants())
            needed -= len(previous.get("rows", [])) if previous else 0
            available = max(0, frozen.max_api_calls - self.store.call_count())
            if needed > available:
                raise ValueError(
                    f"本次计划生成 {needed} 条，项目调用预算仅剩 {available} 次；请缩小题集或实验组。该预算不是云端账户免费额度。"
                )
        eid = previous["id"] if previous else uuid.uuid4().hex
        stop = threading.Event()
        with self.experiment_controls:
            if eid in self.experiment_events:
                raise ValueError("该实验已在队列或运行中，请勿重复提交")
            self.experiment_events[eid] = stop
        queued = {**(previous or {}), "id": eid, "status": "queued", "config": request.model_dump(mode="json"), "error": None}
        self.store.put("experiments", eid, queued)
        pipeline = RAGPipeline(self.retriever, generator, self.store, frozen)

        def work():
            self.store.put("experiments", eid, {**queued, "status": "running"})

            def progress(partial):
                self.store.put("experiments", eid, {"id": eid, "status": "running", **partial})

            try:
                with self.lock:
                    result = run_experiment(request, self.retriever, self.store, pipeline, progress,
                                            resume=previous, should_stop=stop.is_set)
                self.store.put("experiments", eid, {"id": eid, "status": "ready", **result})
            except ExperimentPaused as exc:
                self.store.put("experiments", eid, {"id": eid, "status": "paused", **exc.result})
            except Exception as exc:
                partial = self.store.get("experiments", eid) or {}
                partial.update(id=eid, status="failed", error=str(exc)[:400])
                self.store.put("experiments", eid, partial)
            finally:
                with self.experiment_controls:
                    self.experiment_events.pop(eid, None)

        self.worker.submit(work)
        return {"id": eid, "status": "queued"}

    def pause_experiment(self, eid):
        with self.experiment_controls:
            event = self.experiment_events.get(eid)
            if event is None:
                raise ValueError("该实验当前没有运行")
            event.set()
        return {"id": eid, "status": "pausing", "message": "将在当前题目完成后暂停，已完成结果会保留"}

    def resume_experiment(self, eid):
        previous = self.store.get("experiments", eid)
        if previous is None:
            raise KeyError(eid)
        if previous.get("status") not in {"paused", "failed"} or not previous.get("provenance"):
            raise ValueError("只有保留完整检查点的暂停或中断实验可以继续")
        return self.experiment(ExperimentRequest.model_validate(previous["config"]), previous=previous)

    def close(self):
        with self.experiment_controls:
            for event in self.experiment_events.values():
                event.set()
        self.chat_worker.shutdown(wait=True)
        self.worker.shutdown(wait=True)
        self.instance_lock.close()
