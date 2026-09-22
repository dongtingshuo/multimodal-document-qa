import importlib.util
import io
import json
import queue
import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from PIL import Image, ImageDraw

from .config import GENERATION_MODEL_OPTIONS, Settings
from .evaluation import ExperimentRequest, Review, apply_review
from .generation import generation_status
from .review_io import export_reviews, import_reviews
from .schemas import ChatRequest, GenerationConfigRequest, RetrieveRequest
from .service import Service


def create_app(settings=None):
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app):
        app.state.service = Service(settings)
        try:
            yield
        finally:
            app.state.service.close()

    app = FastAPI(title="多模态文档问答", version="0.1.0", lifespan=lifespan)

    def service():
        return app.state.service

    def document(doc_id):
        doc = service().store.document(doc_id)
        if not doc:
            raise HTTPException(404, "文档不存在")
        return doc

    @app.get("/api/v1/health")
    def health():
        return {
            "status": "ok",
            "parser": settings.parser,
            "parser_version": service().parser.version,
            "generation_configured": settings.generation_configured,
            "api_key_configured": bool(settings.api_key),
            "generation_provider": "ollama" if settings.uses_ollama else "bailian",
            "generation_model": settings.active_generation_model,
            "generation_options": GENERATION_MODEL_OPTIONS,
            "embedding_model": settings.embedding_model,
            "reranker_model": settings.reranker_model,
            "clip_enabled": settings.enable_clip,
            "device": settings.device,
            "api_calls_used": service().store.call_count(),
            "api_calls_limit": settings.max_api_calls,
            "dependencies": {
                m: importlib.util.find_spec(m) is not None
                for m in ["docling", "rapidocr", "cn_clip", "sentence_transformers"]
            },
        }

    @app.get("/api/v1/generation-status")
    def check_generation_status():
        with service().lock:
            return {**service().generation_config(), **generation_status(settings)}

    @app.put("/api/v1/generation-config")
    def generation_config(request: GenerationConfigRequest):
        try:
            with service().lock:
                return service().configure_generation(request.provider, request.model)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/v1/documents", status_code=202)
    def upload(file: Annotated[UploadFile, File()]):
        data = file.file.read(settings.max_upload_bytes + 1)
        file.file.close()
        try:
            return service().import_document(file.filename or "unnamed", data)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/v1/documents")
    def documents():
        return service().store.documents()

    @app.get("/api/v1/documents/{doc_id}/elements")
    def elements(doc_id: str):
        document(doc_id)
        return service().store.elements([doc_id])

    @app.post("/api/v1/documents/{doc_id}/retry", status_code=202)
    def retry(doc_id: str):
        document(doc_id)
        return service().retry(doc_id)

    @app.delete("/api/v1/documents/{doc_id}")
    def delete(doc_id: str):
        document(doc_id)
        service().delete(doc_id)
        return {"deleted": doc_id}

    @app.get("/api/v1/jobs/{job_id}")
    def job(job_id: str):
        result = service().store.get("jobs", job_id)
        if result is None:
            raise HTTPException(404, "任务不存在")
        return result

    @app.post("/api/v1/retrieve")
    def retrieve(request: RetrieveRequest):
        try:
            with service().lock:
                evidence, trace = service().retriever.retrieve(request)
            return {"evidence": evidence, "trace": trace}
        except Exception as exc:
            raise HTTPException(503, "检索失败，请检查模型文件及依赖；" + type(exc).__name__) from exc

    @app.post("/api/v1/chat")
    def chat(request: ChatRequest):
        try:
            with service().chat_lock, service().lock:
                return service().pipeline.run(request)
        except Exception as exc:
            raise HTTPException(503, "问答检索阶段失败，请检查模型文件及依赖；" + type(exc).__name__) from exc

    @app.post("/api/v1/chat/stream")
    def chat_stream(request: ChatRequest):
        events = queue.Queue()
        current_service = service()

        def work():
            try:
                with current_service.chat_lock, current_service.lock:
                    result = current_service.pipeline.run(
                        request, progress=lambda stage: events.put({"type": "progress", "stage": stage}))
                events.put({"type": "result", "answer": result.model_dump()})
            except Exception as exc:
                events.put({"type": "error", "message": "问答处理失败，请检查模型文件及依赖；" + type(exc).__name__})

        def stream():
            started, stage = time.monotonic(), "queued"
            yield json.dumps({"type": "progress", "stage": stage, "elapsed": 0}) + "\n"
            future = current_service.chat_worker.submit(work)
            try:
                while True:
                    try:
                        event = events.get(timeout=1)
                    except queue.Empty:
                        event = {"type": "progress", "stage": stage}
                    stage = event.get("stage", stage)
                    event["elapsed"] = round(time.monotonic() - started, 1)
                    yield json.dumps(event, ensure_ascii=False) + "\n"
                    if event["type"] in {"result", "error"}:
                        break
            finally:
                # Cancel only work that has not started. A running request finishes
                # once and saves its answer; disconnecting never resubmits it.
                future.cancel()

        return StreamingResponse(stream(), media_type="application/x-ndjson",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/v1/evidence/{eid}")
    def evidence(eid: str):
        element = service().store.element(eid)
        if not element:
            raise HTTPException(404, "证据不存在或已删除")
        return element

    @app.get("/api/v1/evidence/{eid}/image")
    def evidence_image(eid: str):
        element = evidence(eid)
        if not element.image_path:
            raise HTTPException(404, "该证据没有裁剪图")
        return FileResponse(element.image_path, media_type="image/png")

    @app.get("/api/v1/documents/{doc_id}/pages/{page}")
    def page_image(doc_id: str, page: int, highlight: str | None = None):
        doc = document(doc_id)
        if page < 1 or page > doc.get("page_count", 0):
            raise HTTPException(404, "页面不存在")
        path = settings.data_dir / "documents" / doc_id / "parsed" / f"page-{page}.png"
        if not path.exists():
            raise HTTPException(404, "页面图尚未生成")
        if not highlight:
            return FileResponse(path, media_type="image/png")
        element = evidence(highlight)
        if element.document_id != doc_id or not any(s.page == page for s in element.sources):
            raise HTTPException(400, "证据不属于此页面")
        with Image.open(path) as source:
            img = source.convert("RGB")
        draw = ImageDraw.Draw(img, "RGBA")
        for position in element.sources:
            if position.page == page:
                x0, y0, x1, y1 = position.bbox
                draw.rectangle(
                    (x0 * img.width, y0 * img.height, x1 * img.width, y1 * img.height),
                    fill=(255, 196, 36, 45),
                    outline=(235, 99, 24, 255),
                    width=4,
                )
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return Response(buffer.getvalue(), media_type="image/png")

    @app.post("/api/v1/experiments", status_code=202)
    def experiment(request: ExperimentRequest):
        try:
            return service().experiment(request)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/v1/experiments")
    def experiments():
        return service().store.records("experiments")

    @app.get("/api/v1/experiments/{experiment_id}")
    def experiment_result(experiment_id: str):
        result = service().store.get("experiments", experiment_id)
        if result is None:
            raise HTTPException(404, "实验不存在")
        return result

    @app.post("/api/v1/experiments/{experiment_id}/pause", status_code=202)
    def pause_experiment(experiment_id: str):
        try:
            return service().pause_experiment(experiment_id)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/v1/experiments/{experiment_id}/resume", status_code=202)
    def resume_experiment(experiment_id: str):
        try:
            return service().resume_experiment(experiment_id)
        except KeyError as exc:
            raise HTTPException(404, "实验不存在") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/v1/experiments/{experiment_id}/review-sheet")
    def review_sheet(experiment_id: str):
        result = experiment_result(experiment_id)
        if result["status"] != "ready":
            raise HTTPException(409, "实验尚未完成")
        return Response(export_reviews(result), media_type="text/csv",
                        headers={"Content-Disposition": 'attachment; filename="review.csv"'})

    @app.post("/api/v1/experiments/{experiment_id}/review-sheet")
    def import_review_sheet(experiment_id: str, file: Annotated[UploadFile, File()]):
        content = file.file.read(10 * 1024 * 1024 + 1)
        file.file.close()
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(400, "评分文件不得超过 10 MB")
        with service().lock:
            result = experiment_result(experiment_id)
            try:
                updated, count = import_reviews(result, content)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            service().store.put("experiments", experiment_id, updated)
        return {"id": experiment_id, "imported": count}

    @app.post("/api/v1/experiments/{experiment_id}/reviews")
    def review_answer(experiment_id: str, review: Review):
        with service().lock:
            result = experiment_result(experiment_id)
            if result["status"] != "ready":
                raise HTTPException(409, "实验尚未完成")
            try:
                result = apply_review(result, review)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            service().store.put("experiments", experiment_id, result)
            return result

    return app


app = create_app()
