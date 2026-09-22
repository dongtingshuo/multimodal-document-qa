import io
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pymupdf as fitz
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from docqa.api import create_app
from docqa.config import Settings
from docqa.evaluation import ExperimentRequest, retrieval_metrics
from docqa.generation import (
    CloudGenerator,
    OllamaGenerator,
    RAGPipeline,
    evidence_context,
    normalize_citation_format,
    validate_citations,
)
from docqa.parsers import LocalParser
from docqa.retrieval import chunks_for, rrf, table_query_hint
from docqa.schemas import ChatRequest, Element, Evidence, Source
from docqa.store import Store


def test_ocr_missing_coordinates_preserves_page_fallback(tmp_path):
    parser = LocalParser(Settings(data_dir=tmp_path, enable_ocr=True))
    parser._ocr = lambda path: SimpleNamespace(txts=["recognized text"], boxes=None)
    source = tmp_path / "scan.pdf"
    with fitz.open() as pdf:
        pdf.new_page()
        pdf.save(source)
    result = parser.parse(source, "scan", tmp_path / "parsed")
    assert result["page_count"] == 1
    assert any(e.kind == "page" and Path(e.image_path).is_file() for e in result["elements"])
    assert not result["warnings"]


def test_local_parser_associates_charts_with_nearby_captions(tmp_path):
    source = tmp_path / "two-charts.pdf"
    with fitz.open() as pdf:
        page = pdf.new_page(width=600, height=800)
        page.insert_text((40, 60), "Unrelated document introduction " * 2)
        page.insert_text((40, 240), "Figure A: library borrowing")
        page.draw_rect((40, 260, 230, 400), fill=(0.2, 0.5, 0.8))
        page.insert_text((340, 240), "Figure B: solar generation")
        page.draw_rect((340, 260, 530, 400), fill=(0.2, 0.8, 0.5))
        pdf.save(source)
    result = LocalParser(Settings(data_dir=tmp_path, enable_ocr=False)).parse(
        source, "charts", tmp_path / "parsed")
    images = sorted((e for e in result['elements'] if e.kind=='image'),
                    key=lambda e:e.sources[0].bbox[0])
    assert len(images)==2
    assert images[0].text.startswith('Figure A: library borrowing')
    assert images[1].text.startswith('Figure B: solar generation')


def pdf_bytes(rotation=0):
    with fitz.open() as pdf:
        page = pdf.new_page(width=600, height=800)
        page.insert_text((50, 70), "The project uses hybrid retrieval with evidence citations.", fontsize=12)
        page.insert_text((50, 140), "Storage: SQLite. Upload limit: 50 megabytes.", fontsize=12)
        page.draw_rect((70, 210, 220, 310), color=(0, 0.3, 0.8), fill=(0.8, 0.9, 1))
        page.set_rotation(rotation)
        return pdf.tobytes()


@pytest.fixture
def client(tmp_path):
    with TestClient(
        create_app(
            Settings(
                data_dir=tmp_path,
                parser="pymupdf",
                enable_ocr=False,
                api_key="",
                generation_provider="bailian",
            )
        )
    ) as client:
        yield client


def upload_ready(client, data=None):
    result = client.post(
        "/api/v1/documents", files={"file": ("demo.pdf", data or pdf_bytes(), "application/pdf")}
    )
    assert result.status_code == 202
    doc = result.json()
    client.app.state.service.worker.submit(lambda: None).result(timeout=30)
    job = client.get("/api/v1/jobs/" + doc["job_id"]).json()
    assert job["status"] == "ready", job
    return doc


def test_upload_retrieve_highlight_duplicate_delete(client):
    content = pdf_bytes()
    doc = upload_ready(client, content)
    duplicate = client.post(
        "/api/v1/documents", files={"file": ("copy.pdf", content, "application/pdf")}
    ).json()
    assert duplicate["duplicate"] and duplicate["id"] == doc["id"]
    result = client.post("/api/v1/retrieve", json={"question": "SQLite", "strategy": "bm25"}).json()
    assert result["evidence"]
    evidence = result["evidence"][0]
    assert "SQLite" in evidence["matched_text"]
    eid = evidence["element"]["id"]
    page = client.get(f"/api/v1/documents/{doc['id']}/pages/1", params={"highlight": eid})
    assert page.status_code == 200
    assert Image.open(io.BytesIO(page.content)).size == (1200, 1600)
    assert client.delete("/api/v1/documents/" + doc["id"]).status_code == 200
    assert client.get("/api/v1/evidence/" + eid).status_code == 404
    assert client.post("/api/v1/retrieve", json={"question": "SQLite"}).json()["evidence"] == []


def test_scope_and_no_evidence(client):
    upload_ready(client)
    assert (
        client.post("/api/v1/retrieve", json={"question": "SQLite", "document_ids": []}).json()["evidence"]
        == []
    )
    answer = client.post("/api/v1/chat", json={"question": "unrelatedxyzzzzz"}).json()
    assert answer["status"] == "no_evidence"
    answer = client.post("/api/v1/chat", json={"question": "SQLite"}).json()
    assert answer["status"] == "evidence_only" and not answer["citations"]


def test_invalid_upload_and_query(client):
    for name, content in [("a.pdf", b"not a pdf"), ("x.exe", b"abc"), ("empty.pdf", b"")]:
        assert client.post("/api/v1/documents", files={"file": (name, content)}).status_code == 400
    assert client.post("/api/v1/retrieve", json={"question": "  "}).status_code == 422
    assert client.post("/api/v1/retrieve", json={"question": "q", "top_k": 0}).status_code == 422


def test_generation_model_can_switch_from_ui(client):
    health = client.get("/api/v1/health").json()
    assert health["generation_provider"] == "bailian"
    assert health["generation_options"] == {
        "bailian": ["qwen3-vl-plus-2025-12-19"],
        "ollama": ["qwen2.5vl:7b"],
    }
    response = client.put(
        "/api/v1/generation-config",
        json={"provider": "ollama", "model": "qwen2.5vl:7b"},
    )
    assert response.status_code == 200
    assert response.json()["provider"] == "ollama"
    service = client.app.state.service
    assert isinstance(service.pipeline.generator, OllamaGenerator)
    assert service.store.get("settings", "generation") == {
        "provider": "ollama",
        "model": "qwen2.5vl:7b",
    }
    response = client.put(
        "/api/v1/generation-config",
        json={"provider": "bailian", "model": "qwen3-vl-plus-2025-12-19"},
    )
    assert response.status_code == 200
    assert response.json()["provider"] == "bailian"


def test_corrupt_pdf_failed_and_retry(client):
    doc = client.post("/api/v1/documents", files={"file": ("broken.pdf", b"%PDF-1.4\ninvalid")}).json()
    service = client.app.state.service
    service.worker.submit(lambda: None).result(timeout=30)
    assert service.store.document(doc["id"])["status"] == "failed"
    assert client.post(f"/api/v1/documents/{doc['id']}/retry").status_code == 202
    service.worker.submit(lambda: None).result(timeout=30)
    assert service.store.document(doc["id"])["status"] == "failed"


def test_rotated_page_coordinates(tmp_path):
    path = tmp_path / "rotated.pdf"
    path.write_bytes(pdf_bytes(rotation=90))
    result = LocalParser(Settings(data_dir=tmp_path, enable_ocr=False)).parse(
        path, "rotation", tmp_path / "out"
    )
    target = next(e for e in result["elements"] if "SQLite" in e.text)
    assert target.sources[0].bbox[0] > 0.7
    assert Image.open(tmp_path / "out/page-1.png").size == (1600, 1200)


def test_table_and_vector_chart(tmp_path):
    path = tmp_path / "table.pdf"
    with fitz.open() as pdf:
        page = pdf.new_page()
        for x in [50, 200, 350]:
            page.draw_line((x, 50), (x, 170))
        for y in [50, 90, 130, 170]:
            page.draw_line((50, y), (350, y))
        for y, row in [(75, ["Method", "Recall"]), (115, ["BM25", "0.60"]), (155, ["Hybrid", "0.85"])]:
            for x, text in zip([60, 210], row, strict=False):
                page.insert_text((x, y), text)
        page.draw_rect((50, 250, 150, 360), fill=(0.2, 0.4, 0.8))
        pdf.save(path)
    parsed = LocalParser(Settings(data_dir=tmp_path, enable_ocr=False)).parse(path, "table", tmp_path / "out")
    tables = [e for e in parsed["elements"] if e.kind == "table"]
    assert tables and "0.85" in tables[0].text
    assert any(e.kind == "image" for e in parsed["elements"])


def test_rrf_and_element_metrics():
    scores = rrf([["a", "a", "b"], ["b", "c"]])
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    metrics = retrieval_metrics(["x", "a", "a", "b"], ["a", "b"])
    assert metrics["recall@3"] == 1
    assert metrics["mrr@10"] == 0.5
    assert retrieval_metrics(["x"], [])["recall@5"] is None


def fixture_evidence(tmp_path):
    img = tmp_path / "figure.png"
    Image.new("RGB", (30, 30), "white").save(img)
    element = Element(
        id="e1",
        document_id="d1",
        kind="table",
        text="Method | Recall\nHybrid | 0.85",
        table=[["Method", "Recall"], ["Hybrid", "0.85"]],
        image_path=str(img),
        sources=[Source(page=1, bbox=(0, 0, 1, 1))],
    )
    return Evidence(
        element=element,
        document_name="demo",
        page_label="页码",
        matched_text=element.text,
        rank=1,
        channels=["bm25"],
        scores={"bm25": 1},
    )


def test_evidence_budget_image_and_citation_validation(tmp_path):
    evidence = fixture_evidence(tmp_path)
    content, context, images = evidence_context([evidence], char_budget=20)
    assert images == 1 and content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert context[0]["truncated"]
    assert context[0]["text"] == "Method | Recall"
    assert validate_citations("结果 [E8]", ["E1"])[1] == {"E8"}
    assert validate_citations("结果无引用", ["E1"])[2]
    assert normalize_citation_format("[证据编号] E1\n答案内容") == "[E1]\n答案内容"
    assert validate_citations("[证据编号] [E1]", ["E1"])[2]


def test_evidence_budget_does_not_reopen_after_exhaustion(tmp_path):
    first = fixture_evidence(tmp_path)
    first.element.kind = "text"
    first.matched_text = "a" * 20
    second_element = Element(
        id="e2",
        document_id="d1",
        kind="text",
        text="b" * 20,
        sources=[Source(page=1, bbox=(0, 0, 1, 1))],
    )
    second = Evidence(
        element=second_element,
        document_name="demo",
        page_label="页码",
        matched_text="b" * 20,
        rank=2,
        channels=["bm25"],
        scores={"bm25": 1},
    )
    _, context, _ = evidence_context([first, second], include_images=False, char_budget=8)
    assert context[0]["text"] == "a" * 8
    assert context[1]["text"] == ""
    assert sum(len(item["text"]) for item in context) <= 8


def test_generation_repair_and_cloud_image_payload(tmp_path):
    evidence = fixture_evidence(tmp_path)
    store = Store(tmp_path / "store")
    settings = Settings(data_dir=tmp_path, api_key="synthetic-test-key")
    calls = []

    def transport(request):
        import json

        calls.append(json.loads(request.content))
        text = "召回率为 0.85 [E9]" if len(calls) == 1 else "召回率为 0.85 [E1]"
        return httpx.Response(
            200, json={"choices": [{"message": {"content": text}}], "usage": {"total_tokens": 10}}
        )

    class Retriever:
        def retrieve(self, request):
            return [evidence], {"timings": {}}

    pipeline = RAGPipeline(
        Retriever(), CloudGenerator(settings, store, httpx.MockTransport(transport)), store, settings
    )
    answer = pipeline.run(ChatRequest(question="召回率多少？", document_ids=["d1"]))
    assert answer.status == "answered" and set(answer.citations) == {"E1"}
    assert len(calls) == 2 and answer.image_count == 1
    assert calls[0]["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_ollama_payload_and_local_calls_do_not_use_project_budget(tmp_path):
    store = Store(tmp_path / "store")
    settings = Settings(
        data_dir=tmp_path,
        api_key="",
        generation_provider="ollama",
        ollama_base="http://ollama.test/api",
        ollama_model="qwen2.5vl:7b",
    )
    calls = []

    def transport(request):
        payload = json.loads(request.content)
        calls.append((str(request.url), payload))
        assert payload["model"] == "qwen2.5vl:7b"
        assert payload["stream"] is False
        assert payload["options"]["num_ctx"] == 8192
        assert payload["messages"][1]["images"]
        assert payload["messages"][1]["content"].startswith("[E1]")
        return httpx.Response(
            200,
            json={
                "model": "qwen2.5vl:7b",
                "message": {"role": "assistant", "content": "召回率为 0.85 [E1]"},
                "prompt_eval_count": 12,
                "eval_count": 8,
            },
        )

    generator = OllamaGenerator(settings, store, httpx.MockTransport(transport))
    result = generator.complete(
        [
            {"role": "system", "content": "仅依据证据回答"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "[E1] 召回率"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,ZmFrZQ=="}},
                ],
            },
        ]
    )
    assert result["choices"][0]["message"]["content"].endswith("[E1]")
    assert result["usage"]["total_tokens"] == 20
    assert calls[0][0].endswith("/api/chat")
    assert store.call_count() == 0


def test_call_budget(tmp_path):
    store = Store(tmp_path)
    store.reserve_call(1)
    with pytest.raises(RuntimeError, match="上限"):
        store.reserve_call(1)


def test_experiment_api_excludes_unanswerable(client):
    doc = upload_ready(client)
    elements = client.get(f"/api/v1/documents/{doc['id']}/elements").json()
    eid = next(e["id"] for e in elements if e["kind"] == "text" and "SQLite" in e["text"])
    request = {
        "strategies": ["bm25"],
        "questions": [
            {"id": "q1", "question": "SQLite", "relevant_element_ids": [eid]},
            {"id": "q2", "question": "unrelatedxyz", "answerable": False},
        ],
    }
    response = client.post("/api/v1/experiments", json=request)
    assert response.status_code == 202
    client.app.state.service.worker.submit(lambda: None).result(timeout=30)
    result = client.get("/api/v1/experiments/" + response.json()["id"]).json()
    assert result["status"] == "ready", result
    assert result["summary"][0]["recall@1"] == 1
    assert result["summary"][0]["answer_accuracy"] is None


def test_invalid_gold_and_table_header():
    with pytest.raises(ValueError):
        ExperimentRequest(questions=[{"id": "q", "question": "x"}])
    e = Element(
        id="t",
        document_id="d",
        kind="table",
        text="",
        sources=[Source(page=1, bbox=(0, 0, 1, 1))],
        table=[["项目", "数值"]] + [["测量项目" * 10, str(i)] for i in range(10)],
    )
    chunks = chunks_for([e], limit=100)
    assert len(chunks) > 1 and all(c.text.startswith("项目 | 数值") for c in chunks)


def test_six_groups_generation_review_and_provenance(tmp_path):
    import json

    from docqa.evaluation import Review, apply_review, run_experiment

    evidence = fixture_evidence(tmp_path)
    store = Store(tmp_path / "store")
    store.put_document({"id": "d1", "sha256": "fixture", "name": "fixture", "status": "ready"})
    store.replace_elements("d1", [evidence.element])
    settings = Settings(data_dir=tmp_path, api_key="mock-only", enable_clip=True)
    calls, retrieval_calls, progress = [], [], []

    def respond(request):
        payload = json.loads(request.content)
        calls.append(payload)
        question = payload["messages"][-1]["content"][-1]["text"]
        answer = "当前文档中没有足够依据" if "未知" in question else "召回率是 0.85 [E1]"
        return httpx.Response(200, json={"choices": [{"message": {"content": answer}}]})

    class Retriever:
        def __init__(self):
            self.settings = settings

        def retrieve(self, request):
            retrieval_calls.append(request)
            return [evidence], {"timings": {"retrieval_seconds": 0.1}}

    retriever = Retriever()
    pipeline = RAGPipeline(
        retriever, CloudGenerator(settings, store, httpx.MockTransport(respond)), store, settings
    )
    request = ExperimentRequest(
        groups=list("ABCDEF"),
        generate=True,
        questions=[
            {
                "id": "q1",
                "question": "召回率？",
                "reference_answer": "0.85",
                "relevant_element_ids": ["e1"],
                "category": "table",
            },
            {"id": "q2", "question": "未知的值？", "answerable": False},
        ],
    )
    result = run_experiment(request, retriever, store, pipeline, lambda r: progress.append(r["completed"]))
    assert len(calls) == len(retrieval_calls) == 12, "生成必须复用本题检索，且不能混入历史会话"
    assert progress == list(range(13))
    assert [r.multimodal for r in retrieval_calls] == [False] * 10 + [True] * 2
    assert [r["answer"]["image_count"] for r in result["rows"]] == [0] * 8 + [1] * 4
    assert len(result["provenance"]["dataset_sha256"]) == 64
    assert result["summary"][0]["answer_accuracy"] is None
    apply_review(
        result,
        Review(
            group="A",
            question_id="q1",
            answer_accuracy=1,
            is_refusal=False,
            citation_supported=True,
            reviewer="tester",
        ),
    )
    assert result["summary"][0]["review_coverage"] == 0.5
    assert result["summary"][0]["correct_refusal_rate"] is None
    assert result["summary"][0]["citation_reviewed"] == 1
    assert result["summary"][0]["citation_supported_rate"] == 1
    apply_review(result, Review(group="A", question_id="q2", is_refusal=True, reviewer="tester"))
    summary = result["summary"][0]
    assert summary["answer_accuracy"] == summary["correct_refusal_rate"] == summary["review_coverage"] == 1
    assert summary["false_refusal_rate"] == 0
    result["rows"][0]["answer"]["status"] = "generation_error"
    with pytest.raises(ValueError, match="失败调用"):
        apply_review(
            result, Review(group="A", question_id="q1", answer_accuracy=0, is_refusal=True, reviewer="tester")
        )


def test_cross_split_leakage_and_cloud_opt_in(client):
    questions = [
        {
            "id": "a",
            "question": "问题一",
            "relevant_element_ids": ["e1"],
            "fact_group": "same",
            "split": "dev",
        },
        {
            "id": "b",
            "question": "问题二",
            "relevant_element_ids": ["e1"],
            "fact_group": "same",
            "split": "test",
        },
    ]
    with pytest.raises(ValueError, match="跨 dev/test"):
        ExperimentRequest(questions=questions)
    response = client.post(
        "/api/v1/experiments",
        json={"generate": True, "questions": [{"id": "a", "question": "不存在？", "answerable": False}]},
    )
    assert response.status_code == 400 and "API_KEY" in response.json()["detail"]


def test_delete_preserves_other_document_conversations(client):
    first = upload_ready(client)
    second = upload_ready(client, pdf_bytes(rotation=90))
    store = client.app.state.service.store
    for doc, session, run in [(first, "s1", "a1"), (second, "s2", "a2")]:
        store.put("sessions", session, {"scope": [doc["id"]], "turns": []})
        store.put("answers", run, {"session_id": session, "evidence": []})
        store.put("retrievals", run, {"final": []})
    assert client.delete("/api/v1/documents/" + first["id"]).status_code == 200
    assert store.get("sessions", "s1") is None and store.get("answers", "a1") is None
    assert store.get("retrievals", "a1") is None
    assert store.get("sessions", "s2") and store.get("answers", "a2") and store.get("retrievals", "a2")


def test_repeated_margin_artwork_filter():
    from docqa.parsers import remove_repeated_margin_artwork

    elements = [
        Element(
            id=f"logo-{i}",
            document_id="d",
            kind="image",
            text="",
            sources=[Source(page=i, bbox=(0.15, 0.04, 0.4, 0.07))],
        )
        for i in range(1, 5)
    ]
    elements += [
        Element(
            id="chart",
            document_id="d",
            kind="image",
            text="图1",
            sources=[Source(page=1, bbox=(0.1, 0.2, 0.8, 0.6))],
        ),
        Element(
            id="unique-margin",
            document_id="d",
            kind="image",
            text="图注",
            sources=[Source(page=1, bbox=(0.5, 0.02, 0.8, 0.05))],
        ),
    ]
    assert {e.id for e in remove_repeated_margin_artwork(elements)} == {"chart", "unique-margin"}


def test_public_dataset_counts_provenance_and_split():
    import json
    from collections import Counter
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / "datasets/public_questions.json").read_text())
    request = ExperimentRequest.model_validate(data)
    assert Counter(q.category for q in request.questions) == {
        "text": 40,
        "table": 25,
        "chart": 20,
        "unanswerable": 15,
    }
    assert Counter(q.split for q in request.questions) == {"dev": 20, "test": 80}
    assert all(q.reference_sources and q.scoring_points for q in request.questions if q.answerable)
    source = json.loads((root / "datasets/public_sources.json").read_text())
    assert len(source["documents"]) == 20 and sum(d["pages"] for d in source["documents"]) == 285


def test_flattened_chinese_headings_preserve_statistical_scope():
    from docqa.structure import restore_section_paths

    def item(index, text, heading, doc="a"):
        return Element(
            id=f"{doc}-{index}",
            document_id=doc,
            kind="text",
            text=text,
            title_path=[heading],
            sources=[Source(page=1, bbox=(0, 0, 1, 1))],
        )

    elements = [
        item(0, "一、全国城市", "一、全国城市"),
        item(1, "总体平均浓度43.1", "一、全国城市"),
        item(2, "二、重点城市", "二、重点城市"),
        item(3, "（一）总体状况", "（一）总体状况"),
        item(4, "（二）主要污染物", "（二）主要污染物"),
        item(5, "样本平均浓度50.7", "（二）主要污染物"),
        item(6, "三、其他区域", "三、其他区域"),
        item(7, "独立报告", "独立报告", doc="b"),
    ]
    result = restore_section_paths(elements)
    assert result[5].title_path == ["二、重点城市", "（二）主要污染物"]
    assert result[1].title_path == ["一、全国城市"]
    assert result[6].title_path == ["三、其他区域"]
    assert result[7].title_path == ["独立报告"]
    assert elements[5].title_path == ["（二）主要污染物"]
    assert [(e.id, e.text, e.sources) for e in result] == [(e.id, e.text, e.sources) for e in elements]


def test_exact_named_document_scope_does_not_guess_or_expand():
    from docqa.retrieval import named_document_scope

    docs = [
        {"id": "a", "name": "2024 年 12 月全国城市空气质量报告.pdf"},
        {"id": "b", "name": "2024年11月全国城市空气质量报告.pdf"},
    ]
    assert named_document_scope("据《2024年12月全国城市空气质量报告》，浓度多少？", docs) == ["a"]
    assert named_document_scope(
        "比较《2024年12月全国城市空气质量报告》和《2024年11月全国城市空气质量报告》", docs
    ) == ["a", "b"]
    assert named_document_scope("《全国城市空气质量报告》的数值？", docs) is None
    assert named_document_scope("《2024年12月全国城市空气质量报告》的数值？", docs[1:]) is None
    assert named_document_scope("《2024年12月全国城市空气质量报告》的数值？", []) is None
    cnnic = [{"id": "c", "name": "第55次中国互联网络发展状况统计报告.pdf"}]
    assert named_document_scope("根据第55次《中国互联网络发展状况统计报告》，图11如何？", cnnic) == ["c"]


def test_table_caption_and_header_survive_each_row_group():
    table = [["物质", "年均限值"], *[[f"样本{i}", str(i)] for i in range(20)]]
    element = Element(
        id="t",
        document_id="doc",
        kind="table",
        text="环境浓度限值\n|物质|年均限值|",
        table=table,
        sources=[Source(page=2, bbox=(0, 0, 1, 1))],
    )
    chunks = chunks_for([element], limit=60, overlap=0)
    assert len(chunks) > 1
    assert all(c.text.startswith("环境浓度限值\n物质 | 年均限值") for c in chunks)
    assert all(len(c.text) <= 60 and c.element_id == "t" for c in chunks)
    assert all(f"样本{i} | {i}" in "\n".join(c.text for c in chunks) for i in range(20))


def test_table_query_hint_requires_explicit_table_language():
    assert table_query_hint("请查浓度限值表：SO2 年平均一级和二级是多少？")
    assert table_query_hint("下表中 PM2.5 的数值是多少？")
    assert not table_query_hint("全国 PM2.5 平均浓度是多少？")
    assert not table_query_hint("表扬本月空气质量改善")


def test_generation_budget_preflight_exposes_remaining_without_spending(tmp_path):
    settings = Settings(
        data_dir=tmp_path,
        parser="pymupdf",
        api_key="test",
        generation_provider="bailian",
        max_api_calls=2,
    )
    with TestClient(create_app(settings)) as client:
        client.app.state.service.store.reserve_call(2)
        response = client.post(
            "/api/v1/experiments",
            json={
                "groups": ["D", "E"],
                "generate": True,
                "questions": [
                    {
                        "id": "q",
                        "question": "预算检查",
                        "reference_answer": "测试答案",
                        "relevant_element_ids": ["test-element"],
                        "split": "dev",
                    }
                ],
            },
        )
        assert response.status_code == 400 and "预算仅剩 1" in response.json()["detail"]
        health = client.get("/api/v1/health").json()
        assert health["api_calls_used"] == 1 and health["api_calls_limit"] == 2
        assert client.get("/api/v1/experiments").json() == []


def test_blind_review_export_separates_answer_key(tmp_path):
    from scripts.export_blind_review import answer_key, blind_rows, export

    result = {
        "provenance": {"retrieval_version": "v", "config": {"questions": ["must not copy"]}},
        "rows": [
            {
                "question_id": "q1",
                "question": "问题",
                "group": "D",
                "strategy": "hybrid_reranker",
                "category": "text",
                "answerable": True,
                "reference_answer": "金标准",
                "scoring_points": ["要点"],
                "gold": ["e1"],
                "gold_evidence": [{"id": "e1"}],
                "answer": {"status": "answered", "text": "回答", "evidence": [{"id": "e2"}]},
            }
        ],
    }
    blind = blind_rows(result)
    key = answer_key(result)
    assert blind[0]["answer_text"] == "回答" and "reference_answer" not in blind[0]
    assert key[0]["reference_answer"] == "金标准"
    input_path = tmp_path / "run.json"
    input_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    blind_path, key_path = export(input_path, tmp_path / "out")
    blind_file = json.loads(blind_path.read_text(encoding="utf-8"))
    key_file = json.loads(key_path.read_text(encoding="utf-8"))
    assert "金标准" not in blind_path.read_text(encoding="utf-8")
    assert key_file["answer_key"][0]["reference_answer"] == "金标准"
    assert "config" not in blind_file["provenance"]


def test_only_one_backend_can_own_a_data_directory(tmp_path):
    from docqa.service import Service

    settings = Settings(data_dir=tmp_path, parser="pymupdf", api_key="")
    first = Service(settings)
    try:
        with pytest.raises(RuntimeError, match="已有后端"):
            Service(settings)
    finally:
        first.close()
    reopened = Service(settings)
    reopened.close()


def test_review_api_persists_and_rejects_invalid_scores(client):
    doc = upload_ready(client)
    service = client.app.state.service
    eid = next(e.id for e in service.store.elements([doc["id"]]) if e.kind == "text" and "SQLite" in e.text)
    service.settings.api_key = "mock-transport-only"
    service.pipeline.generator.transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, json={"choices": [{"message": {"content": "数据库是SQLite [E1]"}}]}
        )
    )
    response = client.post(
        "/api/v1/experiments",
        json={
            "groups": ["A"],
            "generate": True,
            "document_ids": [doc["id"]],
            "questions": [
                {
                    "id": "review-q",
                    "question": "SQLite",
                    "relevant_element_ids": [eid],
                    "reference_answer": "SQLite",
                }
            ],
        },
    )
    experiment_id = response.json()["id"]
    service.worker.submit(lambda: None).result(timeout=30)
    review = {
        "question_id": "review-q",
        "group": "A",
        "answer_accuracy": 1,
        "is_refusal": False,
        "reviewer": "tester",
    }
    response = client.post(f"/api/v1/experiments/{experiment_id}/reviews", json=review)
    assert response.status_code == 200, response.json()
    result = client.get(f"/api/v1/experiments/{experiment_id}").json()
    assert result["summary"][0]["answer_accuracy"] == 1 and result["rows"][0]["review_history"]
    response = client.post(
        f"/api/v1/experiments/{experiment_id}/reviews", json={**review, "is_refusal": True}
    )
    assert response.status_code == 400


@pytest.mark.parametrize("result,expected", [
    ({"models": [{"name": "qwen2.5vl:7b"}]}, "available"),
    ({"models": []}, "missing_model"),
    ({"models": None}, "error"),
    ({"unexpected": True}, "error"),
])
def test_ollama_status_checks_model_without_generation(result, expected, tmp_path):
    from docqa.generation import generation_status
    def respond(request):
        assert request.method == "GET" and request.url.path == "/api/tags"
        return httpx.Response(200, json=result)
    settings = Settings(data_dir=tmp_path, generation_provider="ollama", ollama_model="qwen2.5vl:7b")
    status = generation_status(settings, transport=httpx.MockTransport(respond))
    assert status["status"] == expected


def test_ollama_status_connection_failure(tmp_path):
    from docqa.generation import generation_status
    def respond(request):
        raise httpx.ConnectError("connection refused", request=request)
    settings = Settings(data_dir=tmp_path, generation_provider="ollama")
    assert generation_status(settings, httpx.MockTransport(respond))["status"] == "unavailable"


def test_cloud_status_never_calls_provider(client):
    service = client.app.state.service
    before = service.store.call_count()
    response = client.get("/api/v1/generation-status")
    assert response.json()["status"] == "unconfigured"
    service.settings.api_key = "test-secret-not-to-be-returned"
    response = client.get("/api/v1/generation-status")
    assert response.json()["status"] == "configured"
    assert service.settings.api_key not in response.text
    assert service.store.call_count() == before


def test_invalid_model_switch_preserves_existing_configuration(client):
    before = client.app.state.service.generation_config()
    response = client.put("/api/v1/generation-config", json={"provider": "ollama", "model": "unknown"})
    assert response.status_code == 400
    assert client.app.state.service.generation_config() == before
    assert client.app.state.service.store.get("settings", "generation") is None


def test_model_selection_restores_across_restart(tmp_path):
    from docqa.service import Service
    first = Service(Settings(data_dir=tmp_path, parser="pymupdf", generation_provider="bailian"))
    try:
        first.configure_generation("ollama", "qwen2.5vl:7b")
    finally:
        first.close()
    second = Service(Settings(data_dir=tmp_path, parser="pymupdf", generation_provider="bailian"))
    try:
        assert second.generation_config()["provider"] == "ollama"
        assert second.generation_config()["model"] == "qwen2.5vl:7b"
    finally:
        second.close()
