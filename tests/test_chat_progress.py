import json
import time

from fastapi.testclient import TestClient

from docqa.api import create_app
from docqa.config import Settings


def test_stream_reports_real_stages_heartbeat_and_one_result(tmp_path, monkeypatch):
    app = create_app(Settings(data_dir=tmp_path, parser='pymupdf', api_key=''))
    with TestClient(app) as client:
        pipeline = app.state.service.pipeline
        original = pipeline.run
        calls = []

        def delayed(request, progress=None):
            calls.append(request.question)
            progress('retrieving')
            time.sleep(1.1)
            return original(request, progress=progress)

        monkeypatch.setattr(pipeline, 'run', delayed)
        response = client.post('/api/v1/chat/stream', json={'question':'问题', 'generate':False})
        events = [json.loads(line) for line in response.text.splitlines()]
        assert response.status_code == 200
        assert events[0]['stage'] == 'queued'
        assert any(e.get('stage') == 'retrieving' and e['elapsed'] >= 1 for e in events)
        assert events[-1]['type'] == 'result'
        assert events[-1]['answer']['status'] == 'no_evidence'
        assert sum(e['type'] == 'result' for e in events) == 1
        assert calls == ['问题']
        assert client.get('/api/v1/health').json()['parser_version'] == app.state.service.parser.version


def test_stream_failure_is_terminal_and_does_not_expose_exception_message(tmp_path, monkeypatch):
    app = create_app(Settings(data_dir=tmp_path, parser='pymupdf', api_key=''))
    with TestClient(app) as client:
        def fail(*args, **kwargs):
            raise RuntimeError('private-details')
        monkeypatch.setattr(app.state.service.pipeline, 'run', fail)
        response = client.post('/api/v1/chat/stream', json={'question':'问题'})
        events = [json.loads(line) for line in response.text.splitlines()]
        assert events[-1]['type'] == 'error'
        assert 'private-details' not in response.text
        assert not any(e['type'] == 'result' for e in events)


def test_retry_is_queued_without_waiting_for_processing_lock(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    from docqa.service import Service

    service = Service(Settings(data_dir=tmp_path, parser='pymupdf', api_key=''))
    service.store.put_document({'id':'d','sha256':'d','name':'sample.pdf','status':'ready','job_id':'old'})
    processed = []

    def process(doc_id):
        with service.lock:
            processed.append(doc_id)

    monkeypatch.setattr(service, 'process', process)
    try:
        with ThreadPoolExecutor(max_workers=1) as caller:
            with service.lock:
                first = caller.submit(service.retry, 'd').result(timeout=2)
                assert first['status'] == 'queued'
                second = caller.submit(service.retry, 'd').result(timeout=2)
                assert second['job_id'] == first['job_id']
                assert service.store.document('d')['status'] == 'queued'
                assert not processed
        service.worker.shutdown(wait=True)
        assert processed == ['d']
    finally:
        service.close()
