import csv
import io
import threading

from fastapi.testclient import TestClient
from test_experiment_resume import fixture_data
from test_review_io import result_fixture, sheet_bytes

from docqa.api import create_app
from docqa.config import Settings


def test_pause_resume_through_api(tmp_path):
    settings, store, retriever, request, calls = fixture_data(tmp_path)
    started, release = threading.Event(), threading.Event()
    original = retriever.retrieve

    def retrieve(query):
        if not calls:
            started.set()
            assert release.wait(10)
        return original(query)

    with TestClient(create_app(settings)) as client:
        service = client.app.state.service
        service.retriever.retrieve = retrieve
        response = client.post('/api/v1/experiments', json=request.model_dump())
        assert response.status_code == 202
        eid = response.json()['id']
        path = '/api/v1/experiments/' + eid
        try:
            assert started.wait(10)
            assert client.post(path + '/pause').status_code == 202
        finally:
            release.set()
        service.worker.submit(lambda: None).result(timeout=10)
        checkpoint = client.get(path).json()
        assert checkpoint['status'] == 'paused' and checkpoint['completed'] == 1
        assert client.get(path + '/review-sheet').status_code == 409
        assert client.post(path + '/resume').status_code == 202
        service.worker.submit(lambda: None).result(timeout=10)
        result = client.get(path).json()
        assert result['status'] == 'ready' and result['completed'] == 3
        assert calls == ['value 0?', 'value 1?', 'value 2?']
        assert client.post(path + '/resume').status_code == 409


def test_review_sheet_api_roundtrip_and_atomic_rejection(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path, parser='pymupdf'))) as client:
        original = result_fixture()
        original['rows'][0]['question_id'] = '=formula'
        store = client.app.state.service.store
        store.put('experiments', original['id'], original)
        path = '/api/v1/experiments/run/review-sheet'
        response = client.get(path)
        assert response.status_code == 200
        rows = list(csv.DictReader(io.StringIO(response.content.decode('utf-8-sig'))))
        assert rows[0]['question_id'] == "'=formula"
        rows[0].update(reviewer='API fixture', answer_accuracy='1', is_refusal='0')
        rows[1].update(reviewer='API fixture', answer_accuracy='invalid', is_refusal='0')
        response = client.post(path, files={'file': ('review.csv', sheet_bytes(rows), 'text/csv')})
        assert response.status_code == 400
        assert store.get('experiments', 'run') == original
        rows[1]['answer_accuracy'] = '0.5'
        response = client.post(path, files={'file': ('review.csv', sheet_bytes(rows), 'text/csv')})
        assert response.status_code == 200 and response.json()['imported'] == 2
        saved = client.get('/api/v1/experiments/run').json()
        assert saved['summary'][0]['answer_accuracy'] == 0.75
        assert saved['rows'][0]['question_id'] == '=formula'
