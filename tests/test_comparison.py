import copy
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest
from test_experiment_resume import fixture_data

from docqa.api import create_app
from docqa.comparison import compare
from docqa.evaluation import run_experiment
from docqa.generation import RAGPipeline


def pair_fixture(tmp_path):
    settings, store, retriever, request, _ = fixture_data(tmp_path)
    request.generate = True
    request.expected_provider = 'ollama'
    for q in request.questions:
        q.reference_answer = '18'
    generator = SimpleNamespace(complete=lambda messages: {
        'choices': [{'message': {'content': '数值为 18。[E1]'}}], 'usage': {'total_tokens': 20}})
    result = run_experiment(request, retriever, store,
                            pipeline=RAGPipeline(retriever, generator, store, settings))
    left = {'id': 'local', 'status': 'ready', **result}
    right = copy.deepcopy(left)
    right.update(id='cloud')
    right['config']['expected_provider'] = 'bailian'
    right['provenance']['generation_provider'] = 'bailian'
    right['provenance']['models']['generation'] = 'qwen3-vl-plus-2025-12-19'
    for row in right['rows']:
        row['answer']['model'] = 'qwen3-vl-plus-2025-12-19'
    return settings, left, right


@pytest.mark.parametrize('change', ['same_run', 'code', 'context', 'duplicate'])
def test_comparison_rejects_unmatched_runs(tmp_path, change):
    _, left, right = pair_fixture(tmp_path)
    assert len(compare(left, right)['rows']) == 3
    if change == 'same_run':
        right['id'] = left['id']
    elif change == 'code':
        right['provenance']['code_sha256'] = {}
    elif change == 'context':
        right['rows'][0]['answer']['context'][0]['text'] = 'changed'
    else:
        right['rows'].append(copy.deepcopy(right['rows'][0]))
    with pytest.raises(ValueError):
        compare(left, right)


def test_user_ui_hides_experiment_workspaces(tmp_path, monkeypatch):
    settings, left, right = pair_fixture(tmp_path)
    with TestClient(create_app(settings)) as client:
        service = client.app.state.service
        for run in [left, right]:
            service.store.put('experiments', run['id'], run)
        requests = []

        def local_request(method, url, **kwargs):
            requests.append(method)
            kwargs.pop('timeout', None)
            return client.request(method, '/api/v1' + url.split('/api/v1', 1)[1], **kwargs)

        monkeypatch.setattr(httpx, 'request', local_request)
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'ui/app.py'), default_timeout=15).run()
        assert [radio.label for radio in app.radio] == ['工作区']
        assert app.radio[0].options == ['文档管理', '文档问答', '检索分析']
        assert all('实验结果' not in str(radio.options) and '模型对比' not in str(radio.options)
                   for radio in app.radio)
        assert requests and set(requests) == {'GET'}
        assert service.store.call_count() == 0
