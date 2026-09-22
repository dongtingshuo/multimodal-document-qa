import copy
import threading
from types import SimpleNamespace

import pytest

from docqa.config import Settings
from docqa.evaluation import ExperimentPaused, ExperimentRequest, run_experiment
from docqa.schemas import Element, Evidence, Source
from docqa.service import Service
from docqa.store import Store


def fixture_data(tmp_path):
    settings = Settings(data_dir=tmp_path, parser='pymupdf', generation_provider='ollama')
    store = Store(tmp_path)
    store.put_document({'id': 'doc', 'name': 'fixture', 'sha256': 'a' * 64, 'status': 'ready'})
    element = Element(id='e', document_id='doc', kind='text', text='value 18',
                      sources=[Source(page=1, bbox=(0, 0, 1, 1))])
    store.replace_elements('doc', [element])
    evidence = Evidence(element=element, document_name='fixture', page_label='PDF', matched_text='value 18',
                        rank=1, channels=['bm25'], scores={})
    calls = []
    def retrieve(request):
        calls.append(request.question)
        return [evidence], {'timings': {'retrieval_seconds': 0.01}}
    retriever = SimpleNamespace(settings=settings, retrieve=retrieve)
    request = ExperimentRequest(groups=['A'], questions=[
        {'id': f'q{i}', 'question': f'value {i}?', 'relevant_element_ids': ['e'],
         'reference_sources': [{'document_id': 'doc', 'document_sha256': 'a' * 64,
                                'page': 1, 'bbox': [0, 0, 1, 1]}]}
        for i in range(3)
    ])
    return settings, store, retriever, request, calls


def test_pause_resume_skips_completed_rows_and_records_actual_model(tmp_path):
    settings, store, retriever, request, calls = fixture_data(tmp_path)
    with pytest.raises(ExperimentPaused) as paused:
        run_experiment(request, retriever, store, should_stop=lambda: len(calls) == 1)
    checkpoint = paused.value.result
    assert checkpoint['completed'] == 1 and checkpoint['total'] == 3
    assert checkpoint['provenance']['models']['generation'] == 'qwen2.5vl:7b'
    assert checkpoint['provenance']['generation_provider'] == 'ollama'
    result = run_experiment(request, retriever, store, resume=checkpoint)
    assert calls == ['value 0?', 'value 1?', 'value 2?']
    assert result['completed'] == 3
    assert len(checkpoint['rows']) == 1
    assert result['provenance']['resumed_at']


@pytest.mark.parametrize('change', ['model', 'source', 'code', 'duplicate'])
def test_resume_rejects_changed_inputs_before_retrieval(tmp_path, change):
    settings, store, retriever, request, calls = fixture_data(tmp_path)
    with pytest.raises(ExperimentPaused) as paused:
        run_experiment(request, retriever, store, should_stop=lambda: len(calls) == 1)
    checkpoint = copy.deepcopy(paused.value.result)
    if change == 'model':
        settings.ollama_model = 'different'
    elif change == 'source':
        store.replace_elements('doc', [Element(id='e', document_id='doc', kind='text', text='changed',
                                               sources=[Source(page=1, bbox=(0, 0, 1, 1))])])
    elif change == 'code':
        checkpoint['provenance']['code_sha256'] = {}
    else:
        checkpoint['rows'].append(copy.deepcopy(checkpoint['rows'][0]))
    with pytest.raises(ValueError):
        run_experiment(request, retriever, store, resume=checkpoint)
    assert len(calls) == 1


def test_expected_provider_prevents_accidental_cloud_submission(tmp_path):
    service = Service(Settings(data_dir=tmp_path, parser='pymupdf', generation_provider='bailian', api_key='test'))
    try:
        request = ExperimentRequest(groups=['A'], generate=True, expected_provider='ollama',
                                    questions=[{'id': 'q', 'question': 'unknown?', 'answerable': False}])
        with pytest.raises(ValueError, match='后端已改变'):
            service.experiment(request)
        assert service.store.records('experiments') == []
        assert service.store.call_count() == 0
    finally:
        service.close()


def test_queued_generation_configuration_is_frozen(tmp_path, monkeypatch):
    import docqa.service as module
    service = Service(Settings(data_dir=tmp_path, parser='pymupdf', generation_provider='ollama'))
    gate = threading.Event()
    service.worker.submit(lambda: gate.wait(5))
    seen = []
    def run(request, retriever, store, pipeline, progress, **kwargs):
        seen.append((pipeline.settings.generation_provider, pipeline.settings.active_generation_model,
                     pipeline.generator.settings is pipeline.settings))
        return {'rows': [], 'summary': []}
    monkeypatch.setattr(module, 'run_experiment', run)
    try:
        request = ExperimentRequest(groups=['A'], questions=[{'id': 'q', 'question': 'unknown?', 'answerable': False}])
        service.experiment(request)
        service.configure_generation('bailian', 'qwen3-vl-plus-2025-12-19')
        gate.set()
        service.worker.submit(lambda: None).result(timeout=10)
        assert seen == [('ollama', 'qwen2.5vl:7b', True)]
    finally:
        gate.set()
        service.close()
