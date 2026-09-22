"""Exercise model switching through Streamlit against the real local API layer."""
import httpx
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from docqa.api import create_app
from docqa.config import Settings


def test_sidebar_switches_models_and_keeps_labels_readable(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path, parser="pymupdf", generation_provider="bailian", api_key="")
    with TestClient(create_app(settings)) as client:
        def local_request(method, url, **kwargs):
            kwargs.pop("timeout", None)
            return client.request(method, "/api/v1" + url.split("/api/v1", 1)[1], **kwargs)
        monkeypatch.setattr(httpx, "request", local_request)
        app = AppTest.from_file("../ui/app.py", default_timeout=15).run()
        assert not app.exception
        assert all("项目调用预算" not in element.value and "本地调用不消耗" not in element.value
                   for element in app.caption)
        assert all("账户剩余额度" not in element.value and "Token" not in element.value
                   for element in app.caption)
        assert app.selectbox(key="generation-provider").value == "bailian"
        app.selectbox(key="generation-provider").select("ollama").run()
        assert not app.exception
        next(button for button in app.button if button.label == "应用模型").click().run()
        assert not app.exception
        assert client.get("/api/v1/health").json()["generation_provider"] == "ollama"
        assert any("Qwen2.5-VL 7B" in entry.value for entry in app.success)
        app.selectbox(key="generation-provider").select("bailian").run()
        next(button for button in app.button if button.label == "应用模型").click().run()
        assert not app.exception
        assert client.get("/api/v1/health").json()["generation_provider"] == "bailian"
        assert any("Qwen3-VL Plus" in entry.value for entry in app.success)
        assert all("2025-12-19" not in entry.value for entry in app.success)


def test_processing_document_refreshes_and_disables_retry(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path, parser='pymupdf', api_key='', enable_clip=True)
    with TestClient(create_app(settings)) as client:
        doc = {'id': 'processing-test', 'name': 'sample.pdf', 'status': 'parsing',
               'element_count': 4, 'page_count': 1, 'parse_seconds': 10.0}
        reads = []

        def local_request(method, url, **kwargs):
            kwargs.pop('timeout', None)
            path = '/api/v1' + url.split('/api/v1', 1)[1]
            if method == 'POST' and path.endswith('/retry'):
                doc['status'] = 'queued'
                return httpx.Response(202, json=dict(doc), request=httpx.Request(method, url))
            if method == 'GET' and path == '/api/v1/documents':
                reads.append(doc['status'])
                return httpx.Response(200, json=[dict(doc)], request=httpx.Request(method, url))
            return client.request(method, path, **kwargs)

        monkeypatch.setattr(httpx, 'request', local_request)
        app = AppTest.from_file('../ui/app.py', default_timeout=15).run()
        assert not app.exception
        assert app.button(key='retry-processing-test').disabled
        assert sum('**⏳ 1 份文档处理中**' == entry.value for entry in app.markdown) == 1
        assert not app.info
        assert any('上次解析结果' in entry.value for entry in app.caption)
        assert len(reads) >= 2  # Initial page read plus the polling fragment.
        doc['status'] = 'ready'
        app.run()
        assert not app.exception
        assert not app.button(key='retry-processing-test').disabled
        assert any('解析完成' in entry.label for entry in app.expander)
        assert not any('文档处理中' in entry.value for entry in app.markdown)
        app.button(key='retry-processing-test').click().run()
        assert not app.exception
        assert app.button(key='retry-processing-test').disabled
        assert any('等待处理' in entry.label for entry in app.expander)
        assert sum('**⏳ 1 份文档处理中**' == entry.value for entry in app.markdown) == 1
        assert not app.info


def test_scope_warning_parser_reminder_and_stream_result(tmp_path, monkeypatch):
    import json
    from contextlib import contextmanager

    settings = Settings(data_dir=tmp_path, parser='pymupdf', api_key='', enable_clip=True)
    with TestClient(create_app(settings)) as client:
        docs = [{'id': 'a', 'name': 'economy.pdf', 'status':'ready', 'parser_version':'old'},
                {'id': 'b', 'name': 'education.pdf', 'status':'ready', 'parser_version':'pymupdf-rapidocr-v1'}]
        def local_request(method, url, **kwargs):
            kwargs.pop('timeout', None)
            if url.endswith('/documents'):
                return httpx.Response(200, json=docs, request=httpx.Request(method, url))
            return client.request(method, '/api/v1' + url.split('/api/v1', 1)[1], **kwargs)
        sent = []
        @contextmanager
        def local_stream(method, url, **kwargs):
            sent.append(kwargs['json'])
            answer = {'question':'测试', 'session_id':'s', 'status':'no_evidence',
                      'text':'当前文档中没有足够依据。', 'warnings':[], 'context':[],
                      'evidence':[], 'run_id':'r', 'timings':{'total_seconds':3}, 'image_count':0}
            events = [{'type':'progress','stage':'retrieving','elapsed':1},
                      {'type':'progress','stage':'generating','elapsed':2},
                      {'type':'result','answer':answer,'elapsed':3}]
            yield httpx.Response(200, text='\n'.join(json.dumps(e) for e in events),
                                 request=httpx.Request(method, url))
        monkeypatch.setattr(httpx, 'request', local_request)
        monkeypatch.setattr(httpx, 'stream', local_stream)
        app = AppTest.from_file('../ui/app.py', default_timeout=15).run()
        assert sum('较早或不同的解析规则' in w.value for w in app.warning) == 1
        app.radio[0].set_value('文档问答').run()
        next(c for c in app.checkbox if c.label == '检索全部已解析文档').uncheck().run()
        app.multiselect[0].select('b').run()
        clip = next(c for c in app.checkbox if c.label == '加入 CLIP 视觉召回')
        assert not clip.disabled
        clip.check().run()
        assert not any('本次只查询：' in e.value or '范围已切换' in e.value for e in app.info)
        app.chat_input[0].set_value('测试').run()
        assert not app.exception
        assert len(sent) == 1 and sent[0]['document_ids'] == ['b']
        assert sent[0]['multimodal'] is True
        assert any('用时 3 秒' in entry.label for entry in app.status)
        assert any('没有足够依据' in e.value for e in app.markdown)


def test_clip_option_appears_in_both_query_pages_when_enabled(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path, parser='pymupdf', api_key='', enable_clip=True)
    with TestClient(create_app(settings)) as client:
        client.app.state.service.store.put_document(
            {'id': 'd', 'sha256': 'd', 'name': 'sample.pdf', 'status': 'ready'}
        )

        def local_request(method, url, **kwargs):
            kwargs.pop('timeout', None)
            return client.request(method, '/api/v1' + url.split('/api/v1', 1)[1], **kwargs)

        monkeypatch.setattr(httpx, 'request', local_request)
        app = AppTest.from_file('../ui/app.py', default_timeout=15).run()
        app.radio[0].set_value('文档问答').run()
        clip = next(item for item in app.checkbox if item.label == '加入 CLIP 视觉召回')
        assert not clip.disabled
        app.radio[0].set_value('检索分析').run()
        clip = next(item for item in app.checkbox if item.label == '加入 CLIP 视觉召回')
        assert not clip.disabled
