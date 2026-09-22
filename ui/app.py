import io
import json
import os

import httpx
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API = os.getenv("DOCQA_API_URL", "http://127.0.0.1:8000").rstrip("/") + "/api/v1"
MODEL_LABELS = {
    "qwen3-vl-plus-2025-12-19": "Qwen3-VL Plus",
    "qwen2.5vl:7b": "Qwen2.5-VL 7B",
}
st.set_page_config(page_title="文档问答工作台", page_icon="📚", layout="wide")
st.markdown(
    """<style>
.block-container {padding-top: 2rem; max-width: 1440px;}
[data-testid="stSidebar"] {background: #f3f5f7;}
h1 {letter-spacing: -.04em;} .stMetric {background:#f6f8fa;border-radius:12px;padding:16px;}
</style>""",
    unsafe_allow_html=True,
)


def api(method, path, **kwargs):
    try:
        response = httpx.request(method, API + path, timeout=300, **kwargs)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        try:
            message = exc.response.json().get("detail", "请求失败")
        except ValueError:
            message = "请求失败"
        st.error(str(message))
    except httpx.RequestError:
        st.error("无法连接文档服务，请打开文档问答 App。若刚刚启动，请稍后刷新页面。")
    return None


def chat_with_progress(payload):
    stages = {"queued": "等待处理", "rewriting": "理解追问", "retrieving": "检索原文",
              "preparing": "准备文字与图片证据", "generating": "模型正在回答",
              "validating": "检查引用与计算", "repairing": "模型正在修正回答"}
    with st.status("正在提交问题 · 已等待 0 秒", expanded=False) as status:
        try:
            with httpx.stream("POST", API + "/chat/stream", json=payload, timeout=300) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    event = json.loads(line)
                    elapsed = float(event.get("elapsed", 0))
                    if event["type"] == "progress":
                        status.update(label=f"{stages.get(event['stage'], '处理中')} · 已等待 {elapsed:.0f} 秒")
                    elif event["type"] == "result":
                        answer = event["answer"]
                        failed = answer["status"] in {"generation_error", "citation_error"}
                        status.update(label=("处理完成，请查看结果提示" if failed else "处理完成") + f" · 用时 {elapsed:.0f} 秒",
                                      state="error" if failed else "complete", expanded=False)
                        return answer
                    elif event["type"] == "error":
                        st.error(event["message"])
                        break
                else:
                    st.error("连接已结束，但未收到完整回答。请勿连续重复提交，可先检查服务状态。")
        except (httpx.RequestError, httpx.HTTPStatusError, ValueError, KeyError, TypeError):
            st.error("未能接收完整回答，请检查服务状态。后台可能仍在处理，请勿连续重复提交。")
        status.update(label="本次连接未完成", state="error")
    return None


@st.dialog("原文与证据区域", width="large")
def show_source(item):
    element = item["element"]
    st.write(item["document_name"])
    for source in element["sources"]:
        st.caption(f"{item['page_label']} {source['page']} · 橙色框为证据区域")
        url = API + f"/documents/{element['document_id']}/pages/{source['page']}?highlight={element['id']}"
        try:
            response = httpx.get(url, timeout=30)
            if response.is_success:
                st.image(io.BytesIO(response.content), width="stretch")
            else:
                st.warning("原页不存在或文档已删除")
        except httpx.RequestError:
            st.warning("页面加载失败")


def show_evidence(items, key_prefix):
    for i, item in enumerate(items):
        element = item["element"]
        source = element["sources"][0]
        label = (
            f"E{i + 1} · {item['document_name']} · {item['page_label']} {source['page']} · {element['kind']}"
        )
        with st.expander(label):
            st.text(element["text"])
            st.caption("以上为该证据区域的原文；点击下方按钮查看高亮位置。")
            st.caption("元素 ID：" + element["id"])
            st.json(item["scores"], expanded=False)
            if st.button("查看原页与高亮", key=f"{key_prefix}-{i}"):
                show_source(item)


health = api("GET", "/health")
if health is None:
    st.stop()
docs = api("GET", "/documents") or []
ready = [d for d in docs if d["status"] == "ready"]
with st.sidebar:
    st.title("文档问答工作台")
    st.caption("多模态检索增强生成")
    # 实验评测属于独立评测工作流，不放进面向最终用户的软件工作区。
    # 历史实验数据和 API/CLI 仍保留，便于报告复核与后续追加实验。
    page = st.radio("工作区", ["文档管理", "文档问答", "检索分析"])
    st.divider()
    st.caption(f"解析器：{health['parser']} · 设备：{health['device']}")
    provider = health.get("generation_provider", "bailian")
    if provider == "cloud":
        provider = "bailian"
    provider_name = "Ollama 本地生成" if provider == "ollama" else "云端生成"
    current_model_label = MODEL_LABELS.get(health.get("generation_model", ""), health.get("generation_model", ""))
    st.caption(
        f"当前选择：{provider_name} · {current_model_label}"
        if health["generation_configured"]
        else "生成服务未配置 · 可检索原文"
    )
    generation_options = health.get(
        "generation_options",
        {"bailian": ["qwen3-vl-plus-2025-12-19"], "ollama": ["qwen2.5vl:7b"]},
    )
    with st.expander("切换生成模型", expanded=False):
        provider_labels = {"bailian": "百炼云端", "ollama": "Ollama 本地"}
        provider_choices = list(generation_options)
        selected_provider = st.selectbox(
            "生成后端",
            provider_choices,
            index=provider_choices.index(provider) if provider in provider_choices else 0,
            format_func=provider_labels.get,
            key="generation-provider",
        )
        model_choices = list(generation_options.get(selected_provider, []))
        current_model = health.get("generation_model", "")
        selected_model = st.selectbox(
            "模型",
            model_choices,
            index=(model_choices.index(current_model) if selected_provider == provider and current_model in model_choices else 0),
            format_func=lambda model: MODEL_LABELS.get(model, model),
            key="generation-model",
        )
        if st.button("应用模型", use_container_width=True):
            result = api(
                "PUT",
                "/generation-config",
                json={"provider": selected_provider, "model": selected_model},
            )
            if result:
                st.session_state["model_notice"] = (
                    f"已切换为 {provider_labels[selected_provider]} · {MODEL_LABELS.get(selected_model, selected_model)}"
                )
                st.rerun()
        if selected_provider == "ollama":
            st.caption("请先确认 Ollama 服务正在运行，且已下载所选模型。")
        elif not health.get("api_key_configured", False):
            st.caption("当前未检测到云端 API Key，选择百炼后只能检索原文。")
        if st.button("检查当前模型连接", use_container_width=True):
            status = api("GET", "/generation-status")
            if status:
                if status["status"] == "available":
                    st.success(status["message"])
                elif status["status"] == "configured":
                    st.info(status["message"])
                else:
                    st.warning(status["message"])
        st.caption("连接检查不会生成回答，也不消耗百炼调用额度。")
    if notice := st.session_state.pop("model_notice", None):
        st.success(notice)
    if st.button("刷新状态", use_container_width=True):
        st.rerun()

st.title(page)
if page == "文档管理":
    st.session_state.pop("parse_notice", None)
    pending = {d["id"]: d["status"] for d in docs
               if d["status"] in {"queued", "parsing", "indexing"}}
    if pending:
        @st.fragment(run_every=2)
        def watch_processing():
            latest = api("GET", "/documents")
            if latest is not None:
                states = {d["id"]: d["status"] for d in latest}
                if any(states.get(doc_id) != status for doc_id, status in pending.items()):
                    st.rerun(scope="app")
            st.markdown(f"**⏳ {len(pending)} 份文档处理中**")

        watch_processing()
    else:
        st.caption("上传文档，查看原文与解析结果。")
    a, b, c = st.columns(3)
    a.metric("文档数量", len(docs))
    b.metric("已完成解析", len(ready))
    c.metric("证据元素", sum(d.get("element_count", 0) for d in ready))
    uploaded = st.file_uploader(
        "添加 PDF、Word 或图片（单文件不超过 50 MB）",
        type=["pdf", "docx", "doc", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )
    if st.button("导入文档", type="primary", disabled=not uploaded):
        for file in uploaded:
            result = api("POST", "/documents", files={"file": (file.name, file.getvalue(), file.type)})
            if result:
                st.success(f"{file.name}：" + ("已存在" if result.get("duplicate") else "已加入处理队列"))
        st.rerun()
    statuses = {
        "queued": "等待处理",
        "parsing": "正在解析",
        "indexing": "建立索引",
        "ready": "解析完成",
        "failed": "处理失败",
    }
    for doc in docs:
        with st.expander(
            f"{doc['name']} · {statuses.get(doc['status'], doc['status'])}",
            expanded=doc["status"] == "failed",
        ):
            if doc["status"] in {"queued", "parsing", "indexing"} and doc.get("element_count"):
                st.caption("以下为上次解析结果，本次处理完成后更新。")
            st.caption(
                f"{doc.get('page_count', 0)} 页 · {doc.get('element_count', 0)} 个元素 · {doc.get('parse_seconds', 0):.2f} 秒"
            )
            if (doc["status"] == "ready" and health.get("parser_version")
                    and doc.get("parser_version") != health["parser_version"]):
                st.warning("此文档使用较早或不同的解析规则，建议点击“重新解析”更新。旧结果仍可使用。")
            if doc.get("error"):
                st.error(doc["error"])
            for warning in doc.get("warnings", []):
                st.warning(warning)
            actions = st.columns(3)
            if actions[0].button("查看元素", key="view-" + doc["id"]):
                st.session_state["elements"] = api("GET", f"/documents/{doc['id']}/elements")
            if actions[1].button("重新解析", key="retry-" + doc["id"],
                                 disabled=doc["status"] in {"queued", "parsing", "indexing"}):
                notice = st.empty()
                notice.caption("正在提交…")
                result = api("POST", f"/documents/{doc['id']}/retry")
                if result:
                    st.rerun()
                notice.empty()
            if actions[2].button("删除文档", key="delete-" + doc["id"]):
                api("DELETE", f"/documents/{doc['id']}")
                for key in ["elements", "source", "messages", "retrieval", "session_id"]:
                    st.session_state.pop(key, None)
                st.rerun()
    if st.session_state.get("elements"):
        st.subheader("元素预览")
        st.dataframe(
            pd.DataFrame(
                [
                    {"ID": e["id"], "类型": e["kind"], "页码": e["sources"][0]["page"], "内容": e["text"]}
                    for e in st.session_state["elements"]
                ]
            ),
            hide_index=True,
        )

elif page in {"文档问答", "检索分析"}:
    if not ready:
        st.info("请先在文档管理中导入文档。")
        st.stop()
    names = {d["id"]: d["name"] for d in ready}
    if st.checkbox("检索全部已解析文档", value=True):
        selected = list(names)
        st.caption(f"当前范围：{len(selected)} 份文档")
    else:
        selected = st.multiselect("检索文档范围（至少选择一份）", list(names), format_func=names.get)
    scope = tuple(sorted(selected))
    if st.session_state.get("scope") != scope:
        for key in ["messages", "session_id", "source", "retrieval"]:
            st.session_state.pop(key, None)
        st.session_state["scope"] = scope
    settings_cols = st.columns(3)
    labels = {
        "bm25": "BM25 关键词",
        "dense": "Dense 语义",
        "hybrid": "Hybrid 混合",
        "hybrid_reranker": "Hybrid + Reranker",
    }
    strategy = settings_cols[0].selectbox("检索策略", list(labels), index=3, format_func=labels.get)
    top_k = settings_cols[1].slider("证据数量", 1, 10, 6)
    multimodal = settings_cols[2].checkbox(
        "加入 CLIP 视觉召回",
        disabled=not health["clip_enabled"],
        help="从独立的视觉向量索引补充图片证据；与是否把入选原图发送给生成模型相互独立。",
    )
    if page == "文档问答":
        answer_options = st.columns(2)
        include_images = answer_options[0].checkbox("向生成模型发送入选证据的原图", value=True)
        generate = answer_options[1].checkbox(
            "生成回答", value=health["generation_configured"], disabled=not health["generation_configured"]
        )
        if st.button("开始新会话"):
            st.session_state["messages"] = []
            st.session_state.pop("session_id", None)
        question = st.chat_input("输入问题，例如：系统采用哪些检索策略？", disabled=not selected)
        if question:
            result = chat_with_progress({
                "question": question, "document_ids": selected, "strategy": strategy,
                "top_k": top_k, "multimodal": multimodal, "include_images": include_images,
                "generate": generate, "session_id": st.session_state.get("session_id"),
            })
            if result:
                st.session_state["session_id"] = result["session_id"]
                st.session_state.setdefault("messages", []).append(result)
        for message in st.session_state.get("messages", []):
            with st.chat_message("user"):
                st.write(message["question"])
            with st.chat_message("assistant"):
                for warning in message["warnings"]:
                    st.warning(warning)
                st.write(message["text"])
                status_label = {"answered": "已生成回答（请核对引用）", "no_evidence": "依据不足，未作答",
                                "evidence_only": "已找到原文", "citation_error": "回答依据尚未确认",
                                "generation_error": "生成失败，已保留原文"}.get(message['status'], message['status'])
                st.caption(
                    f"状态：{status_label} · {message['timings']['total_seconds']:.2f} 秒 · 输入图片 {message['image_count']} 张"
                )
                show_evidence(message["evidence"], message["run_id"])
                if message["context"]:
                    with st.expander("实际送入模型的证据"):
                        st.json(message["context"])
    else:
        question = st.text_input("检索问题")
        if st.button("执行检索", type="primary", disabled=not question.strip() or not selected):
            with st.spinner("检索中；首次加载模型需要时间…"):
                st.session_state["retrieval"] = api(
                    "POST",
                    "/retrieve",
                    json={
                        "question": question,
                        "document_ids": selected,
                        "strategy": strategy,
                        "top_k": top_k,
                        "multimodal": multimodal,
                    },
                )
        result = st.session_state.get("retrieval")
        if result:
            st.caption(f"检索耗时 {result['trace']['timings']['retrieval_seconds']:.3f} 秒")
            show_evidence(result["evidence"], "retrieve")
            with st.expander("各阶段候选与排名"):
                st.json(result["trace"])
