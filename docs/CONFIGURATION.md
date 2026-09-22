# Configuration / 配置指南

[English README](../README.md) · [中文 README](../README.zh-CN.md)

## Installation / 安装

Install `.[layout,vision,test]` for the documented full configuration. For a lighter setup, install `.[test]` and set `DOCQA_PARSER=pymupdf` and `DOCQA_ENABLE_CLIP=false`. Python 3.11 is the validated development version. Lock files capture a historical environment and are not cross-platform guarantees.

完整配置安装 `.[layout,vision,test]`；轻量配置安装 `.[test]`，并设置 `DOCQA_PARSER=pymupdf`、`DOCQA_ENABLE_CLIP=false`。Python 3.11 是已验证开发版本；锁文件记录历史环境，不代表跨平台兼容保证。

## Models / 模型

`scripts/download_models.py --reranker` downloads BGE embedding and reranking weights into `models/huggingface`. The launcher uses the same `HF_HOME`. Chinese-CLIP uses `models/chinese-clip`; enabling it may download missing weights. Docling and OCR can also require first-run model downloads. Set `DOCQA_LOCAL_MODELS_ONLY=true` only after preparing the required Hugging Face caches; this setting is not a global network sandbox for every dependency.

下载脚本将 BGE 与重排序权重放入 `models/huggingface`，启动器使用相同 `HF_HOME`。Chinese-CLIP 使用 `models/chinese-clip`，缺失时可能下载权重；Docling 与 OCR 首次运行也可能下载模型。准备好 Hugging Face 缓存后再启用 `DOCQA_LOCAL_MODELS_ONLY`，该选项不是对所有依赖的全局断网控制。

## Local generation / 本地生成

```env
DOCQA_GENERATION_PROVIDER=ollama
DOCQA_OLLAMA_BASE=http://127.0.0.1:11434/api
DOCQA_OLLAMA_MODEL=qwen2.5vl:7b
DOCQA_OLLAMA_CONTEXT_LENGTH=8192
DOCQA_OLLAMA_TIMEOUT=300
```

Start Ollama separately and download the selected model. The desktop launcher does not manage the Ollama service. Local generation avoids cloud quota but still uses local compute.

单独启动 Ollama 并下载模型。桌面启动器不管理 Ollama 服务。本地生成不消耗云端额度，但需要本机计算资源。

## Cloud generation / 云端生成

```env
DOCQA_GENERATION_PROVIDER=bailian
DOCQA_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
DOCQA_API_KEY=
DOCQA_GENERATION_MODEL=qwen3-vl-plus-2025-12-19
DOCQA_MAX_API_CALLS=100
```

Fill the key locally and select a supported model/region. The model above is the repository's recorded configuration. `DOCQA_MAX_API_CALLS` is an application guard, not an account billing cap. Never commit `.env`.

在本机填写密钥，并匹配可用模型与地域；上面的模型是工程记录配置。调用上限是应用保护，不是账号费用硬上限。不要提交 `.env`。

## Troubleshooting / 故障排查

| Symptom / 现象 | Action / 处理 |
| --- | --- |
| Missing `.venv/bin/python` / 缺少解释器 | Create the project environment, or set `DOCQA_PYTHON` to a suitable interpreter. / 创建环境或指定解释器。 |
| Missing model / 模型缺失 | Prepare weights and verify `HF_HOME`; do not enable cache-only mode prematurely. / 准备权重并检查缓存目录。 |
| Legacy DOC fails / DOC 转换失败 | Install LibreOffice and make `soffice` available on PATH. / 安装 LibreOffice 并配置 PATH。 |
| Port occupied / 端口占用 | Use the URL printed by the launcher; it selects available ports. / 以启动器实际地址为准。 |
| Weak chart answer / 图表回答不佳 | Check retrieved source, CLIP availability and image-input control separately. / 分别核对证据、视觉召回和图片输入。 |
| Ollama timeout / 本地生成超时 | Verify service, model, memory and `DOCQA_OLLAMA_TIMEOUT`. / 检查服务、模型、内存和超时设置。 |
