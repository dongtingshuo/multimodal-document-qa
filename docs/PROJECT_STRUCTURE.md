# Architecture and layout / 架构与目录

| Path / 路径 | Responsibility / 职责 |
| --- | --- |
| `docqa/api.py` | HTTP routes / HTTP 接口 |
| `docqa/service.py` | Document and query orchestration / 文档与查询流程 |
| `docqa/parsers.py` | Parsing, rendering and OCR / 解析、渲染与 OCR |
| `docqa/schemas.py` | Shared evidence structures / 统一证据结构 |
| `docqa/store.py` | SQLite metadata / SQLite 元数据 |
| `docqa/retrieval.py` | Text and visual retrieval / 文本与视觉检索 |
| `docqa/generation.py` | Providers and citation checks / 生成后端与引用检查 |
| `docqa/evaluation.py` | Controlled evaluation / 对照评测 |
| `ui/` | Streamlit workspaces / Streamlit 工作区 |
| `scripts/` | Launch, preparation and evaluation tools / 启动、准备与评测工具 |
| `tests/` | Regression tests / 回归测试 |
| `datasets/` | Source manifests and annotations / 来源清单与标注 |
| `assets/` | Desktop launcher build assets / 桌面启动器构建资源 |

The UI calls the API, which delegates to the service layer. Retrieved evidence retains document, page and normalized bounding-box provenance through generation and source display. Text and visual vectors live in separate indices and are fused by rank.

界面通过 API 调用服务层。检索证据在生成和原文展示中持续保留文档、页码与归一化边界框；文本和视觉向量分别建索引，通过排名融合。

`data/`, `models/`, `.venv/` and `.env` are local runtime paths excluded from the repository. The macOS App is built from source with `zsh scripts/build_app.sh`.

数据、模型、依赖环境与密钥均在本地保留，不提交仓库。macOS App 通过构建脚本生成。
