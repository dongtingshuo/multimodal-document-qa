# Contributing / 贡献指南

Use Python 3.11 and install `.[test]` in an isolated environment. Run `python -m pytest -q` and `python -m ruff check .` before submitting changes. Describe the concrete problem, resulting behavior and validation in pull requests. Keep English and Chinese READMEs aligned.

使用 Python 3.11 和独立环境，安装 `.[test]`。提交前运行测试与 Ruff；PR 说明具体问题、修改行为与验证结果，保持中英文 README 同步。

Do not commit credentials, uploaded documents, downloaded models, local databases or generated reports. Preserve frozen experiments; use a new output directory for new runs and document dataset/model changes. Include focused regression tests for behavioral fixes. Model-dependent checks must state their prerequisites and actual provider.

不要提交密钥、上传文档、模型、数据库和生成报告。保留历史冻结实验，新实验使用新输出目录，并记录数据与模型变化。行为修复提供针对性回归测试；依赖真实模型的验证明确写出前提与后端。
