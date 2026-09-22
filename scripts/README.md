# 脚本导航

从项目根目录执行。Python 使用 `.venv/bin/python`，Shell 使用 `bash`。

| 工作 | 入口 |
| --- | --- |
| 日常启动 | `start.sh`、`launch.py`、`api_address.py` |
| 构建桌面入口和图标 | `build_app.sh`、`build_app_icon.sh`、`generate_app_icon.py`、`docqa_launcher.m` |
| 下载模型与准备语料 | `download_models.py`、`prepare_public_corpus.py`、`import_corpus.py`、`refine_corpus.py` |
| 题集与演示数据 | `build_public_questions.py`、`seed_demo.py` |
| 检索和生成评测 | `benchmark.py`、`run_local_acceptance.py`、`run_visual_ablation.py`、`compare_generators.py`、`run_generalization.py`、`run_completion_evaluation.py` |
| 结果汇总与评分 | `summarize_*.py`、`score_answers.py`、`review_experiment.py`、`export_blind_review.py`、`export_evaluation_report.py` |
| 工程检查 | `smoke.py`、`check_*.py` |
| 冻结依赖与归档 | `lock_dependencies.py`、`freeze_environment.py`、`package_source.py` |

具体参数见脚本 `--help`（支持时）和 `docs/evaluation.md`。检查脚本中部分会加载模型或实际调用生成服务，请按目的选用。历史评分脚本绑定特定答案文件，不能直接用于新实验评分。
