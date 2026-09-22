# Evaluation / 评测指南

[English README](../README.md) · [中文 README](../README.zh-CN.md)

## Protocol / 实验协议

| Group / 组别 | Retrieval / 检索 | Generation input / 生成输入 |
| --- | --- | --- |
| A | BM25 | Text / 文本 |
| B | BGE dense | Text / 文本 |
| C | BM25 + BGE + RRF | Text / 文本 |
| D | C + reranker / 重排序 | Text / 文本 |
| E | Same retrieval as D / 同 D | Text + images / 文本与图片 |
| F | Hybrid + Chinese-CLIP + reranker | Text + images / 文本与图片 |

Recall and MRR use the top 10 source elements and exclude unanswerable questions from their denominators. Generation uses selected evidence, without reference answers in the prompt. D/E retrieval is identical when generation is disabled. Unscored answer metrics remain empty; scoring coverage is reported separately.

Recall 与 MRR 使用前 10 个来源元素，不可回答问题不进入分母。生成输入不包含参考答案。关闭生成时 D/E 检索一致；未评分的回答指标保持空值，单独报告评分覆盖率。

## Reproduction / 复现步骤

Start the application with the required models, then run from the repository root:

先准备模型并启动应用，再从项目根目录执行：

```bash
.venv/bin/python scripts/prepare_public_corpus.py
.venv/bin/python scripts/import_corpus.py
.venv/bin/python scripts/build_public_questions.py
.venv/bin/python scripts/benchmark.py datasets/public_questions.json --groups A B C D E F --split dev --output data/public-evaluation/dev
.venv/bin/python scripts/benchmark.py datasets/public_questions.json --groups A B C D E F --split test --output data/public-evaluation/test
```

Group F requires `DOCQA_ENABLE_CLIP=true` and visual model weights. The downloader checks PDF SHA256 and page counts. Stop and review source changes if hashes differ. Rebuilding questions updates the local annotation file to the imported element IDs; inspect the diff before committing it.

F 组需要开启 CLIP 并准备权重。下载脚本校验 PDF 摘要与页数；摘要变化时应停止并检查来源。重建题集会更新本地标注以匹配导入元素 ID，提交前检查差异。

Generation is opt-in with `--generate`; cloud runs consume quota. Full Chinese instructions, scoring and resume commands are in [evaluation.md](evaluation.md).

生成评测通过 `--generate` 显式开启，云端调用消耗额度。评分和续跑详见 [evaluation.md](evaluation.md)。

## Data availability and limits / 数据可用性与限制

The source manifest and 100-question annotations are included. Original third-party PDFs, weights, private documents and historical raw experiment archives are excluded. Reports under `docs/` retain historical measurements and dates; local-only paths are provenance notes rather than downloadable repository artifacts. Exact historical reproduction also depends on parser/model versions and matching element annotations.

仓库提供来源清单与 100 题标注，不附带第三方 PDF、权重、私人文档或历史原始运行归档。`docs/` 保留历史结果和日期；本机路径仅作溯源记录，不是可下载附件。精确复现还依赖解析器、模型版本和元素标注一致性。

The corpus is narrow: 19 air-quality reports and one CNNIC report. Chart questions all come from the latter. Baseline retrieval scores do not measure current answer accuracy, generalization or comprehensive semantic faithfulness.

语料范围有限：19 份空气质量报告与 1 份 CNNIC 报告，图表题均来自后者。基线检索分数不代表当前问答准确率、泛化能力或完整语义忠实度。
