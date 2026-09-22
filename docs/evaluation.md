# 实验与人工评分

评测默认只运行检索，生成需通过 `--generate` 显式开启。

## 题集格式

```json
{
  "document_ids": ["导入后返回的文档ID"],
  "questions": [
    {
      "id": "q001",
      "question": "文档中的具体问题？",
      "relevant_element_ids": ["在元素预览中核对的原始元素ID"],
      "reference_answer": "依据原文填写的答案及数值、单位、评分要点",
      "category": "table",
      "fact_group": "同一事实的改写题共享此值",
      "split": "dev",
      "answerable": true
    },
    {
      "id": "q002",
      "question": "文档未提供的信息？",
      "relevant_element_ids": [],
      "category": "unanswerable",
      "split": "test",
      "answerable": false
    }
  ]
}
```

类别为 `text`、`table`、`chart`、`unanswerable`。同一事实组或完全相同的问题不能跨开发集和测试集。金标准必须属于本次选定的文档范围；重新解析后需核对元素标注。参考答案应直接来自原文，不能用被测系统的回答作为标准答案。原文区域随结果中的证据元素保存。

已归档 20 份真实公开报告、285 页和 100 题（文本 40、表格 25、图表 20、无答案 15），划分开发 20／测试 80。原文件来源及 SHA256 见 `datasets/public_sources.json`，题集见 `datasets/public_questions.json`，与合成演示题集分开。开发集为文本 10、图表 8、无答案 2；测试集为文本 30、表格 25、图表 12、无答案 13，并非分层随机划分。

另有一组独立的本地新语料留出回归：`data/generalization-20260911/` 包含 3 份未进入上述公开题库的构造文档和 12 道题，覆盖文本、表格、图形及不可回答问题。使用 `.venv/bin/python scripts/run_generalization.py --output data/generalization-20260911` 可重新解析并建立独立索引；默认只运行 D 组检索，不消耗云端额度。`REPORT.md` 和 `retrieval-result.json` 保存指标与逐题召回，结果用于工程回归，不能替代外部真实语料泛化评测。

语料主要是 19 份月度空气质量报告及 1 份 CNNIC 报告；所有图表题来自 CNNIC，标准表在多份月报中重复。因此本实验反映特定报告集合上的表现，不能外推为通用文档问答性能。正式语料均为 PDF，其他格式通过独立功能检查验证。标注使用 PDF 实际页序，不一定与页面印刷页码相同。每道可回答题另附 `reference_sources`（文档 ID、文件 SHA256、页码、bbox）及 `scoring_points`。

## 六组固定对比

| 组别 | 召回与排序 | 生成输入 |
|---|---|---|
| A | BM25 | 文本 |
| B | BGE Dense | 文本 |
| C | BM25 + Dense，RRF | 文本 |
| D | C + BGE 重排序 | 文本 |
| E | 与 D 相同 | 文本 + 最多 3 张原图 |
| F | 增加 Chinese-CLIP 召回，融合后重排序 | 文本 + 最多 3 张原图 |

每题检索前 10 个原始元素用于 Recall/MRR，生成仅复用其中前 6 个，不重新检索、不带入上一道题的会话。生成的参考答案不进入模型输入。本轮冻结证据预算为 6,000 **字符**，与计划中的约 6,000 token 有差异，实验元数据明确记录实际值。

关闭生成时，D/E 的检索结果应一致；无法据此比较原图输入的回答效果。启用实验生成需显式使用命令行 `--generate`。使用百炼会消耗云端额度，使用 `DOCQA_GENERATION_PROVIDER=ollama` 时调用本地 `qwen2.5vl:7b`，不消耗百炼额度。F 还要求 `DOCQA_ENABLE_CLIP=true` 并准备本地模型。

```bash
.venv/bin/python scripts/prepare_public_corpus.py
.venv/bin/python scripts/import_corpus.py
.venv/bin/python scripts/build_public_questions.py
.venv/bin/python scripts/freeze_environment.py
.venv/bin/python scripts/benchmark.py datasets/public_questions.json --groups A B C D E F --split dev --output data/public-evaluation/dev
.venv/bin/python scripts/benchmark.py datasets/public_questions.json --groups A B C D E F --split test --output data/public-evaluation/test
# 以下使用百炼时先在界面应用百炼模型，确认账号额度和费用设置：
.venv/bin/python scripts/benchmark.py datasets/public_questions.json --groups D E F --split test --generate --expected-provider bailian
# 受控抽样生成（例如每类抽取若干独立测试题）；不会修改原题集
.venv/bin/python scripts/benchmark.py datasets/public_questions.json --question-ids public-011 public-041 public-074 --groups D --split test --generate --output data/public-evaluation/test-sample
```

命令默认读取 `data/server.lock` 中的实际后端端口；可用 `--api http://127.0.0.1:8100/api/v1` 覆盖。结果保存到 `data/experiments/`。等待超时不会重复提交，结果文件供评测报告使用；软件工作台不展示实验结果。

生成实验完成后，可导出供人工盲评的文件。盲评文件只包含问题、模型回答、回答引用及实际证据，不包含参考答案、评分要点或金标准元素；答案键文件单独保存给评分负责人。两者均保留检索版本和模型配置摘要，便于核对实验身份：

```bash
.venv/bin/python scripts/export_blind_review.py data/experiments/<实验结果>.json --output-dir data/reviews/<实验结果>
```

评分完成后，可按 `POST /api/v1/experiments/{id}/reviews` 写回原实验结果，或使用评分表导入接口；不要修改盲评文件中的回答文本。评分结果随后在报告中汇总，软件工作台不展示。

本轮 Ollama 80 题已生成项目辅助评阅文件：`data/acceptance-20260911/reviews/assisted-review.csv`，摘要为同目录的 `assisted-review-summary.md`，规则和来源摘要为 `assisted-review.json`。评分尺度为 0、0.5、1；该文件不覆盖原始回答，也不覆盖空白 `human-review.csv`。如需在代码更新后重算，增加 `--overwrite` 参数。

导出本轮已归档报告（不会重新运行模型）：

```bash
.venv/bin/python -m pip install -r requirements-report.lock.txt
.venv/bin/python scripts/export_evaluation_report.py --dev data/public-evaluation/dev/85946e3f302948359db5483d0b5b7e6b.json --test data/public-evaluation/test/d170e5e1291a457280a16323de7131f6.json
```

脚本核对完成状态、600 条记录、无重复题目组别、开发测试不重叠，以及两次运行的元素集合、模型和提示词一致性后，生成报告、汇总 CSV、未命中列表及 PNG/PDF 图表。后续新实验应传入新的结果路径。

## 指标与评分

- Recall@K：前 K 个去重元素中命中的金标准数量／金标准总数。
- MRR@10：前 10 个元素中第一条命中的倒数排名；没有命中记 0。
- 无答案题不进入 Recall/MRR 的分母。
- 有答案题人工评分：0 错误或拒答，0.5 部分正确，1 正确。对照数值、单位、事实与预先写明的要点。
- 正确拒答率：已核查的无答案题中明确拒答的比例；错误拒答率：已核查的有答案题中明确拒答的比例。
- 引用支持率：在已填写“引用是否支持答案”的评分中，实际支持答案的比例；未检查项不进入分母，并同时报告核查数量。
- 未评分的值保留 `null`，准确率旁同时报告已评分数量和覆盖率。生成失败或引用校验失败单列，不算作成功拒答，也不允许人工当作正常回答评分。
- 引用编号检查只能排除不存在的编号；引用是否真正支持结论需使用逐题核查材料对照原文。

评分表或 API 保存评分后，使用报告脚本汇总总表和分类表。结果 JSON 保存题集摘要、文档 SHA256、元素摘要、模型名称、依赖版本、提示词摘要、参数和逐题耗时，避免把不同数据和配置混作同一实验。软件工作台只负责日常文档问答。

耗时包含首次模型加载、索引构建和独立 FAISS 进程启动，因此不等于缓存预热后的稳态吞吐量。`datasets/runtime_manifest.json` 归档本地 BGE、重排序和 Chinese-CLIP 的 18 个权重／配置文件摘要、可获得的缓存 revision、依赖锁文件及后端代码摘要。Docling/OCR 权重未逐项锁定；当前解析结果另行归档，并以实验中的元素摘要标识，不能承诺重新下载所有解析模型后的逐字节一致性。

## 暂停、续跑与本地生成（2026-09-11）

运行中的 API 实验可通过 `scripts/benchmark.py --pause-experiment <实验ID>` 暂停；当前题目完成并保存后才会暂停。继续时使用同一实验 ID，只执行尚未保存的题目。已保存的失败回答不会被静默重试。服务重启后，有有效检查点的未完成实验状态为已暂停。

续跑校验题集、文档和元素、模型、生成后端、接口地址摘要、超时设置、代码及依赖等来源信息；发生变化会拒绝拼接结果。请新建实验，或恢复原配置后继续。暂停与崩溃不同：进程被强制结束时，在途请求可能已经计费但尚未保存结果，续跑不能保证这一题只调用一次。

```bash
# 明确限定本地后端，防止界面模型已经切到云端而意外消耗额度
.venv/bin/python scripts/benchmark.py datasets/public_questions.json --groups E --split test --generate --expected-provider ollama --question-ids public-011 public-041 public-074 public-088 --output data/public-evaluation/ollama-test-sample
# 使用原实验配置继续；请将占位符替换为实际 ID
.venv/bin/python scripts/benchmark.py --resume-experiment <实验ID> --output data/experiments
# 只观察和导出，不重新提交
.venv/bin/python scripts/benchmark.py --watch-experiment <实验ID> --output data/experiments
```

本地请求默认超时 `DOCQA_OLLAMA_TIMEOUT=300` 秒，独立于云端超时。提交任务时冻结模型配置，记录实际使用的模型名称，避免本地回答被错误标成百炼模型。检查点持续导出 JSON，已有行同时导出 CSV；退出命令行观察程序不会暂停后台实验。

## 批量人工评分表

已完成的生成实验可通过下方接口或 `scripts/review_experiment.py` 导出 UTF-8 CSV。保留前七列不变，填写评分人、得分、是否拒答、引用支持和说明。得分可为 0、0.5 或 1；布尔项为 1/0；引用未检查可留空。表格含参考答案，适合核查评分；正式盲评继续使用前述 `export_blind_review.py`，把答案键交由评分负责人保管。

下载接口为 `GET /api/v1/experiments/{id}/review-sheet`，导入为同一路径 `POST`，通过 multipart 的 `file` 字段上传文件。系统核对原问题和回答摘要，并在所有行都有效后一次性保存。任何一行错误都不会部分写入；未填写表格不会产生零分。评分历史仍按逐题评分机制保存。自动化测试里的模拟评分只存在于临时数据库，不写入正式实验。

## 同题双模型结果核验

分别在界面应用 Ollama 和百炼，以相同题集、文档范围和实验组执行两次生成。随后离线核对并导出对照表，不重新调用模型：

```bash
.venv/bin/python scripts/compare_generators.py <Ollama结果.json> <百炼结果.json> --output data/model-comparison
```

脚本验证题集、源文档、元素、代码、依赖、提示词、检索模型、逐题召回 ID、实际文字上下文及图片数量一致，然后导出完整回答、状态、耗时、各服务返回的 token 用量和两份空白评分表。生成模型、服务地址、上下文容量和超时可以不同，并保留在原始实验来源记录中。脚本不会把答案自动评为正确，也不会把两家服务的 token 数直接换算成费用。

本次测量仅为同题抽样，Ollama 已预热且重复诊断可能命中缓存，百炼还受网络和服务负载影响；耗时包含引用修正调用。不能将一次平均耗时当作严谨的速度排名。

## 独立副本完整回归

`run_local_acceptance.py` 使用 SQLite 在线备份建立独立数据库快照，保留既有文档和元素的标识，索引与新回答写入单独目录。原始图片从归档路径读取，运行期间请保留原项目文档。它不占用日常后端的实验队列，仍会使用本机 CPU、内存和 Ollama 资源。

```bash
.venv/bin/python scripts/run_local_acceptance.py --output data/acceptance-20260911
.venv/bin/python scripts/freeze_environment.py --output data/acceptance-20260911/runtime_manifest.json
.venv/bin/python scripts/summarize_local_acceptance.py data/acceptance-20260911
```

任务固定执行开发 20 题 × 六组、测试划分 80 题 × 六组检索，再执行 Ollama E 组 80 题回答。启用本地 CLIP，只加载已下载模型，云端密钥为空、调用预算为零。Ctrl+C 会在当前题目保存后暂停；重复相同命令会验证配置后续跑，已经完成的阶段不会重复运行。该独立运行器的结果不加入工作台实验数据库，通过输出目录查看。

本题集已经用于历史实验和部分调试，本轮定义为完整回归；不能将全部 80 题重新表述为严格未见测试。输出目录保留先前有生成记录的题目列表，但此列表不保证覆盖所有人工查看记录。

生成完成后自动提供 `reviews/human-review.csv`、盲评 JSON 与答案键。汇总工具核对完成状态、全部组别与题目、逐题指标、D/E 检索一致性、阶段间代码/模型/证据摘要，并生成 REPORT.md 与逐题阅读材料；不自动填入人工分数。

离线人工评分可使用下面的命令，结果文件名替换为本轮实际路径：

```bash
.venv/bin/python scripts/review_experiment.py <结果.json> --output <空白评分.csv>
.venv/bin/python scripts/review_experiment.py <结果.json> --import-sheet <已填写评分.csv> --output <已评分结果.json>
```

导入时生成新 JSON，拒绝覆盖已有文件；空白评分不记零分，回答摘要不匹配或任意评分行不合法时整次导入失败。此操作不需要启动 App，也不调用任何模型。

多格式功能回归使用 `.venv/bin/python scripts/check_format_acceptance.py --output data/format-acceptance-20260911`，验证合成双栏文本、四种旋转角度、扫描 PDF、PNG/JPG、EXIF 方向及 DOCX/DOC 转换。该检查使用轻量解析路径，不能替代复杂真实文档的版式准确性评测。

源码快照可通过 `.venv/bin/python scripts/package_source.py --output data/acceptance-20260911/source-snapshot-final-v8.zip` 保存。采用指定源码目录与文件清单，包含题集、说明、App 和图标，排除真实 `.env`、文档/聊天数据、模型权重、环境和软链接；归档内记录每个文件的 SHA256，拒绝覆盖同名快照。它是源码快照，在其他机器运行仍需按 README 准备依赖与模型；实验原始数据应在本机单独保留。最终快照 v7 已通过 `testzip`，且不含 `data/`、真实密钥或软链接。

本轮补齐后的最新源码快照：source-snapshot.zip（本地归档 / local archive: `../data/completion-20260911/source-snapshot.zip`）。此前acceptance目录中的快照是历史阶段版本，继续保留。新脚本run_completion_evaluation.py支持prepare、bailian、ollama、synthetic阶段；已经完成的答案不会重复生成，固定题集不能在生成后扩展。


## 2026-09-20 小样本视觉消融

已完成固定16题D/E/F共48条本地生成，配置、逐题辅助评阅、完整性校验和限制见本轮报告（本地归档 / local archive: `../data/visual-ablation-20260920/REPORT.md`）。运行入口为`scripts/run_visual_ablation.py`，配对核验为`scripts/summarize_visual_ablation.py`。本轮另存事实诊断分及生成/引用检查失败计零后的分数，未写回原始人工评分字段；两者均不冒充完整语义支持率。

`scripts/score_answers.py`中的既有判断只针对指定历史80题文件，现按文件SHA256拒绝其他回答载荷，避免新结果套用旧分数。public-054与public-064的编号错配修正见勘误（本地归档 / local archive: `../data/release-20260920/historical-review-correction.json`），两行得分交换、总分63/73不变，旧评分及原始回答均保留。
