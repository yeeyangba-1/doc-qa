# StudyRAG：带检索溯源与质量评估日志的课程资料 RAG 学习助手

一个面向课程 PDF/TXT 资料的 RAG 学习助手，支持文档问答、复习提纲、自动出题、错题解析，并提供检索溯源、低置信度拒答和问答日志导出能力，可通过 RAGOps 完成离线质量诊断与发布建议。

## 项目背景

普通 RAG Demo 通常只完成“检索 + 回答”，但真实应用还需要回答几个更关键的问题：

- 回答依据来自哪里
- 检索片段是否相关
- 什么时候应该拒答
- 如何记录问答日志用于后续效果分析

StudyRAG 在基础文档问答流程上做轻量增强：保留原有学习助手功能，同时补充检索证据展示、低置信度拒答和问答日志导出，让项目更接近真实 RAG 应用的评估闭环。

## 功能亮点

- PDF/TXT 文档上传与解析
- Chunk 切分、Embedding 向量化、FAISS 检索
- 基于 DeepSeek API 的课程资料问答
- 复习提纲生成、自动出题、错题解析
- 检索溯源：展示 top-k 参考片段、chunk_id 和相似度 score
- 低置信度拒答：当检索证据不足时提示资料依据不足
- 问答日志导出：记录 query、answer、retrieved_docs、scores、feedback
- 可通过 RAGOps 完成规则评估、Bad Case 分析、实验对比和发布建议

## RAG 流程

```text
上传 PDF/TXT
→ 文本解析
→ Chunk 切分
→ Embedding 向量化
→ FAISS 相似度检索
→ 低置信度判断
→ 构造 Prompt
→ 调用大模型生成回答
→ 展示参考片段与 score
→ 保存问答日志
```

## 检索溯源

`rag.py` 中的 `search_with_scores(query, top_k)` 会返回命中的 chunk、相似度分数和 chunk_id。Streamlit 页面在智能问答结果下方展示“参考片段 / 检索证据”，方便用户和开发者判断回答是否真的基于资料。

## 低置信度拒答

`chat_with_trace(...)` 会检查最高相似度。如果没有检索结果，或最高 score 低于 `score_threshold`，系统不会调用模型强行回答，而是提示：

```text
资料中没有足够依据回答这个问题，建议补充相关资料或换一种问法。
```

## 问答日志导出

每次智能问答会追加写入：

```text
outputs/rag_qa_logs.csv
```

日志字段包括：

- `query`
- `answer`
- `retrieved_docs`
- `scores`
- `user_feedback`
- `created_at`

页面侧边栏提供“下载 RAG 问答日志 CSV”按钮。`outputs/` 已加入 `.gitignore`，不会提交运行日志。

## 与 RAGOps 的配合

StudyRAG 负责生成包含 query、answer、检索片段、分数和延迟的 Trace；RAGOps 负责对 Trace 进行确定性的规则评估、候选配置 Bad Case 分析、基准与候选实验对比，并生成发布建议。

两者形成“RAG 应用 + 离线质量诊断”的闭环。发布建议只生成本地结果，不会自动部署应用。

## 离线质量门禁

离线质量门禁使用同一组 Benchmark Case 分别运行 baseline 和 candidate。每个 `case_id` 是问题的稳定标识，确保两套配置比较的是同一批问题；baseline 和 candidate 可以使用不同的 `top_k`、`score_threshold` 等参数。

当前 RAGOps 规则评估关注是否有检索结果、最高检索分数和请求延迟。完整流程会生成两份评估报告，分析 candidate 的 Bad Case，对比两组实验结果，并给出确定性的发布建议。当前不提供答案语义正确性评分或 LLM Judge，发布建议也不会自动调参、部署、回滚或触发 CI/CD。

Benchmark Case 使用 UTF-8 JSONL，每行一个问题：

```json
{"case_id":"case_001","question":"资料中的核心概念是什么？"}
{"case_id":"case_002","question":"资料给出了哪些关键步骤？"}
```

可复制 `examples/benchmark_cases.example.jsonl` 作为起点。正式运行前，应将示例问题替换为与待测 PDF/TXT 资料对应的问题。

运行命令：

```bash
python run_quality_gate.py \
  --document path/to/material.txt \
  --cases examples/benchmark_cases.example.jsonl \
  --output-dir outputs/quality_gate \
  --baseline-top-k 3 \
  --candidate-top-k 5 \
  --baseline-score-threshold 0.25 \
  --candidate-score-threshold 0.25
```

可以通过 `--min-candidate-pass-rate`、`--min-pass-rate-delta`、`--max-regressed-trace-count` 和 `--max-total-issue-increase` 调整门禁策略；已有输出需要替换时使用 `--overwrite`。

输出目录包含：

- `baseline_traces.jsonl`：baseline 配置生成的稳定 Trace
- `candidate_traces.jsonl`：candidate 配置生成的稳定 Trace
- `evaluation_reports.jsonl`：按 baseline、candidate 顺序保存的评估报告
- `release_decisions.jsonl`：发布建议及其原因
- `quality_gate_summary.json`：便于脚本直接读取的门禁摘要

退出码：

- `0`：门禁批准
- `1`：门禁拒绝
- `2`：参数、文件、运行或持久化错误

## 快速开始

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 创建 `.env`

```env
DEEPSEEK_API_KEY=your_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

3. 运行项目

```bash
streamlit run app.py
```

浏览器打开：

```text
http://127.0.0.1:8501
```

## 项目结构

```text
├── app.py                         # Streamlit 页面：上传、问答、证据展示、日志下载
├── rag.py                         # RAG 核心：文档解析、向量化、FAISS 检索、trace 问答
├── features.py                    # 问答/提纲/出题/解析的 prompt 模板
├── rag_logger.py                  # RAG 问答日志 CSV 追加写入
├── benchmark.py                   # Benchmark Case 与稳定 Trace 生成
├── quality_gate.py                # RAGOps 离线质量门禁编排
├── run_quality_gate.py            # 离线质量门禁命令行入口
├── examples/
│   └── benchmark_cases.example.jsonl
├── README.md                      # 项目说明
├── docs/
│   ├── interview_notes.md         # 面试讲解速记
│   └── resume_project_description.md
├── requirements.txt
└── .gitignore
```
