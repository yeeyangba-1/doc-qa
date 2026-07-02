# StudyRAG：带检索溯源与质量评估日志的课程资料 RAG 学习助手

一个面向课程 PDF/TXT 资料的 RAG 学习助手，支持文档问答、复习提纲、自动出题、错题解析，并提供检索溯源、低置信度拒答和问答日志导出能力，可与 SearchInsight 配合进行 RAG 效果诊断。

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
- 可接入 SearchInsight 做 Bad Case 分析

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

## 与 SearchInsight 的配合

StudyRAG 负责产生 RAG 问答日志，包括 query、answer、retrieved_docs、scores、user_feedback 等字段；SearchInsight 负责分析这些日志中的 Bad Case，例如检索为空、答非所问、知识库未利用、用户不满意等问题。

两者可以形成“RAG 应用 + RAG 质量诊断”的闭环。

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
├── README.md                      # 项目说明
├── docs/
│   ├── interview_notes.md         # 面试讲解速记
│   └── resume_project_description.md
├── requirements.txt
└── .gitignore
```
