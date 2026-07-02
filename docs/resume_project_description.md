# StudyRAG 简历项目描述

## 项目名称

StudyRAG：带检索溯源与质量评估日志的课程资料 RAG 学习助手

## 技术栈

Python、Streamlit、FAISS、BGE-small-zh、DeepSeek API、PyMuPDF

## 项目描述

- 基于 RAG 流程实现课程 PDF/TXT 文档问答，完成文本解析、Chunk 切分、Embedding 向量化、FAISS 相似度检索与大模型回答生成。
- 新增检索溯源能力，支持展示 top-k 参考片段、chunk_id 与相似度 score，提高回答可解释性。
- 设计低置信度拒答机制，当检索结果相关性不足时提示资料依据不足，降低模型幻觉风险。
- 实现问答日志导出，记录 query、answer、retrieved_docs、scores、feedback 等字段，可接入 SearchInsight 做 Bad Case 分析。
