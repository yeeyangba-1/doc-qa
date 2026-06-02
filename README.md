# DocQA

一个基于 RAG 的文档问答系统，上传 PDF 后可以针对文档内容提问。

## 功能

- 上传 PDF，自动构建知识库
- 基于文档内容的智能问答
- 自动生成复习提纲
- 根据文档内容生成选择题
- 错题解析

## 技术栈

Python / LangChain / FAISS / DeepSeek API / Streamlit / PyMuPDF / BGE-small-zh

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key
# 在项目根目录创建 .env 文件，写入：
# DEEPSEEK_API_KEY=你的key
# DEEPSEEK_BASE_URL=https://api.deepseek.com

# 3. 运行
streamlit run app.py
```

浏览器打开 http://localhost:8501

## 项目结构

```
├── app.py          # 前端界面
├── rag.py          # RAG 核心：文档解析、向量化、检索
├── features.py     # 问答/提纲/出题/解析的 prompt 模板
└── requirements.txt
```
