"""
RAG 核心引擎 —— 快速加载 + 延迟分析
"""
import os, re, fitz, faiss, json, threading
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

load_dotenv(Path(__file__).parent / ".env")

llm = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)

_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print("加载 embedding 模型...")
        _embedding_model = SentenceTransformer("BAAI/bge-small-zh")
    return _embedding_model

EMBEDDING_DIM = 512

splitter = RecursiveCharacterTextSplitter(
    chunk_size=600, chunk_overlap=100,
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
)

# ============================================
# 文件读取
# ============================================
def read_pdf(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    return "".join(page.get_text() for page in doc)

def read_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")

# ============================================
# LLM 章节分析
# ============================================
STRUCTURE_PROMPT = """你是一个文档分析专家。请分析以下文档，识别其中的章节结构和知识点。

返回格式必须是严格的 JSON（不要包含 markdown 代码块标记）：
{
  "sections": [
    {
      "title": "章节标题（简洁概括，不超过20字）",
      "summary": "这一节的一句话概述",
      "knowledge_points": ["知识点1（概括性，不要太细）", "知识点2", "知识点3"]
    }
  ]
}

规则：
1. 如果文档有明确的章节划分（如"第一章"、"任务1"等），按原章节划分
2. 如果文档没有明确章节，根据内容主题自行划分，控制在 3~8 个章节
3. 每个章节的知识点控制在 2~5 个，要概括性的，不要太细碎
4. 标题和知识点都要简洁清晰"""

def analyze_structure_with_llm(text: str) -> list[dict]:
    """用 LLM 分析文档结构，返回章节列表"""
    # 只取前 6000 字分析（足够判断结构，且省 token）
    sample = text[:6000]
    try:
        resp = llm.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": STRUCTURE_PROMPT},
                {"role": "user", "content": f"请分析以下文档的结构：\n\n{sample}\n\n（如果文档后面还有内容，请根据前文推断整体结构）"},
            ],
            temperature=0.1,
            max_tokens=2000,
        )
        result = resp.choices[0].message.content
        # 清理 markdown 代码块标记
        result = re.sub(r'^```(?:json)?\s*', '', result.strip())
        result = re.sub(r'\s*```$', '', result)
        data = json.loads(result)
        sections = data.get("sections", [])
        return sections
    except Exception as e:
        print(f"LLM 结构分析失败: {e}")
        return []

# ============================================
# 知识库
# ============================================
class KnowledgeBase:
    def __init__(self, file_bytes: bytes, filename: str):
        # 阶段1：秒开原文
        if filename.endswith(".pdf"):
            self.raw_text = read_pdf(file_bytes)
        else:
            self.raw_text = read_txt(file_bytes)

        self.filename = filename
        self.chunks = []
        self.chunk_docs = []
        self.n_chunks = 0
        self.index = None
        self.structure = []
        self.ready = False         # 向量索引就绪
        self.structure_ready = False  # 章节结构就绪
        self._building = False

    def build_index(self):
        """构建向量索引（可后台调用）"""
        if self._building:
            return
        self._building = True
        try:
            model = get_embedding_model()
            self.chunks = splitter.split_text(self.raw_text)
            if not self.chunks:
                self.chunks = [self.raw_text]
            # 保存向量下标到 chunk 的映射，后续用于检索溯源和日志分析。
            self.chunk_docs = [
                {"chunk_id": i, "content": chunk}
                for i, chunk in enumerate(self.chunks)
            ]
            vecs = model.encode(self.chunks).astype(np.float32)
            faiss.normalize_L2(vecs)
            self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
            self.index.add(vecs)
            self.n_chunks = len(self.chunks)
            self.ready = True
        finally:
            self._building = False

    def build_structure(self):
        """用 LLM 分析章节结构（可后台调用）"""
        sections = analyze_structure_with_llm(self.raw_text)
        # 为每个章节定位原文大致位置
        structured = []
        for i, sec in enumerate(sections):
            title = sec.get("title", f"第{i+1}章")
            summary = sec.get("summary", "")
            kps = sec.get("knowledge_points", [])
            # 在原文中搜索章节位置
            pos = self.raw_text.find(title)
            if pos < 0:
                # 尝试搜索关键词
                keywords = re.split(r'[，,\s]', title)
                for kw in keywords:
                    if len(kw) > 1:
                        pos = self.raw_text.find(kw)
                        if pos >= 0:
                            break
            if pos < 0:
                pos = 0
            structured.append({
                "title": title,
                "summary": summary,
                "knowledge_points": [{"id": f"kp{i+1}_{j+1}", "title": kp} for j, kp in enumerate(kps)],
                "text_start": pos,
            })
        self.structure = structured
        self.structure_ready = True

    def search_with_scores(self, query: str, top_k: int = 5) -> list[dict]:
        """返回带相似度分数的检索结果，用于页面展示和质量日志。"""
        if not self.ready:
            # 索引未就绪，先构建
            self.build_index()
        if self.index is None or self.n_chunks == 0:
            return []

        q_vec = get_embedding_model().encode([query]).astype(np.float32)
        faiss.normalize_L2(q_vec)
        scores, indices = self.index.search(q_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= self.n_chunks:
                continue
            doc = self.chunk_docs[idx] if idx < len(self.chunk_docs) else {
                "chunk_id": int(idx),
                "content": self.chunks[idx],
            }
            results.append({
                "chunk_id": int(doc["chunk_id"]),
                "content": doc["content"],
                "score": float(score),
            })
        return results

    def search(self, query: str, top_k: int = 5) -> list[str]:
        return [doc["content"] for doc in self.search_with_scores(query, top_k)]

    def get_section_text(self, section_index: int) -> str:
        """获取某个章节的原文内容"""
        if not self.structure:
            return self.raw_text
        if section_index < 0 or section_index >= len(self.structure):
            return self.raw_text
        start = self.structure[section_index]["text_start"]
        if section_index + 1 < len(self.structure):
            end = self.structure[section_index + 1]["text_start"]
        else:
            end = len(self.raw_text)
        return self.raw_text[start:end].strip()


# ============================================
# LLM 调用
# ============================================
def ask_llm(system_prompt: str, user_question: str, context: list[str] | None = None) -> str:
    messages = [{"role": "system", "content": system_prompt}]
    if context:
        ctx = "\n\n---\n\n".join(context)
        messages.append({"role": "system", "content": f"参考资料：\n\n{ctx}"})
    messages.append({"role": "user", "content": user_question})
    resp = llm.chat.completions.create(
        model="deepseek-chat", messages=messages,
        temperature=0.3, max_tokens=2000,
    )
    return resp.choices[0].message.content


def build_kb(file_bytes: bytes, filename: str) -> KnowledgeBase:
    """快速创建知识库（只读原文，不构建索引）"""
    return KnowledgeBase(file_bytes, filename)


def chat(kb: KnowledgeBase, system_prompt: str, question: str) -> str:
    if not kb.ready:
        kb.build_index()
    context = kb.search(question, top_k=5)
    return ask_llm(system_prompt, question, context)


def chat_with_trace(
    kb: KnowledgeBase,
    system_prompt: str,
    question: str,
    top_k: int = 5,
    score_threshold: float = 0.25,
) -> dict:
    """问答入口：返回答案、检索片段、分数和低置信度标记。"""
    if not kb.ready:
        kb.build_index()

    retrieved_docs = kb.search_with_scores(question, top_k=top_k)
    scores = [doc["score"] for doc in retrieved_docs]
    max_score = max(scores) if scores else 0.0
    is_low_confidence = not retrieved_docs or max_score < score_threshold

    if is_low_confidence:
        answer = "资料中没有足够依据回答这个问题，建议补充相关资料或换一种问法。"
    else:
        context = [doc["content"] for doc in retrieved_docs]
        answer = ask_llm(system_prompt, question, context)

    return {
        "answer": answer,
        "retrieved_docs": retrieved_docs,
        "scores": scores,
        "is_low_confidence": is_low_confidence,
    }
