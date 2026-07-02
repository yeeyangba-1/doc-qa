"""
RAG 问答日志：记录 query、answer、检索证据和用户反馈。
"""
import csv
import json
from datetime import datetime
from pathlib import Path


LOG_PATH = Path(__file__).parent / "outputs" / "rag_qa_logs.csv"
FIELDNAMES = ["query", "answer", "retrieved_docs", "scores", "user_feedback", "created_at"]


def append_qa_log(
    query: str,
    answer: str,
    retrieved_docs: list[dict],
    scores: list[float],
    user_feedback: str = "",
) -> Path:
    """追加保存一次 RAG 问答日志，供后续 SearchInsight 做质量分析。"""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = LOG_PATH.exists()

    with LOG_PATH.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "query": query,
            "answer": answer,
            "retrieved_docs": json.dumps(retrieved_docs, ensure_ascii=False),
            "scores": json.dumps(scores, ensure_ascii=False),
            "user_feedback": user_feedback,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        })

    return LOG_PATH


def get_log_path() -> Path:
    return LOG_PATH
