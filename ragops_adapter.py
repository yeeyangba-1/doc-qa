"""StudyRAG-specific mapping and runner construction for RAGOps tracing."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any

from ragops.tracing import RagTracePayload, TraceCollector, TracedRagRunner


__all__ = ["map_studyrag_result", "create_studyrag_runner"]


PROMPT_VERSION = "smart_qa_v1"
MODEL_NAME = "deepseek-v4-flash"
DEFAULT_TRACE_PATH = Path(__file__).resolve().parent / "outputs" / "ragops_traces.jsonl"


def _require_sequence(value: object, field_name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{field_name} must be a sequence")
    return value


def map_studyrag_result(result: Mapping[str, Any]) -> RagTracePayload:
    """Map a ``smart_qa_with_trace`` result to the RAGOps MVP payload."""
    if not isinstance(result, Mapping):
        raise TypeError("result must be a mapping")

    required_fields = ("answer", "retrieved_docs", "scores")
    for field_name in required_fields:
        if field_name not in result:
            raise KeyError(f"missing required field: {field_name}")

    answer = result["answer"]
    if not isinstance(answer, str):
        raise TypeError("answer must be a string")
    if not answer.strip():
        raise ValueError("answer must not be blank")

    retrieved_docs = _require_sequence(result["retrieved_docs"], "retrieved_docs")
    raw_scores = _require_sequence(result["scores"], "scores")

    chunks: list[str] = []
    for index, document in enumerate(retrieved_docs):
        if not isinstance(document, Mapping):
            raise TypeError(f"retrieved_docs[{index}] must be a mapping")
        if "content" not in document:
            raise KeyError(f"retrieved_docs[{index}] is missing content")
        content = document["content"]
        if not isinstance(content, str):
            raise TypeError(f"retrieved_docs[{index}].content must be a string")
        if not content.strip():
            raise ValueError(f"retrieved_docs[{index}].content must not be blank")
        chunks.append(content)

    scores: list[float] = []
    for index, score in enumerate(raw_scores):
        if isinstance(score, bool) or not isinstance(score, Real):
            raise TypeError(f"scores[{index}] must be a real number")
        numeric_score = float(score)
        if not math.isfinite(numeric_score):
            raise ValueError(f"scores[{index}] must be finite")
        scores.append(numeric_score)

    if len(chunks) != len(scores):
        raise ValueError("retrieved_docs and scores must have the same length")

    return RagTracePayload(
        retrieval_chunks=chunks,
        retrieval_scores=scores,
        answer=answer,
    )


def create_studyrag_runner(
    trace_path: str | Path | None = None,
) -> TracedRagRunner[Mapping[str, Any]]:
    """Create a fresh fail-open runner for the StudyRAG question-answer flow."""
    storage_path = DEFAULT_TRACE_PATH if trace_path is None else Path(trace_path)
    collector = TraceCollector(storage_path)
    return TracedRagRunner(
        collector,
        result_mapper=map_studyrag_result,
        prompt_version=PROMPT_VERSION,
        model=MODEL_NAME,
        fail_open=True,
    )
