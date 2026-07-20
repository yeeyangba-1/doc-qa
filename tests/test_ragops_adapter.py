"""Tests for the lightweight StudyRAG-to-RAGOps adapter."""

from __future__ import annotations

from copy import deepcopy

import pytest
from ragops.tracing import TraceCollector

from ragops_adapter import create_studyrag_runner, map_studyrag_result


def test_map_studyrag_result_maps_answer_chunks_and_scores() -> None:
    result = {
        "answer": "期末重点是第三章。",
        "retrieved_docs": [
            {"chunk_id": 3, "content": "第三章介绍核心概念。", "score": 0.92},
            {"chunk_id": 7, "content": "复习时需要掌握例题。", "score": 0.81},
        ],
        "scores": [0.92, 0.81],
        "is_low_confidence": False,
    }

    payload = map_studyrag_result(result)

    assert payload.answer == "期末重点是第三章。"
    assert list(payload.retrieval_chunks) == [
        "第三章介绍核心概念。",
        "复习时需要掌握例题。",
    ]
    assert list(payload.retrieval_scores) == [0.92, 0.81]


def test_map_studyrag_result_accepts_low_confidence_rejection() -> None:
    result = {
        "answer": "资料中没有足够依据回答这个问题。",
        "retrieved_docs": [],
        "scores": [],
        "is_low_confidence": True,
    }

    payload = map_studyrag_result(result)

    assert payload.answer == "资料中没有足够依据回答这个问题。"
    assert list(payload.retrieval_chunks) == []
    assert list(payload.retrieval_scores) == []


@pytest.mark.parametrize("missing_field", ["answer", "retrieved_docs", "scores"])
def test_map_studyrag_result_rejects_missing_required_field(
    missing_field: str,
) -> None:
    result = {
        "answer": "回答",
        "retrieved_docs": [{"content": "资料"}],
        "scores": [0.9],
    }
    del result[missing_field]

    with pytest.raises(KeyError, match=missing_field):
        map_studyrag_result(result)


def test_map_studyrag_result_rejects_document_without_content() -> None:
    result = {
        "answer": "回答",
        "retrieved_docs": [{"chunk_id": 1, "score": 0.9}],
        "scores": [0.9],
        "is_low_confidence": False,
    }

    with pytest.raises(KeyError, match=r"retrieved_docs\[0\].*content"):
        map_studyrag_result(result)


def test_runner_persists_fake_pipeline_result_without_modifying_it(tmp_path) -> None:
    trace_path = tmp_path / "ragops_traces.jsonl"
    runner = create_studyrag_runner(trace_path)
    pipeline_calls = 0
    original_result = {
        "answer": "期末重点是第三章。",
        "retrieved_docs": [
            {"chunk_id": 3, "content": "第三章介绍核心概念。", "score": 0.92}
        ],
        "scores": [0.92],
        "is_low_confidence": False,
    }
    original_snapshot = deepcopy(original_result)

    def fake_pipeline(query: str) -> dict:
        nonlocal pipeline_calls
        pipeline_calls += 1
        assert query == "这门课期末考试的重点是什么？"
        return original_result

    traced_result = runner.run(
        "这门课期末考试的重点是什么？",
        fake_pipeline,
    )

    assert pipeline_calls == 1
    assert traced_result.result is original_result
    assert original_result == original_snapshot
    assert traced_result.trace_id is not None

    trace = TraceCollector(trace_path).get_trace(traced_result.trace_id)
    assert trace is not None
    assert trace.query == "这门课期末考试的重点是什么？"
    assert trace.answer == "期末重点是第三章。"
    assert trace.retrieval_chunks == ["第三章介绍核心概念。"]
    assert trace.retrieval_scores == [0.92]
    assert trace.prompt_version == "smart_qa_v1"
    assert trace.model == "deepseek-chat"
