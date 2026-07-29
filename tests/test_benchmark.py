"""Offline tests for benchmark case loading and stable trace generation."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from ragops.tracing import RagTracePayload, TraceCollector

import benchmark
from benchmark import (
    BenchmarkCase,
    BenchmarkConfig,
    StudyRagBenchmarkRunner,
    load_benchmark_cases,
)


def _write_cases(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _case_payload(case_id: str, question: str) -> str:
    return json.dumps(
        {"case_id": case_id, "question": question},
        ensure_ascii=False,
    )


def _config(name: str = "baseline") -> BenchmarkConfig:
    return BenchmarkConfig(
        name=name,
        top_k=3,
        score_threshold=0.25,
        prompt_version="smart_qa_v1",
    )


def _qa_result(answer: str = "答案") -> dict:
    return {
        "answer": answer,
        "retrieved_docs": [{"content": "资料片段", "score": 0.9}],
        "scores": [0.9],
        "is_low_confidence": False,
    }


def test_load_benchmark_cases_reads_valid_jsonl(tmp_path: Path) -> None:
    path = _write_cases(
        tmp_path / "cases.jsonl",
        [_case_payload("case_001", "问题一"), _case_payload("case-002", "问题二")],
    )

    assert load_benchmark_cases(path) == (
        BenchmarkCase("case_001", "问题一"),
        BenchmarkCase("case-002", "问题二"),
    )


def test_load_benchmark_cases_ignores_blank_lines_and_preserves_order(
    tmp_path: Path,
) -> None:
    path = _write_cases(
        tmp_path / "cases.jsonl",
        ["", _case_payload("second", "第二题"), "   ", _case_payload("first", "第一题")],
    )

    assert [case.case_id for case in load_benchmark_cases(path)] == ["second", "first"]


def test_load_benchmark_cases_reports_invalid_json_physical_line(
    tmp_path: Path,
) -> None:
    path = _write_cases(tmp_path / "cases.jsonl", ["", "{broken"])

    with pytest.raises(ValueError, match=r"physical line 2"):
        load_benchmark_cases(path)


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "missing id"},
        {"case_id": "bad id", "question": "invalid id"},
        {"case_id": "valid", "question": "   "},
        {"case_id": "valid"},
    ],
)
def test_load_benchmark_cases_reports_invalid_fields_with_line(
    tmp_path: Path,
    payload: dict,
) -> None:
    path = _write_cases(
        tmp_path / "cases.jsonl",
        ["", json.dumps(payload)],
    )

    with pytest.raises(ValueError, match=r"physical line 2"):
        load_benchmark_cases(path)


def test_load_benchmark_cases_rejects_duplicate_case_id(tmp_path: Path) -> None:
    path = _write_cases(
        tmp_path / "cases.jsonl",
        [_case_payload("same", "一"), _case_payload("same", "二")],
    )

    with pytest.raises(ValueError, match=r"duplicate case_id.*line 2"):
        load_benchmark_cases(path)


def test_load_benchmark_cases_rejects_empty_and_missing_files(tmp_path: Path) -> None:
    empty = _write_cases(tmp_path / "empty.jsonl", ["", "  "])

    with pytest.raises(ValueError, match="at least one"):
        load_benchmark_cases(empty)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_benchmark_cases(tmp_path / "missing.jsonl")


@pytest.mark.parametrize(
    ("kwargs", "error_type"),
    [
        ({"name": "", "top_k": 1, "score_threshold": 0.0, "prompt_version": "v"}, ValueError),
        ({"name": "n", "top_k": True, "score_threshold": 0.0, "prompt_version": "v"}, TypeError),
        ({"name": "n", "top_k": 0, "score_threshold": 0.0, "prompt_version": "v"}, ValueError),
        ({"name": "n", "top_k": 1, "score_threshold": True, "prompt_version": "v"}, TypeError),
        ({"name": "n", "top_k": 1, "score_threshold": float("nan"), "prompt_version": "v"}, ValueError),
        ({"name": "n", "top_k": 1, "score_threshold": float("inf"), "prompt_version": "v"}, ValueError),
        ({"name": "n", "top_k": 1, "score_threshold": 1.1, "prompt_version": "v"}, ValueError),
        ({"name": "n", "top_k": 1, "score_threshold": 0.0, "prompt_version": ""}, ValueError),
    ],
)
def test_benchmark_config_rejects_invalid_values(kwargs: dict, error_type: type[Exception]) -> None:
    with pytest.raises(error_type):
        BenchmarkConfig(**kwargs)


def test_runner_calls_qa_once_per_case_and_persists_mapped_traces(
    tmp_path: Path,
) -> None:
    calls: list[tuple[object, str, int, float]] = []
    kb = object()

    def fake_qa(kb_arg: object, question: str, *, top_k: int, score_threshold: float) -> dict:
        calls.append((kb_arg, question, top_k, score_threshold))
        return _qa_result(f"回答：{question}")

    cases = (BenchmarkCase("one", "问题一"), BenchmarkCase("two", "问题二"))
    trace_path = tmp_path / "traces.jsonl"

    traces = StudyRagBenchmarkRunner(fake_qa, iter([1.0, 1.01, 2.0, 2.02]).__next__).run(
        kb, cases, _config(), trace_path
    )

    assert calls == [(kb, "问题一", 3, 0.25), (kb, "问题二", 3, 0.25)]
    assert [trace.trace_id for trace in traces] == [
        "trc_benchmark_one",
        "trc_benchmark_two",
    ]
    assert [trace.latency_ms for trace in traces] == pytest.approx([10.0, 20.0])
    assert TraceCollector(trace_path).list_traces() == list(traces)


def test_runner_consumes_generator_once(tmp_path: Path) -> None:
    yielded: list[str] = []

    def generate() -> Iterator[BenchmarkCase]:
        for case_id in ("one", "two"):
            yielded.append(case_id)
            yield BenchmarkCase(case_id, case_id)

    traces = StudyRagBenchmarkRunner(lambda *_args, **_kwargs: _qa_result()).run(
        object(), generate(), _config(), tmp_path / "traces.jsonl"
    )

    assert yielded == ["one", "two"]
    assert len(traces) == 2


def test_baseline_and_candidate_have_same_trace_ids_but_distinct_prompts(
    tmp_path: Path,
) -> None:
    cases = (BenchmarkCase("one", "问题"),)
    runner = StudyRagBenchmarkRunner(lambda *_args, **_kwargs: _qa_result())

    baseline = runner.run(object(), cases, _config("baseline"), tmp_path / "base.jsonl")
    candidate = runner.run(object(), cases, _config("candidate"), tmp_path / "candidate.jsonl")

    assert [trace.trace_id for trace in baseline] == [trace.trace_id for trace in candidate]
    assert baseline[0].prompt_version == "smart_qa_v1:baseline"
    assert candidate[0].prompt_version == "smart_qa_v1:candidate"


def test_runner_uses_existing_result_mapper(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mapped: list[dict] = []

    def fake_mapper(result: dict) -> RagTracePayload:
        mapped.append(result)
        return RagTracePayload(
            retrieval_chunks=["mapped chunk"],
            retrieval_scores=[0.8],
            answer="mapped answer",
        )

    monkeypatch.setattr(benchmark, "map_studyrag_result", fake_mapper)
    result = _qa_result()
    trace = StudyRagBenchmarkRunner(lambda *_args, **_kwargs: result).run(
        object(), [BenchmarkCase("one", "问题")], _config(), tmp_path / "trace.jsonl"
    )[0]

    assert mapped == [result]
    assert trace.answer == "mapped answer"
    assert trace.retrieval_chunks == ["mapped chunk"]


def test_runner_refuses_nonempty_file_without_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "traces.jsonl"
    path.write_text("old data\n", encoding="utf-8")
    calls = 0

    def fake_qa(*_args, **_kwargs) -> dict:
        nonlocal calls
        calls += 1
        return _qa_result()

    with pytest.raises(FileExistsError, match="already contains data"):
        StudyRagBenchmarkRunner(fake_qa).run(
            object(), [BenchmarkCase("one", "问题")], _config(), path
        )
    assert calls == 0


def test_runner_overwrite_replaces_old_content(tmp_path: Path) -> None:
    path = tmp_path / "traces.jsonl"
    path.write_text("old data\n", encoding="utf-8")

    traces = StudyRagBenchmarkRunner(lambda *_args, **_kwargs: _qa_result()).run(
        object(), [BenchmarkCase("one", "问题")], _config(), path, overwrite=True
    )

    assert TraceCollector(path).list_traces() == list(traces)
    assert "old data" not in path.read_text(encoding="utf-8")


def test_runner_propagates_midstream_error(tmp_path: Path) -> None:
    calls = 0

    def failing_qa(*_args, **_kwargs) -> dict:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("pipeline failed")
        return _qa_result()

    with pytest.raises(RuntimeError, match="pipeline failed"):
        StudyRagBenchmarkRunner(failing_qa).run(
            object(),
            [BenchmarkCase("one", "一"), BenchmarkCase("two", "二")],
            _config(),
            tmp_path / "traces.jsonl",
        )
    assert calls == 2


def test_runner_rejects_duplicate_cases_before_calling_qa(tmp_path: Path) -> None:
    calls = 0

    def fake_qa(*_args, **_kwargs) -> dict:
        nonlocal calls
        calls += 1
        return _qa_result()

    with pytest.raises(ValueError, match="duplicate case_id"):
        StudyRagBenchmarkRunner(fake_qa).run(
            object(),
            [BenchmarkCase("same", "一"), BenchmarkCase("same", "二")],
            _config(),
            tmp_path / "traces.jsonl",
        )
    assert calls == 0
