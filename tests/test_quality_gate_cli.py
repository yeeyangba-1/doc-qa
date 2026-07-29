"""CLI tests that never load a model or call a remote API."""

from __future__ import annotations

import argparse
import importlib
import sys
import types
from pathlib import Path

import pytest

import run_quality_gate


def test_cli_parser_accepts_document_cases_configs_and_policy() -> None:
    args = run_quality_gate.build_parser().parse_args(
        [
            "--document", "material.txt",
            "--cases", "cases.jsonl",
            "--output-dir", "out",
            "--baseline-top-k", "2",
            "--candidate-top-k", "4",
            "--baseline-score-threshold", "0.1",
            "--candidate-score-threshold", "0.3",
            "--min-candidate-pass-rate", "0.9",
            "--min-pass-rate-delta", "0.0",
            "--max-regressed-trace-count", "1",
            "--max-total-issue-increase", "2",
            "--overwrite",
        ]
    )

    assert args.document == Path("material.txt")
    assert args.cases == Path("cases.jsonl")
    assert args.baseline_top_k == 2
    assert args.candidate_top_k == 4
    assert args.overwrite is True


@pytest.mark.parametrize(("approved", "exit_code"), [(True, 0), (False, 1)])
def test_cli_main_returns_decision_exit_code(
    monkeypatch: pytest.MonkeyPatch,
    approved: bool,
    exit_code: int,
) -> None:
    monkeypatch.setattr(run_quality_gate, "run", lambda _args: approved)

    assert run_quality_gate.main(["--document", "doc.txt", "--cases", "cases.jsonl"]) == exit_code


def test_cli_main_returns_two_and_prints_concise_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(_args: argparse.Namespace) -> bool:
        raise RuntimeError("offline failure")

    monkeypatch.setattr(run_quality_gate, "run", fail)

    assert run_quality_gate.main(["--document", "doc.txt", "--cases", "cases.jsonl"]) == 2
    assert "quality gate failed: offline failure" in capsys.readouterr().err


def test_import_run_quality_gate_does_not_import_rag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "rag", raising=False)

    importlib.reload(run_quality_gate)

    assert "rag" not in sys.modules


def test_cli_run_uses_fake_kb_and_offline_runners(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    document = tmp_path / "material.txt"
    cases = tmp_path / "cases.jsonl"
    document.write_text("material", encoding="utf-8")
    cases.write_text('{"case_id":"one","question":"question"}\n', encoding="utf-8")
    build_calls: list[tuple[bytes, str]] = []

    class FakeKb:
        def __init__(self) -> None:
            self.build_count = 0

        def build_index(self) -> None:
            self.build_count += 1

    fake_kb = FakeKb()

    def fake_build_kb(data: bytes, filename: str) -> FakeKb:
        build_calls.append((data, filename))
        return fake_kb

    monkeypatch.setitem(sys.modules, "rag", types.SimpleNamespace(build_kb=fake_build_kb))

    trace = types.SimpleNamespace(trace_id="trc_benchmark_one")

    class FakeBenchmarkRunner:
        calls: list[tuple] = []

        def run(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return (trace,)

    decision = types.SimpleNamespace(approved=True, reasons=())
    comparison = types.SimpleNamespace(
        pass_rate_delta=0.0,
        improved_trace_ids=(),
        regressed_trace_ids=(),
    )
    report = types.SimpleNamespace(pass_rate=1.0)
    analysis = types.SimpleNamespace(issue_groups={})
    gate_result = types.SimpleNamespace(
        baseline_report=report,
        candidate_report=report,
        comparison=comparison,
        candidate_analysis=analysis,
        decision=decision,
        summary_path=tmp_path / "out" / "quality_gate_summary.json",
    )

    class FakeQualityGateRunner:
        def run(self, *_args, **_kwargs):
            return gate_result

    monkeypatch.setattr(run_quality_gate, "StudyRagBenchmarkRunner", FakeBenchmarkRunner)
    monkeypatch.setattr(run_quality_gate, "StudyRagQualityGateRunner", FakeQualityGateRunner)
    args = run_quality_gate.build_parser().parse_args(
        ["--document", str(document), "--cases", str(cases), "--output-dir", str(tmp_path / "out")]
    )

    assert run_quality_gate.run(args) is True
    assert build_calls == [(b"material", "material.txt")]
    assert fake_kb.build_count == 1
    assert len(FakeBenchmarkRunner.calls) == 2
