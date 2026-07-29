"""Offline tests for the RAGOps quality gate orchestration."""

from __future__ import annotations

import json
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path

import pytest
from ragops.evaluation import EvaluationReportCollector
from ragops.experiments import IncomparableEvaluationReportsError
from ragops.release import ReleaseDecisionCollector
from ragops.schemas import EvaluationIssueCode, ReleasePolicy, Trace

from quality_gate import StudyRagQualityGateRunner


def _trace(
    trace_id: str,
    *,
    score: float | None = 0.9,
    latency_ms: float = 10.0,
) -> Trace:
    return Trace(
        trace_id=trace_id,
        query=f"question {trace_id}",
        retrieval_chunks=[] if score is None else ["chunk"],
        retrieval_scores=[] if score is None else [score],
        prompt_version="smart_qa_v1:test",
        model="deepseek-v4-flash",
        answer="answer",
        latency_ms=latency_ms,
    )


def test_quality_gate_generates_and_persists_reports_in_order(tmp_path: Path) -> None:
    result = StudyRagQualityGateRunner().run(
        [_trace("one")],
        [_trace("one")],
        tmp_path,
    )

    reports = EvaluationReportCollector(
        tmp_path / "evaluation_reports.jsonl"
    ).list_reports()
    assert reports == [result.baseline_report, result.candidate_report]
    assert result.baseline_report.total_count == 1
    assert result.candidate_report.total_count == 1


def test_quality_gate_analyzes_candidate_bad_cases_and_all_issue_groups(
    tmp_path: Path,
) -> None:
    result = StudyRagQualityGateRunner().run(
        [_trace("empty"), _trace("slow")],
        [_trace("empty", score=None), _trace("slow", latency_ms=40000)],
        tmp_path,
    )

    assert result.candidate_analysis.total_bad_cases == 2
    assert result.candidate_analysis.issue_groups[EvaluationIssueCode.NO_RETRIEVAL] == ("empty",)
    assert result.candidate_analysis.issue_groups[EvaluationIssueCode.HIGH_LATENCY] == ("slow",)
    assert result.candidate_analysis.issue_groups[EvaluationIssueCode.LOW_RETRIEVAL_SCORE] == ()


def test_quality_gate_comparison_tracks_improvement_and_regression(tmp_path: Path) -> None:
    result = StudyRagQualityGateRunner().run(
        [_trace("improved", score=None), _trace("regressed")],
        [_trace("improved"), _trace("regressed", score=None)],
        tmp_path,
    )

    assert result.comparison.improved_trace_ids == ("improved",)
    assert result.comparison.regressed_trace_ids == ("regressed",)


def test_quality_gate_persists_release_decision(tmp_path: Path) -> None:
    result = StudyRagQualityGateRunner().run([_trace("one")], [_trace("one")], tmp_path)

    decisions = ReleaseDecisionCollector(
        tmp_path / "release_decisions.jsonl"
    ).list_decisions()
    assert decisions == [result.decision]


def test_quality_gate_summary_has_complete_json_safe_fields(tmp_path: Path) -> None:
    result = StudyRagQualityGateRunner().run(
        [_trace("one")],
        [_trace("one", score=0.1)],
        tmp_path,
        ReleasePolicy(min_candidate_pass_rate=0.0, min_pass_rate_delta=-1.0, max_regressed_trace_count=1, max_total_issue_increase=1),
    )
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))

    expected_fields = {
        "baseline_report_id", "candidate_report_id", "comparison_id", "decision_id",
        "baseline_pass_rate", "candidate_pass_rate", "pass_rate_delta",
        "improved_trace_ids", "regressed_trace_ids", "unchanged_passed_trace_ids",
        "unchanged_failed_trace_ids", "candidate_failed_trace_ids",
        "candidate_issue_groups", "issue_count_deltas", "approved", "reasons",
        "regressed_trace_count", "total_issue_increase",
    }
    assert expected_fields <= summary.keys()
    assert set(summary["candidate_issue_groups"]) == {
        issue.value for issue in EvaluationIssueCode
    }
    assert set(summary["issue_count_deltas"]) == {
        issue.value for issue in EvaluationIssueCode
    }
    assert all(isinstance(reason, str) for reason in summary["reasons"])
    assert isinstance(summary["regressed_trace_ids"], list)


def test_quality_gate_approved_and_rejected_scenarios(tmp_path: Path) -> None:
    approved = StudyRagQualityGateRunner().run(
        [_trace("one")], [_trace("one")], tmp_path / "approved"
    )
    rejected = StudyRagQualityGateRunner().run(
        [_trace("one")], [_trace("one", score=None)], tmp_path / "rejected"
    )

    assert approved.decision.approved is True
    assert rejected.decision.approved is False
    assert rejected.decision.reasons


def test_quality_gate_rejects_mismatched_trace_sets(tmp_path: Path) -> None:
    with pytest.raises(IncomparableEvaluationReportsError):
        StudyRagQualityGateRunner().run(
            [_trace("baseline")], [_trace("candidate")], tmp_path
        )


def test_quality_gate_does_not_modify_input_traces(tmp_path: Path) -> None:
    baseline = [_trace("one")]
    candidate = [_trace("one")]
    before = (deepcopy(baseline), deepcopy(candidate))

    StudyRagQualityGateRunner().run(baseline, candidate, tmp_path)

    assert baseline == before[0]
    assert candidate == before[1]


def test_quality_gate_consumes_each_iterable_once(tmp_path: Path) -> None:
    consumed: list[str] = []

    def traces(label: str) -> Iterator[Trace]:
        consumed.append(label)
        yield _trace("one")

    StudyRagQualityGateRunner().run(traces("baseline"), traces("candidate"), tmp_path)

    assert consumed == ["baseline", "candidate"]
