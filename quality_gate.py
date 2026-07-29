"""End-to-end offline quality gate orchestration using RAGOps components."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ragops.analysis import IssueAnalyzer
from ragops.evaluation import EvaluationReportCollector, RuleBasedEvaluator
from ragops.experiments import ExperimentComparator
from ragops.release import (
    ReleaseDecisionCollector,
    ReleaseGate,
    ReleaseGateRunner,
)
from ragops.schemas import (
    EvaluationIssueCode,
    EvaluationReport,
    ExperimentComparison,
    IssueAnalysisReport,
    ReleaseDecision,
    ReleasePolicy,
    Trace,
)


@dataclass(frozen=True)
class QualityGateResult:
    baseline_report: EvaluationReport
    candidate_report: EvaluationReport
    candidate_analysis: IssueAnalysisReport
    comparison: ExperimentComparison
    decision: ReleaseDecision
    summary_path: Path


class StudyRagQualityGateRunner:
    """Evaluate, analyze, compare, and gate two matching trace batches."""

    def run(
        self,
        baseline_traces: Iterable[Trace],
        candidate_traces: Iterable[Trace],
        output_dir: str | Path,
        policy: ReleasePolicy | None = None,
    ) -> QualityGateResult:
        baseline_trace_tuple = tuple(baseline_traces)
        candidate_trace_tuple = tuple(candidate_traces)
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)

        evaluator = RuleBasedEvaluator()
        baseline_report = evaluator.evaluate_many(baseline_trace_tuple)
        candidate_report = evaluator.evaluate_many(candidate_trace_tuple)

        report_collector = EvaluationReportCollector(
            destination / "evaluation_reports.jsonl"
        )
        report_collector.save(baseline_report)
        report_collector.save(candidate_report)

        candidate_analysis = IssueAnalyzer().analyze(
            candidate_report,
            candidate_trace_tuple,
        )
        comparison = ExperimentComparator().compare(
            baseline_report,
            candidate_report,
        )
        decision_collector = ReleaseDecisionCollector(
            destination / "release_decisions.jsonl"
        )
        decision = ReleaseGateRunner(
            ReleaseGate(),
            decision_collector,
        ).run(comparison, policy)

        summary_path = destination / "quality_gate_summary.json"
        summary = {
            "baseline_report_id": baseline_report.report_id,
            "candidate_report_id": candidate_report.report_id,
            "comparison_id": comparison.comparison_id,
            "decision_id": decision.decision_id,
            "baseline_pass_rate": baseline_report.pass_rate,
            "candidate_pass_rate": candidate_report.pass_rate,
            "pass_rate_delta": comparison.pass_rate_delta,
            "improved_trace_ids": list(comparison.improved_trace_ids),
            "regressed_trace_ids": list(comparison.regressed_trace_ids),
            "unchanged_passed_trace_ids": list(
                comparison.unchanged_passed_trace_ids
            ),
            "unchanged_failed_trace_ids": list(
                comparison.unchanged_failed_trace_ids
            ),
            "candidate_failed_trace_ids": list(candidate_report.failed_trace_ids),
            "candidate_issue_groups": {
                issue.value: list(candidate_analysis.issue_groups[issue])
                for issue in EvaluationIssueCode
            },
            "issue_count_deltas": {
                issue.value: comparison.issue_count_deltas[issue]
                for issue in EvaluationIssueCode
            },
            "approved": decision.approved,
            "reasons": [reason.value for reason in decision.reasons],
            "regressed_trace_count": decision.regressed_trace_count,
            "total_issue_increase": decision.total_issue_increase,
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        return QualityGateResult(
            baseline_report=baseline_report,
            candidate_report=candidate_report,
            candidate_analysis=candidate_analysis,
            comparison=comparison,
            decision=decision,
            summary_path=summary_path,
        )
