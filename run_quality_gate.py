"""Command-line entry point for the StudyRAG offline quality gate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from benchmark import (
    BenchmarkConfig,
    StudyRagBenchmarkRunner,
    load_benchmark_cases,
)
from quality_gate import StudyRagQualityGateRunner
from ragops.schemas import ReleasePolicy
from ragops_adapter import PROMPT_VERSION


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the offline StudyRAG quality gate")
    parser.add_argument("--document", required=True, type=Path)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/quality_gate"))
    parser.add_argument("--baseline-top-k", type=int, default=3)
    parser.add_argument("--candidate-top-k", type=int, default=5)
    parser.add_argument("--baseline-score-threshold", type=float, default=0.25)
    parser.add_argument("--candidate-score-threshold", type=float, default=0.25)
    parser.add_argument("--min-candidate-pass-rate", type=float)
    parser.add_argument("--min-pass-rate-delta", type=float)
    parser.add_argument("--max-regressed-trace-count", type=int)
    parser.add_argument("--max-total-issue-increase", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _release_policy(args: argparse.Namespace) -> ReleasePolicy | None:
    option_names = (
        "min_candidate_pass_rate",
        "min_pass_rate_delta",
        "max_regressed_trace_count",
        "max_total_issue_increase",
    )
    values = {
        name: getattr(args, name)
        for name in option_names
        if getattr(args, name) is not None
    }
    return ReleasePolicy(**values) if values else None


def run(args: argparse.Namespace) -> bool:
    if not args.document.is_file():
        raise FileNotFoundError(f"document does not exist: {args.document}")
    if args.document.suffix.lower() not in {".pdf", ".txt"}:
        raise ValueError("document must be a PDF or TXT file")

    from rag import build_kb

    kb = build_kb(args.document.read_bytes(), args.document.name)
    kb.build_index()
    cases = load_benchmark_cases(args.cases)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        for filename in (
            "evaluation_reports.jsonl",
            "release_decisions.jsonl",
            "quality_gate_summary.json",
        ):
            (output_dir / filename).unlink(missing_ok=True)

    benchmark_runner = StudyRagBenchmarkRunner()
    baseline_traces = benchmark_runner.run(
        kb,
        cases,
        BenchmarkConfig(
            name="baseline",
            top_k=args.baseline_top_k,
            score_threshold=args.baseline_score_threshold,
            prompt_version=PROMPT_VERSION,
        ),
        output_dir / "baseline_traces.jsonl",
        overwrite=args.overwrite,
    )
    candidate_traces = benchmark_runner.run(
        kb,
        cases,
        BenchmarkConfig(
            name="candidate",
            top_k=args.candidate_top_k,
            score_threshold=args.candidate_score_threshold,
            prompt_version=PROMPT_VERSION,
        ),
        output_dir / "candidate_traces.jsonl",
        overwrite=args.overwrite,
    )
    result = StudyRagQualityGateRunner().run(
        baseline_traces,
        candidate_traces,
        output_dir,
        _release_policy(args),
    )

    comparison = result.comparison
    print(f"baseline_pass_rate={result.baseline_report.pass_rate}")
    print(f"candidate_pass_rate={result.candidate_report.pass_rate}")
    print(f"pass_rate_delta={comparison.pass_rate_delta}")
    print(f"improved_count={len(comparison.improved_trace_ids)}")
    print(f"regressed_count={len(comparison.regressed_trace_ids)}")
    print(
        "candidate_issue_groups="
        + str(
            {
                issue.value: list(trace_ids)
                for issue, trace_ids in result.candidate_analysis.issue_groups.items()
            }
        )
    )
    print(f"approved={result.decision.approved}")
    print(f"reasons={[reason.value for reason in result.decision.reasons]}")
    print(f"summary_path={result.summary_path}")
    return result.decision.approved


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        return 0 if run(args) else 1
    except Exception as error:
        print(f"quality gate failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
