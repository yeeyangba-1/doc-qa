"""Offline benchmark cases and deterministic StudyRAG trace generation."""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ragops.schemas import Trace
from ragops.tracing import TraceCollector

from ragops_adapter import MODEL_NAME, map_studyrag_result

if TYPE_CHECKING:
    from rag import KnowledgeBase


_CASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    question: str

    def __post_init__(self) -> None:
        case_id = self.case_id.strip() if isinstance(self.case_id, str) else ""
        question = self.question.strip() if isinstance(self.question, str) else ""
        if not case_id:
            raise ValueError("case_id must not be blank")
        if not _CASE_ID_PATTERN.fullmatch(case_id):
            raise ValueError(
                "case_id may contain only letters, numbers, underscores, and hyphens"
            )
        if not question:
            raise ValueError("question must not be blank")
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "question", question)


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    top_k: int
    score_threshold: float
    prompt_version: str

    def __post_init__(self) -> None:
        name = self.name.strip() if isinstance(self.name, str) else ""
        prompt_version = (
            self.prompt_version.strip()
            if isinstance(self.prompt_version, str)
            else ""
        )
        if not name:
            raise ValueError("name must not be blank")
        if not prompt_version:
            raise ValueError("prompt_version must not be blank")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int):
            raise TypeError("top_k must be a positive integer")
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if isinstance(self.score_threshold, bool) or not isinstance(
            self.score_threshold, Real
        ):
            raise TypeError("score_threshold must be a finite number")
        score_threshold = float(self.score_threshold)
        if not math.isfinite(score_threshold):
            raise ValueError("score_threshold must be a finite number")
        if not -1 <= score_threshold <= 1:
            raise ValueError("score_threshold must be between -1 and 1")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "prompt_version", prompt_version)
        object.__setattr__(self, "score_threshold", score_threshold)


def load_benchmark_cases(path: str | Path) -> tuple[BenchmarkCase, ...]:
    """Load validated benchmark cases from a UTF-8 JSONL file."""
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"benchmark cases file does not exist: {source_path}")

    cases: list[BenchmarkCase] = []
    seen_case_ids: set[str] = set()
    with source_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            payload = line.strip()
            if not payload:
                continue
            try:
                data = json.loads(payload)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON at physical line {line_number}: {error.msg}"
                ) from error
            if not isinstance(data, dict):
                raise ValueError(
                    f"invalid benchmark case at physical line {line_number}: "
                    "expected a JSON object"
                )
            try:
                case = BenchmarkCase(**data)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid benchmark case at physical line {line_number}: {error}"
                ) from error
            if case.case_id in seen_case_ids:
                raise ValueError(
                    f"duplicate case_id at physical line {line_number}: {case.case_id}"
                )
            seen_case_ids.add(case.case_id)
            cases.append(case)

    if not cases:
        raise ValueError("benchmark cases file must contain at least one case")
    return tuple(cases)


QaCallable = Callable[..., Mapping[str, Any]]
Clock = Callable[[], float]


class StudyRagBenchmarkRunner:
    """Run StudyRAG cases and persist stable, comparable RAGOps traces."""

    def __init__(
        self,
        qa_callable: QaCallable | None = None,
        clock: Clock = time.perf_counter,
    ) -> None:
        if qa_callable is None:
            from features import smart_qa_with_trace

            qa_callable = smart_qa_with_trace
        self._qa_callable = qa_callable
        self._clock = clock

    def run(
        self,
        kb: KnowledgeBase,
        cases: Iterable[BenchmarkCase],
        config: BenchmarkConfig,
        trace_path: str | Path,
        *,
        overwrite: bool = False,
    ) -> tuple[Trace, ...]:
        storage_path = Path(trace_path)
        if storage_path.exists() and storage_path.stat().st_size > 0:
            if not overwrite:
                raise FileExistsError(
                    f"trace file already contains data: {storage_path}"
                )
            storage_path.write_text("", encoding="utf-8")
        storage_path.parent.mkdir(parents=True, exist_ok=True)

        case_tuple = tuple(cases)
        seen_case_ids: set[str] = set()
        for case in case_tuple:
            if case.case_id in seen_case_ids:
                raise ValueError(f"duplicate case_id: {case.case_id}")
            seen_case_ids.add(case.case_id)

        collector = TraceCollector(storage_path)
        traces: list[Trace] = []
        for case in case_tuple:
            started_at = self._clock()
            result = self._qa_callable(
                kb,
                case.question,
                top_k=config.top_k,
                score_threshold=config.score_threshold,
            )
            payload = map_studyrag_result(result)
            latency_ms = (self._clock() - started_at) * 1000.0
            trace = Trace(
                trace_id=f"trc_benchmark_{case.case_id}",
                query=case.question,
                retrieval_chunks=list(payload.retrieval_chunks),
                retrieval_scores=list(payload.retrieval_scores),
                prompt_version=f"{config.prompt_version}:{config.name}",
                model=MODEL_NAME,
                answer=payload.answer,
                latency_ms=latency_ms,
            )
            traces.append(collector.save(trace))

        return tuple(traces)
