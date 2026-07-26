"""Checks for StudyRAG's declared direct runtime dependencies."""

from __future__ import annotations

import re
from pathlib import Path


REQUIREMENTS_PATH = Path(__file__).resolve().parents[1] / "requirements.txt"
EXPECTED_DIRECT_DEPENDENCIES = {
    "numpy",
    "pymupdf",
    "faiss-cpu",
    "streamlit",
    "sentence-transformers",
    "langchain",
    "langchain-community",
    "langchain-text-splitters",
    "openai",
    "python-dotenv",
    "torch",
    "ragops",
}
RAGOPS_COMMIT = "fbefc21dad5b604c2000f51d101dfb8c26626dc3"


def _requirement_lines() -> list[str]:
    return [
        line
        for raw_line in REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines()
        if (line := raw_line.split("#", 1)[0].strip())
    ]


def _normalize_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _declared_package_names(lines: list[str]) -> set[str]:
    names: set[str] = set()
    for line in lines:
        match = re.match(r"([A-Za-z0-9][A-Za-z0-9._-]*)", line)
        if match:
            names.add(_normalize_package_name(match.group(1)))
    return names


def test_direct_runtime_dependencies_are_declared() -> None:
    declared = _declared_package_names(_requirement_lines())

    assert EXPECTED_DIRECT_DEPENDENCIES <= declared


def test_ragops_uses_the_pinned_github_commit_without_local_paths() -> None:
    lines = _requirement_lines()
    ragops_lines = [
        line
        for line in lines
        if _normalize_package_name(line.split(maxsplit=1)[0]) == "ragops"
    ]

    assert ragops_lines == [
        "ragops @ git+https://github.com/yeeyangba-1/"
        f"RAGOps.git@{RAGOPS_COMMIT}"
    ]

    requirements_text = REQUIREMENTS_PATH.read_text(encoding="utf-8")
    normalized_text = requirements_text.replace("\\", "/")
    assert "e:/a_project" not in normalized_text.lower()
    assert not re.search(
        r"(?im)(?:^|[\s@])(?:file:/+|[A-Z]:/)",
        normalized_text,
    )
    assert not any(
        line.lower().startswith(("-e ", "--editable ")) for line in lines
    )
