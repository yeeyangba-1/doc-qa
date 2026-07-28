"""Static checks for the DeepSeek model used by StudyRAG and its traces."""

from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAG_PATH = PROJECT_ROOT / "rag.py"
RAGOPS_ADAPTER_PATH = PROJECT_ROOT / "ragops_adapter.py"
EXPECTED_MODEL = "deepseek-v4-flash"
EXPECTED_EXTRA_BODY = {"thinking": {"type": "disabled"}}
LEGACY_MODEL = "-".join(("deepseek", "chat"))


def _read_source(path: Path) -> tuple[str, ast.Module]:
    source = path.read_text(encoding="utf-8")
    return source, ast.parse(source, filename=path.name)


def _attribute_path(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _attribute_path(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def test_all_deepseek_completion_calls_use_the_expected_model() -> None:
    source, tree = _read_source(RAG_PATH)
    completion_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _attribute_path(node.func) == "llm.chat.completions.create"
    ]

    assert len(completion_calls) >= 2
    for call in completion_calls:
        model_values = [
            keyword.value for keyword in call.keywords if keyword.arg == "model"
        ]
        assert len(model_values) == 1
        assert isinstance(model_values[0], ast.Constant)
        assert model_values[0].value == EXPECTED_MODEL

        extra_body_values = [
            keyword.value
            for keyword in call.keywords
            if keyword.arg == "extra_body"
        ]
        assert len(extra_body_values) == 1
        assert ast.literal_eval(extra_body_values[0]) == EXPECTED_EXTRA_BODY

    assert LEGACY_MODEL not in source


def test_ragops_trace_uses_the_expected_model() -> None:
    source, tree = _read_source(RAGOPS_ADAPTER_PATH)
    model_assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "MODEL_NAME"
            for target in node.targets
        )
    ]

    assert len(model_assignments) == 1
    model_value = model_assignments[0].value
    assert isinstance(model_value, ast.Constant)
    assert model_value.value == EXPECTED_MODEL
    assert LEGACY_MODEL not in source
