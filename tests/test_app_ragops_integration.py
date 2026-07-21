"""Static checks for the lightweight RAGOps integration in ``app.py``."""

from __future__ import annotations

import ast
from pathlib import Path


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _read_app() -> tuple[str, ast.Module]:
    source = APP_PATH.read_text(encoding="utf-8")
    return source, ast.parse(source, filename="app.py")


def _attribute_path(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _attribute_path(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def _assigned_value(
    tree: ast.AST,
    target_path: str,
) -> list[tuple[ast.Assign, ast.AST]]:
    assignments: list[tuple[ast.Assign, ast.AST]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if any(_attribute_path(target) == target_path for target in node.targets):
            assignments.append((node, node.value))
    return assignments


def _find_smart_qa_branch(tree: ast.AST) -> ast.If:
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        test = node.test
        if (
            isinstance(test.left, ast.Name)
            and test.left.id == "feature"
            and any(
                isinstance(comparator, ast.Constant)
                and comparator.value == "💬 智能问答"
                for comparator in test.comparators
            )
        ):
            return node
    raise AssertionError("smart QA branch was not found")


def test_app_source_compiles_without_importing_heavy_dependencies() -> None:
    source, _ = _read_app()

    compile(source, "app.py", "exec")


def test_app_imports_runner_factory_and_initializes_trace_id() -> None:
    _, tree = _read_app()

    imports_factory = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "ragops_adapter"
        and any(alias.name == "create_studyrag_runner" for alias in node.names)
        for node in tree.body
    )
    assert imports_factory

    initialization = [
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Constant)
        and node.test.left.value == "last_trace_id"
        and any(isinstance(operator, ast.NotIn) for operator in node.test.ops)
        and any(
            _attribute_path(comparator) == "st.session_state"
            for comparator in node.test.comparators
        )
    ]
    assert len(initialization) == 1
    assert any(
        isinstance(statement, ast.Assign)
        and any(
            _attribute_path(target) == "st.session_state.last_trace_id"
            for target in statement.targets
        )
        and isinstance(statement.value, ast.Constant)
        and statement.value.value is None
        for statement in initialization[0].body
    )


def test_smart_qa_branch_runs_pipeline_once_and_preserves_existing_flow() -> None:
    _, tree = _read_app()
    smart_branch = _find_smart_qa_branch(tree)

    runner_assignment = next(
        node
        for node in smart_branch.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "runner" for target in node.targets)
    )
    assert isinstance(runner_assignment.value, ast.Call)
    assert isinstance(runner_assignment.value.func, ast.Name)
    assert runner_assignment.value.func.id == "create_studyrag_runner"

    traced_run_assignment = next(
        node
        for node in smart_branch.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "traced_run"
            for target in node.targets
        )
    )
    runner_call = traced_run_assignment.value
    assert isinstance(runner_call, ast.Call)
    assert _attribute_path(runner_call.func) == "runner.run"
    assert len(runner_call.args) == 2
    pipeline = runner_call.args[1]
    assert isinstance(pipeline, ast.Lambda)
    assert [argument.arg for argument in pipeline.args.args] == ["query"]
    assert isinstance(pipeline.body, ast.Call)
    assert isinstance(pipeline.body.func, ast.Name)
    assert pipeline.body.func.id == "smart_qa_with_trace"

    pipeline_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "smart_qa_with_trace"
    ]
    assert pipeline_calls == [pipeline.body]

    trace_assignment = next(
        node
        for node in smart_branch.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "trace" for target in node.targets)
    )
    assert _attribute_path(trace_assignment.value) == "traced_run.result"

    trace_id_assignment = next(
        node
        for node, _ in _assigned_value(
            smart_branch,
            "st.session_state.last_trace_id",
        )
    )
    assert _attribute_path(trace_id_assignment.value) == "traced_run.trace_id"

    append_log_calls = [
        node
        for node in ast.walk(smart_branch)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "append_qa_log"
    ]
    assert len(append_log_calls) == 1

    assert runner_assignment.lineno < traced_run_assignment.lineno
    assert traced_run_assignment.lineno < trace_assignment.lineno
    assert trace_assignment.lineno < trace_id_assignment.lineno
    assert trace_id_assignment.lineno < append_log_calls[0].lineno


def test_new_request_clears_trace_id_before_runner_executes() -> None:
    _, tree = _read_app()
    smart_branch = _find_smart_qa_branch(tree)
    send_branch = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(test_node, ast.Name) and test_node.id == "ask_btn"
            for test_node in ast.walk(node.test)
        )
        and smart_branch in ast.walk(node)
    )

    runner_call = next(
        node
        for node in ast.walk(smart_branch)
        if isinstance(node, ast.Call) and _attribute_path(node.func) == "runner.run"
    )
    request_resets = [
        assignment
        for assignment, value in _assigned_value(
            send_branch,
            "st.session_state.last_trace_id",
        )
        if isinstance(value, ast.Constant) and value.value is None
    ]
    assert len(request_resets) == 1
    assert request_resets[0].lineno < runner_call.lineno

    all_trace_id_resets = [
        assignment
        for assignment, value in _assigned_value(
            tree,
            "st.session_state.last_trace_id",
        )
        if isinstance(value, ast.Constant) and value.value is None
    ]
    assert len(all_trace_id_resets) == 5
