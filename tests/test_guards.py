# SPDX-License-Identifier: MPL-2.0
import ast
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def yaml_safe_load(path: Path) -> Any:
    return yaml.safe_load(path.read_text("utf-8"))


def _py_files(*dirs: str) -> list[Path]:
    return [f for d in dirs for f in (ROOT / d).rglob("*.py")]


def test_src_has_no_forbidden_constructs() -> None:  # AST-based: model.eval() and the word "pickle" in prose are fine
    files = _py_files("src")
    assert files  # an empty glob must not pass
    for f in files:
        for node in ast.walk(ast.parse(f.read_text("utf-8"))):
            assert not (isinstance(node, (ast.Import, ast.ImportFrom)) and "pickle" in ast.unparse(node)), f
            assert not (
                isinstance(node, ast.Attribute) and node.attr == "load" and ast.unparse(node.value) == "torch"
            ), f
            assert not (
                isinstance(node, ast.ImportFrom)
                and node.module == "torch"
                and any(a.name == "load" for a in node.names)
            ), f
            assert not (isinstance(node, ast.keyword) and node.arg == "allow_pickle"), f
            assert not (
                isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec"}
            ), f


def test_spdx_headers() -> None:
    files = _py_files("src", "tools", "tests")
    assert files
    for f in files:
        assert f.read_text("utf-8").startswith("# SPDX-License-Identifier: MPL-2.0"), f


def test_workflows_do_not_interpolate_event_text_in_run() -> None:
    wfs = list((ROOT / ".github" / "workflows").glob("*.yml"))
    assert wfs  # an empty glob must not pass
    for wf in wfs:
        for job in yaml_safe_load(wf)["jobs"].values():
            for step in job.get("steps", []):
                assert "${{ github.event" not in step.get("run", ""), (wf, step)
