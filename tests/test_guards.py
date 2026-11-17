# SPDX-License-Identifier: MPL-2.0
import ast
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def yaml_safe_load(path: Path) -> Any:
    return yaml.safe_load(path.read_text("utf-8"))


# Attacker-controlled contexts (GitHub's "untrusted input" list): any github.event field and the PR head ref.
_UNTRUSTED = re.compile(r"\$\{\{\s*github\.(event\b|head_ref\b)")


def interpolates_untrusted(run: str) -> bool:
    """True if a run script expands an untrusted context, however the expression is spaced."""
    return bool(_UNTRUSTED.search(run))


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


@pytest.mark.parametrize(
    ("run", "untrusted"),
    [
        ('echo "${{ github.event.issue.title }}"', True),
        ('echo "${{github.event.pull_request.body}}"', True),  # no spaces is the same expression
        ('echo "${{   github.event.head_commit.message }}"', True),
        ('git checkout "${{ github.head_ref }}"', True),  # the PR branch name is attacker-chosen
        ('echo "${{ matrix.python }}"', False),
        ('echo "${{ env.ESPEAK_TAG }}"', False),
    ],
)
def test_untrusted_interpolation_detector(run: str, untrusted: bool) -> None:
    assert interpolates_untrusted(run) is untrusted


def test_workflows_do_not_interpolate_event_text_in_run() -> None:
    wf_dir = ROOT / ".github" / "workflows"
    wfs = [*wf_dir.glob("*.yml"), *wf_dir.glob("*.yaml")]
    assert wfs  # an empty glob must not pass
    for wf in wfs:
        for job in yaml_safe_load(wf)["jobs"].values():
            for step in job.get("steps", []):
                assert not interpolates_untrusted(step.get("run", "")), (wf, step)
