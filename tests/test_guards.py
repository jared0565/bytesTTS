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


def test_derived_from_lines_follow_the_notice_format() -> None:  # NOTICE promises "<version>, <original path>"
    files = _py_files("src", "tools", "tests")
    lines = [(f, line) for f in files for line in f.read_text("utf-8").splitlines()[:3] if "Derived from" in line]
    assert lines  # the port's files carry them: an empty list means the scan broke
    for f, line in lines:
        assert re.match(r"# Derived from Coqui TTS v0\.22\.0, TTS/\S+", line), (f, line)


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


def _workflow_steps() -> list[dict[str, Any]]:
    wf_dir = ROOT / ".github" / "workflows"
    wfs = [*wf_dir.glob("*.yml"), *wf_dir.glob("*.yaml")]
    assert wfs  # an empty glob must not pass
    return [step for wf in wfs for job in yaml_safe_load(wf)["jobs"].values() for step in job.get("steps", [])]


def test_workflow_actions_are_pinned_to_commit_shas() -> None:  # a tag can be moved; a commit SHA cannot
    uses = [step["uses"] for step in _workflow_steps() if "uses" in step]
    assert uses
    for ref in uses:
        assert re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", ref), ref


def test_espeak_source_build_verifies_the_reviewed_commit() -> None:
    builds = [s["run"] for s in _workflow_steps() if "git clone" in s.get("run", "") and "espeak-ng" in s["run"]]
    assert builds
    for run in builds:
        assert 'test "$(git rev-parse HEAD)" = "$ESPEAK_COMMIT"' in run


def test_espeak_build_is_never_restored_from_cache() -> None:  # a restored cache would skip the commit check
    steps = _workflow_steps()
    assert not [s for s in steps if "actions/cache" in s.get("uses", "") and "espeak" in str(s.get("with", {}))]
    for s in steps:
        if "git clone" in s.get("run", "") and "espeak-ng" in s["run"]:
            assert "if" not in s or "cache" not in str(s["if"]), s


def test_workflows_do_not_interpolate_event_text_in_run() -> None:
    wf_dir = ROOT / ".github" / "workflows"
    wfs = [*wf_dir.glob("*.yml"), *wf_dir.glob("*.yaml")]
    assert wfs  # an empty glob must not pass
    for wf in wfs:
        for job in yaml_safe_load(wf)["jobs"].values():
            for step in job.get("steps", []):
                assert not interpolates_untrusted(step.get("run", "")), (wf, step)


def test_uncapped_ort_session_is_refused() -> None:  # tests/conftest.py guard: this PC resets under all-core ORT
    import onnxruntime as ort

    with pytest.raises(pytest.fail.Exception, match="intra_op_num_threads"):
        ort.InferenceSession(b"not a model")
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    with pytest.raises(pytest.fail.Exception, match="intra_op_num_threads"):
        ort.InferenceSession(b"not a model", sess_options=so)


def test_capped_ort_session_reaches_onnx_runtime() -> None:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    with pytest.raises(Exception) as e:  # any ORT parse error; only the guard's failure would be wrong
        ort.InferenceSession(b"not a model", sess_options=so)
    assert not isinstance(e.value, pytest.fail.Exception)


def test_thread_environment_is_capped() -> None:
    import os

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        assert os.environ.get(var) == "1", var
