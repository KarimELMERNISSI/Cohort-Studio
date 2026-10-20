"""The CI type check compares pyright's errors per file with a baseline that only falls (M5b)."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "check_type_baseline", ROOT / "dev_scripts" / "check_type_baseline.py"
)
assert _SPEC is not None and _SPEC.loader is not None
baseline = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(baseline)


def _error(file: str, line: int = 3) -> dict:
    return {"file": file, "message": "an error", "range": {"start": {"line": line}}, "rule": "x"}


@pytest.fixture
def counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "pyright_baseline.json"
    path.write_text(json.dumps({"a.py": 2, "b.py": 1}), encoding="utf-8")
    monkeypatch.setattr(baseline, "BASELINE", path)
    return path


def test_the_counts_as_recorded_pass(counts: Path) -> None:
    assert baseline.check(errors=[_error("a.py"), _error("a.py", 9), _error("b.py")]) == 0


def test_a_new_error_or_a_new_file_fails(counts: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        baseline.check(errors=[_error("a.py"), _error("a.py"), _error("a.py"), _error("b.py")]) == 1
    )
    assert (
        baseline.check(errors=[_error("a.py"), _error("a.py"), _error("b.py"), _error("c.py")]) == 1
    )
    assert "NEW ERRORS c.py: 1, baseline 0" in capsys.readouterr().out


def test_fewer_errors_fail_until_the_baseline_is_lowered(counts: Path) -> None:
    errors = [_error("a.py")]
    assert baseline.check(errors=errors) == 1  # a.py 1 < 2, b.py 0 < 1: lower it
    assert baseline.check(update=True, errors=errors) == 0
    assert json.loads(counts.read_text(encoding="utf-8")) == {"a.py": 1}
    assert baseline.check(errors=errors) == 0


def test_an_update_never_raises_a_count_or_adds_a_file(counts: Path) -> None:
    errors = [_error("a.py")] * 5 + [_error("c.py")]
    assert baseline.check(update=True, errors=errors) == 1  # the new errors still fail
    assert json.loads(counts.read_text(encoding="utf-8")) == {"a.py": 2}
