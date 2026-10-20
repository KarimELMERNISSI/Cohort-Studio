"""The CI coverage-floor check sums pytest-cov's XML per top-level package."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("check_coverage_floors", ROOT / "dev_scripts" / "check_coverage_floors.py")
assert _SPEC is not None and _SPEC.loader is not None
floors = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(floors)

XML = """<?xml version="1.0" ?>
<coverage line-rate="0.5"><packages>
<package name="biostats"><classes>
<class filename="biostats/a.py"><lines><line number="1" hits="3"/><line number="2" hits="1"/></lines></class>
<class filename="biostats/mi/b.py"><lines><line number="1" hits="0"/><line number="2" hits="1"/></lines></class>
</classes></package>
<package name="ui"><classes>
<class filename="ui\\c.py"><lines><line number="1" hits="0"/><line number="2" hits="0"/></lines></class>
</classes></package>
</packages></coverage>"""


@pytest.fixture
def report(tmp_path: Path) -> Path:
    path = tmp_path / "coverage.xml"
    path.write_text(XML, encoding="utf-8")
    return path


def test_coverage_is_summed_per_top_level_package(report: Path) -> None:
    assert floors.package_coverage(report) == {"biostats": (3, 4), "ui": (0, 2)}


def test_packages_under_their_floor_are_reported(report: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert floors.check(report, {"biostats": 75.0, "ui": 10.0}) == ["ui"]
    assert floors.check(report, {"biostats": 76.0}) == ["biostats"]
    assert "UNDER THE FLOOR" in capsys.readouterr().out


def test_every_floor_names_a_package() -> None:
    assert all((ROOT / package).is_dir() for package in floors.FLOORS)


def test_a_package_far_above_its_floor_asks_for_a_higher_floor(
    report: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # biostats is at 75 %: a floor of 70 is within the margin, a floor of 65 must rise
    assert floors.check(report, {"biostats": 75.0 - floors.MARGIN}) == []
    assert floors.check(report, {"biostats": 65.0}) == ["biostats"]
    assert "RAISE THE FLOOR to 73" in capsys.readouterr().out
