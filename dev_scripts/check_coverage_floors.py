"""
Per-package coverage floors, checked in CI after the test run.

Reads the ``coverage.xml`` written by pytest-cov (the same measurement as ``--cov-fail-under``; the
``.coverage`` data file can miss lines run in Streamlit AppTest threads) and fails when a package falls
under its floor. Floors sit a little under the measured values: raise a floor when its package exceeds
it by 2 points.

Usage: python dev_scripts/check_coverage_floors.py [coverage.xml]
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

FLOORS: dict[str, float] = {"biostats": 90.0, "core": 90.0, "graph": 91.0, "ingest": 87.0, "ui": 75.0}


def package_coverage(xml_path: Path) -> dict[str, tuple[int, int]]:
    """Covered and total statements per top-level package."""
    totals: dict[str, tuple[int, int]] = {}
    for cls in ET.parse(xml_path).getroot().iter("class"):
        lines = cls.find("lines")
        if lines is None:
            continue
        package = cls.get("filename", "").replace("\\", "/").split("/", 1)[0]
        covered = sum(1 for line in lines if line.get("hits") != "0")
        done, total = totals.get(package, (0, 0))
        totals[package] = (done + covered, total + len(lines))
    return totals


def check(xml_path: Path, floors: dict[str, float] = FLOORS) -> list[str]:
    """Print one line per package and return the packages under their floor."""
    coverage = package_coverage(xml_path)
    failures = []
    for package, floor in floors.items():
        covered, total = coverage.get(package, (0, 0))
        percent = 100.0 * covered / total if total else 0.0
        print(f"{package:10s} {percent:5.1f} %  (floor {floor:.0f} %)  {'ok' if percent >= floor else 'UNDER THE FLOOR'}")
        if percent < floor:
            failures.append(package)
    return failures


if __name__ == "__main__":
    sys.exit(1 if check(Path(sys.argv[1] if len(sys.argv) > 1 else "coverage.xml")) else 0)
