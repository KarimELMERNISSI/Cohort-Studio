"""
Per-package coverage floors, checked in CI after the test run.

Reads the ``coverage.xml`` written by pytest-cov (the same measurement as ``--cov-fail-under``; the
``.coverage`` data file can miss lines run in Streamlit AppTest threads) and fails when a package falls
under its floor. Every package has one (tests/test_quality_gates.py). Floors sit about 2 points under the
measured values, and only rise: a package more than ``MARGIN`` points above its floor fails too, until the
floor is raised.

Usage: python dev_scripts/check_coverage_floors.py [coverage.xml]
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Measured 2026-10-20 (M5b): app_pages 53.2, biostats 95.0, core 94.3, enrich 69.9, explore 77.0, graph 93.7,
# ingest 90.7, manage 74.3, monitor 8.3 (it needs tests), prompts 87.0, ui 84.9, utils 63.2 (72.4 overall).
# app_pages raised to 54 at 56.3 measured, when the smoke test began opening every tab of the lazy pages.
FLOORS: dict[str, float] = {
    "app_pages": 54.0,
    "biostats": 93.0,
    "core": 92.0,
    "enrich": 68.0,
    "explore": 75.0,
    "graph": 92.0,
    "ingest": 89.0,
    "manage": 72.0,
    "monitor": 6.0,
    "prompts": 85.0,
    "ui": 83.0,
    "utils": 61.0,
}
MARGIN = 4.0  # a package more than this above its floor fails until the floor is raised


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
    """Print one line per package and return the packages under their floor or too far above it."""
    coverage = package_coverage(xml_path)
    failures = []
    for package, floor in floors.items():
        covered, total = coverage.get(package, (0, 0))
        percent = 100.0 * covered / total if total else 0.0
        if percent < floor:
            verdict = "UNDER THE FLOOR"
        elif percent > floor + MARGIN:
            verdict = f"RAISE THE FLOOR to {int(percent) - 2}"
        else:
            verdict = "ok"
        print(f"{package:10s} {percent:5.1f} %  (floor {floor:.0f} %)  {verdict}")
        if verdict != "ok":
            failures.append(package)
    return failures


if __name__ == "__main__":
    sys.exit(1 if check(Path(sys.argv[1] if len(sys.argv) > 1 else "coverage.xml")) else 0)
