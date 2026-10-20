"""
Type-checking errors per file, against a baseline that only falls (M5b).

pyright checks every package (pyrightconfig.json). The errors of today's code are counted per file in
``dev_scripts/pyright_baseline.json``. The check fails when a file outside the baseline has errors, and when a
file's count differs from it: a rise is a new error, and a fall must lower the baseline (``--update``), so that the
count cannot rise back unseen. ``--update`` only lowers counts: it never raises one or adds a file. pyright's own
``ignore`` would silence the new errors of those files, in the editor too.

Usage: python dev_scripts/check_type_baseline.py [--update]
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "dev_scripts" / "pyright_baseline.json"


def diagnostics() -> list[dict]:
    """pyright's errors, each with its file relative to the repository."""
    run = subprocess.run(
        [sys.executable, "-m", "pyright", "--outputjson"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if run.returncode not in (0, 1):  # 0: no error, 1: errors; anything else is pyright failing
        raise RuntimeError(f"pyright failed ({run.returncode}): {run.stderr.strip()[:500]}")
    report = json.loads(run.stdout)
    if not report.get("summary", {}).get("filesAnalyzed"):
        raise RuntimeError("pyright analysed no file: check pyrightconfig.json")
    errors = []
    for item in report.get("generalDiagnostics", []):
        if item.get("severity") == "error":
            item["file"] = Path(item["file"]).resolve().relative_to(ROOT).as_posix()
            errors.append(item)
    return errors


def lowered(baseline: dict[str, int], counts: Counter[str]) -> dict[str, int]:
    """The baseline after ``--update``: counts only fall, and no file is added."""
    return {
        f: min(n, counts.get(f, 0)) for f, n in sorted(baseline.items()) if counts.get(f, 0) > 0
    }


def check(update: bool = False, errors: list[dict] | None = None) -> int:
    errors = diagnostics() if errors is None else errors
    counts = Counter(item["file"] for item in errors)
    baseline: dict[str, int] = json.loads(BASELINE.read_text(encoding="utf-8"))
    risen = {f: n for f, n in counts.items() if n > baseline.get(f, 0)}
    fallen = {f: n for f, n in baseline.items() if counts.get(f, 0) < n}
    for item in errors:
        if item["file"] in risen:
            line = item["range"]["start"]["line"] + 1
            print(
                f"{item['file']}:{line}: {item['message'].splitlines()[0]} ({item.get('rule', '')})"
            )
    for file, n in sorted(risen.items()):
        print(f"NEW ERRORS {file}: {n}, baseline {baseline.get(file, 0)}")
    if update:
        new = lowered(baseline, counts)
        BASELINE.write_text(json.dumps(new, indent=2) + "\n", encoding="utf-8")
        print(f"Baseline lowered: {sum(new.values())} errors in {len(new)} files.")
        return 1 if risen else 0
    for file, n in sorted(fallen.items()):
        print(f"FEWER ERRORS {file}: {counts.get(file, 0)}, baseline {n}: lower it (--update)")
    print(
        f"{sum(counts.values())} errors in {len(counts)} files; baseline {sum(baseline.values())}."
    )
    return 1 if risen or fallen else 0


if __name__ == "__main__":
    sys.exit(check(update="--update" in sys.argv[1:]))
