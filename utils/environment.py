"""
The software a dataset was loaded and analysed with: CohortStudio's version and commit, Python and the key libraries.
It is recorded in the trace when a dataset is loaded (the ``initial_load`` step) and printed in the study dossier,
so that a reader knows which versions produced the results.
"""

from __future__ import annotations

import platform
import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIBRARIES = ("numpy", "pandas", "scipy", "statsmodels", "scikit-learn", "lifelines", "streamlit")


def app_version() -> str:
    """CohortStudio's version: the installed package, else pyproject.toml (a source checkout)."""
    try:
        return version("cohortstudio")
    except PackageNotFoundError:
        pass
    try:
        return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        return "unknown"


def git_commit(root: Path = ROOT) -> str | None:
    """The short commit of a source checkout (read from .git, without running git), or None."""
    git = root / ".git"
    try:
        head = (git / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref: "):
            return head[:7] or None
        ref = head.removeprefix("ref: ")
        loose = git / ref
        if loose.is_file():
            return loose.read_text(encoding="utf-8").strip()[:7] or None
        for line in (git / "packed-refs").read_text(encoding="utf-8").splitlines():
            if line.endswith(f" {ref}"):
                return line.split(" ", 1)[0][:7]
    except OSError:
        return None
    return None


def environment() -> dict[str, str]:
    """CohortStudio, its commit (when known), Python and the versions of the key libraries that are installed."""
    found = {"cohortstudio": app_version()}
    commit = git_commit()
    if commit:
        found["commit"] = commit
    found["python"] = platform.python_version()
    for name in LIBRARIES:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            continue
    return found


def environment_text() -> str:
    """The environment in one line, for reports: "CohortStudio 2.4.1 (commit abc1234), Python 3.14.3, numpy 2.4, ..."."""
    env = environment()
    head = f"CohortStudio {env['cohortstudio']}" + (f" (commit {env['commit']})" if "commit" in env else "")
    rest = [f"Python {env['python']}"] + [f"{name} {env[name]}" for name in LIBRARIES if name in env]
    return ", ".join([head, *rest])


__all__ = ["LIBRARIES", "app_version", "environment", "environment_text", "git_commit"]
