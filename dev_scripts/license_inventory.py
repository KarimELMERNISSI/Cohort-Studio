"""
Licence inventory of the dependencies: documentation/developer/License_Dependencies.md.

    uv run python dev_scripts/license_inventory.py

The package sets come from uv.lock: the packages the core application installs, the packages each optional
group adds, and the development tools. The versions come from the lock file, the licences from the metadata of
the installed packages, so run it in an environment with every group:
`uv sync --extra all --extra dev --extra graph-yfiles`. A package that is not installed (another platform) is
listed without its licence. `tests/test_license_inventory.py` fails when the lock file lists a package or a
version that the document does not.
"""

from __future__ import annotations

import re
import tomllib
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "uv.lock"
OUT = ROOT / "documentation" / "developer" / "License_Dependencies.md"
PROJECT = "cohortstudio"
OPTIONAL = ("ai", "ml", "survival", "graph-yfiles")  # optional groups ("all" is ai + ml + survival)
COPYLEFT = re.compile(r"\b(A?GPL|LGPL|MPL|EPL|EUPL|CDDL)", re.IGNORECASE)

# Licence classifiers and free-text licences, shortened to their usual names
SHORT = {
    "MIT License": "MIT",
    "BSD License": "BSD",
    "Apache Software License": "Apache",
    "ISC License (ISCL)": "ISC",
    "Python Software Foundation License": "PSF",
    "Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
    "The Unlicense (Unlicense)": "Unlicense",
    "GNU Lesser General Public License v3 (LGPLv3)": "LGPL-3.0",
    "GNU Library or Lesser General Public License (LGPL)": "LGPL",
    "Historical Permission Notice and Disclaimer (HPND)": "HPND",
}

YFILES = """\
## Proprietary and network-dependent component

### yfiles-graphs-for-streamlit (optional group `graph-yfiles`)

License: *yWorks yFiles Streamlit Component Free License v1.0* (October 2025), shipped in the package as `LICENSE.md`.
Obligations and constraints that matter for this project:

| Clause | Consequence for CohortStudio |
|---|---|
| 1.2 Internet requirement | The component loads its "Cloud Runtime" from `https://www.yworks.com/streamlit-yfiles-core/` at render time. It cannot work on an air-gapped network. |
| 1.4 No redistribution | Only designated redistributables may be shipped; the package is therefore **not** part of the default install or Docker image (`requirements.txt`), only of `requirements-yfiles.txt` / the `graph-yfiles` extra. |
| 1.7 End-user notice | Users must be informed of the yWorks terms and the internet requirement: done on the About page and in a caption under every yFiles-rendered graph. |
| 2 Discontinuation | yWorks may modify, suspend or discontinue the component at any time; the application keeps working through the offline Plotly engine. |
| 3 Analytics | yWorks collects anonymised structural/geometric information about user-created graphs (no text), settings and interactions. The "zero telemetry" statement only holds with the default engine (`CMP_GRAPH_RENDERER=auto|plotly`, or `CMP_OFFLINE=1`). |

The graph connector (`graph/`) keeps the engine swappable; see `documentation/developer/Graph_Rendering.md`.
"""


PLATFORMS = (  # where CohortStudio runs: a dependency counts when its marker holds on one of them
    {
        "sys_platform": "linux",
        "platform_system": "Linux",
        "os_name": "posix",
        "platform_machine": "x86_64",
    },
    {
        "sys_platform": "win32",
        "platform_system": "Windows",
        "os_name": "nt",
        "platform_machine": "AMD64",
    },
    {
        "sys_platform": "darwin",
        "platform_system": "Darwin",
        "os_name": "posix",
        "platform_machine": "arm64",
    },
)


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _applies(dep: dict) -> bool:
    """False for a dependency only installed elsewhere (for example in the browser, sys_platform emscripten)."""
    marker = dep.get("marker")
    if not marker:
        return True
    from packaging.markers import Marker

    python = {
        "python_version": "3.14",
        "python_full_version": "3.14.0",
        "implementation_name": "cpython",
        "platform_python_implementation": "CPython",
    }
    return any(Marker(marker).evaluate({**python, **platform}) for platform in PLATFORMS)


def _lock() -> dict[str, dict]:
    data = tomllib.loads(LOCK.read_text(encoding="utf-8"))
    return {_norm(p["name"]): p for p in data["package"]}


def _closure(lock: dict[str, dict], seeds: list[dict]) -> set[str]:
    """Every package installed for these dependency entries, with the extras they ask for."""
    seen: set[str] = set()
    todo = list(seeds)
    while todo:
        dep = todo.pop()
        if not _applies(dep):
            continue
        name = _norm(dep["name"])
        package = lock.get(name)
        for extra in dep.get("extra", []):
            todo.extend((package or {}).get("optional-dependencies", {}).get(extra, []))
        if name in seen or package is None:
            continue
        seen.add(name)
        todo.extend(package.get("dependencies", []))
    return seen


def package_groups() -> dict[str, list[tuple[str, str]]]:
    """(name, version) per group: core, each optional group (what it adds), dev (what the tools add)."""
    lock = _lock()
    root = lock[PROJECT]
    optional = root.get("optional-dependencies", {})
    core = _closure(lock, root.get("dependencies", [])) - {PROJECT}
    groups: dict[str, set[str]] = {"core": core}
    taken = set(core)
    for group in OPTIONAL:
        groups[group] = _closure(lock, optional.get(group, [])) - taken - {PROJECT}
        taken |= groups[group]
    groups["dev"] = _closure(lock, optional.get("dev", [])) - taken - {PROJECT}
    return {
        group: sorted((name, str(lock[name]["version"])) for name in names)
        for group, names in groups.items()
    }


def licence(name: str) -> str:
    try:
        meta = metadata.metadata(name)
    except metadata.PackageNotFoundError:
        return "(not installed on this machine)"
    expression = (meta.get("License-Expression") or "").strip()
    if expression:
        return expression
    classifiers = sorted(
        {
            SHORT.get(c.split(" :: ")[-1], c.split(" :: ")[-1])
            for c in meta.get_all("Classifier") or []
            if c.startswith("License ::") and c.split(" :: ")[-1] != "OSI Approved"
        }
    )
    if classifiers:
        return " / ".join(classifiers)
    text = (meta.get("License") or "").strip()
    first = text.splitlines()[0].strip() if text else ""
    if first and len(first) <= 60:
        return SHORT.get(first, first)
    return "see the package"


def render(groups: dict[str, list[tuple[str, str]]]) -> str:
    licences = {name: licence(name) for packages in groups.values() for name, _ in packages}
    shipped = [name for group in ("core", "ai", "ml", "survival") for name, _ in groups[group]]
    copyleft = sorted(name for name in shipped if COPYLEFT.search(licences[name]))
    lines = [
        "# License Dependencies",
        "",
        "Generated by `dev_scripts/license_inventory.py` from `uv.lock` (the packages and their versions) and the",
        "metadata of the installed packages (their licences). Regenerate it after a change of the dependencies:",
        "`uv run python dev_scripts/license_inventory.py`.",
        "",
        "CohortStudio is released under the Apache License 2.0.",
        "",
        "## Summary",
        "",
        f"- The core application installs {len(groups['core'])} packages. The optional groups add "
        + ", ".join(f"{len(groups[g])} (`{g}`)" for g in OPTIONAL)
        + f". The development tools add {len(groups['dev'])}; they are not shipped.",
    ]
    if copyleft:
        lines.append(
            "- Copyleft licences among the shipped packages: "
            + ", ".join(f"`{name}` ({licences[name]})" for name in copyleft)
            + ". They are weak, file-level licences: they apply to the library's own files, which CohortStudio "
            "uses unmodified, and require keeping their notices."
        )
    else:
        lines.append("- No copyleft licence among the shipped packages.")
    lines += [
        "- One optional component is proprietary and needs an internet connection: the yFiles graph engine "
        "(group `graph-yfiles`, not installed by default). See below.",
        "",
        YFILES,
    ]
    titles = {
        "core": "Core application",
        "ai": "Optional group `ai` (packages it adds)",
        "ml": "Optional group `ml` (packages it adds)",
        "survival": "Optional group `survival` (packages it adds)",
        "graph-yfiles": "Optional group `graph-yfiles` (packages it adds)",
        "dev": "Development tools (not shipped)",
    }
    for group, title in titles.items():
        lines += [
            "",
            f"## {title}",
            "",
            "| Package | Version | Licence |",
            "| :--- | :--- | :--- |",
        ]
        lines += [f"| {name} | {version} | {licences[name]} |" for name, version in groups[group]]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.write_text(render(package_groups()), encoding="utf-8", newline="\n")
    print(f"written {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
