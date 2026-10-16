"""
Optional dependency groups (the extras of pyproject.toml, follow-up programme M4): which are installed, and what to
tell the user when a feature needs one that is not.

- ``ai``: LLM connectors, document library and vector databases (variable map, documents, AI suggestions);
- ``ml``: MissForest imputation (LightGBM), UMAP and FAMD (prince);
- ``survival``: the proportional-hazards test and the concordance of the Cox model (lifelines).

The statistics engine, the data pipeline and the analysis pages need none of them.
"""

from __future__ import annotations

import importlib.util
from functools import cache

EXTRAS: dict[str, tuple[str, ...]] = {
    "ai": ("langchain_core", "langchain_text_splitters", "pypdf", "lancedb", "qdrant_client"),
    "ml": ("lightgbm", "umap", "prince"),
    "survival": ("lifelines",),
}

# module needed by one method of the pages, and the extra that provides it
METHOD_MODULES: dict[str, tuple[str, str]] = {
    "MissForest": ("lightgbm", "ml"),
    "UMAP": ("umap", "ml"),
    "FAMD": ("prince", "ml"),
}


@cache
def available(module: str) -> bool:
    """True when ``module`` can be imported (checked without importing it)."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def extra_installed(extra: str) -> bool:
    return all(available(m) for m in EXTRAS[extra])


def method_available(method: str) -> bool:
    """False for a method whose optional library is missing (MissForest, UMAP, FAMD)."""
    module = METHOD_MODULES.get(method)
    return module is None or available(module[0])


def available_methods(methods: list[str]) -> tuple[list[str], list[str]]:
    """The methods that can run here, and those left out because their optional library is missing."""
    return [m for m in methods if method_available(m)], [m for m in methods if not method_available(m)]


def missing_message(feature: str, extra: str) -> str:
    return (
        f"{feature} needs the optional '{extra}' dependencies, which are not installed. Install them with "
        f"`uv sync --extra {extra}` (or `--extra all`), then restart the application."
    )


def missing_methods_message(methods: list[str]) -> str:
    extras = sorted({METHOD_MODULES[m][1] for m in methods if m in METHOD_MODULES})
    names = ", ".join(methods)
    install = " ".join(f"--extra {e}" for e in extras)
    return f"Not available here: {names} (install the optional dependencies with `uv sync {install}`)."


__all__ = [
    "EXTRAS",
    "METHOD_MODULES",
    "available",
    "available_methods",
    "extra_installed",
    "method_available",
    "missing_message",
    "missing_methods_message",
]
