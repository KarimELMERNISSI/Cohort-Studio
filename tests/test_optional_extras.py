"""Optional dependency groups: what is installed, which methods are offered, what the user is told (M4)."""

from __future__ import annotations

import pytest

import utils.optional as opt


def test_installed_modules_are_found_without_importing_them() -> None:
    assert opt.available("json") and not opt.available("surely_not_a_module_cmp")
    assert opt.method_available("PCA")  # a method without an optional library


def test_methods_whose_library_is_missing_are_left_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(opt, "available", lambda module: module != "umap")
    offered, missing = opt.available_methods(["PCA", "t-SNE", "UMAP", "FAMD"])
    assert offered == ["PCA", "t-SNE", "FAMD"] and missing == ["UMAP"]
    assert "`uv sync --extra ml`" in opt.missing_methods_message(missing)


def test_the_message_names_the_extra_and_the_command() -> None:
    text = opt.missing_message("The AI engine", "ai")
    assert "'ai'" in text and "uv sync --extra ai" in text and "--extra all" in text
    assert set(opt.EXTRAS) == {"ai", "ml", "survival"}
