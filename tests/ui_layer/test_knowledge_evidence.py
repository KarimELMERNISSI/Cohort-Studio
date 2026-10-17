"""The 'Your documents' path of the Start page workflow is done only when the library can be used: indexed
documents still in the library, and the AI engine on."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import ui.progress as progress
import utils.data_paths as dp
from core.workspace import UserScope, remove_pdf


@pytest.fixture
def scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> UserScope:
    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path))
    user = UserScope.for_user("reader")
    monkeypatch.setattr(progress, "get_current_scope", lambda: user)
    return user


def _index(scope: UserScope, *documents: str) -> None:
    folder = scope.vectors_dir() / "lancedb"
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {"engine": "lancedb", "documents": list(documents)}
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_the_documents_path_is_done_only_when_the_library_can_be_used(scope: UserScope) -> None:
    assert progress.knowledge_evidence(ai_on=True)["docs"] == (
        False,
        "No document in your library yet.",
    )
    (scope.documents_dir() / "cardiac-bmi.pdf").write_bytes(b"%PDF-1.4")
    done, text = progress.knowledge_evidence(ai_on=True)["docs"]
    assert not done and "not indexed yet" in text

    _index(scope, "cardiac-bmi.pdf")
    done, text = progress.knowledge_evidence(ai_on=False)["docs"]
    assert not done and text == (
        "cardiac-bmi.pdf indexed in your library: initialise the AI engine to use it."
    )
    done, text = progress.knowledge_evidence(ai_on=True)["docs"]
    assert done and text == "cardiac-bmi.pdf indexed for chat and search."

    assert remove_pdf(scope.documents_dir(), "cardiac-bmi.pdf")  # the index still lists it
    assert progress.knowledge_evidence(ai_on=True)["docs"] == (
        False,
        "No document in your library yet.",
    )


def test_the_documents_are_named(scope: UserScope) -> None:
    names = ["a.pdf", "b.pdf", "c.pdf", "d.pdf", "e.pdf"]
    for name in names:
        (scope.documents_dir() / name).write_bytes(b"%PDF-1.4")
    _index(scope, *names)
    done, text = progress.knowledge_evidence(ai_on=True)["docs"]
    assert done and text.startswith("a.pdf, b.pdf, c.pdf and 2 more indexed")
