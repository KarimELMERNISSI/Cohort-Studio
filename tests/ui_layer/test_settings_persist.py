"""
Settings keeps its choices across pages (M5, AppTest).

Streamlit drops the value of a widget whose page is not shown. The day-first default was lost on the way to Import,
the page that reads it; a revisit of Settings showed Ollama, the first model and an empty consent box, and the
consent card withdrew the running engine's consent. Each cloud provider has its own consent box, and unticking the
one of the engine that runs stops it.
"""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import utils.data_paths as dp
from ui import nav

ROOT = Path(__file__).resolve().parents[2]
MAIN = str(ROOT / "main.py")


def _visit(at: AppTest, page: str) -> None:
    at.switch_page(nav.page_path(page)).run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]


@pytest.fixture
def signed_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path))
    monkeypatch.setitem(os.environ, "CMP_IAM_PROVIDER", "local")
    at = AppTest.from_file(MAIN, default_timeout=240)
    at.run()
    at.session_state["user_authenticated"] = True
    at.session_state["username"] = "persist_user"
    at.session_state["user_role"] = "researcher"
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    return at


@pytest.fixture
def app(signed_in: AppTest) -> AppTest:
    _visit(signed_in, "settings")
    return signed_in


def test_the_day_first_default_reaches_the_import_page(app: AppTest) -> None:
    app.toggle(key="import_default_dayfirst").set_value(True).run()
    _visit(app, "import")
    assert app.session_state["import_default_dayfirst"] is True  # what a new import draft reads
    _visit(app, "settings")
    assert app.toggle(key="import_default_dayfirst").value is True


def test_the_ai_provider_model_and_consent_survive_a_revisit(app: AppTest) -> None:
    pytest.importorskip("langchain_core", reason="the AI engine is in the optional 'ai' extra")
    app.selectbox(key="rag_provider_select").select("gemini").run()
    app.selectbox(key="rag_gemini_model_select").select("gemini-flash-lite-latest").run()
    app.checkbox(key="rag_gemini_consent").check().run()
    assert app.session_state["rag_manager"].cloud_consent is True

    _visit(app, "start")
    _visit(app, "settings")
    assert app.selectbox(key="rag_provider_select").value == "gemini"
    assert app.selectbox(key="rag_gemini_model_select").value == "gemini-flash-lite-latest"
    assert app.checkbox(key="rag_gemini_consent").value is True
    assert app.session_state["rag_manager"].cloud_consent is True  # not withdrawn by the visit
    assert not [w for w in app.warning if "Session State API" in str(w.value)]


def test_settings_start_from_the_engine_that_runs(signed_in: AppTest, tmp_path: Path) -> None:
    """An engine running before Settings is first shown in the session: the page shows it, not Ollama."""
    pytest.importorskip("langchain_core", reason="the AI engine is in the optional 'ai' extra")
    from types import SimpleNamespace

    from manage.rag_manager import RAGManager

    manager = RAGManager(documents_dir=str(tmp_path / "documents"), provider="mistral")
    manager.initialized = True
    manager.cloud_consent = True
    manager.llm = SimpleNamespace(config=SimpleNamespace(model_name="mistral-large-latest"))
    signed_in.session_state["rag_manager"] = manager
    _visit(signed_in, "settings")
    assert signed_in.selectbox(key="rag_provider_select").value == "mistral"
    assert signed_in.selectbox(key="rag_mistral_model_select").value == "mistral-large-latest"
    assert signed_in.checkbox(key="rag_mistral_consent").value is True
    assert signed_in.session_state["rag_manager"].cloud_consent is True


def test_unticking_the_consent_stops_the_engine_that_runs(
    signed_in: AppTest, tmp_path: Path
) -> None:
    """The consent was checked only when the engine started: unticking it left the engine answering."""
    pytest.importorskip("langchain_core", reason="the AI engine is in the optional 'ai' extra")
    from types import SimpleNamespace

    from manage.rag_manager import RAGManager

    manager = RAGManager(documents_dir=str(tmp_path / "documents"), provider="mistral")
    manager.initialized = True
    manager.cloud_consent = True
    manager.llm = SimpleNamespace(config=SimpleNamespace(model_name="mistral-large-latest"))
    manager.vector_store = SimpleNamespace()
    signed_in.session_state["rag_manager"] = manager
    _visit(signed_in, "settings")

    signed_in.selectbox(key="rag_provider_select").select("gemini").run()
    assert signed_in.checkbox(key="rag_gemini_consent").value is False  # one consent per provider
    # looking at another provider stops nothing
    assert signed_in.session_state["rag_manager"].initialized

    signed_in.selectbox(key="rag_provider_select").select("mistral").run()
    signed_in.checkbox(key="rag_mistral_consent").uncheck().run()
    assert not signed_in.exception, [str(e.value)[:300] for e in signed_in.exception]
    stopped = signed_in.session_state["rag_manager"]
    assert not stopped.initialized and stopped.llm is None and stopped.vector_store is None
    assert stopped.cloud_consent is False
    assert any("nothing more is sent" in t.value for t in signed_in.toast)
    assert any(
        "The engine stopped because the data-sharing consent was withdrawn" in c.value
        for c in signed_in.caption
    )
