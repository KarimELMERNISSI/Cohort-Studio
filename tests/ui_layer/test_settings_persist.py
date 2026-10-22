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
    app.checkbox(key="rag_consent::gemini").check().run()
    assert app.session_state["rag_manager"].cloud_consent is True

    _visit(app, "start")
    _visit(app, "settings")
    assert app.selectbox(key="rag_provider_select").value == "gemini"
    assert app.selectbox(key="rag_gemini_model_select").value == "gemini-flash-lite-latest"
    assert app.checkbox(key="rag_consent::gemini").value is True
    assert app.session_state["rag_manager"].cloud_consent is True  # not withdrawn by the visit
    assert not [w for w in app.warning if "Session State API" in str(w.value)]


def test_settings_start_from_the_engine_that_runs(signed_in: AppTest, tmp_path: Path) -> None:
    """An engine running before Settings is first shown in the session: the page shows it, not Ollama."""
    pytest.importorskip("langchain_core", reason="the AI engine is in the optional 'ai' extra")
    from types import SimpleNamespace

    from manage.rag_manager import RAGManager

    manager = RAGManager(documents_dir=str(tmp_path / "documents"), provider="mistral")
    manager.initialized = True
    manager.confirm("mistral")
    manager.llm = SimpleNamespace(config=SimpleNamespace(model_name="mistral-large-latest"))
    signed_in.session_state["rag_manager"] = manager
    _visit(signed_in, "settings")
    assert signed_in.selectbox(key="rag_provider_select").value == "mistral"
    assert signed_in.selectbox(key="rag_mistral_model_select").value == "mistral-large-latest"
    assert signed_in.checkbox(key="rag_consent::mistral").value is True
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
    manager.confirm("mistral")
    manager.llm = SimpleNamespace(config=SimpleNamespace(model_name="mistral-large-latest"))
    manager.vector_store = SimpleNamespace()
    signed_in.session_state["rag_manager"] = manager
    _visit(signed_in, "settings")

    signed_in.selectbox(key="rag_provider_select").select("gemini").run()
    assert signed_in.checkbox(key="rag_consent::gemini").value is False  # one consent per provider
    # looking at another provider stops nothing
    assert signed_in.session_state["rag_manager"].initialized

    signed_in.selectbox(key="rag_provider_select").select("mistral").run()
    signed_in.checkbox(key="rag_consent::mistral").uncheck().run()
    assert not signed_in.exception, [str(e.value)[:300] for e in signed_in.exception]
    stopped = signed_in.session_state["rag_manager"]
    assert not stopped.initialized and stopped.llm is None and stopped.vector_store is None
    assert stopped.cloud_consent is False
    assert any("Nothing more is sent from this tab" in t.value for t in signed_in.toast)
    assert any(
        "The engine stopped because the data-sharing confirmation was withdrawn" in c.value
        for c in signed_in.caption
    )


def _running_mistral(at: AppTest, tmp_path: Path):
    from types import SimpleNamespace

    from manage.rag_manager import RAGManager

    manager = RAGManager(documents_dir=str(tmp_path / "documents"), provider="mistral")
    manager.initialized = True
    manager.confirm("mistral")
    manager.llm = SimpleNamespace(config=SimpleNamespace(model_name="mistral-large-latest"))
    manager.vector_store = SimpleNamespace()
    at.session_state["rag_manager"] = manager
    _visit(at, "settings")


def test_the_running_engine_can_be_stopped_whatever_provider_is_shown(
    signed_in: AppTest, tmp_path: Path
) -> None:
    """With Ollama selected, the page said "no data leaves this server" while a cloud engine ran."""
    pytest.importorskip("langchain_core", reason="the AI engine is in the optional 'ai' extra")
    _running_mistral(signed_in, tmp_path)
    signed_in.selectbox(key="rag_provider_select").select("ollama").run()
    assert any("The AI engine runs with Mistral AI" in i.value for i in signed_in.info)
    signed_in.button(key="rag_withdraw_stop").click().run()
    assert not signed_in.exception, [str(e.value)[:300] for e in signed_in.exception]
    stopped = signed_in.session_state["rag_manager"]
    assert not stopped.holds_connections() and stopped.confirmed == set()
    signed_in.selectbox(key="rag_provider_select").select("mistral").run()
    assert signed_in.checkbox(key="rag_consent::mistral").value is False  # to confirm again


def test_claude_with_cloud_embeddings_names_both_recipients(signed_in: AppTest) -> None:
    pytest.importorskip("langchain_core", reason="the AI engine is in the optional 'ai' extra")
    _visit(signed_in, "settings")
    signed_in.selectbox(key="rag_provider_select").select("anthropic").run()
    signed_in.selectbox(key="rag_claude_embed_provider").select("openai").run()
    box = signed_in.checkbox(key="rag_consent::anthropic+openai")
    names = "Anthropic Claude and OpenAI (embeddings)"
    assert box.label == f"I confirm this data may be sent to {names}"
    notice = next(w for w in signed_in.warning if "What leaves this server" in (w.title or ""))
    assert "OpenAI, for the embeddings" in notice.value
    box.check().run()
    confirmed = signed_in.session_state["rag_manager"].confirmed
    assert confirmed == {frozenset({"anthropic", "openai"})}
    # confirmed together as Claude's embedding provider, OpenAI is not confirmed to answer on its own
    signed_in.selectbox(key="rag_provider_select").select("openai").run()
    assert signed_in.checkbox(key="rag_consent::openai").value is False


def test_an_untouched_box_of_another_configuration_stops_nothing(
    signed_in: AppTest, tmp_path: Path
) -> None:
    """Every render withdrew the recipients of an unticked box: looking at Claude with Mistral embeddings
    stopped a Mistral engine."""
    pytest.importorskip("langchain_core", reason="the AI engine is in the optional 'ai' extra")
    _running_mistral(signed_in, tmp_path)
    signed_in.selectbox(key="rag_provider_select").select("anthropic").run()
    signed_in.selectbox(key="rag_claude_embed_provider").select("mistral").run()
    assert signed_in.checkbox(key="rag_consent::anthropic+mistral").value is False
    manager = signed_in.session_state["rag_manager"]
    assert manager.initialized and manager.is_confirmed(frozenset({"mistral"}))
    assert not signed_in.toast


def test_a_withdrawal_unticks_the_other_boxes_that_name_the_recipient(
    signed_in: AppTest, tmp_path: Path
) -> None:
    """Unticking OpenAI withdrew the Claude + OpenAI confirmation too, but its box stayed ticked and confirmed
    the pair again, without a click, as soon as it was shown."""
    pytest.importorskip("langchain_core", reason="the AI engine is in the optional 'ai' extra")
    _visit(signed_in, "settings")
    signed_in.selectbox(key="rag_provider_select").select("anthropic").run()
    signed_in.selectbox(key="rag_claude_embed_provider").select("openai").run()
    signed_in.checkbox(key="rag_consent::anthropic+openai").check().run()
    signed_in.selectbox(key="rag_provider_select").select("openai").run()
    signed_in.checkbox(key="rag_consent::openai").check().run()
    signed_in.checkbox(key="rag_consent::openai").uncheck().run()
    manager = signed_in.session_state["rag_manager"]
    assert manager.confirmed == set()  # every confirmation naming OpenAI went
    signed_in.selectbox(key="rag_provider_select").select("anthropic").run()
    assert signed_in.checkbox(key="rag_consent::anthropic+openai").value is False
    # not confirmed again by showing it
    assert signed_in.session_state["rag_manager"].confirmed == set()


def _new_session(username: str = "persist_user") -> AppTest:
    """Another session of the same user (a new browser, the next day): only the files are shared."""
    at = AppTest.from_file(MAIN, default_timeout=240)
    at.run()
    at.session_state["user_authenticated"] = True
    at.session_state["username"] = username
    at.session_state["user_role"] = "researcher"
    at.run()
    _visit(at, "settings")
    return at


def _saved(tmp_path: Path, username: str = "persist_user") -> str:
    path = tmp_path / "users" / username / "preferences.json"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def test_the_choices_of_settings_are_kept_for_the_next_session(
    app: AppTest, tmp_path: Path
) -> None:
    """Settings kept its choices for the session only: the next sign-in started from the defaults (M5b)."""
    app.toggle(key="import_default_dayfirst").set_value(True).run()
    assert '"import_dayfirst": true' in _saved(tmp_path)
    later = _new_session()
    assert later.session_state["import_default_dayfirst"] is True  # what the Import page reads
    assert later.toggle(key="import_default_dayfirst").value is True


def test_a_saved_cloud_provider_starts_nothing_and_no_secret_is_saved(
    app: AppTest, tmp_path: Path
) -> None:
    """The provider and its model are kept; keys and consent stay in the session (audit E6)."""
    pytest.importorskip("langchain_core", reason="the AI engine is in the optional 'ai' extra")
    app.selectbox(key="rag_provider_select").select("gemini").run()
    app.text_input(key="rag_gemini_api_key").input("AIza-CANARY-KEY-0123456789").run()
    app.selectbox(key="rag_gemini_model_select").select("gemini-flash-lite-latest").run()
    app.checkbox(key="rag_consent::gemini").check().run()
    if any(t.key == "rag_qdrant_api_key" for t in app.text_input):
        app.text_input(key="rag_qdrant_api_key").input("QDRANT-CANARY-KEY").run()
    saved = _saved(tmp_path)
    assert '"ai_provider": "gemini"' in saved and "gemini-flash-lite-latest" in saved
    assert "CANARY" not in saved and "consent" not in saved
    later = _new_session()
    assert later.selectbox(key="rag_provider_select").value == "gemini"
    assert later.selectbox(key="rag_gemini_model_select").value == "gemini-flash-lite-latest"
    assert (
        later.checkbox(key="rag_consent::gemini").value is False
    )  # confirmed again in each session
    assert (
        later.session_state["rag_manager"].confirmed == set()
    )  # no recipient confirmed by an earlier session


def test_a_saved_model_no_longer_offered_is_named_and_kept(
    signed_in: AppTest, tmp_path: Path
) -> None:
    pytest.importorskip("langchain_core", reason="the AI engine is in the optional 'ai' extra")
    folder = tmp_path / "users" / "persist_user"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "preferences.json").write_text(
        '{"ai_provider": "gemini", "ai_chat_model": {"gemini": "gemini-0.1-retired"}}',
        encoding="utf-8",
    )
    later = _new_session()
    assert later.selectbox(key="rag_provider_select").value == "gemini"
    model = later.selectbox(key="rag_gemini_model_select")
    assert model.value == model.options[0]
    assert any(
        "`gemini-0.1-retired` is not offered here" in c.value for c in later.caption
    )  # as code
    assert "gemini-0.1-retired" in _saved(tmp_path)  # kept until the user chooses again


def _prefs(tmp_path: Path, text: str, username: str = "persist_user") -> None:
    folder = tmp_path / "users" / username
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "preferences.json").write_text(text, encoding="utf-8")


def test_no_address_and_no_qdrant_key_is_saved(app: AppTest, tmp_path: Path) -> None:
    """An address decides where the data go and can carry credentials: it is entered again in each session."""
    pytest.importorskip("langchain_core", reason="the AI engine is in the optional 'ai' extra")
    app.text_input(key="rag_ollama_base_url").input("http://user:CANARY-PASS@127.0.0.1:9").run()
    app.segmented_control(key="rag_vector_engine").set_value("qdrant").run()
    app.text_input(key="rag_qdrant_url").input("http://CANARY-HOST:6333").run()
    app.text_input(key="rag_qdrant_api_key").input("QDRANT-CANARY-KEY").run()
    saved = _saved(tmp_path)
    assert '"vector_store": "qdrant"' in saved
    assert "CANARY" not in saved and "127.0.0.1" not in saved


def test_a_saved_choice_this_server_cannot_use_is_named(
    signed_in: AppTest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each kept choice the server no longer offers is not used, and Settings says which (the models, the provider,
    Claude's embedding provider, the vector database and the graph engine)."""
    pytest.importorskip("langchain_core", reason="the AI engine is in the optional 'ai' extra")
    import graph.manager as gm

    monkeypatch.setenv("CMP_OFFLINE", "1")  # yFiles needs internet access
    monkeypatch.setattr(gm, "_GLOBAL_MANAGER", None)
    _prefs(
        tmp_path, '{"ai_provider": "cohere", "vector_store": "chroma", "graph_renderer": "yfiles"}'
    )
    later = _new_session()
    captions = " ".join(c.value for c in later.caption)
    assert "`cohere` is not offered here" in captions
    assert "`chroma` is not offered here" in captions
    assert later.selectbox(key="rag_provider_select").value == "ollama"
    from ui.graph_view import RENDERER_OVERRIDE_KEY

    assert later.session_state[RENDERER_OVERRIDE_KEY] is None  # the automatic engine
    assert "`yfiles` is not available here" in captions
    monkeypatch.setattr(gm, "_GLOBAL_MANAGER", None)


def test_the_saved_vector_database_and_claude_embedding_provider_come_back(
    signed_in: AppTest, tmp_path: Path
) -> None:
    pytest.importorskip("langchain_core", reason="the AI engine is in the optional 'ai' extra")
    pytest.importorskip("qdrant_client", reason="the Qdrant engine is in the optional 'ai' extra")
    _prefs(
        tmp_path,
        '{"ai_provider": "anthropic", "ai_claude_embedding_provider": "gemini", "vector_store": "qdrant"}',
    )
    later = _new_session()
    assert later.selectbox(key="rag_provider_select").value == "anthropic"
    assert later.selectbox(key="rag_claude_embed_provider").value == "gemini"
    assert later.segmented_control(key="rag_vector_engine").value == "qdrant"


@pytest.mark.parametrize(
    ("provider", "key_variable"),
    [
        ("mistral", "MISTRAL_API_KEY"),
        ("gemini", "GOOGLE_API_KEY"),  # these two list models with a key: not with the server's
        ("anthropic", "ANTHROPIC_API_KEY"),
    ],
)
def test_the_server_s_provider_is_preselected_and_starts_nothing(
    provider: str, key_variable: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CMP_AI_PROVIDER preselects the provider of a user who chose none; a cloud one is not confirmed by it, and nothing
    is asked of it, with the server's key planted: no model list, no connector."""
    pytest.importorskip("langchain_core", reason="the AI engine is in the optional 'ai' extra")
    from manage.rag.factory import ConnectorFactory
    from manage.rag_manager import RAGManager

    asked: list[str] = []
    monkeypatch.setattr(
        RAGManager, "get_available_models", lambda self, *a, **k: asked.append("models") or []
    )
    monkeypatch.setattr(
        ConnectorFactory, "get_llm_connector", classmethod(lambda cls, cfg: asked.append("chat"))
    )
    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path))
    monkeypatch.setitem(os.environ, "CMP_IAM_PROVIDER", "local")
    monkeypatch.setenv("CMP_AI_PROVIDER", provider)
    monkeypatch.setenv(key_variable, "PLANTED-SERVER-KEY-0123456789")
    at = _new_session("server_choice_user")
    assert at.selectbox(key="rag_provider_select").value == provider
    consent = next(c for c in at.checkbox if (c.key or "").startswith("rag_consent::"))
    assert provider in consent.key and consent.value is False
    assert at.session_state["rag_manager"].confirmed == set() and asked == []
    # a user's own choice wins over the server's
    at.selectbox(key="rag_provider_select").select("ollama").run()
    later = _new_session("server_choice_user")
    assert later.selectbox(key="rag_provider_select").value == "ollama"
