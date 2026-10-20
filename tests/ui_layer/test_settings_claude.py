"""Settings > AI engine offers Anthropic Claude with a separate embedding provider (AppTest)."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import utils.data_paths as dp
from ui import nav

pytest.importorskip("langchain_core", reason="the AI engine is in the optional 'ai' extra")

ROOT = Path(__file__).resolve().parents[2]
MAIN = str(ROOT / "main.py")


@pytest.fixture(scope="module")
def app(tmp_path_factory: pytest.TempPathFactory):
    previous = dp.DATA_ROOT
    dp.DATA_ROOT = str(tmp_path_factory.mktemp("data"))
    os.environ.setdefault("CMP_IAM_PROVIDER", "local")
    at = AppTest.from_file(MAIN, default_timeout=240)
    at.run()
    at.session_state["user_authenticated"] = True
    at.session_state["username"] = "claude_user"
    at.session_state["user_role"] = "researcher"
    at.run()
    at.switch_page(nav.page_path("settings")).run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    yield at
    dp.DATA_ROOT = previous


def test_claude_provider_with_local_embeddings(app: AppTest) -> None:
    assert "Anthropic Claude (cloud)" in app.selectbox(key="rag_provider_select").options
    app.selectbox(key="rag_provider_select").select("anthropic").run()
    assert not app.exception, [str(e.value)[:300] for e in app.exception]
    assert app.selectbox(key="rag_anthropic_model_select").value == "claude-fable-5-1"
    embedding = app.selectbox(key="rag_claude_embed_provider")
    assert embedding.value == "ollama"
    assert app.selectbox(key="rag_anthropic_ollama_embed_select").value == "nomic-embed-text:latest"
    assert any(
        "What leaves this server" in (w.title or "") for w in app.warning
    )  # consent required
    assert any(c.key == "rag_anthropic_consent" for c in app.checkbox)

    embedding.select("openai").run()
    assert not app.exception
    assert app.selectbox(key="rag_anthropic_openai_embed_select").value == "text-embedding-3-small"
    assert any(t.key == "rag_claude_openai_embed_key" for t in app.text_input)
