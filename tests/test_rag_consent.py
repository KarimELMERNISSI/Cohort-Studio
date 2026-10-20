"""
The data-sharing confirmation of the AI engine (M5, with the data-protection review).

- Withdrawing it stops the engine: it was checked only when the engine started, so an engine kept
  answering, and sending data to its cloud provider, after the box was unticked.
- Every recipient outside this machine and the private network needs it, as the exact set of one
  box: the embedding provider too (Claude answering, OpenAI embedding), and an Ollama elsewhere.
- A refused, failed or interrupted start leaves no connection behind and never relabels the engine.
- Model discovery never sends a cloud key to the Ollama address of an earlier engine.
- No prompt carries a statistic that the notice says is never sent (the extremes of a column).
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("langchain_core", reason="the AI stack is in the optional 'ai' extra")

import manage.rag_manager as rm


def _fake_connectors(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Chat and embedding connectors that run offline."""
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    import utils.data_paths as dp
    from tests.test_rag_vector_integration import FakeLoader
    from tests.test_vectorstores import WordHashEmbeddings

    def create(cls, **kwargs):
        return FakeListChatModel(responses=["{}"] * 20), WordHashEmbeddings()

    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.delenv("CMP_VECTOR_STORE", raising=False)
    monkeypatch.setattr(rm, "PDFDocumentLoader", FakeLoader)
    monkeypatch.setattr(rm.ConnectorFactory, "create_connectors", classmethod(create))
    monkeypatch.setattr(
        rm.RAGManager,
        "_analyze_global_context",
        lambda self, texts, dataset_columns=None: {"roles": {}},
    )


def _start(manager: rm.RAGManager, provider: str, embedding: str | None = None) -> tuple[bool, str]:
    return manager.initialize_system(
        provider=provider,
        model_name="model",
        embedding_model="embedding",
        vector_engine="lancedb",
        embedding_provider=embedding,
        embedding_api_key="sk-embed" if embedding else None,
    )


def _engine(monkeypatch, tmp_path, provider: str, embedding: str | None = None) -> rm.RAGManager:
    _fake_connectors(monkeypatch, tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir(exist_ok=True)
    manager = rm.RAGManager(documents_dir=str(docs), provider=provider, api_key="sk-test")
    manager.confirm(*rm.RAGManager.recipients_of(provider, None, embedding or provider))
    ok, message = _start(manager, provider, embedding)
    assert ok, message
    return manager


def test_withdrawing_the_confirmation_stops_a_cloud_engine(monkeypatch, tmp_path) -> None:
    manager = _engine(monkeypatch, tmp_path, "gemini")
    llm = manager.llm
    assert manager.initialized and manager.recipients() == {"gemini"} and manager.cloud_consent

    assert manager.withdraw_consent() is True
    assert not manager.initialized and not manager.holds_connections()
    assert manager.confirmed == set() and manager.llm is None and manager.vector_store is None
    assert "confirmation was withdrawn" in (manager.stop_reason or "")
    answers_before = llm.i
    assert manager.suggest_column_renaming(["age", "sex"]) == (None, "RAG system not initialized.")
    assert manager.suggest_proxy_variable("bmi", ["weight"]) == {"proxy_found": False}
    assert manager.suggest_alternative_formula("egfr", "creat", ["age"]) == {"alternative_found": False}
    assert llm.i == answers_before  # nothing more was sent to the provider

    manager.confirm("gemini")  # confirmed again, and started again
    ok, message = _start(manager, "gemini")
    assert ok, message
    assert manager.initialized and manager.stop_reason is None


def test_a_local_engine_keeps_running(monkeypatch, tmp_path) -> None:
    manager = _engine(monkeypatch, tmp_path, "ollama")
    assert not manager.uses_cloud() and manager.recipients() == set()
    assert manager.withdraw_consent("gemini") is False
    assert manager.initialized and manager.llm is not None and manager.stop_reason is None


def test_the_embedding_provider_needs_the_confirmation_too(monkeypatch, tmp_path) -> None:
    # Claude answers, OpenAI embeds the documents and every search text: one box covered only Claude
    _fake_connectors(monkeypatch, tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    manager = rm.RAGManager(documents_dir=str(docs), provider="anthropic", api_key="sk-test")
    manager.confirm("anthropic")
    ok, message = _start(manager, "anthropic", "openai")
    assert not ok and "explicit confirmation" in message and "Anthropic Claude and OpenAI" in message
    manager.confirm("anthropic", "openai")
    ok, message = _start(manager, "anthropic", "openai")
    assert ok, message
    assert manager.recipients() == {"anthropic", "openai"}

    assert manager.withdraw_consent("openai") is True  # the embedding provider's withdrawal stops it
    assert not manager.holds_connections()
    assert manager.confirmed == {frozenset({"anthropic"})}  # every confirmation naming OpenAI went


def test_a_confirmation_covers_its_recipients_together_not_each_alone(monkeypatch, tmp_path) -> None:
    # OpenAI confirmed as Claude's embedding provider made the OpenAI chat box show ticked, and that
    # engine start: answering, it also receives the column summaries, descriptions and criteria
    _fake_connectors(monkeypatch, tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    manager = rm.RAGManager(documents_dir=str(docs), provider="openai", api_key="sk-test")
    manager.confirm("anthropic", "openai")
    ok, message = _start(manager, "openai")
    assert not ok and "may leave this server (OpenAI)" in message
    assert not manager.is_confirmed(frozenset({"openai"}))


def test_an_ollama_outside_the_private_network_needs_the_confirmation(monkeypatch, tmp_path) -> None:
    _fake_connectors(monkeypatch, tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    manager = rm.RAGManager(documents_dir=str(docs), provider="ollama")
    outside = "https://ollama.example.com"
    ok, message = manager.initialize_system(
        provider="ollama", base_url=outside, model_name="m", vector_engine="lancedb"
    )
    assert not ok and "Ollama at ollama.example.com" in message
    manager.confirm("ollama@ollama.example.com")
    ok, message = manager.initialize_system(
        provider="ollama", base_url=outside, model_name="m", vector_engine="lancedb"
    )
    assert ok, message
    assert manager.recipients() == {"ollama@ollama.example.com"} and manager.uses_cloud()
    for private in ("http://192.168.1.20:11434", "http://gpu-box.lan:11434", "http://ollama:11434"):
        assert rm.RAGManager.recipients_of("ollama", private) == frozenset(), private


def test_a_refused_start_leaves_the_running_engine_as_it_was(monkeypatch, tmp_path) -> None:
    # the provider fields were written before the checks: a refused switch relabelled the running
    # engine, and a later withdrawal of its real provider missed it
    manager = _engine(monkeypatch, tmp_path, "gemini")
    ok, message = _start(manager, "openai")
    assert not ok and "OpenAI" in message
    assert manager.initialized and manager.provider == "gemini"
    assert manager.recipients() == {"gemini"}
    assert manager.withdraw_consent("gemini") is True


def test_a_failed_or_interrupted_start_keeps_no_connection(monkeypatch, tmp_path) -> None:
    manager = _engine(monkeypatch, tmp_path, "gemini")

    def no_answer(self, texts, dataset_columns=None):
        raise rm.ProviderError("quota exceeded")

    monkeypatch.setattr(rm.RAGManager, "_analyze_global_context", no_answer)
    ok, message = _start(manager, "gemini")
    assert not ok and "did not answer" in message
    assert not manager.initialized and not manager.holds_connections()

    class Rerun(BaseException):  # Streamlit stops a run with a BaseException: `except Exception` misses it
        pass

    def interrupted(self, texts, dataset_columns=None):
        raise Rerun

    monkeypatch.setattr(rm.RAGManager, "_analyze_global_context", interrupted)
    with pytest.raises(Rerun):
        _start(manager, "gemini")
    assert not manager.holds_connections()


def test_an_unknown_provider_needs_the_confirmation(tmp_path) -> None:
    manager = rm.RAGManager(documents_dir=str(tmp_path), provider="custom")
    ok, message = manager.initialize_system(
        provider="custom", model_name="m", embedding_provider="custom"
    )
    assert not ok and "explicit confirmation" in message  # only the local providers are exempt


def test_model_discovery_sends_no_cloud_key_to_the_ollama_address(monkeypatch, tmp_path) -> None:
    seen: list = []

    def connector(cls, cfg):
        seen.append(cfg)
        return SimpleNamespace(get_available_models=lambda: [])

    monkeypatch.setattr(rm.ConnectorFactory, "get_llm_connector", classmethod(connector))
    manager = rm.RAGManager(documents_dir=str(tmp_path), provider="ollama", api_key="sk-typed")
    manager.base_url = "http://gpu-box.lan:11434"  # left by an earlier Ollama engine
    manager.get_available_models("anthropic")
    manager.get_available_models("ollama")
    assert seen[0].base_url is None  # the Anthropic key goes to Anthropic, not to the Ollama host
    assert seen[1].base_url == "http://gpu-box.lan:11434"


def test_cloud_embeddings_count_as_cloud(tmp_path) -> None:
    manager = rm.RAGManager(documents_dir=str(tmp_path), provider="ollama")
    manager.embedding_provider = "openai"  # the embedding provider embeds every question
    manager.embedding_base_url = None
    assert manager.uses_cloud() and manager.recipients() == {"openai"}


def test_no_prompt_carries_the_extremes_of_a_column(tmp_path) -> None:
    # refining the variable map sent the minimum, maximum and mean of every numeric column, and later
    # prompts sent the whole map with them, though the notice says the extremes are never sent
    prompts: list[str] = []

    def invoke(prompt: str):
        prompts.append(prompt)
        return SimpleNamespace(content="{}")

    manager = rm.RAGManager(documents_dir=str(tmp_path), provider="gemini")
    manager.initialized = True
    manager.adherence_guidance = ""
    manager.llm = SimpleNamespace(invoke=invoke)
    stats = {"min": 31.5, "max": 912.0, "mean": 88.4}
    taxonomy = {"creatinine": {"description": "serum creatinine", "stats": stats}}
    columns_info = [{"name": "creatinine", "stats": {"min": 31.5, "max": 912.0, "p50": 88.0}}]
    manager.refine_variable_taxonomy(taxonomy, {"creatinine": "it is in umol/L"}, columns_info)
    imported = {"age": {"stats": {"min": 18, "max": 101}}}  # a map imported from a file
    manager.variable_taxonomy = taxonomy | imported
    manager._map_formulas_to_data([{"name": "eGFR"}], ["creatinine", "age"], False, 3)
    manager._get_theoretical_formulas("eGFR", "Go To Target", "context", 3, ["creatinine"])
    assert len(prompts) == 3
    for prompt in prompts:
        for extreme in ("912", "31.5", "88.4", "101"):
            assert extreme not in prompt, (extreme, prompt[:300])
    assert '"p50": 88.0' in prompts[0]  # the percentiles of the notice still reach the model


def test_the_uploaded_pdf_is_deleted_when_the_extraction_fails(tmp_path) -> None:
    pytest.importorskip("google.genai", reason="the Gemini SDK is in the optional 'ai' extra")
    deleted: list[str] = []
    # (the provider decides the file API: test_claude_never_uploads_the_pdf)
    uploaded = SimpleNamespace(
        name="files/1", state=SimpleNamespace(name="ACTIVE"), uri="u", mime_type="application/pdf"
    )

    def generate_content(**kwargs):
        raise RuntimeError("model overloaded")

    files = SimpleNamespace(
        upload=lambda file: uploaded,
        get=lambda name: uploaded,
        delete=lambda name: deleted.append(name),
    )
    client = SimpleNamespace(files=files, models=SimpleNamespace(generate_content=generate_content))
    (tmp_path / "paper.pdf").write_bytes(b"%PDF-1.4")
    manager = rm.RAGManager(documents_dir=str(tmp_path), provider="gemini")
    manager.initialized = True
    manager.llm = SimpleNamespace(client=client, model_name="gemini-flash-lite-latest")
    graph, error = manager.extract_custom_graph_from_doc("paper.pdf")
    assert graph is None and error
    assert deleted == ["files/1"]  # the PDF itself was sent: Google is asked to delete it anyway


def test_claude_never_uploads_the_pdf(tmp_path) -> None:
    # Anthropic's client has a files API too: the graph extraction uploaded each PDF to Anthropic and left it
    # there, though only the passages of a question are meant to reach Claude
    uploads: list[str] = []
    prompts: list[str] = []
    client = SimpleNamespace(files=SimpleNamespace(upload=lambda file: uploads.append(file)))

    def invoke(prompt: str):
        prompts.append(prompt)
        return SimpleNamespace(content="[]")

    manager = rm.RAGManager(documents_dir=str(tmp_path), provider="anthropic")
    manager.initialized = True
    manager.llm = SimpleNamespace(client=client, invoke=invoke)
    (tmp_path / "paper.pdf").write_bytes(b"%PDF-1.4")
    manager.extract_custom_graph_from_doc("paper.pdf")
    assert uploads == []  # the text path (or nothing, when no text could be read)
