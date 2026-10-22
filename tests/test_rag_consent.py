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


class _Files:
    """A fake of Google's file API: the uploaded file goes through ``states`` (one per check); what was uploaded, how,
    and the names Google is asked to delete are kept."""

    def __init__(self, states: list[str], *, fail_upload: bool = False) -> None:
        self.states, self.fail_upload = list(states), fail_upload
        self.uploads: list[tuple[object, object]] = []  # (the file argument, the config)
        self.deleted: list[str] = []

    def _stored(self, name: str) -> SimpleNamespace:
        state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
        return SimpleNamespace(
            name=name, state=SimpleNamespace(name=state), uri="u", mime_type="application/pdf"
        )

    def upload(self, file, config=None):
        self.uploads.append((file, config))
        if self.fail_upload:  # the answer lost after Google stored the file
            raise ConnectionError("connection reset")
        return self._stored(f"files/{config.name}")

    def get(self, name):
        return self._stored(name)

    def delete(self, name):
        self.deleted.append(name)


def _gemini(generate_content, files: _Files):
    """A Gemini chat connector whose client is a fake."""
    pytest.importorskip("google.genai", reason="the Gemini SDK is in the optional 'ai' extra")
    from manage.rag.providers import gemini
    from manage.rag.schemas import ProviderConfig

    connector = gemini.GeminiChatConnector(
        config=ProviderConfig(provider="gemini", model_name="models/gemini-flash-lite-latest")
    )
    connector.client = SimpleNamespace(
        files=files, models=SimpleNamespace(generate_content=generate_content)
    )
    return connector


def _pdf(tmp_path, name: str = "patient_smith_report.pdf") -> str:
    path = tmp_path / name
    path.write_bytes(b"%PDF-1.4")
    return str(path)


def test_gemini_reads_the_pdf_file_and_google_is_asked_to_delete_it(tmp_path) -> None:
    asked: list[dict] = []

    def generate_content(**kwargs):
        asked.append(kwargs)
        return SimpleNamespace(text='{"nodes": []}')

    files = _Files(["PROCESSING", "ACTIVE"])
    progress: list[str] = []
    answer = _gemini(generate_content, files).read_document(
        _pdf(tmp_path),
        "the graph",
        lambda pct, message: progress.append(message),
        as_json=True,
        temperature=0.2,
    )
    assert answer == '{"nodes": []}'
    # sent as a stream under a name chosen here: Google is sent neither the file's name nor its path
    ((stream, config),) = files.uploads
    assert not isinstance(stream, str) and hasattr(stream, "read")
    assert len(config.name) == 32 and config.mime_type == "application/pdf"
    assert "patient" not in config.name and files.deleted == [f"files/{config.name}"]
    (call,) = asked
    assert call["model"] == "gemini-flash-lite-latest"  # without "models/"
    assert call["config"].response_mime_type == "application/json"
    assert call["config"].temperature == 0.2
    assert [p.text for p in call["contents"][0].parts if p.text] == ["the graph"]
    assert progress == [
        "Uploading the file to Google Gemini...",
        "Google is processing the file...",
        "Waiting for the model's answer...",
    ]  # no file name: the caller says which document


def test_google_is_asked_to_delete_the_pdf_whatever_happens(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from manage.rag.providers import gemini

    def overloaded(**kwargs):
        raise RuntimeError("model overloaded")

    def chosen(files: _Files) -> str:
        return f"files/{files.uploads[0][1].name}"

    files = _Files(["ACTIVE"])
    with pytest.raises(RuntimeError, match="overloaded"):
        _gemini(overloaded, files).read_document(_pdf(tmp_path), "p")
    assert files.deleted == [chosen(files)]  # the PDF itself was sent: Google is asked to delete it anyway
    files = _Files(["FAILED"])
    with pytest.raises(RuntimeError, match="File processing failed by Google"):
        _gemini(overloaded, files).read_document(_pdf(tmp_path), "p")
    assert files.deleted == [chosen(files)]
    # the upload failing after Google stored the file: its name is known, so it is deleted too
    files = _Files(["ACTIVE"], fail_upload=True)
    with pytest.raises(ConnectionError):
        _gemini(overloaded, files).read_document(_pdf(tmp_path), "p")
    assert files.deleted == [chosen(files)]
    # a file Google is still processing after five minutes: given up (the loop used to wait for ever)
    monkeypatch.setattr(gemini, "_PROCESSING_SECONDS", 0.0)
    files = _Files(["PROCESSING"])
    with pytest.raises(RuntimeError, match="still processing the file after five minutes"):
        _gemini(overloaded, files).read_document(_pdf(tmp_path), "p")
    assert files.deleted == [chosen(files)]


def test_a_file_ready_at_the_last_check_is_read(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from manage.rag.providers import gemini

    clock = iter([0.0, 1.0, 299.0, 301.0])  # the deadline, then one check before it
    monkeypatch.setattr(gemini.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(gemini, "_POLL_SECONDS", 0.0)
    files = _Files(["PROCESSING", "PROCESSING", "ACTIVE"])
    answer = _gemini(lambda **kwargs: SimpleNamespace(text="ok"), files).read_document(
        _pdf(tmp_path), "p"
    )
    assert answer == "ok"


def test_the_graph_extraction_sends_the_file_only_to_a_provider_that_declares_it(tmp_path) -> None:
    from manage.rag.providers import file_readers

    assert file_readers() == {"gemini"}
    calls: list[tuple[str, dict]] = []

    class Reader:  # a connector that reads files: what the extraction asks it
        def read_document(self, path, prompt, progress=None, **kwargs):
            calls.append((path, kwargs))
            return '{"nodes": [], "edges": []}'

    (tmp_path / "paper.pdf").write_bytes(b"%PDF-1.4")
    manager = rm.RAGManager(documents_dir=str(tmp_path), provider="gemini")
    manager.initialized = True
    manager.llm = Reader()
    graph, error = manager.extract_custom_graph_from_doc("paper.pdf")
    assert error is None and graph == {"nodes": [], "edges": []}
    assert calls == [(str(tmp_path / "paper.pdf"), {"as_json": True, "temperature": 0.2})]


def test_the_consent_notice_says_what_document_insight_sends_each_provider() -> None:
    from app_pages.rag_sidebar import _notice

    assert (
        "the PDF file itself, not its name, while its graph is extracted; the app then asks Google Gemini to "
        "delete it" in _notice(["gemini"], {"provider": "gemini"})
    )
    for provider in ("openai", "mistral", "anthropic", "ollama"):
        notice = _notice([provider], {"provider": provider})
        assert "the PDF file itself" not in notice
        assert (
            "the text of the first 30 pages of each PDF whose graph you extract, 35,000 characters at most, "
            "whether the PDF is indexed or not" in notice
        )


def test_the_documents_left_out_of_a_graph_are_named_with_the_reason(tmp_path) -> None:
    manager = rm.RAGManager(documents_dir=str(tmp_path), provider="gemini")
    graphs = {"a.pdf": ({"nodes": [{"id": "egfr", "label": "eGFR"}], "edges": []}, None)}
    failure = "Google was still processing the file after five minutes."
    manager.extract_custom_graph_from_doc = lambda doc, progress_callback=None: graphs.get(
        doc, (None, failure)
    )
    skipped: list[tuple[str, str]] = []
    graph, error = manager.extract_merged_graph_from_docs(["a.pdf", "b.pdf"], skipped=skipped)
    assert error is None and [n["id"] for n in graph["nodes"]] == ["egfr"]
    assert skipped == [("b.pdf", failure)]
    # none gives a graph: an error naming each, not an empty graph
    graph, error = manager.extract_merged_graph_from_docs(["b.pdf", "c.pdf"])
    assert graph is None and error == f"No graph could be extracted: b.pdf: {failure}; c.pdf: {failure}"


def test_a_provider_s_error_reaches_the_page_without_a_key(tmp_path) -> None:
    def leaky(*args, **kwargs):
        key = "AIza" + "SyA1234567890abcdefghijklmnopqrstuv"  # a Google key's shape: 35 characters after AIza
        raise RuntimeError(f"403 for https://generativelanguage.googleapis.com/?key={key}")

    (tmp_path / "paper.pdf").write_bytes(b"%PDF-1.4")
    manager = rm.RAGManager(documents_dir=str(tmp_path), provider="gemini")
    manager.initialized = True
    manager.llm = SimpleNamespace(read_document=leaky)
    graph, error = manager.extract_custom_graph_from_doc("paper.pdf")
    assert graph is None and "403" in error and "AIzaSy" not in error


def test_claude_never_uploads_the_pdf(tmp_path) -> None:
    # Anthropic's client has a files API too: the graph extraction uploaded each PDF to Anthropic and left it
    # there, though only the passages of a question are meant to reach Claude
    uploads: list[str] = []
    read: list[str] = []
    prompts: list[str] = []
    client = SimpleNamespace(files=SimpleNamespace(upload=lambda file: uploads.append(file)))

    def invoke(prompt: str):
        prompts.append(prompt)
        return SimpleNamespace(content="[]")

    manager = rm.RAGManager(documents_dir=str(tmp_path), provider="anthropic")
    manager.initialized = True
    manager.llm = SimpleNamespace(
        client=client, invoke=invoke, read_document=lambda path, *args, **kwargs: read.append(path)
    )
    (tmp_path / "paper.pdf").write_bytes(b"%PDF-1.4")
    manager.extract_custom_graph_from_doc("paper.pdf")
    assert uploads == [] and read == []  # the text path (or nothing, when no text could be read)


def test_an_engine_without_a_provider_uses_the_server_s_and_sends_nothing(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RAGManager() was Google Gemini's, a cloud provider nobody chose: it is the server's (CMP_AI_PROVIDER), local
    Ollama by default, and a cloud one named there still needs the session's confirmation."""
    from utils import settings as cfg

    monkeypatch.delenv("CMP_AI_PROVIDER", raising=False)
    cfg.reset_settings()
    assert rm.RAGManager(documents_dir=str(tmp_path)).provider == "ollama"
    monkeypatch.setenv("CMP_AI_PROVIDER", "gemini")
    cfg.reset_settings()
    manager = rm.RAGManager(documents_dir=str(tmp_path))
    assert manager.provider == "gemini" and manager.confirmed == set()
    ok, message = manager.initialize_system()  # its declared models, once confirmed
    assert not ok and "explicit confirmation" in message
    cfg.reset_settings()
