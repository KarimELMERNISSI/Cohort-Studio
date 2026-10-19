"""
Each AI provider's SDK is installed by the extra it belongs to (M5).

The OpenAI and Mistral connectors import the `openai` SDK, which no extra installed: choosing them in Settings gave
"'openai' package is not installed". The Ollama connector imported httpx, which arrived only through other packages.
utils/optional.py EXTRAS, which tells whether an extra is installed, checks a module of each package of the extra.
"""

from __future__ import annotations

import ast
import base64
import importlib.util
import json
import re
import struct
import sys
import tomllib
from functools import cache
from importlib.metadata import PackageNotFoundError, distribution, packages_distributions
from pathlib import Path

import pytest

import utils.optional as opt

ROOT = Path(__file__).resolve().parents[1]
PROVIDERS = ROOT / "manage" / "rag" / "providers"


@cache
def _project() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _packages(extra: str | None) -> set[str]:
    """The packages of the core dependencies (None) or of an extra, by their canonical names."""
    project = _project()["project"]
    requirements = project["dependencies"] if extra is None else project["optional-dependencies"][extra]
    return {_canonical(re.match(r"[A-Za-z0-9._-]+", r).group(0)) for r in requirements}


def _installed(package: str) -> bool:
    try:
        distribution(package)
    except PackageNotFoundError:
        return False
    return True


@cache
def _top_level() -> dict[str, list[str]]:
    return packages_distributions()


def _package_of(module: str) -> str:
    """The installed package that provides ``module``, inside a namespace package (google.genai) too."""
    path = module.replace(".", "/")
    for name in _top_level().get(module.split(".")[0], []):
        files = distribution(name).files or []
        if any(f.as_posix() == f"{path}.py" or f.as_posix().startswith(f"{path}/") for f in files):
            return _canonical(name)
    raise AssertionError(f"no installed package provides {module}")


def _is_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _third_party_imports(file: Path) -> set[str]:
    first_party = set(_project()["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"])
    found: set[str] = set()
    for node in ast.walk(ast.parse(file.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            # `from google import genai` imports the module google.genai
            for alias in node.names:
                sub = f"{node.module}.{alias.name}"
                found.add(sub if _is_module(sub) else node.module)
    return {
        m for m in found if m.split(".")[0] not in sys.stdlib_module_names | first_party | {"__future__"}
    }


def test_the_extras_checked_are_extras_of_the_project() -> None:
    assert set(opt.EXTRAS) <= set(_project()["project"]["optional-dependencies"])


@pytest.mark.parametrize("extra", sorted(opt.EXTRAS))
def test_each_extra_checks_a_module_of_each_of_its_packages(extra: str) -> None:
    packages = _packages(extra)
    if not all(_installed(p) for p in packages):
        pytest.skip(f"the '{extra}' packages are not all installed (uv sync --extra {extra})")
    assert {_package_of(m) for m in opt.EXTRAS[extra]} == packages


@pytest.mark.parametrize("file", sorted(PROVIDERS.glob("*.py")), ids=lambda p: p.name)
def test_each_ai_provider_imports_only_what_the_ai_extra_installs(file: Path) -> None:
    if not all(_installed(p) for p in _packages("ai")):
        pytest.skip("the 'ai' packages are not all installed (uv sync --extra ai)")
    declared = _packages(None) | _packages("ai")
    undeclared = {m: _package_of(m) for m in _third_party_imports(file) if _package_of(m) not in declared}
    assert not undeclared, f"{file.name} imports packages that the 'ai' extra does not install: {undeclared}"


def _answer(request):
    """The OpenAI-compatible API, as OpenAI and Mistral serve it."""
    import httpx

    path = request.url.path
    if path.endswith("/chat/completions"):
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 0,
                "model": "chat-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Salut"},
                    }
                ],
            },
        )
    if path.endswith("/embeddings"):
        body = json.loads(request.content)
        vector: object = [0.25, 0.5]
        if body.get("encoding_format") == "base64":  # what the SDK asks for unless told otherwise
            vector = base64.b64encode(struct.pack("<2f", 0.25, 0.5)).decode()
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": vector}],
                "model": "embedding-model",
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )
    models = ["gpt-5-mini", "mistral-small-latest", "mistral-embed"]
    return httpx.Response(
        200,
        json={
            "object": "list",
            "data": [{"id": m, "object": "model", "created": 0, "owned_by": "x"} for m in models],
        },
    )


@pytest.mark.parametrize("provider", ["openai", "mistral"])
def test_the_connectors_work_with_the_installed_openai_sdk(provider: str) -> None:
    """Offline: the SDK the extra installs builds the requests these connectors send, and parses the answers."""
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")
    pytest.importorskip("langchain_core")
    from manage.rag import ProviderConfig, ProviderType
    from manage.rag.providers import mistral
    from manage.rag.providers.openai import OpenAIChatConnector, OpenAIEmbeddingConnector

    chat_class, embedding_class = (
        (OpenAIChatConnector, OpenAIEmbeddingConnector)
        if provider == "openai"
        else (mistral.MistralChatConnector, mistral.MistralEmbeddingConnector)
    )
    config = ProviderConfig(
        provider=ProviderType(provider),
        api_key="sk-test",
        model_name="chat-model",
        embedding_model="embedding-model",
    )
    chat, embeddings = chat_class(config=config), embedding_class(config)
    if provider == "mistral":
        assert str(chat.client.base_url).rstrip("/") == mistral.MISTRAL_API_BASE
    sent = []

    def answer(request):
        sent.append(request)
        return _answer(request)

    offline = openai.OpenAI(
        api_key="sk-test",
        base_url=str(chat.client.base_url),
        http_client=httpx.Client(transport=httpx.MockTransport(answer)),
    )
    chat.client = embeddings.client = offline
    assert chat.invoke("Bonjour").content == "Salut"
    request = json.loads(sent[-1].content)
    assert request["model"] == "chat-model"
    assert request["messages"] == [{"role": "user", "content": "Bonjour"}]
    assert embeddings.embed_documents(["Bonjour"]) == [[0.25, 0.5]]
    assert json.loads(sent[-1].content)["model"] == "embedding-model"
    listed = chat.get_available_models()
    assert "mistral-embed" not in listed and ("gpt-5-mini" in listed)
