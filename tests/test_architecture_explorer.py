"""
The architecture explorer (dev_scripts/architecture_explorer): built from the code, and true to it.

The Pages workflow builds the page after CI has passed and publishes it with the site. This test runs the same
build at each push, so that what would stop it fails here, on the commit that causes it: a module that belongs to
no brick, a brick that is not on the layer of its modules, the developer guide's layer table or the README's scheme
disagreeing with the layering test, a connector registry or a catalogue of the code that the explorer does not
describe, a connector placed against what its class declares, a library it cannot name, a name its words give that
the code does not hold. The direction of the imports is left to tests/test_layering.py: one violation fails one
test.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

from tests.test_docs_site import SIGNED_NEGATION, UNSUPPORTED

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "dev_scripts" / "architecture_explorer"
# a test's name that the modules' docstrings, which the page shows, may hold: not the claim the site refuses
SIGNED_NAMES = ("signed-rank",)


@pytest.fixture(scope="module")
def explorer():
    spec = importlib.util.spec_from_file_location("build_explorer", FOLDER / "build_explorer.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_explorer"] = module
    spec.loader.exec_module(module)
    module.load(ROOT)
    return module


@pytest.fixture(scope="module")
def data(explorer) -> dict:
    try:
        built = explorer.build(check_arrows=False)
    except explorer.NotBuilt as refused:
        pytest.fail(f"The architecture explorer would not be built: {refused}")
    return built


@pytest.fixture(scope="module")
def page(explorer, data, tmp_path_factory) -> str:
    output = tmp_path_factory.mktemp("explorer") / "index.html"
    explorer.write(data, output)
    return output.read_text(encoding="utf-8")


def test_the_explorer_is_built_from_the_code(explorer, data) -> None:
    """Its checks pass (the fixture builds it), and it holds every module at each level of detail."""
    modules = set(data["modules"])
    assert modules == set(explorer.MODULES) and set(data["moduleOrder"]) == modules
    assert [level["id"] for level in data["levels"]] == [
        "overview",
        "packages",
        "bricks",
        "modules",
    ]
    for level, view in data["views"].items():
        held = {name for node in view["nodes"].values() for name in node["members"]}
        assert held == modules, level


def test_the_page_loads_nothing_from_elsewhere_and_makes_no_unsupported_claim(page) -> None:
    """One file: no script, style or picture fetched from another site, which its own policy forbids too, and the
    site's rule on claims, the docstrings the page shows included."""
    assert explorer_token() not in page
    policy = re.search(r'<meta http-equiv="Content-Security-Policy" content="([^"]+)"', page)
    assert (
        policy and "default-src 'none'" in policy.group(1) and "base-uri 'none'" in policy.group(1)
    )
    assert re.findall(r"<link\b[^>]*>", page) == ['<link rel="icon" href="data:," />']
    tags = r"<script[^>]+src=|<img\b|<iframe\b|<object\b|<embed\b|<base\b|<form\b|srcset=|http-equiv=\"refresh\""
    assert not re.search(tags, page, flags=re.IGNORECASE)
    assert not re.search(r"@import|url\(\s*['\"]?(https?:)?//", page)
    calls = (
        r"fetch\(|XMLHttpRequest|sendBeacon|WebSocket|EventSource|\bimport\(|localStorage|sessionStorage|"
        r"indexedDB|serviceWorker|document\.cookie"
    )
    assert not re.search(calls, page)
    text = page.replace(SIGNED_NEGATION, "")
    for name in SIGNED_NAMES:
        text = text.replace(name, "")
    found = [pattern for pattern in UNSUPPORTED if re.search(pattern, text, flags=re.IGNORECASE)]
    assert not found, found
    assert not re.search(r"\bsigned\b", text, flags=re.IGNORECASE), "'signed' outside the negation"


def test_the_page_names_no_address_machine_person_or_secret(explorer, page) -> None:
    """The page is public and shows the first paragraph of every module's docstring: a host, a path, an e-mail
    address or a key written in one would be published with it."""
    urls = set(re.findall(r"https?://[^\s\"'<>)\\]+", page))
    assert all(url.startswith(explorer.levels.REPOSITORY) for url in urls), (
        urls
    )  # the repository's own links
    assert not re.search(r"[\"'(=]\s*//[\w-]+\.", page)  # no address without its scheme
    assert not re.search(
        r"[\w.+-]+@[\w-]+\.[a-z]{2,}", page, flags=re.IGNORECASE
    )  # no e-mail address
    assert not re.search(
        r"\b[A-Za-z]:\\{1,2}[A-Za-z]|/home/|/Users/|AppData", page
    )  # no local path
    assert not re.search(
        r"\blocalhost\b|(?<![\d.])\d{1,3}(\.\d{1,3}){3}(?![\d.])", page
    )  # no machine
    assert not re.search(
        r"AIza[\w-]{30,}|\bsk-[\w-]{20,}|\bgh[pousr]_\w{30,}|-----BEGIN", page
    )  # no key


def test_the_site_publishes_the_page_it_builds(explorer) -> None:
    """The workflow builds the page where the site's upload takes it, and git never holds it."""
    workflow = (ROOT / ".github" / "workflows" / "deploy-pages.yml").read_text(encoding="utf-8")
    build, deploy = workflow.split("\n  build:")[1].split("\n  deploy:")
    assert "python dev_scripts/architecture_explorer/build_explorer.py" in build
    assert "path: './docs'" in build
    # after CI has passed on a push to this repository's main, from the commit CI checked: never a pull request's
    # code, never a fork's, never a failed run's
    assert "workflow_run:" in workflow
    for guard in (
        "github.event.workflow_run.conclusion == 'success'",
        "github.event.workflow_run.event == 'push'",
        "github.event.workflow_run.head_repository.full_name == github.repository",
        "github.event_name == 'workflow_dispatch' && github.ref == 'refs/heads/main'",
        "ref: ${{ github.event.workflow_run.head_sha || github.sha }}",
    ):
        assert guard in build, guard
    # only the tip of main is published, and the job that may publish runs no code of the repository
    assert 'if [ "$(git rev-parse FETCH_HEAD)" = "$(git rev-parse HEAD)" ]' in build
    assert "pages: write" in deploy and "pages: write" not in build
    assert "uv " not in deploy and "checkout" not in deploy
    assert explorer.OUTPUT.as_posix() == "docs/architecture/index.html"
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "docs/architecture/" in ignored


def explorer_token() -> str:
    return "__EXPLORER_DATA__"
