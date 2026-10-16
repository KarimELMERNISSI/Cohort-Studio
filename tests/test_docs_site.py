"""The documentation site (docs/): no unsupported claims, nothing loaded from other sites, local assets present,
images sized, the current version in the citation."""

from __future__ import annotations

import re
import tomllib
from html.parser import HTMLParser
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PAGE = DOCS / "index.html"

# Regulatory and marketing claims the software does not back (follow-up programme M0).
UNSUPPORTED = [
    r"21\s*CFR",
    r"Part\s*11",
    r"\bcomplian(t|ce)\b",
    r"\bimmutable\b",
    r"\bcertif(y|ied|ies|icate)",
    r"\bHIPAA\b",
    r"\bGxP\b",
    r"\bFDA\b",
    r"enterprise-grade",
    r"production-ready",
    r"\bguarantee",
    r"100\s*%\s*(test\s*)?coverage",
]
SIGNED_NEGATION = "not an electronically signed record"


class _Page(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.loads: list[tuple[str, str]] = []  # resources the browser fetches: (tag, url)
        self.links: list[str] = []  # anchors
        self.images: list[dict[str, str | None]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if tag == "img":
            self.images.append(a)
        if tag in ("img", "script", "source", "iframe", "audio", "video") and a.get("src"):
            self.loads.append((tag, a["src"] or ""))
        if tag == "link" and a.get("href") and a.get("rel") != "canonical":
            self.loads.append((tag, a["href"] or ""))
        if tag == "a" and a.get("href"):
            self.links.append(a["href"] or "")


@pytest.fixture(scope="module")
def html() -> str:
    return PAGE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def page(html: str) -> _Page:
    parser = _Page()
    parser.feed(html)
    return parser


def test_the_site_makes_no_unsupported_claims(html: str) -> None:
    text = html.replace(SIGNED_NEGATION, "")
    found = [p for p in UNSUPPORTED if re.search(p, text, flags=re.IGNORECASE)]
    assert not found, found
    assert not re.search(r"\bsigned\b", text, flags=re.IGNORECASE), "'signed' outside the negation"


def test_the_page_loads_nothing_from_other_sites(page: _Page) -> None:
    remote = [url for _, url in page.loads if re.match(r"^(https?:)?//", url)]
    assert not remote, remote
    css = (DOCS / "styles.css").read_text(encoding="utf-8")
    assert "@import" not in css and not re.search(r"url\(\s*['\"]?(https?:)?//", css)
    script = (DOCS / "app.js").read_text(encoding="utf-8")
    assert not re.search(r"https?://|fetch\(|XMLHttpRequest|sendBeacon", script)


def test_every_local_file_exists(page: _Page) -> None:
    local = [url for _, url in page.loads] + [
        url for url in page.links if not re.match(r"^(https?:|mailto:|#)", url)
    ]
    missing = [url for url in local if not (DOCS / url.split("#")[0]).is_file()]
    assert not missing, missing
    anchors = [url[1:] for url in page.links if url.startswith("#") and len(url) > 1]
    html = PAGE.read_text(encoding="utf-8")
    assert all(f'id="{anchor}"' in html for anchor in anchors), anchors


def test_images_have_a_text_alternative_and_their_real_size(page: _Page) -> None:
    for image in page.images:
        assert "alt" in image, image  # empty for decorative images
        if not image.get("src"):
            continue  # the lightbox image, filled in by the script
        with Image.open(DOCS / (image["src"] or "")) as data:
            assert (int(image["width"] or 0), int(image["height"] or 0)) == data.size, image["src"]


def test_the_citation_names_the_current_version(html: str) -> None:
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "version"
    ]
    assert f"Version {version}" in html
    assert f"version = {{{version}}}" in html
    assert "assets/logo.png" not in html  # the old logo; the brand assets come from assets/brand
