"""The documentation site (docs/): no unsupported claims, nothing loaded from other sites, local assets present,
images sized, the current version in the citation."""

from __future__ import annotations

import importlib.util
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


# The README and the handbook chapters: the regulatory and marketing claims removed in M0 must not come back.
# The audit report quotes them on purpose, and documentation/legacy keeps old texts.
DOCUMENT_CLAIMS = [
    r"21\s*CFR",
    r"Part\s*11",
    r"\bcompliant\b",
    r"\bimmutable\b",
    r"\bHIPAA\b",
    r"enterprise-grade",
    r"production-ready",
    r"zero-copy",
]


def _documents() -> list[Path]:
    chapters = sorted((ROOT / "documentation").glob("*.md"))
    return [ROOT / "README.md", *[p for p in chapters if not p.name.startswith("13_Audit_Report")]]


@pytest.mark.parametrize("path", _documents(), ids=lambda p: p.name)
def test_the_documentation_makes_no_regulatory_or_marketing_claims(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    found = [p for p in DOCUMENT_CLAIMS if re.search(p, text, flags=re.IGNORECASE)]
    assert not found, found


# The README images: every one exists, the animations loop and stay light (the large-file hook allows 2 MB), and
# the screenshot library holds every screenshot the image builders use, so that they rebuild without the app.
RAW = "https://raw.githubusercontent.com/KarimELMERNISSI/Cohort-Studio/main/"


def _readme_images() -> list[str]:
    found = re.findall(r'(?:src|srcset)="([^"]+)"', (ROOT / "README.md").read_text(encoding="utf-8"))
    return [url.removeprefix(RAW) for url in found if url.startswith(RAW) or "://" not in url]


def test_the_readme_images_exist() -> None:
    images = _readme_images()
    assert {"assets/readme/tour.webp", "assets/readme/describe.webp", "assets/readme/workflow.webp"} <= set(images)
    missing = [path for path in images if not (ROOT / path).is_file()]
    assert not missing, missing


@pytest.mark.parametrize("name", ["tour", "describe"])
def test_the_readme_animations_loop_and_stay_light(name: str) -> None:
    path = ROOT / "assets" / "readme" / f"{name}.webp"
    with Image.open(path) as image:
        assert image.format == "WEBP" and image.n_frames > 10 and image.info.get("loop") == 0
    assert path.stat().st_size < 2_000_000


def _builder(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "dev_scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_screenshot_library_holds_what_the_image_builders_use() -> None:
    readme, site = _builder("make_readme_assets"), _builder("make_site_assets")
    needed = {shot for shot, _, _ in readme.TOUR + readme.DESCRIBE} | {"workflow", "survival"}
    needed |= {f"plot_{shot}" for shot in readme.PLOTS.values()} | {shot for shot, _, _ in site.SHOTS.values()}
    library = {path.stem for path in (ROOT / "assets" / "screenshots").glob("*.webp")}
    assert needed <= library, sorted(needed - library)
