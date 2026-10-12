"""Handbook page: the researcher documentation rendered in the app (works offline)."""

import re
from pathlib import Path

import streamlit as st

from ui.components import page_header

DOCS_DIR = Path(__file__).resolve().parents[2] / "documentation"
HIDDEN = {"13_Audit_Report_2026-10", "12_IAM_Architecture_and_BetterAuth_Evaluation"}


def _chapters() -> list[tuple[str, str]]:
    """(stem, title) for every top-level chapter, in file order."""
    items = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        if path.stem in HIDDEN:
            continue
        title = path.stem
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.startswith("# "):
                    title = line[2:].strip()
                    break
        except OSError:
            continue
        items.append((path.stem, title))
    return items


def _anchor(text: str) -> str:
    slug = re.sub(r"[^a-z0-9\s-]", "", text.lower()).strip()
    return re.sub(r"[\s]+", "-", slug)


page_header("Handbook", "How to prepare, validate, analyse and report a cohort with this platform.")

chapters = _chapters()
if not chapters:
    st.info("No documentation files were found.")
    st.stop()

stems = [s for s, _ in chapters]
wanted = st.session_state.get("handbook_chapter")
default_index = stems.index(wanted) if wanted in stems else 0
chosen = st.selectbox("Chapter", stems, index=default_index, format_func=lambda s: dict(chapters)[s], key="handbook_select")
st.session_state["handbook_chapter"] = chosen

text = (DOCS_DIR / f"{chosen}.md").read_text(encoding="utf-8")
headings = [(len(m.group(1)), m.group(2).strip()) for m in re.finditer(r"^(#{2,3})\s+(.+)$", text, flags=re.MULTILINE)]

toc, body = st.columns([1, 4], gap="large")
with toc:
    st.caption("On this page")
    for level, heading in headings[:40]:
        indent = "&nbsp;&nbsp;&nbsp;" if level == 3 else ""
        st.markdown(f"{indent}[{heading}](#{_anchor(heading)})", unsafe_allow_html=False)
with body:
    st.markdown(text)
