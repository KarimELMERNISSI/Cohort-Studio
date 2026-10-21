"""Text from outside the application (a language model's answer, an imported file) shown on a page as it is written."""

from __future__ import annotations

import re

# CommonMark lets a backslash escape any ASCII punctuation; these start a link, an image, a format, a heading, a list,
# a table, HTML, a formula, or one of Streamlit's colour and icon directives (":red[...]", ":material/...:")
_MARKDOWN = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~$:])")


def plain(text: object, *, limit: int | None = None) -> str:
    """
    ``text`` for Markdown (st.markdown, a caption, a notice, a button label), shown as written: its punctuation is
    escaped, so a link, an image, a colour, an icon or a heading in it stays text. White space is collapsed, and a text
    longer than ``limit`` characters is cut.
    """
    value = " ".join(str(text if text is not None else "").split())
    if limit is not None and len(value) > limit:
        value = value[: limit - 1].rstrip() + "…"
    return _MARKDOWN.sub(r"\\\1", value)


__all__ = ["plain"]
