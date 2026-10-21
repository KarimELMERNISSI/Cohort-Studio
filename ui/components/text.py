"""Text from outside the application (a language model's answer, an imported file) shown on a page as it is written."""

from __future__ import annotations

import re
import unicodedata

# CommonMark lets a backslash escape any ASCII punctuation; these start a link, an image, a format, a heading, a list,
# a table, HTML, a formula, or one of Streamlit's colour and icon directives (":red[...]", ":material/...:")
_MARKDOWN = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~$:])")
# in an answer: a link or an image, a colour, badge or icon directive, a formula
_IN_ANSWER = re.compile(r"([\[\]:$])")
# shown as written: Markdown draws nothing in code
_CODE = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)


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


def answer(text: object, *, limit: int = 20_000) -> str:
    """
    A language model's answer (a chat reply, a summary) for st.markdown, with its paragraphs, lists, emphasis and code:
    its ``[`` and ``]`` are escaped, so it draws no image (the browser would fetch the image's address, which can carry
    the conversation out) and no link whose text hides where it goes; ``:`` too, so that no colour, badge or icon
    imitates the application's own notices, and ``$``, so that it draws no formula. Code is shown as written. Control
    and bidirectional characters are dropped (new lines and tabs kept), and a text longer than ``limit`` characters is
    cut. Never for a label, st.latex or HTML: :func:`plain` is for those.
    """
    value = "".join(
        c
        for c in str(text if text is not None else "")
        if c in "\n\t" or unicodedata.category(c) not in ("Cc", "Cf")
    )
    if len(value) > limit:
        value = value[: limit - 1].rstrip() + "…"
    # text, code, text...: what is taken for code may not be code to Markdown (an escaped or unmatched backtick, a
    # fence opened within a line), so an image is escaped there too
    parts = _CODE.split(value)
    return "".join(
        part.replace("![", "!\\[") if i % 2 else _IN_ANSWER.sub(r"\\\1", part)
        for i, part in enumerate(parts)
    )


__all__ = ["answer", "plain"]
