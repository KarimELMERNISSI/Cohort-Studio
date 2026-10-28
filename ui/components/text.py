"""Text from outside the application (a language model's answer, an imported file) shown on a page as it is written."""

from __future__ import annotations

import re
import unicodedata

# CommonMark lets a backslash escape any ASCII punctuation; these start a link, an image, a format, a heading, a list,
# a table, HTML, a formula, a character reference ("&colon;"), or one of Streamlit's colour and icon directives
# (":red[...]", ":material/...:")
_MARKDOWN = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~$:/&])")
# in an answer: a link or an image, a colour, badge or icon directive, a formula
_IN_ANSWER = re.compile(r"([\[\]:$])")
# shown as written: Markdown draws nothing in code
_CODE = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)
# Streamlit draws some text anew once Markdown has read it, where an escape no longer counts: ":material_name:" as a
# Material icon (":material/name:" is made so in the source, code included) and ":streamlit:" as its logo, which a
# zero-width space after the colon stops; and, between spaces, "<->", "->", "<-", "--", ">=", "<=" and "~=" as an
# arrow, a dash, ≥, ≤ or ≈, which code keeps as written
_ICON = re.compile(r":(?=material|streamlit)")
_SYMBOL = re.compile(
    r"(?<![^\s\N{ZERO WIDTH NO-BREAK SPACE}])(<->|->|<-|--|>=|<=|~=)(?![^\s\N{ZERO WIDTH NO-BREAK SPACE}])"
)
# in an answer: a character reference ("&#58;", "&colon;"), read after the checks above; a fence whose language
# Streamlit draws, as a Mermaid diagram (the browser fetches its image shapes) or a formula, wherever it opens (after
# a quote's or a list's marker too)
_REFERENCE = re.compile(r"&(?=[#A-Za-z])")
_DRAWN_FENCE = re.compile(r"((?:`{3,}|~{3,})[ \t]*)(?=(?:mermaid|math)\b)", re.IGNORECASE)
_BREAK = "\N{ZERO WIDTH SPACE}"


def plain(text: object, *, limit: int | None = None) -> str:
    """
    ``text`` for Markdown (st.markdown, a caption, a notice, a button label), shown as written: its punctuation is
    escaped, so a link, an image, a colour, an icon, a character reference or a heading in it stays text; a zero-width
    space follows the colon of an icon's or the logo's name, which Streamlit reads after the escapes; and "<=", "->",
    "--"... between spaces, which it would draw as symbols ("age <= 85" read "age ≤ 85"), are put in code. White space
    is collapsed, and a text longer than ``limit`` characters is cut.
    """
    value = " ".join(str(text if text is not None else "").split())
    if limit is not None and len(value) > limit:
        value = value[: limit - 1].rstrip() + "…"
    # text, then a symbol in code: the text's own backticks are escaped, so each code span ends where it should
    parts = _SYMBOL.split(_ICON.sub(":" + _BREAK, value))
    return "".join(
        f"`{part}`" if i % 2 else _MARKDOWN.sub(r"\\\1", part) for i, part in enumerate(parts)
    )


def answer(text: object, *, limit: int = 20_000) -> str:
    """
    A language model's answer (a chat reply, a summary) for st.markdown, with its paragraphs, lists, emphasis and code:
    its ``[`` and ``]`` are escaped, so it draws no image (the browser would fetch the image's address, which can carry
    the conversation out) and no link whose text hides where it goes; ``:`` too, so that no colour, badge or icon
    imitates the application's own notices, and ``$``, so that it draws no formula. A zero-width space follows the
    colon of an icon's or the logo's name and the ``&`` of a character reference, which Streamlit reads after the
    escapes, and comes before the language of a ``mermaid`` or ``math`` fence, which would draw a diagram (whose image
    shapes the browser fetches) or a formula. Code is shown as written, but for these zero-width spaces. Control and
    bidirectional characters are dropped (new lines and tabs kept), and a text longer than ``limit`` characters is cut.
    Never for a label, st.latex or HTML: :func:`plain` is for those.
    """
    value = "".join(
        c
        for c in str(text if text is not None else "")
        if c in "\n\t" or unicodedata.category(c) not in ("Cc", "Cf")
    )
    if len(value) > limit:
        value = value[: limit - 1].rstrip() + "…"
    # in code too: what is taken for code may be text to Markdown (below)
    value = _REFERENCE.sub("&" + _BREAK, _ICON.sub(":" + _BREAK, value))
    value = _DRAWN_FENCE.sub(r"\1" + _BREAK, value)
    # text, code, text...: what is taken for code may not be code to Markdown (an escaped or unmatched backtick, a
    # fence opened within a line), so an image is escaped there too
    parts = _CODE.split(value)
    return "".join(
        part.replace("![", "!\\[") if i % 2 else _IN_ANSWER.sub(r"\\\1", part)
        for i, part in enumerate(parts)
    )


__all__ = ["answer", "plain"]
