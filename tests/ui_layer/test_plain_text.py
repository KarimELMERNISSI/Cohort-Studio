"""Text from outside the application (a language model's answer, an imported file) is shown as it is written."""

import html
import re

import pytest

from ui.components.text import answer, plain

ZW = "\N{ZERO WIDTH SPACE}"


@pytest.mark.parametrize(
    ("text", "shown"),
    [
        ("[click](http://e.x)", r"\[click\]\(http\:\/\/e\.x\)"),
        ("![beacon](http://e.x/p.png)", r"\!\[beacon\]\(http\:\/\/e\.x\/p\.png\)"),
        (":red[alarm] :material/home:", r"\:red\[alarm\] \:" + ZW + r"material\/home\:"),
        ("# heading\n- item", r"\# heading \- item"),
        ("*bold* and _it_ `code` <b>", r"\*bold\* and \_it\_ \`code\` \<b\>"),
        ("a &amp; b", r"a \&amp; b"),
        ("plain words", "plain words"),
        (None, ""),
    ],
)
def test_outside_text_stays_text(text: object, shown: str) -> None:
    assert plain(text) == shown


def test_a_long_text_is_cut() -> None:
    assert plain("x" * 50, limit=10) == "x" * 9 + "…"


@pytest.mark.parametrize(
    ("text", "shown"),
    [
        # Streamlit draws these anew once the escapes are read: the flow showed "18 ≤ age ≤ 85" for the rule's
        # "18 <= age <= 85", and an expression its user did not write. In code, they stay as written
        ("18 <= age <= 85", "18 `<=` age `<=` 85"),
        ("df['a'] >= 2", r"df\['a'\] `>=` 2"),
        ("a -> b <- c <-> d -- e ~= f", "a `->` b `<-` c `<->` d `--` e `~=` f"),
        ("<= first and last >=", "`<=` first and last `>=`"),
        # an icon and the logo, drawn whatever the escapes: a zero-width space after the colon
        (
            ":material_warning: alert :streamlit:",
            r"\:" + ZW + r"material\_warning\: alert \:" + ZW + r"streamlit\:",
        ),
        # not drawn anew without a space on each side: escaped only
        ("x<=5 and -->", r"x\<=5 and \-\-\>"),
    ],
)
def test_what_streamlit_draws_anew_stays_as_written(text: str, shown: str) -> None:
    assert plain(text) == shown


# What Streamlit's own passes see once Markdown has read a text (StreamlitMarkdown, Streamlit 1.66.1): ":material/" is
# made ":material_" in the source, code included; then, outside code, each text node with its escapes and character
# references read. They draw ":material_name:" as an icon and ":streamlit:" as the logo, and "<=", "->", "--"...
# between spaces or at a node's ends as symbols; an emoji shortcode such as ":warning:" is read when the source holds
# one unescaped.
_READ = re.compile(r"\\([!-/:-@\[-`{-~])|(&(?:#\d+|#[xX][0-9a-fA-F]+|[A-Za-z][A-Za-z0-9]*);)")
_DRAWN = re.compile(
    r":material_\w+:|:streamlit:"
    r"|(?:^|[\s\N{ZERO WIDTH NO-BREAK SPACE}])(?:<->|->|<-|--|>=|<=|~=)(?:[\s\N{ZERO WIDTH NO-BREAK SPACE}]|$)"
)
_EMOJI = re.compile(r":(?!material/|streamlit:)[\w+-][\w_+-]*:")


def _read(source: str) -> str:
    return _READ.sub(lambda m: m.group(1) or html.unescape(m.group(2)), source)


def _streamlit_reads(markdown: str) -> tuple[list[str], list[str]]:
    """The text nodes and the code of ``markdown``, as Streamlit's passes see them."""
    parts = re.split(r"(?<!\\)`([^`]*)`", markdown.replace(":material/", ":material_"))
    return [_read(part) for part in parts[::2]], parts[1::2]


@pytest.mark.parametrize(
    "text",
    [
        "18 <= age <= 85, then a -> b <- c <-> d -- e ~= f",
        "<= at the ends >=",
        ":material/warning: Validated",  # the documented form: ":material_warning:" before the escapes are read
        "&colon;material_check_circle&colon; and &#58;streamlit&#x3a;",  # references, read after the escapes
        ":material_check\\_circle: and :material_a::material_b:",
        ":streamlit:-- x and :material_x:<= b",
        ":material_x: then :warning: and :white_check_mark:",
        "x&nbsp;<=&nbsp;y and a &lt;= b",
        "x<=5 and -->",
    ],
)
def test_plain_text_is_shown_as_written_once_streamlit_has_read_it(text: str) -> None:
    """The escapes stop Markdown, but Streamlit draws some text anew after them: the flow showed "≤" for a rule's "<="
    (M5b browser check), and the data-protection review had icons drawn through the escapes."""
    shown = plain(text)
    nodes, code = _streamlit_reads(shown)
    assert not [node for node in nodes if _DRAWN.search(node)], nodes
    assert not _EMOJI.search(shown)
    read = "".join(node + (code[i] if i < len(code) else "") for i, node in enumerate(nodes))
    assert read.replace(ZW, "") == text


def test_an_answer_draws_no_icon_and_keeps_its_symbols() -> None:
    shown = answer("ok :material_warning: and :streamlit: and `:material_check:`; x >= 5 -> y")
    assert (
        shown.replace(ZW, "")
        == r"ok \:material_warning\: and \:streamlit\: and `:material_check:`; x >= 5 -> y"
    )


@pytest.mark.parametrize(
    "text",
    [
        "ok :material/warning: and :material_check\\_circle: and :streamlit:",
        "&colon;material_check&colon; &#58;streamlit&#58; &#x3a;material_x&#x3a;",
        "in code `:material_check:` and ```\n:streamlit:\n```",
    ],
)
def test_an_answer_draws_no_icon(text: str) -> None:
    """In code too: what is taken for code may be text to Markdown."""
    shown = answer(text)
    assert not re.search(
        r":material_\w+:|:streamlit:", _read(shown.replace(":material/", ":material_"))
    )
    assert shown.replace(ZW, "").count("material") == text.count("material")


@pytest.mark.parametrize(
    "fence",
    [
        "```mermaid",
        "~~~ Mermaid",
        "```\tmermaid",
        "> ```math",
        "```MATH {display}",
        "- ```mermaid",
        "1. ```mermaid",
    ],
)
def test_an_answer_draws_no_diagram_or_formula_from_a_fence(fence: str) -> None:
    """Streamlit draws a mermaid fence with Mermaid, whose image shapes the browser fetches (data-protection review:
    the address was fetched; a fence after a list's marker too, in the second round), and a math fence as a formula."""
    first = answer(f'{fence}\nflowchart TD\n A@{{ img: "http://e.x/p.png" }}\n```').splitlines()[0]
    assert not re.search(r"(?:`{3,}|~{3,})[ \t]*(?:mermaid|math)\b", first, re.IGNORECASE)
    assert first.replace(ZW, "") == fence


@pytest.mark.parametrize(
    ("text", "shown"),
    [
        # an image the browser would fetch, a link hiding its address, by inline or reference form
        ("see ![b](http://e.x/p.png?q=1)", r"see !\[b\](http\://e.x/p.png?q=1)"),
        ("[safe](http://e.x) and ![r][1]", r"\[safe\](http\://e.x) and !\[r\]\[1\]"),
        # a directive imitating the application's notices, a formula
        (":green-badge[OK] :material/check:", r"\:green-badge\[OK\] \:" + ZW + r"material/check\:"),
        ("costs $5 or $x^2$", r"costs \$5 or \$x^2\$"),
        # the answer's own formatting is kept, and code is shown as written
        ("**bold**\n- one\n- two", "**bold**\n- one\n- two"),
        ("use `x[0]` or\n```\nd = {'a': [1]}\n```", "use `x[0]` or\n```\nd = {'a': [1]}\n```"),
    ],
)
def test_an_answer_keeps_its_formatting_but_draws_nothing_it_would_fetch(
    text: str, shown: str
) -> None:
    assert answer(text) == shown


def test_an_answer_drops_invisible_characters_and_is_cut() -> None:
    assert answer("a‮b‏c\x07d\n\te") == "abcd\n\te"
    assert answer("x" * 50, limit=10) == "x" * 9 + "…"


@pytest.mark.parametrize(
    "text",
    [
        r"\`![x](http://e.x/p.png)\`",  # escaped backticks: no code to Markdown
        "`![x](http://e.x/p.png)``",  # one backtick opens, two close: no code either
        "see ```\n\n![x](http://e.x/p.png)\n\n```",  # a fence opened within a line
    ],
)
def test_an_image_in_what_looks_like_code_is_not_drawn(text: str) -> None:
    # data-protection review: a CommonMark parser drew these images. An image starts with "![": none is left
    shown = answer(text)
    assert "![" not in shown and "x](http" in shown
