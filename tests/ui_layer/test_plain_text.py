"""Text from outside the application (a language model's answer, an imported file) is shown as it is written."""

import pytest

from ui.components.text import answer, plain


@pytest.mark.parametrize(
    ("text", "shown"),
    [
        ("[click](http://e.x)", r"\[click\]\(http\://e\.x\)"),
        ("![beacon](http://e.x/p.png)", r"\!\[beacon\]\(http\://e\.x/p\.png\)"),
        (":red[alarm] :material/home:", r"\:red\[alarm\] \:material/home\:"),
        ("# heading\n- item", r"\# heading \- item"),
        ("*bold* and _it_ `code` <b>", r"\*bold\* and \_it\_ \`code\` \<b\>"),
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
        # an image the browser would fetch, a link hiding its address, by inline or reference form
        ("see ![b](http://e.x/p.png?q=1)", r"see !\[b\](http\://e.x/p.png?q=1)"),
        ("[safe](http://e.x) and ![r][1]", r"\[safe\](http\://e.x) and !\[r\]\[1\]"),
        # a directive imitating the application's notices, a formula
        (":green-badge[OK] :material/check:", r"\:green-badge\[OK\] \:material/check\:"),
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
