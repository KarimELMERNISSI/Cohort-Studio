"""Text from outside the application (a language model's answer, an imported file) is shown as it is written."""

import pytest

from ui.components.text import plain


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
