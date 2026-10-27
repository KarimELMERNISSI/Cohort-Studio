"""
The control that chooses a connector (ui.components.connector_selector, M5b step 4), on a fake family (AppTest): the
choice shown and why, a saved choice or the server's setting set aside and said, what cannot run here and how to
install it, a change kept for the next session, the session's own choice, and the identity providers refused.
"""

import inspect
import json
import re
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import utils.data_paths as dp

FAKE = """
from core.registry import Availability


class Good:
    name = "good"
    display_name = "Good"

    @classmethod
    def availability(cls):
        return Availability(True)


class Other:
    name = "other"
    display_name = "Other"

    @classmethod
    def availability(cls):
        return Availability(True)


class Down:
    name = "down"
    display_name = "Down"

    @classmethod
    def availability(cls):
        return Availability(False, "library too old", "graph-yfiles")
"""
USER = "selector_user"


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """A fake connector module on the path, one whose package is not installed, and a workspace for the user."""
    module = f"fake_selector_{tmp_path.name.replace('-', '_')}"
    folder = tmp_path / "modules"
    folder.mkdir()
    (folder / f"{module}.py").write_text(FAKE, encoding="utf-8")
    (folder / f"{module}_extra.py").write_text("import not_installed_pkg_zz\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(folder))
    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path / "data"))
    yield module
    for name in (module, f"{module}_extra", f"{module}_pages"):
        sys.modules.pop(name, None)


def _script(
    module: str,
    only: tuple[str, ...] = ("good", "other", "down", "ext"),
    order: tuple[str, ...] = ("good", "other"),
    **options,
) -> None:
    from core.registry import ConnectorRegistry
    from ui import preferences
    from ui.components.connector_selector import connector_selector

    registry = ConnectorRegistry("test engine", setting="CMP_TEST_ENGINE", auto=order)
    targets = {"good": "Good", "other": "Other", "down": "Down"}
    for name in only:
        if name == "ext":
            registry.register("ext", f"{module}_extra:X", extra="ai")
        else:
            registry.register(name, f"{module}:{targets[name]}")
    connector_selector(
        registry,
        key="engine",
        label="Engine",
        preference=preferences.current().graph_renderer,
        preference_name="graph_renderer",
        **options,
    )


def _app(module: str, *, engine: str | None = None, **options) -> AppTest:
    """A session of the signed-in user (``engine``: the control's value before the first run)."""
    at = AppTest.from_function(_script, kwargs={"module": module, **options}, default_timeout=60)
    at.session_state["username"] = USER
    at.session_state["user_role"] = "researcher"
    if engine is not None:
        at.session_state["engine"] = engine
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


def _saved_file(tmp_path: Path) -> Path:
    return tmp_path / "data" / "users" / USER / "preferences.json"


def _save(tmp_path: Path, text: str) -> None:
    _saved_file(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    _saved_file(tmp_path).write_text(text, encoding="utf-8")


def _saved(tmp_path: Path) -> str | None:
    return json.loads(_saved_file(tmp_path).read_text(encoding="utf-8")).get("graph_renderer")


def _shown(markdown: str) -> str:
    """Markdown as it reads on the page: names and reasons are escaped (ui.components.text.plain)."""
    return re.sub(r"\\(.)", r"\1", markdown)


def _why(at: AppTest) -> list[str]:
    return [_shown(m.value) for m in at.expander[0].markdown]


def test_the_choice_is_shown_with_why_and_what_cannot_run_here(fake: str, tmp_path: Path) -> None:
    at = _app(fake)
    box = at.selectbox(key="engine")
    assert box.value == "auto"
    assert box.options == ["Automatic: Good", "Good", "Other"]  # only what can run here
    assert [e.label for e in at.expander] == ["Why Good", "Not available here (2)"]
    assert _why(at) == ["- the first of Good and Other that can run here"]
    missing = [m.value for m in at.expander[1].markdown]
    assert missing[0] == (
        "**Down**: library too old. Install it on the server with `uv sync --extra graph-yfiles`."
    )
    assert missing[1].startswith("**ext**: not\\_installed\\_pkg\\_zz is not installed")
    assert missing[1].endswith("Install it on the server with `uv sync --extra ai`.")
    assert not at.caption  # nothing set aside
    assert not _saved_file(tmp_path).exists()  # seeding never fires on_change


def test_a_change_is_kept_for_the_next_session_and_automatic_forgets_it(
    fake: str, tmp_path: Path
) -> None:
    at = _app(fake)
    at.selectbox(key="engine").select("other").run()
    assert _saved(tmp_path) == "other"
    assert at.expander[0].label == "Why Other" and _why(at) == ["- your saved choice"]
    at.run()
    assert at.selectbox(key="engine").value == "other"  # survives a rerun
    later = _app(fake)  # the user's next session
    assert later.selectbox(key="engine").value == "other"
    later.selectbox(key="engine").select("auto").run()
    assert _saved(tmp_path) is None
    assert later.expander[0].label == "Why Good"


@pytest.mark.parametrize(
    ("saved", "notice"),
    [
        (
            "down",
            "Your saved test engine `down` is not available here (library too old): Good is used.",
        ),
        ("cohere", "Your saved test engine `cohere` is not offered here: Good is used."),
    ],
)
def test_a_saved_choice_this_server_cannot_use_is_said_until_the_user_chooses_again(
    fake: str, tmp_path: Path, saved: str, notice: str
) -> None:
    _save(tmp_path, json.dumps({"graph_renderer": saved}))
    at = _app(fake)
    assert at.selectbox(key="engine").value == "auto"
    assert [c.value for c in at.caption] == [notice]
    at.run()
    assert [c.value for c in at.caption] == [notice]  # the file keeps it
    assert _saved(tmp_path) == saved
    at.selectbox(key="engine").select("other").run()
    assert not at.caption and _saved(tmp_path) == "other"


def test_the_server_s_setting_is_the_default_unless_it_cannot_run_here(fake: str) -> None:
    at = _app(fake, setting="other")
    assert at.selectbox(key="engine").options[0] == "Server default: Other"
    assert _why(at) == ["- the server's setting (CMP_TEST_ENGINE)"]
    down = _app(fake, setting="down")
    assert down.selectbox(key="engine").options[0] == "Automatic: Good"
    assert [c.value for c in down.caption] == [
        "The server's test engine `down` is not available here (library too old): Good is used."
    ]


def test_the_session_s_own_choice_seeds_the_control(fake: str, tmp_path: Path) -> None:
    """The engine that runs, or the graph engine chosen on another page, wins over the saved choice."""
    _save(tmp_path, json.dumps({"graph_renderer": "good"}))
    at = _app(fake, seed="other")
    assert at.selectbox(key="engine").value == "other"
    assert _why(at) == ["- chosen for this session"]
    assert _saved(tmp_path) == "good"  # seeding saves nothing


def test_a_choice_that_can_no_longer_run_is_replaced(fake: str, tmp_path: Path) -> None:
    at = _app(fake, engine="down")
    assert at.selectbox(key="engine").value == "auto"
    assert not _saved_file(tmp_path).exists()


def test_a_session_choice_that_cannot_run_here_falls_through(fake: str, tmp_path: Path) -> None:
    at = _app(fake, seed="down")
    assert at.selectbox(key="engine").value == "auto"
    assert not at.caption and not _saved_file(tmp_path).exists()


def _browser_shows(at: AppTest) -> str:
    """The option the browser shows: the value the server sets on the widget, else its default. (``value`` reads the
    session, which keeps another page's value: the browser's widget of this page starts from its own state.)"""
    proto = at.selectbox(key="engine").proto
    return proto.raw_value if proto.set_value else proto.options[proto.default]


def test_the_choice_follows_the_user_from_page_to_page(fake: str, tmp_path: Path) -> None:
    """The graph engine's control is on four pages under one key, and a widget's identity includes its page: each
    page's control shows the session's choice (it showed "automatic", and the next rerun forgot the saved choice)."""
    pages = f"{fake}_pages"
    folder = tmp_path / "modules"
    (folder / f"{pages}.py").write_text(inspect.getsource(_script), encoding="utf-8")
    for page in ("a", "b"):
        (folder / f"page_{page}.py").write_text(
            f"from {pages} import _script\n\n_script({fake!r})\n", encoding="utf-8"
        )
    (folder / "main.py").write_text(
        "import streamlit as st\n\n"
        'st.navigation([st.Page("page_a.py", default=True), st.Page("page_b.py")]).run()\n',
        encoding="utf-8",
    )
    _save(tmp_path, json.dumps({"graph_renderer": "other"}))
    at = AppTest.from_file(str(folder / "main.py"), default_timeout=60)
    at.session_state["username"] = USER
    at.session_state["user_role"] = "researcher"
    at.run()
    assert _browser_shows(at) == "Other"
    at.switch_page("page_b.py").run()
    assert _browser_shows(at) == "Other"
    at.run()
    assert _browser_shows(at) == "Other" and _saved(tmp_path) == "other"
    at.selectbox(key="engine").select("good").run()
    at.switch_page("page_a.py").run()
    assert _browser_shows(at) == "Good" and _saved(tmp_path) == "good"
    assert not at.exception, [str(e.value) for e in at.exception]


@pytest.mark.parametrize(
    ("offer_auto", "options", "value"), [(True, ["Automatic"], "auto"), (False, [], None)]
)
def test_nothing_can_run_here(
    fake: str, tmp_path: Path, offer_auto: bool, options: list[str], value: str | None
) -> None:
    at = _app(fake, only=("down", "ext"), order=("down",), offer_auto=offer_auto)
    box = at.selectbox(key="engine")
    assert box.options == options and box.value == value
    assert [e.label for e in at.expander] == ["Not available here (2)"]  # no "Why": nothing is used
    assert not _saved_file(tmp_path).exists()


def test_without_automatic_nothing_is_preselected_when_the_default_cannot_run(
    fake: str, tmp_path: Path
) -> None:
    """Never a connector nobody chose (the AI provider's default is local Ollama, never a cloud provider): with the
    default unable to run, the user chooses among the others."""
    at = _app(fake, order=("down",), offer_auto=False)
    box = at.selectbox(key="engine")
    assert box.options == ["Good", "Other"] and box.value is None
    assert not _saved_file(tmp_path).exists()


def test_without_automatic_a_recommendation_is_preselected_with_its_reasons(
    fake: str, tmp_path: Path
) -> None:
    at = _app(
        fake,
        offer_auto=False,
        widget="segmented",
        auto="other",
        auto_reasons=("Suits this machine.", "See [the guide](https://example.org)."),
    )
    control = at.segmented_control(key="engine")
    assert control.value == "other" and control.options == ["Good", "Other"]
    assert _why(at) == ["- Suits this machine.", "- See [the guide](https://example.org)."]
    assert "](" not in at.expander[0].markdown[1].value  # a reason is text, never a link
    control.set_value("good").run()
    assert _saved(tmp_path) == "good"  # no "automatic": the name itself is kept


def test_explain_off_shows_the_control_only(fake: str) -> None:
    at = _app(fake, explain=False, setting="down")
    assert at.selectbox(key="engine").value == "auto"
    assert not at.expander and not at.caption


def test_the_identity_providers_have_no_control() -> None:
    """The deployment chooses them (CMP_IAM_PROVIDER): a session or a saved choice never does."""
    from manage.iam.providers import IAM_PROVIDERS
    from ui.components.connector_selector import connector_selector

    with pytest.raises(ValueError, match="chosen by CMP_IAM_PROVIDER only"):
        connector_selector(
            IAM_PROVIDERS, key="iam", label="Identity", preference=None, preference_name="x"
        )
