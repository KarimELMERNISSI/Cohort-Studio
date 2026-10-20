"""Only the open tab of a page runs (M5b, Streamlit 1.66 tab state), and the widgets of a closed tab keep their values."""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import utils.data_paths as dp
from ui import nav

ROOT = Path(__file__).resolve().parents[2]
WIDGETS = {"selectbox", "multiselect", "slider", "number_input", "text_input", "text_area", "checkbox", "radio",
           "toggle", "select_slider", "date_input", "segmented_control", "pills", "color_picker", "time_input",
           "file_uploader", "data_editor"}  # fmt: skip
NO_PERSIST = {
    "file_uploader",
    "data_editor",
}  # Streamlit cannot keep them while their tab is closed
# the modules of the pages that run only their open tab
LAZY = ("app_pages/data_monitoring.py", "app_pages/data_quality_dashboard.py")
# widgets deliberately not kept when their tab is closed: a confirmation is asked again
CONFIRMATIONS = ("mask_overwrite_",)


def _ok(at: AppTest) -> None:
    assert not at.exception, [str(e.value)[:300] for e in at.exception]


@pytest.fixture(scope="module")
def app(tmp_path_factory: pytest.TempPathFactory):
    previous = dp.DATA_ROOT
    dp.DATA_ROOT = str(tmp_path_factory.mktemp("data"))
    os.environ.setdefault("CMP_IAM_PROVIDER", "local")
    at = AppTest.from_file(str(ROOT / "main.py"), default_timeout=300)
    at.run()
    at.session_state["user_authenticated"] = True
    at.session_state["username"] = "lazy_tabs_user"
    at.session_state["user_role"] = "researcher"
    at.run()
    at.switch_page(nav.page_path("start")).run()
    at.button(key="start_tutorial").click().run()
    _ok(at)
    yield at
    dp.DATA_ROOT = previous


def test_only_the_open_tab_of_validate_runs(app: AppTest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every click on Validate ran all six tabs: the dashboard's audit, the checks, the outlier detection."""
    from app_pages import data_monitoring as dm

    calls: list[str] = []
    for name in ("render_dashboard", "render_population_panel", "add_outlier_handling_ui",
                 "add_domain_expert_based_anomaly_mask", "add_study_inclusion_mask"):  # fmt: skip
        original = getattr(dm, name)

        def counted(*args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(dm, name, counted)
    app.switch_page(nav.page_path("validate")).run()
    _ok(app)
    assert calls == ["render_dashboard"]  # the first tab is open
    for label, work in (
        ("Analysis population", "render_population_panel"),
        ("Handle Outliers", "add_outlier_handling_ui"),
        ("Handle Clinical Anomalies", "add_domain_expert_based_anomaly_mask"),
        ("Handle Inclusion Criteria", "add_study_inclusion_mask"),
    ):
        calls.clear()
        app.session_state[dm.TAB_KEY] = label
        app.run()
        _ok(app)
        assert calls == [work], label


def test_a_widget_keeps_its_value_while_its_tab_is_closed(app: AppTest) -> None:
    from app_pages import data_monitoring as dm

    app.switch_page(nav.page_path("validate")).run()
    app.session_state[dm.TAB_KEY] = "Handle Outliers"
    app.run()
    app.selectbox(key="outlier_detection_method").set_value("quantile").run()
    _ok(app)
    app.session_state[dm.TAB_KEY] = "Analysis population"
    app.run()
    app.session_state[dm.TAB_KEY] = "Handle Outliers"
    app.run()
    _ok(app)
    assert app.selectbox(key="outlier_detection_method").value == "quantile"


def _calls(path: str):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    return tree, [
        n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    ]


def test_the_widgets_of_lazy_pages_have_a_key_and_keep_their_value() -> None:
    """A widget of a closed tab without persist_state loses its value; without a key it cannot keep one."""
    missing = []
    for path in LAZY:
        _, calls = _calls(path)
        for call in calls:
            if call.func.attr not in WIDGETS:
                continue
            kws = {k.arg: k.value for k in call.keywords}
            key = kws.get("key")
            if key is None:
                missing.append(f"{path}:{call.lineno} {call.func.attr} has no key")
                continue
            text = ast.unparse(key)
            confirmation = any(prefix in text for prefix in CONFIRMATIONS)
            if call.func.attr not in NO_PERSIST and not confirmation and "persist_state" not in kws:
                missing.append(
                    f"{path}:{call.lineno} {call.func.attr} ({text}) has no persist_state"
                )
    assert not missing, missing


def test_the_labels_of_lazy_tabs_are_fixed() -> None:
    """The labels are part of the tabs' identity: a label with a count would reset the open tab when it changes."""
    for path in LAZY:
        tree, calls = _calls(path)
        constants = {
            target.id: node.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        for call in calls:
            lazy = any(k.arg == "on_change" for k in call.keywords)
            if call.func.attr != "tabs" or not lazy:
                continue
            labels = call.args[0]
            if (
                isinstance(labels, ast.Call)
                and isinstance(labels.func, ast.Name)
                and labels.func.id == "list"
            ):
                labels = labels.args[0]
            if isinstance(labels, ast.Name):
                labels = constants[labels.id]
            assert isinstance(labels, ast.List | ast.Tuple), f"{path}:{call.lineno}"
            assert all(
                isinstance(e, ast.Constant) and isinstance(e.value, str) for e in labels.elts
            )
