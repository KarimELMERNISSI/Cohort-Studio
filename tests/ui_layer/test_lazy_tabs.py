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
LAZY = (
    "app_pages/data_monitoring.py",
    "app_pages/data_quality_dashboard.py",
    "app_pages/validate_dates.py",
    "app_pages/home.py",
    "app_pages/epidemiology.py",
    # the panels of its tabs
    "app_pages/analysis_config.py",
    "app_pages/analysis_diagnostic.py",
    "app_pages/analysis_extras.py",
    "app_pages/analysis_pooled.py",
    "app_pages/analysis_guidance_ui.py",
    # the panels of the Analysis population tab of Validate
    "ui/components/population.py",
    "ui/components/flow.py",
)
# widgets deliberately not kept when their tab is closed: a confirmation is asked again
CONFIRMATIONS = ("mask_overwrite_", "ai_replace_", "_FREEZE_CONFIRM")
# nor an input that mirrors a session key holding its value (value= comes from it): a kept value would come back stale
MIRRORS = (
    "rename_",
    "es_input_",
    "es_chi2_input",
    "power_chi2_min_dim",
    "family_box",  # the sidebar's population boxes, set from the ticked criteria on every run
    "criterion_box",
)


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
                 "add_domain_expert_based_anomaly_mask", "add_study_inclusion_mask",
                 "render_date_checks"):  # fmt: skip
        original = getattr(dm, name)

        def counted(*args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(dm, name, counted)
    app.switch_page(nav.page_path("validate")).run()
    _ok(app)
    assert calls == ["render_dashboard"]  # the first tab is open
    for label, work in (
        ("Date checks", "render_date_checks"),
        ("Analysis population", "render_population_panel"),
        ("Handle Outliers", "add_outlier_handling_ui"),
        ("Handle Implausible Values", "add_domain_expert_based_anomaly_mask"),
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
            exempt = any(prefix in text for prefix in (*CONFIRMATIONS, *MIRRORS))
            if call.func.attr not in NO_PERSIST and not exempt and "persist_state" not in kws:
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


def test_the_detection_settings_survive_a_page_switch(app: AppTest) -> None:
    """The quantile inputs started at 0.01 and 0.99 whatever the stored settings: after a page switch the scorecard
    kept the user's 0.05 while the input showed 0.01, and the next click audited with 0.01 (review, M5b)."""
    from app_pages import data_monitoring as dm

    app.switch_page(nav.page_path("validate")).run()
    app.session_state[dm.TAB_KEY] = "Data Quality Dashboard"
    app.run()
    app.selectbox(key="validity_detection_method").set_value("quantile").run()
    app.number_input(key="validity_q_lower").set_value(0.05).run()
    _ok(app)
    app.switch_page(nav.page_path("explore")).run()  # the controls' values are discarded
    app.switch_page(nav.page_path("validate")).run()
    _ok(app)
    assert app.selectbox(key="validity_detection_method").value == "quantile"
    assert app.number_input(key="validity_q_lower").value == 0.05
    next(s for s in app.slider if s.label.endswith("Weight")).set_value(2.0).run()  # another click
    assert app.session_state["validity_outlier_params"] == {"lower": 0.05, "upper": 0.99}


def test_only_the_open_tab_of_statistics_and_correlations_runs(
    app: AppTest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each click ran the statistics, the renaming table and the benchmark (the benchmark times every analysis)."""
    from app_pages import home

    calls: list[str] = []
    display = home.RenameColumnsComponent.display
    benchmark = home.run_benchmark

    def counted_display(self, *args, **kwargs):
        calls.append("rename")
        return display(self, *args, **kwargs)

    def counted_benchmark(*args, **kwargs):
        calls.append("benchmark")
        return benchmark(*args, **kwargs)

    monkeypatch.setattr(home.RenameColumnsComponent, "display", counted_display)
    monkeypatch.setattr(home, "run_benchmark", counted_benchmark)
    app.switch_page(nav.page_path("explore")).run()
    _ok(app)
    assert calls == []  # the statistics tab is open
    for label, work in (("Columns Renaming", "rename"), ("Performance Benchmark", "benchmark")):
        calls.clear()
        app.session_state[home.TAB_KEY] = label
        app.run()
        _ok(app)
        assert calls == [work], label


def test_only_the_open_tab_of_hypothesis_tests_runs(
    app: AppTest, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app_pages import epidemiology as epi

    calls: list[str] = []
    work = {
        "Analysis": "run_univariate_analysis",
        "Diagnostic accuracy & agreement": "run_diagnostic_accuracy",
        "Paired, rates & stratified": "run_paired_and_rates",
        "Power & Sample Size": "run_power_analysis",
        "Z-Score & Reference": "run_zscore_analysis",
        "Test Guide & Education": "show_educational_content",
    }
    for name in work.values():
        original = getattr(epi, name)

        def counted(*args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(epi, name, counted)
    app.switch_page(nav.page_path("analyze")).run()
    _ok(app)
    for label, name in work.items():
        calls.clear()
        app.session_state[epi.TAB_KEY] = label
        app.run()
        _ok(app)
        assert calls == [name], label


def _power_tab(app: AppTest) -> None:
    from app_pages import epidemiology as epi

    app.switch_page(nav.page_path("analyze")).run()
    app.session_state[epi.TAB_KEY] = "Power & Sample Size"
    app.run()
    _ok(app)


def _load(app: AppTest, option: str) -> None:
    app.selectbox(key="power_import_result").set_value(option).run()
    next(b for b in app.button if b.label == "Load Parameters into Calculator").click().run()
    _ok(app)


def test_a_comparison_run_in_the_analysis_tab_loads_into_the_power_tab(app: AppTest) -> None:
    """The Power tab reads the Analysis tab's results from the session (ui/keys.py): they no longer run together."""
    from app_pages import epidemiology as epi

    app.switch_page(nav.page_path("analyze")).run()
    app.session_state[epi.TAB_KEY] = "Analysis"
    app.run()
    app.selectbox(key="group_col").select("arm").run()
    app.button(key="univariate_run").click().run()
    _ok(app)
    results = app.session_state["epidemiology_results"]
    t_test = next(r for r in results if r["Test Used"] in ("Student's t-test", "Welch's t-test"))
    _power_tab(app)
    _load(app, f"{t_test['Variable']} ({t_test['Test Used']})")
    assert app.selectbox(key="power_test_type_selector").value == "T-Test (Independent)"
    assert app.session_state["es_ttest"] == pytest.approx(abs(float(t_test["Effect Size"])))
    assert app.session_state["power_group_col"] == "arm"


def test_a_loaded_chi_square_result_reaches_the_calculator_on_screen(app: AppTest) -> None:
    """The minimum dimension and the effect mirror session keys: Load writes the inputs' own keys, or an input
    already on screen keeps its old value under "Parameters loaded!" (review, M5b)."""
    _power_tab(app)
    app.selectbox(key="power_test_type_selector").select("Chi-Square").run()
    app.selectbox(key="es_type_chi2").select("Cohen's w").run()
    app.number_input(key="es_chi2_input").set_value(0.10).run()
    _ok(app)
    cramer = {
        "Variable": "smoker",
        "Test Used": "Chi-Square",
        "Effect Size": 0.25,
        "Effect Type": "Cramer's V",
        "Sample Info": {"N_Cats": 3, "N_Total": 200, "Min_Dim": 2},
    }
    app.session_state["epidemiology_results"] = [cramer]
    app.run()
    _load(app, "smoker (Chi-Square)")
    assert app.selectbox(key="es_type_chi2").value == "Cramer's V"
    assert app.number_input(key="power_chi2_min_dim").value == 2
    assert app.number_input(key="es_chi2_input").value == pytest.approx(0.25)
    assert app.session_state["chi2_w"] == pytest.approx(0.25 * 2**0.5)
    assert app.number_input(key="df_chi2").value == 2


def test_the_mann_whitney_effect_follows_its_scale(app: AppTest) -> None:
    """The effect input mirrors mwu_p_xy: a kept value came back stale after a change of scale (review, M5b)."""
    _power_tab(app)
    app.selectbox(key="power_test_type_selector").select("Mann-Whitney U (Non-parametric)").run()
    app.selectbox(key="es_type_mwu").select("P(X < Y)").run()
    app.number_input(key="es_input_P(X < Y)").set_value(0.70).run()
    app.selectbox(key="es_type_mwu").select("Rank-Biserial r").run()
    assert app.number_input(key="es_input_Rank-Biserial r").value == pytest.approx(0.40)
    app.number_input(key="es_input_Rank-Biserial r").set_value(0.20).run()
    app.selectbox(key="es_type_mwu").select("P(X < Y)").run()
    _ok(app)
    assert app.number_input(key="es_input_P(X < Y)").value == pytest.approx(0.60)
    assert app.session_state["mwu_p_xy"] == pytest.approx(0.60)


def test_a_column_renamed_and_renamed_back_is_not_renamed_again(app: AppTest) -> None:
    """The rename inputs mirror rename_dict: a kept input showed the old new name and renamed the column again."""
    from app_pages import home

    app.switch_page(nav.page_path("explore")).run()
    app.session_state[home.TAB_KEY] = "Columns Renaming"
    app.run()
    apply = "Apply Renaming"
    app.text_input(key="rename_age").set_value("age_years")
    next(b for b in app.button if b.label == apply).click().run()
    app.run()  # the rename is applied after the inputs of that run: the inputs of the new names come next
    app.text_input(key="rename_age_years").set_value("age")
    next(b for b in app.button if b.label == apply).click().run()
    _ok(app)
    assert "age" in app.session_state["_data_session"].base.columns
    app.run()
    assert app.text_input(key="rename_age").value == "age"  # not the old age_years
    next(b for b in app.button if b.label == apply).click().run()
    _ok(app)
    columns = app.session_state["_data_session"].base.columns
    assert "age" in columns and "age_years" not in columns


def test_the_date_statistics_can_be_downloaded_as_excel(app: AppTest) -> None:
    """The quantitative and qualitative tables had their Excel download, the date table none."""
    from io import BytesIO

    import pandas as pd

    from app_pages import home

    app.session_state[home.TAB_KEY] = "Dataset Statistics"
    app.switch_page(nav.page_path("explore")).run()
    _ok(app)
    assert any(m.value == "#### Date Statistics" for m in app.markdown)
    keys = [b.key for b in app.get("download_button")]
    assert "date_download_ungrouped" in keys
    stats = pd.DataFrame(
        {"Min Date": pd.to_datetime(["2020-01-03"]), "Duration": pd.to_timedelta(["1214 days"])},
        index=pd.Index(["enrol_date"], name="Feature"),
    )
    workbook = pd.read_excel(BytesIO(home.to_excel(home._date_stats_for_excel(stats))), index_col=0)
    assert list(workbook.columns) == ["Min Date", "Duration (days)"] and workbook.iloc[0, 1] == 1214
    # grouped: one sheet per group, as the quantitative statistics
    app.selectbox(key="select_group_quant").select("arm").run()
    _ok(app)
    assert "date_download_grouped" in [b.key for b in app.get("download_button")]
    app.selectbox(key="select_group_quant").select("None").run()
