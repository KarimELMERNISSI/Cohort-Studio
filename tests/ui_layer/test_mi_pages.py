"""Pooled analyses on the analysis pages once an imputation exists (M2, AppTest on the whole app)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import utils.data_paths as dp
from ui import nav

ROOT = Path(__file__).resolve().parents[2]
MAIN = str(ROOT / "main.py")
# the tutorial's four sites; "Metz " and the other spelling variants hold one to three rows
SITES = ["Esch-sur-Alzette", "Ettelbruck", "Luxembourg", "Metz"]


def _ok(at: AppTest) -> None:
    assert not at.exception, [str(e.value)[:300] for e in at.exception]


@pytest.fixture(scope="module")
def app(tmp_path_factory: pytest.TempPathFactory):
    previous = dp.DATA_ROOT
    dp.DATA_ROOT = str(tmp_path_factory.mktemp("data"))
    os.environ.setdefault("CMP_IAM_PROVIDER", "local")
    at = AppTest.from_file(MAIN, default_timeout=300)
    at.run()
    at.session_state["user_authenticated"] = True
    at.session_state["username"] = "mi_pages_user"
    at.session_state["user_role"] = "researcher"
    at.run()
    at.switch_page(nav.page_path("start")).run()
    at.button(key="start_tutorial").click().run()
    _ok(at)
    at.switch_page(nav.page_path("enrich")).run()
    _ok(at)
    at.number_input(key="mi_m").set_value(3)
    at.number_input(key="mi_maxit").set_value(2)
    at.button(key="mi_run").click().run()
    _ok(at)
    assert at.session_state["_mi"].mi.m == 3
    yield at
    dp.DATA_ROOT = previous


def _stored(at: AppTest, key: str):
    return at.session_state["_results"][key].value


def test_survival_page_pools_kaplan_meier_and_cox(app: AppTest) -> None:
    app.session_state["analysis_data_mode"] = "multiple"
    app.switch_page(nav.page_path("survival")).run()
    _ok(app)
    assert any("pool **3 imputed datasets**" in i.value for i in app.info)
    app.selectbox(key="surv_time").select("fu_days").run()
    app.selectbox(key="surv_event").select("death").run()
    app.selectbox(key="surv_group").select("arm").run()
    app.button(key="surv_run").click().run()
    _ok(app)
    km = _stored(app, "survival_mi")
    assert (
        km.kind == "survival"
        and km.m == 3
        and set(km.extra["curves"])
        == set(km.complete_case.curves[i].label for i in range(len(km.complete_case.curves)))
    )
    assert km.tests["log_rank"].method == "D2"
    app.selectbox(key="cox_time").select("fu_days").run()
    app.selectbox(key="cox_event").select("death").run()
    app.multiselect(key="cox_covs").select("age").select("hba1c_pct").select("arm").run()
    app.button(key="cox_run").click().run()
    _ok(app)
    cox = _stored(app, "cox_mi")
    hba1c = cox.term("hba1c_pct")
    assert (
        hba1c.scale == "log" and hba1c.natural_estimate is not None and hba1c.fmi > 0
    )  # imputed covariate
    assert (
        cox.complete_case is not None and cox.complete_case.n < 1206
    )  # complete cases drop the missing HbA1c
    assert any("Complete-case analysis" in e.label for e in app.expander)


def test_analyze_page_pools_group_comparisons_and_ancova(app: AppTest) -> None:
    app.session_state["analysis_data_mode"] = "multiple"
    app.session_state["epi_analysis_mode"] = "Univariate (Group Comparison)"
    app.switch_page(nav.page_path("analyze")).run()
    _ok(app)
    assert any("No transformation with imputed data" in c.value for c in app.caption)
    app.selectbox(key="mi_cmp_group").select("arm").run()
    app.multiselect(key="mi_cmp_targets").set_value(
        ["hba1c_pct", "diabetes", "smoking_status"]
    ).run()
    app.button(key="mi_cmp_run").click().run()
    _ok(app)
    stored = _stored(app, "univariate_mi")
    kinds = {target: item.kind for target, item in stored["results"].items()}
    assert kinds == {"hba1c_pct": "welch", "diabetes": "twobytwo", "smoking_status": "chisq"}, (
        stored["errors"]
    )
    welch = stored["results"]["hba1c_pct"].result.extra["result"]
    assert welch.mi is not None and welch.mi.m == 3 and welch.mi.fmi > 0  # HbA1c has imputed values
    assert not any(
        "selected manually" in w.value for w in app.warning
    )  # Welch is the pooled test by design
    for target in ("diabetes", "smoking_status"):  # every detail view renders
        app.selectbox(key="mi_cmp_detail").select(target).run()
        _ok(app)

    # three or more groups: D1 on a linear model with robust (HC3) variances, as Welch's ANOVA on the observed data
    app.selectbox(key="mi_cmp_group").select("site").run()
    app.multiselect(key="mi_cmp_levels::site").set_value(
        SITES
    ).run()  # the four sites, not the variants
    app.button(key="mi_cmp_run").click().run()
    _ok(app)
    by_site = _stored(app, "univariate_mi")["results"]["hba1c_pct"]
    assert by_site.kind == "hc3" and by_site.result.tests["site"].method == "D1"
    assert by_site.result.extra["cov_type"] == "HC3"

    app.session_state["epi_analysis_mode"] = "Multivariate (ANCOVA/Regression)"
    app.run()
    app.selectbox(key="mi_anc_target").select("hba1c_pct").run()
    app.selectbox(key="mi_anc_group").select("arm").run()
    app.multiselect(key="mi_anc_covs").select("age").run()
    app.button(key="mi_anc_run").click().run()
    _ok(app)
    pooled = _stored(app, "ancova_mi")
    assert pooled.kind == "linear" and pooled.tests["arm"].method == "D1"
    assert len(pooled.extra["adjusted_means"]) == 2 and pooled.complete_case is not None
    app.session_state["epi_analysis_mode"] = "Univariate (Group Comparison)"


def test_the_pooled_two_group_test_can_be_chosen_with_a_warning(app: AppTest) -> None:
    """Recommended (Welch) or chosen: Student's t-test, pooled the same way, with why Welch is recommended."""
    from manage.report_tables import _pooled_comparison_blocks

    app.session_state["analysis_data_mode"] = "multiple"
    app.session_state["epi_analysis_mode"] = "Univariate (Group Comparison)"
    app.switch_page(nav.page_path("analyze")).run()
    _ok(app)
    app.selectbox(key="mi_cmp_group").select("arm").run()
    app.multiselect(key="mi_cmp_targets").set_value(["hba1c_pct"]).run()
    assert app.radio(key="mi_cmp_selection").value == "Recommended"
    app.radio(key="mi_cmp_selection").set_value("Choose the method").run()
    assert app.selectbox(key="mi_cmp_two_groups").value == "welch"
    app.selectbox(key="mi_cmp_two_groups").select("student").run()
    assert any("Student's t-test assumes equal variances" in w.value for w in app.warning)
    app.button(key="mi_cmp_run").click().run()
    _ok(app)
    item = _stored(app, "univariate_mi")["results"]["hba1c_pct"]
    test = item.result.extra["result"]
    assert item.kind == "student" and "Student" in test.name and test.mi.m == 3
    _, table = _pooled_comparison_blocks(_stored(app, "univariate_mi")).tables[0]  # the dossier
    assert table["Analysis"].tolist() == ["Student t-test (equal variances, chosen)"]
    app.radio(key="mi_cmp_selection").set_value("Recommended").run()


def test_the_pooled_test_of_three_groups_can_be_chosen_with_a_warning(app: AppTest) -> None:
    """Recommended (robust HC3 variances) or chosen: the equal-variance model, with why HC3 is recommended."""
    from manage.report_tables import _pooled_comparison_blocks

    app.session_state["analysis_data_mode"] = "multiple"
    app.session_state["epi_analysis_mode"] = "Univariate (Group Comparison)"
    app.switch_page(nav.page_path("analyze")).run()
    _ok(app)
    app.selectbox(key="mi_cmp_group").select("site").run()
    app.multiselect(key="mi_cmp_levels::site").set_value(SITES).run()
    app.multiselect(key="mi_cmp_targets").set_value(["hba1c_pct"]).run()
    app.radio(key="mi_cmp_selection").set_value("Choose the method").run()
    assert app.selectbox(key="mi_cmp_more_groups").value == "hc3"
    app.selectbox(key="mi_cmp_more_groups").select("anova").run()
    assert any("The one-way ANOVA assumes equal variances" in w.value for w in app.warning)
    app.button(key="mi_cmp_run").click().run()
    _ok(app)
    item = _stored(app, "univariate_mi")["results"]["hba1c_pct"]
    assert item.kind == "anova" and item.result.extra["cov_type"] == "nonrobust"
    _, table = _pooled_comparison_blocks(_stored(app, "univariate_mi")).tables[0]  # the dossier
    assert table["Analysis"].tolist() == [
        "Linear model on the groups, equal variances (chosen; D1)"
    ]
    app.radio(key="mi_cmp_selection").set_value("Recommended").run()


def test_a_group_of_one_value_is_refused_by_the_robust_model() -> None:
    """HC3 gives each group its own variance, which one value does not have: the message says what to do."""
    import numpy as np
    import pandas as pd

    from app_pages.analysis_pooled import compare_pooled
    from biostats.mi import MIAnalysisError, MIData

    rng = np.random.default_rng(5)
    base = pd.DataFrame(
        {"site": ["Metz"] * 20 + ["Luxembourg"] * 20 + ["Metz "], "sbp": rng.normal(130, 10, 41)}
    )
    base.loc[[3, 25], "sbp"] = np.nan
    completed = [base.fillna({"sbp": 130.0 + i}) for i in range(3)]
    mi = MIData.from_completed(base, completed, methods={"sbp": "pmm"})
    levels = ["Luxembourg", "Metz", "Metz "]
    with pytest.raises(MIAnalysisError, match="Metz␣ has a single value, while the robust"):
        compare_pooled(mi, "site", levels, "sbp", numeric=True)
    chosen = compare_pooled(mi, "site", levels, "sbp", numeric=True, more_groups="anova")
    assert chosen.kind == "anova"  # the equal-variance model still runs


def test_paired_and_one_sample_t_tests_are_pooled(app: AppTest) -> None:
    app.session_state["analysis_data_mode"] = "multiple"
    app.switch_page(nav.page_path("analyze")).run()
    _ok(app)
    assert any("analyse the observed data" in i.value for i in app.info)  # diagnostic & agreement
    app.selectbox(key="pair_before").select("sbp_baseline").run()
    app.selectbox(key="pair_after").select("sbp_6m").run()  # 183 imputed values
    app.button(key="pair_run").click().run()
    _ok(app)
    paired = _stored(app, "paired_mi").extra["result"]
    assert (
        "pooled over 3 imputations" in paired.name and paired.mi is not None and paired.mi.fmi > 0
    )
    app.radio(key="pair_kind").set_value("Wilcoxon signed-rank").run()
    _ok(app)
    assert any("Not pooled with imputed data" in i.value for i in app.info)
    app.radio(key="pair_kind").set_value("Paired t-test").run()
    app.selectbox(key="one_col").select("hba1c_pct").run()
    app.number_input(key="one_ref").set_value(7.0).run()
    app.button(key="one_run").click().run()
    _ok(app)
    one = _stored(app, "one_sample_mi").extra["result"]
    assert one.mi is not None and one.mi.m == 3 and one.estimate_ci is not None


def test_observed_mode_is_unchanged_and_a_stale_imputation_is_not_used(app: AppTest) -> None:
    app.session_state["analysis_data_mode"] = "observed"
    app.switch_page(nav.page_path("survival")).run()
    _ok(app)
    assert not any("imputed datasets" in i.value for i in app.info)
    app.session_state["analysis_data_mode"] = "multiple"
    app.checkbox(
        key=next(c.key for c in app.checkbox if c.key and c.key.startswith("popfam::"))
    ).check().run()
    _ok(app)  # a narrower population: the imputation covers more rows than it now analyses
    assert any("fitted on more rows" in c.label for c in app.checkbox)
    assert any("analyses below use the observed data" in w.value for w in app.warning)
