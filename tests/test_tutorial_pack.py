"""The tutorial pack stays consistent: generator = files, configuration rules = planted problems, chapter = numbers."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from statsmodels.stats.contingency_tables import StratifiedTable

from app_pages.data_enrichment import evaluate_formula_safely
from core.population import family_masks

ROOT = Path(__file__).resolve().parents[1]
TUTORIAL = ROOT / "assets" / "tutorial"
CHAPTER = (ROOT / "documentation" / "14_Full_Tutorial.md").read_text(encoding="utf-8")


def _generator():
    spec = importlib.util.spec_from_file_location(
        "make_tutorial_cohort", ROOT / "dev_scripts" / "make_tutorial_cohort.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@pytest.fixture(scope="module")
def cohort() -> pd.DataFrame:
    return pd.read_csv(TUTORIAL / "tutorial_cohort.csv")


@pytest.fixture(scope="module")
def config() -> dict:
    return json.loads((TUTORIAL / "tutorial_config.json").read_text(encoding="utf-8"))


def _masks(df: pd.DataFrame, config: dict, family: str) -> pd.DataFrame:
    masks = family_masks(df, config["mask_families"][family])
    assert not masks.errors, masks.errors
    return pd.DataFrame(masks.rules)


@pytest.fixture(scope="module")
def analysis_set(cohort: pd.DataFrame, config: dict) -> pd.DataFrame:
    clean = cohort[~_masks(cohort, config, "clinical_anomalies").any(axis=1)]
    return clean[_masks(clean, config, "study_inclusion").all(axis=1)].copy()


def test_files_match_the_generator(cohort: pd.DataFrame) -> None:
    df, followup = _generator().make_tutorial_cohort()

    def lf(text: str) -> str:
        return text.replace("\r\n", "\n")

    assert lf(df.to_csv(index=False)) == lf(
        (TUTORIAL / "tutorial_cohort.csv").read_text(encoding="utf-8")
    )
    assert lf(followup.to_csv(index=False)) == lf(
        (TUTORIAL / "tutorial_followup_labs.csv").read_text(encoding="utf-8")
    )
    assert (
        cohort.shape == (1206, 38) and int(cohort.duplicated().sum()) == 6 and len(followup) == 1017
    )
    dictionary = (TUTORIAL / "tutorial_dictionary.md").read_text(encoding="utf-8")
    for column in cohort.columns:
        assert column in dictionary, column


def test_anomaly_rules_flag_exactly_the_planted_errors(cohort: pd.DataFrame, config: dict) -> None:
    masks = _masks(cohort, config, "clinical_anomalies")
    counts = {name: int(m.sum()) for name, m in masks.items()}
    assert counts == {
        "Implausible age": 2,
        "Height not in centimetres": 3,
        "Implausible weight": 2,
        "Diastolic above systolic": 3,
        "lvef_pct": 1,
        "potassium_mmol_l": 2,
        "Negative follow-up": 1,
        "Last contact before enrolment": 2,
    }
    assert int(masks.any(axis=1).sum()) == 16


def test_inclusion_criteria(cohort: pd.DataFrame, config: dict) -> None:
    assert (
        int(_masks(cohort, config, "study_inclusion").all(axis=1).sum()) == 939
    )  # without the anomaly filter
    clean = cohort[~_masks(cohort, config, "clinical_anomalies").any(axis=1)]
    masks = _masks(clean, config, "study_inclusion")
    assert len(clean) == 1190 and int(masks.all(axis=1).sum()) == 928
    failing = {name: int((~m).sum()) for name, m in masks.items()}
    assert failing == {
        "Written consent": 14,
        "age": 71,
        "Symptomatic heart failure (NYHA II-IV)": 183,
        "No severe kidney disease": 8,
    }
    soft = (clean["diabetes"] == "No") & (clean["hba1c_pct"] >= 6.5)
    assert int(soft.sum()) == 16


def test_chapter_formulas_work_in_the_enrich_evaluator(analysis_set: pd.DataFrame) -> None:
    df = analysis_set.copy()
    formulas = {
        "site_clean": "site.str.strip()",
        "sex_clean": "np.where(sex.str.strip().str.lower().str.startswith('f'), 'F', 'M')",
        "bmi": "weight_kg / (height_cm / 100) ** 2",
        "egfr": "142 * np.minimum(creatinine_umol_l / 88.4 / np.where(sex_clean == 'F', 0.7, 0.9), 1) ** "
        "np.where(sex_clean == 'F', -0.241, -0.302) * np.maximum(creatinine_umol_l / 88.4 / "
        "np.where(sex_clean == 'F', 0.7, 0.9), 1) ** -1.2 * 0.9938 ** age * np.where(sex_clean == 'F', 1.012, 1.0)",
    }
    for name, formula in formulas.items():
        ok, result, _ = evaluate_formula_safely(formula, name, df.copy(), return_series=True)
        assert ok, (name, result)
        df[name] = result
        assert f"`{formula}`" in CHAPTER, name
    assert sorted(df["site_clean"].unique()) == [
        "Esch-sur-Alzette",
        "Ettelbruck",
        "Luxembourg",
        "Metz",
    ]
    assert set(df["sex_clean"]) == {"F", "M"}


def test_the_chapter_numbers(analysis_set: pd.DataFrame) -> None:
    a = analysis_set
    assert a["arm"].value_counts().to_dict() == {"Intervention": 466, "Control": 462}
    followup = pd.read_csv(TUTORIAL / "tutorial_followup_labs.csv")
    assert int(a["patient_id"].isin(followup["patient_id"]).sum()) == 778

    t = pd.crosstab(a["arm"], a["hf_hospitalisation"])
    r_int = t.loc["Intervention", "Yes"] / t.loc["Intervention"].sum()
    r_ctl = t.loc["Control", "Yes"] / t.loc["Control"].sum()
    odds = (t.loc["Intervention", "Yes"] * t.loc["Control", "No"]) / (
        t.loc["Intervention", "No"] * t.loc["Control", "Yes"]
    )
    assert (round(r_int * 100, 1), round(r_ctl * 100, 1)) == (27.7, 37.4)
    assert round(r_int / r_ctl, 2) == 0.74 and round(odds, 2) == 0.64
    assert round(stats.chi2_contingency(t, correction=False)[1], 4) == 0.0015

    x = a.dropna(subset=["sbp_6m"])
    ancova = smf.ols("sbp_6m ~ C(arm) + sbp_baseline", data=x).fit()
    assert round(ancova.params["C(arm)[T.Intervention]"], 1) == -5.9
    assert round((x["sbp_6m"] - x["sbp_baseline"]).mean(), 1) == -5.0
    quit_smoking = int(((a["smoker_baseline"] == "Yes") & (a["smoker_6m"] == "No")).sum())
    started = int(((a["smoker_baseline"] == "No") & (a["smoker_6m"] == "Yes")).sum())
    assert (quit_smoking, started) == (58, 0)

    tables = [
        pd.crosstab(g["arm"] == "Intervention", g["hf_hospitalisation"] == "Yes")
        .reindex(index=[True, False], columns=[True, False])
        .to_numpy()
        for _, g in a.groupby(a["site"].str.strip())
    ]
    st_ = StratifiedTable(tables)
    assert round(st_.oddsratio_pooled, 2) == 0.64 and round(st_.test_equal_odds().pvalue, 2) == 0.33
    assert a["site"].nunique() == 7  # uncleaned site names: why the chapter creates site_clean

    assert round(roc_auc_score(a["mi_adjudicated"] == "Yes", a["troponin_ng_l"]), 2) == 0.93
    assert round(cohen_kappa_score(a["nyha_class"], a["nyha_class_central"]), 2) == 0.71
    assert (
        round(cohen_kappa_score(a["nyha_class"], a["nyha_class_central"], weights="linear"), 2)
        == 0.76
    )
    assert round((a["sbp_baseline"] - a["sbp_ambulatory"]).mean(), 1) == 6.0

    died = (a["death"] == "Yes").astype(int)
    cox = sm.PHReg(
        a["fu_days"], (a[["arm"]] == "Intervention").astype(float), status=died, ties="efron"
    ).fit()
    assert round(float(np.exp(cox.params[0])), 2) == 0.88
    poisson = smf.glm(
        "n_hospital_visits ~ C(arm)",
        data=a,
        family=sm.families.Poisson(),
        offset=np.log(a["fu_days"] / 365.25),
    ).fit()
    assert round(float(np.exp(poisson.params["C(arm)[T.Intervention]"])), 2) == 0.80

    for number in (
        "1,206",
        "16 rows",
        "1,190",
        "928",
        "462 Control and 466 Intervention",
        "778 of the 928",
        "27.7% vs 37.4%",
        "0.74",
        "0.64",
        "-5.9 mmHg",
        "-5.0 mmHg",
        "58 patients quit",
        "0.93",
        "0.71",
        "0.76",
        "6 mmHg higher",
        "0.88",
        "0.80",
        "2,660",
    ):
        assert number in CHAPTER, number


def test_the_csv_reads_with_day_first_dates(cohort: pd.DataFrame) -> None:
    from core.state import ParseOptions
    from ingest.loaders import read_table

    data = (TUTORIAL / "tutorial_cohort.csv").read_bytes()
    result = read_table(data, "tutorial_cohort.csv", ParseOptions(dayfirst=True))
    assert result.df.shape == (1206, 38)
    parsed = pd.to_datetime(cohort["enrol_date"], dayfirst=True)
    assert parsed.min() >= pd.Timestamp("2021-01-01") and parsed.max() <= pd.Timestamp("2024-01-05")
