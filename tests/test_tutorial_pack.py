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
from core.derived import DerivedVariable
from core.population import family_masks
from enrich.derived import (
    AGREES,
    DISAGREES,
    MOSTLY,
    check_derived,
    check_inverses,
    fill_steps,
    plan_exact_fills,
)

ROOT = Path(__file__).resolve().parents[1]
TUTORIAL = ROOT / "assets" / "tutorial"
CHAPTER = (ROOT / "documentation" / "14_Full_Tutorial.md").read_text(encoding="utf-8")
DICTIONARY = (TUTORIAL / "tutorial_dictionary.md").read_text(encoding="utf-8")

# the derived variables of the chapter (step 3.3) and the dictionary
BMI = "weight_kg / (height_cm / 100) ** 2"
HEIGHT = "100 * (weight_kg / bmi_kg_m2) ** 0.5"
EGFR = (
    "np.where(age >= 18, 142 * np.minimum(creatinine_umol_l / 88.4 / np.where(sex_clean == 'F', 0.7, 0.9), 1) ** "
    "np.where(sex_clean == 'F', -0.241, -0.302) * np.maximum(creatinine_umol_l / 88.4 / "
    "np.where(sex_clean == 'F', 0.7, 0.9), 1) ** -1.2 * 0.9938 ** age * np.where(sex_clean == 'F', 1.012, 1.0), "
    "np.nan)"
)
PHENOTYPE = "np.where(lvef_pct <= 40, 'HFrEF', np.where(lvef_pct <= 49, 'HFmrEF', 'HFpEF'))"
SMOKER = "np.where(smoking_status == 'Current', 'Yes', 'No')"


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
        cohort.shape == (1206, 41) and int(cohort.duplicated().sum()) == 6 and len(followup) == 1017
    )
    for column in cohort.columns:
        assert column in DICTIONARY, column


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
    followup = pd.read_csv(TUTORIAL / "tutorial_followup_labs.csv")
    df = analysis_set.merge(followup, on="patient_id", how="left")  # step 3.1
    formulas = {
        "site_clean": "site.str.strip()",
        "sex_clean": "np.where(sex.str.strip().str.lower().str.startswith('f'), 'F', 'M')",
        "egfr_6m": EGFR.replace("creatinine_umol_l", "creatinine_6m_umol_l"),
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
    egfr_6m = pd.to_numeric(df["egfr_6m"])
    assert int(egfr_6m.notna().sum()) == 778 and round(float(egfr_6m.mean())) == 68
    assert "`egfr_6m` averages about 68 mL/min/1.73 m2 in\nthe 778 patients" in CHAPTER


def test_the_derived_variables_agree_but_in_the_planted_errors(
    cohort: pd.DataFrame, analysis_set: pd.DataFrame
) -> None:
    """Step 3.3 and the dictionary: each definition gives the recorded value but where an input was mistyped."""
    data = cohort.assign(
        sex_clean=np.where(cohort["sex"].str.strip().str.lower().str.startswith("f"), "F", "M")
    )
    population = analysis_set.index

    def check(target: str, formula: str, **inverses: str):
        assert f"`{formula}`" in DICTIONARY, target
        return check_derived(DerivedVariable(target, formula, inverses=inverses), data)

    def to_fill(check) -> int:
        return len(check.fillable_rows.intersection(population))

    for formula in (BMI, HEIGHT, EGFR, PHENOTYPE):
        assert f"`{formula}`" in CHAPTER, formula
    bmi = check("bmi_kg_m2", BMI, height_cm=HEIGHT)
    assert (bmi.verdict, bmi.agreeing, bmi.compared) == (AGREES, 1101, 1106)
    assert bmi.strict and bmi.tolerance == pytest.approx(0.05)
    assert sorted(bmi.disagreeing) == [
        41,
        133,
        512,
        760,
        990,
    ]  # heights in metres, weights 0 and 350
    height = check_inverses(bmi.definition, data)["height_cm"]
    assert height.verdict == AGREES
    neither = analysis_set["height_cm"].isna() & analysis_set["bmi_kg_m2"].isna()
    assert (to_fill(bmi), to_fill(height), int(neither.sum())) == (34, 10, 24)

    egfr = check("egfr_ml_min_1_73m2", EGFR)
    assert (egfr.verdict, egfr.agreeing, egfr.compared, list(egfr.disagreeing)) == (
        AGREES,
        1143,
        1144,
        [17],  # the age 212
    )
    raw_sex = check_derived(
        DerivedVariable("egfr_ml_min_1_73m2", EGFR.replace("sex_clean", "sex")), data
    )
    assert raw_sex.verdict == AGREES and sorted(raw_sex.disagreeing) == [17, 211, 377, 701, 1024]
    ckd_epi_2009 = (
        EGFR.replace("142 *", "141 *")
        .replace("-0.241", "-0.329")
        .replace("-0.302", "-0.411")
        .replace("** -1.2 *", "** -1.209 *")
        .replace("0.9938", "0.993")
        .replace("1.012", "1.018")
    )
    assert check_derived(DerivedVariable("egfr_ml_min_1_73m2", ckd_epi_2009), data).verdict == (
        DISAGREES
    )
    assert to_fill(egfr) == 49 and cohort.loc[[95, 700, 1150], "egfr_ml_min_1_73m2"].isna().all()

    phenotype = check("hf_phenotype", PHENOTYPE)
    assert (phenotype.verdict, phenotype.agreeing, phenotype.compared) == (AGREES, 1171, 1172)
    assert list(phenotype.disagreeing) == [615] and to_fill(phenotype) == 30  # the LVEF 155
    esc_2016 = check_derived(
        DerivedVariable("hf_phenotype", PHENOTYPE.replace("lvef_pct <= 40", "lvef_pct < 40")), data
    )
    assert esc_2016.verdict == MOSTLY and int((cohort["lvef_pct"] == 40).sum()) == 39
    assert "`lvef_pct < 40`" in CHAPTER

    smoker = check("smoker_baseline", SMOKER)
    assert (smoker.verdict, smoker.agreeing, smoker.compared) == (AGREES, 1206, 1206)

    for number in (
        "1,101 of 1,106 rows",
        "1,143 of 1,144 rows",
        "1,171 of 1,172 rows",
        "**34** BMIs, **49** eGFRs and **30** phenotypes",
        "the **10** heights the\ninverse recovers",
        "24 patients have neither",
    ):
        assert number in CHAPTER, number


def test_the_chapter_fills_exactly_what_it_says(
    cohort: pd.DataFrame, analysis_set: pd.DataFrame
) -> None:
    """Step 3.3, Fill exactly: the values the definitions give in the analysis population, all counted as observed,
    and the share of incomplete rows the multiple imputation then starts from."""
    data = cohort.assign(
        sex_clean=np.where(cohort["sex"].str.strip().str.lower().str.startswith("f"), "F", "M")
    )
    definitions = [
        DerivedVariable("bmi_kg_m2", BMI, inverses={"height_cm": HEIGHT}),
        DerivedVariable("egfr_ml_min_1_73m2", EGFR),
        DerivedVariable("hf_phenotype", PHENOTYPE),
    ]
    plan = plan_exact_fills(definitions, data, analysis_set.index)
    assert [(f.target, f.filled, f.exact) for f in plan.fills] == [
        ("bmi_kg_m2", 34, True),
        ("height_cm", 10, True),
        ("egfr_ml_min_1_73m2", 49, True),
        ("hf_phenotype", 30, True),
    ]
    assert not any(f.single for f in plan.fills)  # every value counted as observed
    assert plan.filled == 123 and "**Fill 123" in CHAPTER
    filled = fill_steps(plan, data, analysis_set.index, "r")[-1][0].loc[analysis_set.index]
    incomplete = filled[[c for c in filled.columns if filled[c].isna().any()]].isna().any(axis=1)
    assert round(float(incomplete.mean()), 2) == 0.50 and "about half of them miss" in CHAPTER


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
    assert result.df.shape == (1206, 41)
    parsed = pd.to_datetime(cohort["enrol_date"], dayfirst=True)
    assert parsed.min() >= pd.Timestamp("2021-01-01") and parsed.max() <= pd.Timestamp("2024-01-05")
