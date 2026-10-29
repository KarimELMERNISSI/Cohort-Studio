"""
Generate the tutorial pack (assets/tutorial/): a richer synthetic cohort for the full walk-through.

    python dev_scripts/make_tutorial_cohort.py

Outputs:
    assets/tutorial/tutorial_cohort.csv         1,206 rows (1,200 patients + 6 exact duplicates), 41 columns
    assets/tutorial/tutorial_followup_labs.csv  6-month laboratory visit, to merge on patient_id (Enrich)
    assets/tutorial/tutorial_dictionary.md      data dictionary, planted problems and expected findings

A synthetic, randomised heart-failure cohort (4 sites in Luxembourg and Lorraine, enrolled 2021 to early 2024). No real
patients. Deterministic (seed 2027). Every analysis of the app has something to find: data-entry errors
matching the plausibility rules of tutorial_config.json, patients failing its inclusion criteria, MCAR / MAR /
structural missingness, uniformity and duplicate problems, a paired before/after design, two raters,
two measurement methods, a diagnostic marker against an adjudicated diagnosis, binary, count and
time-to-event outcomes with real treatment effects, a stratifying variable (site), and variables computed from
others (BMI, eGFR, heart-failure phenotype) with gaps that their formulas can fill.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 2027
N = 1200
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "assets" / "tutorial"
CSV_PATH = OUT_DIR / "tutorial_cohort.csv"
FOLLOWUP_PATH = OUT_DIR / "tutorial_followup_labs.csv"
STUDY_END = pd.Timestamp("2024-12-31")

# Rows (0-based, in the 1,200-patient table) that receive the planted data-entry errors.
PLANTED = {
    "age_212": 17,
    "age_negative": 288,
    "height_in_metres": (41, 512, 990),
    "weight_zero": 133,
    "weight_350": 760,
    "dbp_above_sbp": (77, 404, 1101),
    "lvef_155": 615,
    "potassium_high": 230,
    "potassium_low": 845,
    "fu_negative": 1033,
    "contact_before_enrolment": (352, 958),
}
MINORS = (95, 700, 1150)
SEVERE_CKD = (
    64,
    222,
    433,
    571,
    689,
    902,
    1121,
)  # creatinine 300-450 umol/L: fail the inclusion criteria  # ages 16-17: plausible values that fail the inclusion criteria
NO_CONSENT = (12, 140, 266, 391, 503, 618, 737, 829, 911, 1012, 1088, 1140, 1177, 1199)
SEX_VARIANTS = {
    60: "male",
    211: "Female",
    377: "f",
    548: "M ",
    701: "F ",
    866: "MALE",
    1024: "female",
    1178: "m",
}
SITE_TRAILING_SPACE = (5, 130, 255, 480, 640, 805, 1066, 1190)
DUPLICATED = (3, 250, 499, 750, 1000, 1150)


def _expit(x):
    return 1 / (1 + np.exp(-x))


def _ckd_epi_2021(creatinine_umol_l, age, female):
    """eGFR in mL/min/1.73 m2, race-free CKD-EPI 2021 (Inker et al. 2021), from a creatinine in umol/L."""
    ratio = np.asarray(creatinine_umol_l, dtype=float) / 88.4 / np.where(female, 0.7, 0.9)
    return (
        142
        * np.minimum(ratio, 1) ** np.where(female, -0.241, -0.302)
        * np.maximum(ratio, 1) ** -1.2
        * 0.9938 ** np.asarray(age, dtype=float)
        * np.where(female, 1.012, 1.0)
    )


def _with_derived(df: pd.DataFrame, true: pd.DataFrame) -> pd.DataFrame:
    """
    Three variables computed from others, as the hospital record and the laboratory hold them: from the values before
    the planted typing errors, so each mistyped input disagrees with its formula. Their gaps, and the heights not
    recorded, come from a random stream of their own (no existing value moves) and never fall on a row with a planted
    problem, a duplicate or an exclusion.
    """
    n = len(df)
    # one draw per patient for each reason a value is missing
    u = np.random.default_rng([SEED, 1]).random((5, n))
    special = np.zeros(n, dtype=bool)
    for rows in (
        *PLANTED.values(),
        MINORS,
        SEVERE_CKD,
        NO_CONSENT,
        tuple(SEX_VARIANTS),
        SITE_TRAILING_SPACE,
        DUPLICATED,
    ):
        special[list(np.atleast_1d(rows))] = True
    minors = np.isin(np.arange(n), MINORS)
    # 212 and -3 are typing errors, the minors' ages are not; the sex is what each code means, however typed
    age = np.where(minors, df["age"], true["age"]).astype(float)
    female = df["sex"].str.strip().str.lower().str.startswith("f").to_numpy()

    bmi = (true["weight_kg"] / (true["height_cm"] / 100) ** 2).round(1).to_numpy()
    egfr = _ckd_epi_2021(df["creatinine_umol_l"], age, female).round()
    lvef = true["lvef_pct"].to_numpy()
    phenotype = np.where(lvef <= 40, "HFrEF", np.where(lvef <= 49, "HFmrEF", "HFpEF"))  # ESC 2021

    # the height not measured, more often in older patients (no BMI either), or measured but not transcribed;
    # no eGFR under 18 either: CKD-EPI is an adult equation
    not_measured = (u[0] < _expit(-3.7 + 0.07 * (age - 68))) & ~special
    height_not_copied = (u[1] < 0.02) & ~special & ~not_measured
    bmi_not_copied = (u[2] < 0.035) & ~special & ~not_measured & ~height_not_copied
    no_egfr = ((u[3] < 0.05) & ~special) | minors
    no_phenotype = (u[4] < 0.03) & ~special

    df = df.copy()
    df.loc[not_measured | height_not_copied, "height_cm"] = np.nan
    df.insert(
        df.columns.get_loc("weight_kg") + 1,
        "bmi_kg_m2",
        np.where(not_measured | bmi_not_copied, np.nan, bmi),
    )
    df.insert(
        df.columns.get_loc("lvef_reader2_pct") + 1,
        "hf_phenotype",
        np.where(no_phenotype, None, phenotype),
    )
    df.insert(
        df.columns.get_loc("creatinine_umol_l") + 1,
        "egfr_ml_min_1_73m2",
        np.where(no_egfr, np.nan, egfr),
    )
    return df


def make_tutorial_cohort(n: int = N, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    patient_id = [f"HF-{i + 1:04d}" for i in range(n)]
    site = rng.choice(
        ["Luxembourg", "Esch-sur-Alzette", "Ettelbruck", "Metz"], size=n, p=[0.35, 0.25, 0.2, 0.2]
    )
    enrol = pd.Timestamp("2021-01-04") + pd.to_timedelta(rng.integers(0, 3 * 365, n), unit="D")
    arm = rng.permutation(np.repeat(["Control", "Intervention"], n // 2))
    is_int = arm == "Intervention"

    sex = rng.choice(["F", "M"], size=n, p=[0.4, 0.6])
    female = sex == "F"
    age = np.clip(rng.normal(68, 11, n), 30, 95).round().astype(int)
    height = np.where(female, rng.normal(163, 6, n), rng.normal(176, 7, n)).round(1)
    bmi = np.clip(rng.normal(28.5, 5, n), 17, 50)
    weight = (bmi * (height / 100) ** 2).round(1)

    smoking = rng.choice(["Never", "Former", "Current"], size=n, p=[0.45, 0.35, 0.20])
    diabetes = rng.random(n) < 0.35
    hypertension = rng.random(n) < 0.70
    prior_mi = rng.random(n) < 0.30

    nyha_levels = np.array(["I", "II", "III", "IV"])
    nyha_idx = rng.choice(4, size=n, p=[0.15, 0.45, 0.32, 0.08])
    nyha = nyha_levels[nyha_idx]
    # Central adjudication agrees with the investigator about 80% of the time, otherwise one class apart.
    shift = np.where(rng.random(n) < 0.8, 0, rng.choice([-1, 1], size=n))
    nyha_central = nyha_levels[np.clip(nyha_idx + shift, 0, 3)]

    lvef = np.clip(rng.normal(42 - 4 * nyha_idx, 10, n), 10, 75).round()
    lvef_reader2 = np.clip(lvef + rng.normal(1.0, 3.5, n), 8, 80).round()

    sbp = (rng.normal(130, 17, n) + 8 * hypertension).round()
    dbp = (0.55 * sbp + rng.normal(5, 7, n)).round()
    sbp_ambulatory = (
        sbp - 6 + rng.normal(0, 7, n)
    ).round()  # white-coat effect: office reads higher
    heart_rate = np.clip(rng.normal(74, 12, n), 45, 130).round()

    creatinine = np.exp(
        np.log(92) + 0.006 * (age - 68) + 0.10 * diabetes + rng.normal(0, 0.3, n)
    ).round()
    potassium = np.clip(rng.normal(4.4, 0.45, n), 3.0, 6.2).round(1)
    hba1c = np.where(diabetes, rng.normal(7.6, 1.1, n), rng.normal(5.6, 0.45, n)).round(1)
    ldl = np.clip(rng.normal(2.8, 0.9, n), 0.6, 7.0).round(2)
    ntprobnp = np.exp(
        np.log(900) + 0.45 * nyha_idx - 0.02 * (lvef - 35) + rng.normal(0, 0.7, n)
    ).round()

    mi = rng.random(n) < _expit(-2.6 + 0.9 * prior_mi + 0.02 * (age - 68))
    troponin = np.exp(np.where(mi, np.log(60), np.log(14)) + rng.normal(0, 0.8, n)).round(1)

    current = smoking == "Current"
    quits = rng.random(n) < np.where(is_int, 0.35, 0.15)
    smoker_baseline = np.where(current, "Yes", "No")
    smoker_6m = np.where(current & ~quits, "Yes", "No")

    adherence = np.clip(rng.normal(np.where(is_int, 84, 88), 12, n), 20, 100).round()

    # 12-month heart-failure hospitalisation (intervention: OR about 0.6).
    lp = (
        -1.5
        - 0.5 * is_int
        + 0.45 * nyha_idx
        - 0.025 * (lvef - 38)
        + 0.015 * (age - 68)
        + 0.3 * diabetes
    )
    hf_hosp = rng.random(n) < _expit(lp)

    # Survival: exponential hazards, intervention HR about 0.72; administrative censoring at study end.
    hazard = 0.00028 * np.exp(
        np.log(0.72) * is_int + 0.35 * nyha_idx + 0.03 * (age - 68) - 0.02 * (lvef - 38)
    )
    t_death = rng.exponential(1 / hazard)
    t_admin = (STUDY_END - enrol).days.to_numpy()
    t_dropout = np.where(rng.random(n) < 0.08, rng.uniform(30, 900, n), np.inf)
    fu = np.minimum.reduce([t_death, t_admin, t_dropout])
    death = t_death <= np.minimum(t_admin, t_dropout)
    fu_days = np.maximum(fu.round(), 1).astype(int)
    last_contact = enrol + pd.to_timedelta(fu_days, unit="D")

    # Hospital visits over follow-up (Poisson, intervention IRR about 0.8).
    visit_rate = 1.6 * np.exp(np.log(0.8) * is_int + 0.25 * nyha_idx) * fu_days / 365.25
    visits = rng.poisson(visit_rate)

    # 6-month office SBP: missing when the patient died or left before 6 months, or skipped the visit.
    sbp_6m = (sbp - np.where(is_int, 9, 3) + rng.normal(0, 9, n)).round()
    absent_6m = (fu_days < 180) | (rng.random(n) < _expit(-3.2 + 0.03 * (90 - adherence)))
    sbp_6m = np.where(absent_6m, np.nan, sbp_6m)

    angioedema = rng.random(n) < np.where(is_int, 0.025, 0.008)

    # Missingness: LDL MCAR (6%), HbA1c MAR on diabetes, NT-proBNP MAR on NYHA class.
    ldl = np.where(rng.random(n) < 0.06, np.nan, ldl)
    hba1c = np.where(rng.random(n) < np.where(diabetes, 0.05, 0.30), np.nan, hba1c)
    ntprobnp = np.where(rng.random(n) < np.where(nyha_idx <= 1, 0.25, 0.05), np.nan, ntprobnp)

    yes_no = lambda flag: np.where(flag, "Yes", "No")
    df = pd.DataFrame(
        {
            "patient_id": patient_id,
            "site": site,
            "enrol_date": enrol.strftime("%d/%m/%Y"),
            "consent": "Yes",
            "arm": arm,
            "age": age,
            "sex": sex,
            "height_cm": height,
            "weight_kg": weight,
            "smoking_status": smoking,
            "diabetes": yes_no(diabetes),
            "hypertension": yes_no(hypertension),
            "prior_mi": yes_no(prior_mi),
            "nyha_class": nyha,
            "nyha_class_central": nyha_central,
            "lvef_pct": lvef,
            "lvef_reader2_pct": lvef_reader2,
            "sbp_baseline": sbp,
            "dbp_baseline": dbp,
            "sbp_ambulatory": sbp_ambulatory,
            "heart_rate": heart_rate,
            "creatinine_umol_l": creatinine,
            "potassium_mmol_l": potassium,
            "hba1c_pct": hba1c,
            "ldl_mmol_l": ldl,
            "nt_probnp_pg_ml": ntprobnp,
            "troponin_ng_l": troponin,
            "mi_adjudicated": yes_no(mi),
            "smoker_baseline": smoker_baseline,
            "smoker_6m": smoker_6m,
            "adherence_pct": adherence,
            "sbp_6m": sbp_6m,
            "hf_hospitalisation": yes_no(hf_hosp),
            "n_hospital_visits": visits,
            "fu_days": fu_days,
            "death": yes_no(death),
            "last_contact_date": last_contact.strftime("%d/%m/%Y"),
            "angioedema": yes_no(angioedema),
        }
    )
    # the values before the planted problems: what the hospital record and the laboratory hold
    true = df.copy()

    # ---- planted problems (documented in the dictionary, caught by tutorial_config.json)
    df.loc[PLANTED["age_212"], "age"] = 212
    df.loc[PLANTED["age_negative"], "age"] = -3
    for row in PLANTED["height_in_metres"]:
        df.loc[row, "height_cm"] = round(df.loc[row, "height_cm"] / 100, 2)
    df.loc[PLANTED["weight_zero"], "weight_kg"] = 0.0
    df.loc[PLANTED["weight_350"], "weight_kg"] = 350.0
    for row in PLANTED["dbp_above_sbp"]:  # systolic and diastolic entered in the wrong fields
        systolic, diastolic = float(df.loc[row, "sbp_baseline"]), float(df.loc[row, "dbp_baseline"])
        df.loc[row, "sbp_baseline"], df.loc[row, "dbp_baseline"] = diastolic, systolic
    df.loc[PLANTED["lvef_155"], "lvef_pct"] = 155.0
    df.loc[PLANTED["potassium_high"], "potassium_mmol_l"] = 9.8
    df.loc[PLANTED["potassium_low"], "potassium_mmol_l"] = 0.9
    df.loc[PLANTED["fu_negative"], "fu_days"] = -12
    for row in PLANTED["contact_before_enrolment"]:
        d = pd.to_datetime(df.loc[row, "enrol_date"], dayfirst=True) - pd.Timedelta(days=400)
        df.loc[row, "last_contact_date"] = d.strftime("%d/%m/%Y")
    for row, value in zip(SEVERE_CKD, (312, 448, 365, 301, 420, 389, 336), strict=True):
        df.loc[row, "creatinine_umol_l"] = float(value)
    for row, age_value in zip(MINORS, (16, 17, 17), strict=True):
        df.loc[row, "age"] = age_value
    df.loc[list(NO_CONSENT), "consent"] = "No"
    for row, value in SEX_VARIANTS.items():
        df.loc[row, "sex"] = value
    for row in SITE_TRAILING_SPACE:
        df.loc[row, "site"] = df.loc[row, "site"] + " "

    df = _with_derived(df, true)
    df = pd.concat([df, df.iloc[list(DUPLICATED)]], ignore_index=True)  # exact duplicate rows

    # ---- 6-month laboratory visit, for the merge step (patients alive and seen at 6 months)
    seen = ~absent_6m
    ntprobnp_6m = np.exp(
        np.log(900) + 0.45 * nyha_idx - 0.02 * (lvef - 35) - 0.25 * is_int + rng.normal(0, 0.6, n)
    )
    creatinine_6m = creatinine * np.exp(rng.normal(0.02, 0.08, n))
    followup = pd.DataFrame(
        {
            "patient_id": np.array(patient_id)[seen],
            "visit_6m_date": (enrol[seen] + pd.Timedelta(days=182)).strftime("%d/%m/%Y"),
            "nt_probnp_6m_pg_ml": ntprobnp_6m.round()[seen],
            "creatinine_6m_umol_l": creatinine_6m.round()[seen],
        }
    )
    return df, followup


DICTIONARY = """# Tutorial cohort - data dictionary

Synthetic data generated by `dev_scripts/make_tutorial_cohort.py` (seed 2027): a randomised heart-failure
cohort, 1,200 patients at 4 sites (Luxembourg, Esch-sur-Alzette, Ettelbruck, Metz), enrolled from January 2021 to
early January 2024 and followed until 31/12/2024. **No real patients.** The file has 1,206 rows because 6 patients were entered twice.

Files: `tutorial_cohort.csv` (this table), `tutorial_followup_labs.csv` (6-month laboratory visit, merge on
`patient_id`), `tutorial_config.json` (plausibility rules and inclusion criteria). The step-by-step tutorial is the
handbook chapter *14 Full tutorial*.

| Column | Type | Meaning | Notes |
|---|---|---|---|
| patient_id | identifier | Pseudonymous code HF-0001 to HF-1200 | 6 patients duplicated (exact copies) |
| site | categorical | Recruiting site | 8 rows have a trailing space ("Luxembourg ") |
| enrol_date | date dd/mm/yyyy | Enrolment (randomisation) date | day-first format |
| consent | binary | Written informed consent | 14 "No": fail the inclusion criteria |
| arm | categorical | Randomised arm | Control / Intervention, 600 each |
| age | numeric (years) | Age at enrolment | errors: 212 and -3; 3 minors (16-17) fail inclusion |
| sex | categorical | Sex | F / M; 8 rows coded inconsistently (male, Female, f, M , ...) |
| height_cm | numeric (cm) | Height | 3 heights typed in metres (1.6-1.8); **MAR**: missing in 55 rows, more often in older patients: 38 not measured (no BMI either), 17 measured but not transcribed (their BMI is recorded) |
| weight_kg | numeric (kg) | Weight | errors: 0 and 350 |
| bmi_kg_m2 | numeric (kg/m2) | Body-mass index, from the hospital record | computed from the weight and the height (see below); missing in 83 rows: 38 without a height, 45 not transcribed |
| smoking_status | categorical | Never / Former / Current | |
| diabetes, hypertension, prior_mi | binary | Comorbidities | Yes / No |
| nyha_class | ordinal | NYHA class by the investigator | I < II < III < IV; class I fails inclusion |
| nyha_class_central | ordinal | NYHA class by the central committee | second rater: weighted kappa |
| lvef_pct | numeric (%) | Ejection fraction, echo reader 1 | error: 155 |
| lvef_reader2_pct | numeric (%) | Same echo read by reader 2 | reads about +1 point: ICC, Bland-Altman |
| hf_phenotype | categorical | Heart-failure phenotype (ESC 2021) | HFrEF (ejection fraction 40% or less), HFmrEF (41-49%), HFpEF (50% or more), from `lvef_pct` (see below); 34 not entered |
| sbp_baseline, dbp_baseline | numeric (mmHg) | Office blood pressure | 3 rows with diastolic > systolic (swapped) |
| sbp_ambulatory | numeric (mmHg) | 24-hour ambulatory SBP | about 6 mmHg below office SBP (white-coat effect) |
| heart_rate | numeric (bpm) | Resting heart rate | |
| creatinine_umol_l | numeric (umol/L) | Serum creatinine | > 265 fails inclusion (severe kidney disease) |
| egfr_ml_min_1_73m2 | numeric (mL/min/1.73 m2) | eGFR reported by the laboratory | CKD-EPI 2021 without race, from creatinine, age and sex (see below); not reported for the 3 minors (an adult equation) nor for 58 other patients |
| potassium_mmol_l | numeric (mmol/L) | Serum potassium | errors: 9.8 and 0.9 |
| hba1c_pct | numeric (%) | HbA1c | **MAR**: missing in 30% of non-diabetics vs 5% of diabetics |
| ldl_mmol_l | numeric (mmol/L) | LDL cholesterol | **MCAR**: about 6% missing at random |
| nt_probnp_pg_ml | numeric (pg/mL) | NT-proBNP | **MAR**: missing in 25% of NYHA I-II vs 5% of III-IV; right-skewed |
| troponin_ng_l | numeric (ng/L) | High-sensitivity troponin at admission | index test for `mi_adjudicated` |
| mi_adjudicated | binary | Myocardial infarction confirmed by adjudication | reference standard |
| smoker_baseline, smoker_6m | binary | Current smoker at baseline and at 6 months | paired: McNemar; `smoker_baseline` is *Yes* exactly when `smoking_status` is *Current* |
| adherence_pct | numeric (%) | Adherence to the study treatment | |
| sbp_6m | numeric (mmHg) | Office SBP at 6 months | paired with `sbp_baseline`; **structurally missing** if the patient died or left before 6 months |
| hf_hospitalisation | binary | Heart-failure hospitalisation within 12 months | lower with Intervention |
| n_hospital_visits | count | Hospital visits during follow-up | count model with `fu_days` as exposure |
| fu_days | numeric (days) | Follow-up time | from `enrol_date` to `last_contact_date`; error: -12 |
| death | binary | Death during follow-up | event indicator for survival analysis |
| last_contact_date | date dd/mm/yyyy | Last contact (enrolment + follow-up) | 2 dates before enrolment |
| angioedema | binary | Angioedema (rare adverse event) | small counts: Fisher's exact test |

## Planted problems

| Problem | Where | Found by |
|---|---|---|
| Impossible ages (212, -3) | 2 rows | plausibility rule *Implausible age* |
| Heights typed in metres | 3 rows | plausibility rule *Height not in centimetres* |
| Impossible weights (0, 350 kg) | 2 rows | plausibility rule *Implausible weight* |
| Diastolic above systolic | 3 rows | plausibility rule *Diastolic above systolic* |
| Ejection fraction 155% | 1 row | plausibility rule *Ejection fraction out of range* |
| Potassium 9.8 and 0.9 mmol/L | 2 rows | plausibility rule *Potassium out of range* |
| Negative follow-up | 1 row | plausibility rule *Negative follow-up* |
| Last contact before enrolment | 2 rows | plausibility rule *Last contact before enrolment* |
| HbA1c in the diabetic range without a diabetes diagnosis | a few rows | soft rule *Possible undiagnosed diabetes* (clinical review, not an error) |
| 6 duplicated patients | 6 rows | Data Quality Dashboard, Uniqueness |
| Inconsistent sex codes, site with trailing spaces | 16 rows | Data Quality Dashboard, Uniformity |
| No consent, minors, NYHA I, creatinine > 265, age > 85 | see the tutorial | inclusion criteria |

## Variables computed from others

The hospital record and the laboratory computed them from the true values, before the typing errors above: each
mistyped input disagrees with its formula. Declare them on Data > Enrich > Handle Missing Data > Derived variables
(step 3.3 of the tutorial), where each definition is checked against the rows that have all its variables.

| Variable | Definition | Agrees with | Missing values it can fill in the analysis population |
|---|---|---|---|
| bmi_kg_m2 | `weight_kg / (height_cm / 100) ** 2` | 1,101 of 1,106 rows: the 3 heights in metres and the weights 0 and 350 disagree | 34, and 10 heights by its inverse `height_cm = 100 * (weight_kg / bmi_kg_m2) ** 0.5` |
| egfr_ml_min_1_73m2 | CKD-EPI 2021 for adults (below) | 1,143 of 1,144 rows: the age 212 disagrees | 49 |
| hf_phenotype | `np.where(lvef_pct <= 40, 'HFrEF', np.where(lvef_pct <= 49, 'HFmrEF', 'HFpEF'))` | 1,171 of 1,172 rows: the ejection fraction 155 disagrees | 30 |
| smoker_baseline | `np.where(smoking_status == 'Current', 'Yes', 'No')` | 1,206 of 1,206 rows | none |

The eGFR, with `sex_clean` from step 3.2 of the tutorial (with the raw `sex`, the 4 women typed *Female*, *f*, *F*
with a trailing space and *female* disagree too):
`np.where(age >= 18, 142 * np.minimum(creatinine_umol_l / 88.4 / np.where(sex_clean == 'F', 0.7, 0.9), 1) ** np.where(sex_clean == 'F', -0.241, -0.302) * np.maximum(creatinine_umol_l / 88.4 / np.where(sex_clean == 'F', 0.7, 0.9), 1) ** -1.2 * 0.9938 ** age * np.where(sex_clean == 'F', 1.012, 1.0), np.nan)`.
The CKD-EPI 2009 equation disagrees, and the ESC 2016 cut-offs (HFrEF below 40%) only mostly agree with the
phenotype: the 39 patients at exactly 40% are HFmrEF in 2016 and HFrEF in 2021. 24 patients of the analysis
population have neither a height nor a BMI: they are left to multiple imputation. `fu_days` is the number of days
from `enrol_date` to `last_contact_date` once both are converted to dates on Data > Validate.
"""


def main() -> None:
    df, followup = make_tutorial_cohort()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    newline = chr(10)  # the same files on every operating system
    df.to_csv(CSV_PATH, index=False, lineterminator=newline)
    followup.to_csv(FOLLOWUP_PATH, index=False, lineterminator=newline)
    (OUT_DIR / "tutorial_dictionary.md").write_text(DICTIONARY, encoding="utf-8")
    print(
        f"wrote {CSV_PATH} ({len(df)} rows, {df.shape[1]} columns), {FOLLOWUP_PATH} ({len(followup)} rows)"
    )


if __name__ == "__main__":
    main()
