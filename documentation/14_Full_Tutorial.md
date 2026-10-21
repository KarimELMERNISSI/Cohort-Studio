# Full tutorial: from a raw file to a reproducible report

This tutorial walks through every step of CohortStudio with the **tutorial cohort**: a synthetic, randomised
heart-failure study run at 4 sites (Luxembourg, Esch-sur-Alzette, Ettelbruck, Metz). There are 1,200 patients
in 1,206 rows, because 6 patients were entered twice. There are 41 variables, and errors and gaps were planted
in the data on purpose. No real patient is involved. Each step says what you should see, so you can check that you are on track.

**Time needed:** about 45 minutes. **AI engine:** not needed except for the optional step 7.

## The tutorial files

| File | What it is | Where it is used |
| --- | --- | --- |
| `tutorial_cohort.csv` | The cohort (1,206 rows, 41 columns, dates in dd/mm/yyyy) | Step 1 |
| `tutorial_config.json` | The study configuration: 8 clinical anomaly rules and 4 inclusion criteria | Step 2 |
| `tutorial_followup_labs.csv` | The 6-month laboratory visit (1,017 patients), to merge on `patient_id` | Step 3 |
| `tutorial_dictionary.md` | The data dictionary and the list of planted problems | Throughout |

Download them from **Start > Tutorial cohort > Tutorial files**. They are also in `assets/tutorial/` in the repository.

---

## Step 1 - Load the cohort and its configuration

1. On **Start**, in the *Tutorial cohort* card, click **Load the tutorial cohort**. This loads the cohort **and**
   its study configuration.
2. The sidebar shows `tutorial_cohort`, **1,206 rows · 41 columns**. On the workflow map, *Load data* is ticked and
   *Validate* is the suggested next step.

*Doing it by hand instead:* go to **Data > Import**, upload `tutorial_cohort.csv` and tick *day first* (the dates are
dd/mm/yyyy). Then, on **Data > Validate**, choose *Load Existing Config* in the sidebar and upload
`tutorial_config.json`.

---

## Step 2 - Validate the data

Open **Data > Validate**.

### 2.1 Data Quality Dashboard

The global score is about **97/100** (*excellent*). Hover the segments of the ring, then click the bars of
**Where the score loses points**:

- **Uniqueness 99.5%**: 6 rows are exact duplicates of another row.
- **Clinical Validity 98.67%**: 16 of the 1,206 rows break at least one anomaly rule of the configuration.
- **Uniformity**: the `sex` column mixes codes (*male*, *Female*, *f*, *M* with a trailing space, ...) and 8 `site`
  values end with a space. You will clean both in step 3. Until then, the lists of groups on Analyze show such a
  space as ␣ (*Luxembourg␣*), and warn that these groups differ only by spaces or capitals.
- **Completeness**: the cells missing in `hba1c_pct`, `nt_probnp_pg_ml`, `sbp_6m`, `ldl_mmol_l` and `height_cm`, and in
  three variables computed from others: `bmi_kg_m2`, `egfr_ml_min_1_73m2` and `hf_phenotype` (see 2.5).

### 2.2 Clinical anomalies

1. Open the **Handle Clinical Anomalies** tab. Under *Declared Anomaly Criteria* you will find the 8 rules of the
   configuration. Six are expression rules, such as `df['dbp_baseline'] >= df['sbp_baseline']`. Two are numeric
   range rules, on `lvef_pct` and `potassium_mmol_l`: with *exclude*, values outside the range are flagged.
2. Expand *Apply Anomaly Detection based on Domain-Expert Criteria* and click **Run Anomaly Detection**.
   **16 rows** are flagged:

| Rule | Rows |
| --- | --- |
| Implausible age (212, -3) | 2 |
| Height not in centimetres (typed in metres) | 3 |
| Implausible weight (0, 350 kg) | 2 |
| Diastolic above systolic (fields swapped) | 3 |
| `lvef_pct` outside 5-85% (155) | 1 |
| `potassium_mmol_l` outside 2.5-7.0 (9.8, 0.9) | 2 |
| Negative follow-up | 1 |
| Last contact before enrolment | 2 |

3. In *Results*, click **Exclude these anomalies from the analysis population**. Criteria are filters: the 8 rules
   are now ticked in the sidebar (*Population*), and every analysis uses the **1,190 rows** they keep. The dataset
   itself still has its 1,206 rows, and you can untick a rule at any time. In a real study you would correct these
   values at the source; here, excluding them keeps the tutorial simple.
4. **Add your own rule.** In *New Anomaly Criterion*, create an expression rule named *Possible undiagnosed
   diabetes*: `(df['diabetes'] == 'No') & (df['hba1c_pct'] >= 6.5)`. Run the detection again: **16 patients**
   are flagged. These are not data-entry errors but patients to review clinically, so **leave this rule unticked**
   in the sidebar.

### 2.3 Inclusion criteria

1. Open **Handle Inclusion Criteria**. It lists the 4 criteria: written consent; `age` from 18 to 85 (a numeric
   rule with *include*); NYHA class II to IV; creatinine of 265 umol/L or less.
2. Expand *Apply Data Validation based on Inclusion Criteria* and click **Run Inclusion Criteria Check**. On the
   1,190 rows, **928 meet every criterion** and 262 do not:

| Criterion not met | Rows |
| --- | --- |
| Written consent | 14 |
| Age 18 to 85 (3 minors, older patients) | 71 |
| Symptomatic heart failure (NYHA II-IV) | 183 |
| No severe kidney disease | 8 |

A patient can fail several criteria, so the rows do not add up to 262.

3. Click **Restrict the analysis population to these criteria**. With the anomaly rules, 12 criteria are ticked and
   the analysis set has **928 patients**: 462 Control and 466 Intervention. The sidebar shows *928 of 1,206 rows*,
   and every analysis page shows the same population in a banner.
4. Open the **Analysis population** tab: it shows the participant flow (1,206 rows, 16 excluded by the anomaly
   rules, 1,190, 262 excluded by the inclusion criteria, 928), what each criterion excludes, and the excluded rows
   with their reasons. *Keep only these rows* would delete the excluded rows from the dataset; you do not need it,
   because the criteria already filter every analysis.

### 2.4 Outliers

In **Handle Outliers**, run the IQR method on `nt_probnp_pg_ml`. Many high values are flagged because NT-proBNP is
strongly right-skewed. These are real patients, not errors: **do not exclude them** from the analysis population
(the button would add them as an *outliers* criterion) and analyse NT-proBNP on a log scale.

### 2.5 Missing data

The dashboard's missing-data views show these documented patterns:

| Variable | Missing | Pattern |
| --- | --- | --- |
| `ldl_mmol_l` | about 6% | completely at random (MCAR) |
| `hba1c_pct` | about 22% | depends on `diabetes`: rarely measured in patients without diabetes (MAR) |
| `nt_probnp_pg_ml` | about 15% | depends on NYHA class: less often measured in mild disease (MAR) |
| `sbp_6m` | about 16% | **structural**: the patient died or left before the 6-month visit |
| `height_cm` | about 5% | depends on `age`: less often measured in older patients (MAR); a few heights were measured but not transcribed |
| `bmi_kg_m2`, `egfr_ml_min_1_73m2`, `hf_phenotype` | 3 to 7% | not transcribed, no BMI without a height, no eGFR under 18: each is **computed from other variables**, which can fill most of them (step 3.3) |

Finally, save a version with **Save** in the sidebar. Validate is now ticked on the Start page.

---

## Step 3 - Enrich the cohort

Open **Data > Enrich**.

### 3.1 Merge the 6-month laboratory visit

Changes made on this page go into the **Enrich draft**: a banner at the top of the page lists them, and they reach
the analyses when you click **Apply to the dataset** (each becomes one step of the trace). If the data change while
a draft is open, the banner offers to redo the changes on the new data instead.

In **Import External Data**, upload `tutorial_followup_labs.csv`. Choose to add *Columns*, the *left* join and a
*Common Row Identifier* on `patient_id`. This adds `visit_6m_date`, `nt_probnp_6m_pg_ml` and
`creatinine_6m_umol_l`. **778 of the 928 patients** of the analysis population have a 6-month visit; the others
keep empty cells.

### 3.2 Derive variables

In **Create New Variables > Variable Computation**, add these one at a time. Formulas use column names directly:
as you type, the editor offers the matching column names, and **Compute & Preview** shows the new variable next to
its source columns, with its statistics on the analysis population, before **Confirm & Add Variable to Dataset**.

| New variable | Formula |
| --- | --- |
| `site_clean` | `site.str.strip()` |
| `sex_clean` | `np.where(sex.str.strip().str.lower().str.startswith('f'), 'F', 'M')` |
| `egfr_6m` (CKD-EPI 2021, at the 6-month visit) | `np.where(age >= 18, 142 * np.minimum(creatinine_6m_umol_l / 88.4 / np.where(sex_clean == 'F', 0.7, 0.9), 1) ** np.where(sex_clean == 'F', -0.241, -0.302) * np.maximum(creatinine_6m_umol_l / 88.4 / np.where(sex_clean == 'F', 0.7, 0.9), 1) ** -1.2 * 0.9938 ** age * np.where(sex_clean == 'F', 1.012, 1.0), np.nan)` |

The BMI and the eGFR at enrolment are already in the file, with gaps: step 3.3 declares how they are computed rather
than computing them a second time.

With the AI engine on, you can also start from **AI Variable Suggestions (RAG)** above the editor: search for
*eGFR*, read the formula and its source, and **Apply to Editor** puts it in the editor, to check against the formula
above before you compute it.

`site_clean` has the 4 sites again, and `sex_clean` only *F* and *M*. `egfr_6m` averages about 68 mL/min/1.73 m2 in
the 778 patients with a 6-month visit; `np.where(age >= 18, ..., np.nan)` keeps this adult equation away from
minors. Apply the draft. On the Start page, *Enrich* is ticked and *Validate* turns amber with **run again**: the
data changed, so check the quality once more (step 2.1).

### 3.3 Missing values

In **Handle Missing Data**, start with the **Derived variables** tab. Three variables of the file are computed from
others: `bmi_kg_m2` from the weight and the height, `egfr_ml_min_1_73m2` (CKD-EPI 2021) from creatinine, age and sex,
and `hf_phenotype` (ESC 2021) from the ejection fraction. Declaring them is optional. The app then checks each formula
against the rows where the variable and all its variables are recorded, and you know which gaps a formula can fill
rather than an imputation. Under **Add a definition**, add:

| Variable | Formula |
| --- | --- |
| `bmi_kg_m2` | `weight_kg / (height_cm / 100) ** 2` |
| `egfr_ml_min_1_73m2` | `np.where(age >= 18, 142 * np.minimum(creatinine_umol_l / 88.4 / np.where(sex_clean == 'F', 0.7, 0.9), 1) ** np.where(sex_clean == 'F', -0.241, -0.302) * np.maximum(creatinine_umol_l / 88.4 / np.where(sex_clean == 'F', 0.7, 0.9), 1) ** -1.2 * 0.9938 ** age * np.where(sex_clean == 'F', 1.012, 1.0), np.nan)` |
| `hf_phenotype` | `np.where(lvef_pct <= 40, 'HFrEF', np.where(lvef_pct <= 49, 'HFmrEF', 'HFpEF'))` |

- `bmi_kg_m2` **agrees**: it gives the recorded value in **1,101 of 1,106 rows** (±0.05). Open *The 5 rows that
  disagree*: they are the 3 heights typed in metres and the weights of 0 and 350 kg found in step 2.2. Then **Add
  an inverse** for `height_cm`, `100 * (weight_kg / bmi_kg_m2) ** 0.5`: it recovers a height from the BMI and the
  weight, and agrees too.
- `egfr_ml_min_1_73m2` agrees in **1,143 of 1,144 rows**: the one that disagrees has the age 212. The
  `np.where(age >= 18, ...)` keeps the adult equation away from the 3 minors, whose eGFR the laboratory did not
  report. With `sex` instead of `sex_clean`, the 4 women typed *Female*, *f*, *F* with a trailing space and *female*
  disagree too: clean the codes first.
- `hf_phenotype` agrees in **1,171 of 1,172 rows**: the one that disagrees is the ejection fraction of 155%. Try the
  ESC 2016 cut-off instead, `lvef_pct < 40` for HFrEF: it only **mostly agrees**, because the 39 patients at
  exactly 40% are HFmrEF in 2016 and HFrEF in 2021.

The line of each definition says how many missing values of the analysis population its formula can compute:
**34** BMIs, **49** eGFRs and **30** phenotypes. Under **2. Fill exactly**, the table adds the **10** heights the
inverse recovers; each fill is counted as observed (exact) and rounded as its variable is written. Click **Fill 123
values**: four steps join the draft, each recorded with the definition it used, so that a replay computes the same
values. Apply the draft. 24 patients have neither a height nor a BMI: an imputation below gives them a height, and
their BMI is then computed from it. With the AI engine on, **Find them with the AI engine**
proposes definitions from the column names and their typical values, each checked in the same way: if it proposes
both CKD-EPI equations, only the 2021 one agrees. The definitions are saved with the criteria, in the configuration
downloaded from Data > Validate.

Then:

- `ldl_mmol_l` is missing completely at random: single imputation (KNN or MICE) is acceptable for exploration.
  The imputation is fitted on, and fills, the 928 rows of the analysis population only; the excluded patients
  keep their missing values. Every result computed on the filled values then shows a warning, because the
  analyses treat them as observed. With the derived variables declared, keep **Compute the declared derived
  variables' missing values from the imputed values** ticked: the BMI, the one still missing, is kept out of the
  imputation, and its 24 missing values are computed from the imputed heights, single-imputed as those heights are. If you plan inferential analyses, skip
  this and use the multiple imputation below for every incomplete variable.
- **Do not impute `sbp_6m`.** Its values are missing because the patient died or left. Analyse the patients who
  have the measurement and say so.
- For inferential models with MAR variables (`hba1c_pct`, `nt_probnp_pg_ml`), use **multiple imputation** on
  the **Multiple imputation (for inference)** tab:
  1. Step 1 shows the 928 rows of the analysis population, the share of incomplete rows and the suggested
     number of imputations.
  2. Step 2 keeps the variables of the planned analyses in the model, outcomes included (`death`, `fu_days`,
     `hf_hospitalisation`): leaving an outcome out would bias its associations towards zero. Set the method of
     `sbp_6m` to *do not impute*: its values are missing because the patient died or left. Keep **Compute the
     declared derived variables in each imputed dataset from its imputed values** ticked: `bmi_kg_m2` leaves the
     methods table and each imputed dataset computes its 24 missing BMIs from its own imputed heights, instead of
     imputing a BMI that would not be the weight over the height squared. Once filled exactly,
     `egfr_ml_min_1_73m2` and `hf_phenotype` have no missing value left in the analysis population.
  3. In step 3, keep the suggested m (the percentage of incomplete rows: about half of them miss at least one
     value), tick the planned Cox model with time `fu_days` and event `death`, and use several processor
     cores.
  4. Run it. In step 4 the chain traces should mix without a trend. The imputed HbA1c values are lower than the
     observed ones: HbA1c is mostly missing in patients without diabetes, whose values are lower.
  5. Step 5 saves the imputation with a version, exports it for R or Stata, and writes a methods paragraph.

  The dataset itself is unchanged: the analyses use the imputed datasets when you choose **Analysis data:
  Imputed** in the sidebar (step 6.1).

---

## Step 4 - Explore

Open **Explore > Statistics & correlations**. The table and plots show the descriptive statistics by variable
type, the distribution of each variable and the correlation network. Expect, for example, ejection fraction
falling as NYHA class rises, and NT-proBNP rising with it. Open **Explore > Visualize** to compare `sbp_baseline`
and `sbp_6m` across arms.

---

## Step 5 - Hypothesis tests and epidemiology

Open **Analyze > Hypothesis tests & epidemiology**. Leave **Guidance** on: before each run the page shows the
test it will use, and why.

### 5.1 Compare the arms

On the **Analysis** tab:

1. Set **Select Grouping Variable** to `arm` and **Select Variables to Test** to `sbp_6m`, `hf_hospitalisation`,
   `n_hospital_visits` and `age`.
2. Read **Before you run: the plan**. Each numeric variable gets Welch's t-test, `hf_hospitalisation` gets a
   chi-square test, and Holm is the suggested correction for 4 tests.
3. Click **Run analysis**.
4. In the detailed view, choose `hf_hospitalisation`. Set the reference group to *Control* and the event level
   to *Yes*.

| Measure (Intervention vs Control) | Value |
| --- | --- |
| Risk of HF hospitalisation | 27.7% vs 37.4% |
| Risk ratio | 0.74 |
| Odds ratio | 0.64 |
| Risk difference | -9.8 percentage points (NNT about 10) |
| p-value (chi-square) | 0.0015 |

### 5.2 Adjust for the baseline (ANCOVA)

Switch **Analysis type** to *Multivariate (ANCOVA/Regression)*. Set the target to `sbp_6m`, the group to `arm` and
the covariate to `sbp_baseline`. The *Model check* reports enough observations per parameter.

**Result:** the adjusted difference is **-5.9 mmHg** (95% CI -7.2 to -4.7) for Intervention. Adjusting for the
baseline makes this estimate more precise than the crude difference of 128.2 vs 134.8 mmHg.

### 5.3 Paired data and stratified analysis

On the **Paired, rates & stratified** tab:

- **Paired t-test**, `sbp_baseline` vs `sbp_6m`: the mean change is **-5.0 mmHg** (p < 0.001).
- **McNemar**, `smoker_baseline` vs `smoker_6m`: **58 patients quit and none started** (p < 0.001).
- **Mantel-Haenszel**, exposure `arm`, outcome `hf_hospitalisation`, stratum **`site_clean`**: the pooled odds
  ratio is about **0.64**. Breslow-Day p = 0.33, so the effect is consistent across sites. If you choose `site`
  instead, the uncleaned names give 7 strata: that is why you created `site_clean`.

### 5.4 Diagnostic accuracy and agreement

On the **Diagnostic accuracy & agreement** tab:

- **Diagnostic accuracy:** reference standard `mi_adjudicated`, index test `troponin_ng_l`. **AUC about 0.93**.
  Read the sensitivity and specificity at the suggested cut-off, and note that the cut-off is optimistic in the
  same data.
- **Bland-Altman:** office `sbp_baseline` against `sbp_ambulatory`. Office readings are about **6 mmHg higher**
  (the white-coat effect), with limits of agreement of about -8 to +20 mmHg.
- **Kappa:** `nyha_class` (investigator) against `nyha_class_central` (central committee). Kappa is about
  **0.71**, and the linearly weighted kappa about **0.76** (substantial agreement).
- **ICC:** `lvef_pct` and `lvef_reader2_pct` (two echo readers, reader 2 about 1 point higher).

### 5.5 Power

On the **Power & Sample Size** tab, choose *Chi-Square* with *Expected event proportions* `0.33, 0.28`. That is
roughly the observed 2-year mortality by arm. About **2,660 patients** would be needed for 80% power. The study
has 928, so the mortality question in step 6 was always going to be inconclusive.

---

## Step 6 - Survival and regression

Open **Analyze > Survival & regression**.

- **Kaplan-Meier:** time `fu_days`, event `death`, group `arm`. The curves separate a little; the log-rank
  **p = 0.25**.
- **Cox model:** covariate `arm`. **HR about 0.88 (95% CI 0.70 to 1.10)**. This result is inconclusive, not
  "no effect": the interval includes both a 30% reduction and a 10% increase (see the power calculation in 5.5).
- **Logistic regression:** outcome `hf_hospitalisation` (event *Yes*), covariates `arm`, `age`, `diabetes` and
  `nyha_class`. You get adjusted odds ratios.
- **Count model:** count `n_hospital_visits`, exposure (person-time) `fu_days`, covariate `arm`. The incidence
  rate ratio is about **0.80**.

### 6.1 With multiple imputation

In the sidebar, set **Analysis data** to *Imputed*. The banner says that the analyses pool the imputed datasets
of the 928 rows. Fit the Cox model again with `arm`, `age`, `diabetes` and `hba1c_pct`:

- each hazard ratio comes with its 95% CI, the Barnard-Rubin df, the fraction of missing information (FMI) and
  the relative increase in variance (RIV); `hba1c_pct` has the largest FMI;
- *Complete-case analysis, for comparison* fits the same model on the patients with every value: about a
  fifth fewer patients and wider intervals.

On **Hypothesis tests & epidemiology**, compare `hba1c_pct` between the arms: the Welch t-test is pooled, and
the Mann-Whitney test is listed as not poolable, with the reason. Untick a criterion in the sidebar: the pages
say that the imputation no longer fits the population and analyse the observed data until you run it again.

---

## Step 7 - Knowledge (optional, needs the AI engine)

Configure the AI engine in **Settings > AI engine**. Then:

- **Knowledge > Variable map** builds a taxonomy of the variables, with definitions, units and derived
  formulas. eGFR from creatinine is one of them.
- **Knowledge > Documents** lets you add guidelines or papers about heart failure, chat with them and extract a
  knowledge graph.

---

## Step 8 - Reproduce and report

Open **Reproduce > Versions & traces**. The trace of the session is marked *integrity verified*: its steps are
chained by hashes, so a later modification would show. It lists every step, in order:

1. the load;
2. the analysis population (the 12 ticked criteria and the counts of each step), recorded when a result or a
   version first used it;
3. the merge;
4. the derived variables;
5. any imputation, with the population it was fitted on;
6. the multiple imputation, with its settings and a fingerprint of the imputed cells. A full replay re-runs it
   and checks that it finds the same cells, and the Word report gives its methods paragraph.

Replay a step: the engine recomputes the data and checks it against the fingerprint recorded for every step, and
says when the replayed table is exactly the one you analysed.

Then build the study dossier. Under the Cox model of step 6 (and any other result you want to report), click
**Add to the study dossier**. Open **Reproduce > Study dossier**: the participant flow goes from the 1,206
records of `tutorial_cohort.csv` to 1,190 after the clinical anomalies (16 excluded) and 928 after the inclusion
criteria (262 not meeting them), with a box per analysis below. Click the *Not meeting: study inclusion* box to
see how many patients each criterion excluded first. **Download the Word dossier**: it has the provenance, the
flow figure, the criteria, the missing data and the multiple imputation with its methods paragraph, your
results and the references. Download the **Word report**: it documents the whole
pipeline with what each step did to the table. On **Start**, every core step is now ticked.

---

## What the tutorial cohort contains

The data dictionary (`tutorial_dictionary.md`, also in **Start > Tutorial files**) describes every column and
every planted problem. The cohort is generated by `dev_scripts/make_tutorial_cohort.py` (seed 2027). The numbers
in this chapter are checked by `tests/test_tutorial_pack.py`, so they stay true if the generator changes.
