# Changelog

All notable changes to CohortStudio (named Cohort Monitoring Package until 2.3) are documented here.

---

## [Unreleased]

### Packaging, deployment and CI (follow-up programme M4, in progress)

- **Optional dependency groups** (`pyproject.toml`). The statistics engine, the data pipeline and the analysis
  pages need none of them; `uv sync --extra all` installs everything (the Docker image, `requirements.txt`).
  - `ai`: LLM connectors, document library and vector databases.
  - `ml`: MissForest (LightGBM), UMAP, FAMD (prince).
  - `survival`: the proportional-hazards test and concordance (lifelines).
  - `all`: the three above.
  - Without an extra, its methods are not offered, with the install command (`utils/optional.py`). The AI pages
    and Settings say that the AI engine is not installed, and LightGBM is imported only when MissForest runs.
  - The unused `langchain` and `langchain-community` packages are removed, with LangGraph, SQLAlchemy and about
    15 packages they pulled in.
  - Verified in an environment without any extra: the whole suite passes, with the AI tests skipped.
- **First-run administrator** (`manage/iam/setup_token.py`). A fresh installation without `CMP_ADMIN_PASSWORD`
  writes a one-time setup token to the server log. The login page then creates the first administrator with it.
  Only a hash of the token is stored, and it stops working once used or when an administrator exists.
  - The Docker image runs the bootstrap before the server (`python manage_users.py bootstrap`), so the token is in
    `docker compose logs cohortstudio` from the start.
  - A page load never replaces a pending token. `python manage_users.py setup-token` issues a new one, for a lost
    log.

### Fixed: statistics of a new variable dominated by entry errors (Enrich > Create New Variables)

- The preview statistics of a new variable used every row with the mean and SD only. A few entry errors made
  them meaningless: on the tutorial cohort, BMI had mean 660 and maximum 276,715 because three heights are
  entered in metres.
- The preview now uses the analysis population. It adds the median, quartiles, missing and infinite values
  (`enrich.formulas.numeric_summary`).
- It lists the values beyond 3 interquartile ranges from the quartiles, with their source values
  (`far_outside`). Without ticked criteria it suggests excluding entry errors with the clinical-anomaly criteria.
  The tutorial's BMI then reads mean 28.6, SD 5.1, range 17.0 to 45.7.

### Fixed: a failed parallel imputation no longer stops the multiple imputation

- When the separate process that runs the chains in parallel fails, the chains now run in one process, with a
  warning. This happened intermittently under load in a full test run. The result is the same, since each chain
  has its own random stream.

### Participant flow and study dossier (follow-up programme M3, awaiting the owner's review)

- **Participant flow model** (`core/attrition.py`, STROBE item 13). The flow runs from the records read in the
  file (malformed lines skipped) through the rows removed or added by recorded steps and the rows excluded by
  each family of criteria, to the analysis population and the N of each reported analysis (or the imputed
  datasets it pooled).
  - Each exclusion lists the criteria that excluded the rows first, with their definitions.
  - Every box is checked to add up.
- **Participant flow diagram** on Validate > Analysis population (`ui/components/flow.py`). Clicking a box, or
  choosing it in the menu below, shows its details: the criteria with the rows each excluded first, and their
  definitions. The same diagram is drawn as a picture for the Word dossier (`manage/flow_figure.py`).
- **Results go into a study dossier** (`ui/results.py`, Reproduce > Study dossier):
  - Every stored result has an **Add to the study dossier** button under it. Each set of settings is a separate
    item, kept with the stamp of what it was computed on.
  - The new page shows the participant flow with a box for each analysis in the dossier (N analysed, or the
    imputed datasets pooled), and lists the results.
  - A result computed on other data or another population is marked stale.
  - The observed group comparison is now kept in the result store too, so that it can be added.
- **Word study dossier** (`manage/dossier.py`, `manage/report_tables.py`), downloaded from Reproduce > Study
  dossier. Its sections:
  1. Provenance: dataset, source file and hash, parse options, data fingerprint, trace with its integrity,
     software versions.
  2. Analysis population: the participant flow figure, the exclusions with the criteria behind them, and every
     criterion's definition.
  3. Data handling: the recorded steps that changed the data, with their formulas.
  4. Missing data: missing values per variable, values filled by single imputation, and the multiple imputation
     with its settings, methods per variable, status and methods paragraph.
  5. Results: each result with its settings, what it was computed on, its tables and, for survival analyses, the
     Kaplan-Meier figure.
  6. References: those of the methods used, plus STROBE and, with multiple imputation, Rubin, Barnard-Rubin,
     van Buuren, White and Sterne.
  - Results computed on other data or another population are left out and listed. They can be kept instead,
    marked as not current.
  - The dossier says it supports STROBE item 13 and the Sterne et al. (2009) items, without claiming compliance.
- The Word trace report summarises long nested parameters (row-id ranges) instead of printing them whole.
- Documentation: the README, handbook chapters 2 (the flow diagram) and 8 (the Study dossier page), the tutorial
  (step 8 builds the dossier), the documentation site and the audit report (M3 done, pending the owner's review).

### Multiple imputation end-to-end (follow-up programme M2, awaiting the owner's review)

- **Pooling layer** (`biostats/mi`): Rubin's rules with the formulas of R mice, including Barnard-Rubin
  degrees of freedom from the complete-data df (as mice 3.19: imputations that agree give infinite df). Each
  term reports riv, lambda and FMI. Tables of regression terms come with a D1 Wald test of
  several terms (a factor; Reiter's small-sample df) and a D2 combination of chi-square statistics. R^2 is
  pooled with Harel's method. Ratios, proportions, correlations and survival probabilities are pooled on the
  log, logit, Fisher z and cloglog scales and reported back. Analyses that cannot be pooled (rank tests,
  bootstrap intervals, ICC, Bland-Altman limits, standardised variables) are listed with the reason and the
  alternative.
- `rubin_pool` keeps its signature. The pooled means of the Enrich MI panel now use the complete-data df n - 1.
- **R parity harness**: `tests/reference/r/make_mi_fixtures.R` computes reference values with R mice and
  mitml: pool.scalar cases, and linear, logistic, Poisson and Cox models pooled on datasets that mice imputed,
  with D1, D2 and pooled R2. The manual workflow "R reference fixtures" runs it and pushes the values to the
  `r-fixtures` branch under the repository owner's identity. `test_mi_r_parity.py` fits the same models on
  the same completed datasets and compares every pooled quantity.
  - With R 4.6.1, mice 3.19.0 and mitml 0.4.5 every quantity matches: 1e-10 for pooling, 1e-8 for the
    linear model, 1e-6 for the logistic model, 1e-5 for Poisson and Cox. R's glm convergence tolerance limits
    the last two.
  - The comparison corrected three points to match R: no floor on lambda in the Barnard-Rubin df; mitml's
    full form of Reiter's df for D1; events minus parameters as the complete-data df of a Cox model.
- **What pooling needs from each analysis** (additive; existing callers unchanged):
  - `biostats/design.py` builds the design matrices of the regression and Cox models. Factor levels can be
    fixed from the observed data, so every imputed dataset estimates the same terms. A level absent from
    the rows or a value outside the levels is named in the error (instead of "singular matrix").
  - Logistic, Poisson, negative-binomial and Cox results carry their coefficients, covariance matrix and
    complete-data df (n - p, as R's df.residual and mice).
  - t-test results carry the standard error of their estimate. The 2x2 analysis keeps the SEs of the log
    odds ratio, the log risk ratio and the risk difference. Kaplan-Meier curves keep their Greenwood SE.
    `MIInfo` describes a pooled result.
- **Linear regression and ANCOVA** (`biostats/linear.py`): OLS with the coefficient table, covariance and
  residual df, a Type II table (Wald F of each covariate, equal to the extra-sum-of-squares test in a model
  without interactions) with partial eta squared, and adjusted (least-squares) means with their SE. It
  reproduces statsmodels' `ols`, `anova_lm(typ=2)` and `get_prediction`. The ANCOVA page switches to it when
  it gains its multiple-imputation mode.
- DeLong AUC: placement values by sorted search (identical, ties included): 25 ms at 50,000 rows instead of
  seconds, so the AUC can be computed on every imputed dataset.
- **Analyses over imputed datasets** (`biostats/mi/data.py`, `analyze.py`, `adapters.py`):
  - `MIData` stores a multiply imputed dataset compactly: the observed table plus the imputed cells per column.
    It rebuilds any completed dataset, and exports and imports the mice long layout (`.imp`, `.id`) that R
    and Stata read.
  - `analyze_mi(fn, mi)` runs one analysis on every completed dataset and pools it through an adapter for
    its result type:
    - linear regression and ANCOVA: D1 test of each covariate (Type II), Harel R^2, pooled adjusted means;
    - logistic and count regression and Cox: ratios reported back from the log scale, D1 test of each
      factor;
    - t-tests: a pooled test result with its MI information;
    - 2x2 tables: odds ratio, risk ratio and risk difference, D2 on the chi-square;
    - Kaplan-Meier: curves pooled on the cloglog scale, median read from the pooled curve, D2 on the
      log-rank.
  - A failure names the imputed dataset. Rank tests are refused with the reason and the alternative.
    Warnings are counted across imputations, and the complete-case analysis is kept for comparison. A time
    budget gives a result labelled as a preview.
- **Imputation engine** (`biostats/mi/engine.py`, `methods.py`, `spec.py`): chained equations like R mice.
  - Methods: `pmm` (Bayesian regression draw, type-1 matching, 5 donors), `norm`, `logreg` (augmented against
    perfect prediction, White, Daniel & Royston 2010), `polyreg` (augmented, bootstrap of the observed rows
    for parameter uncertainty) and `rf` (random-forest donors, Doove et al. 2014). miceforest was evaluated
    and not adopted: its imputations carry no parameter uncertainty and its inference is unvalidated.
  - The m chains are independent, one random stream each (`SeedSequence(seed, (stratum, chain))`): results
    are reproducible, do not depend on m, and are identical when the chains run in parallel processes
    (`workers`). 50,000 rows x 20 variables, m = 20, 10 iterations: 2.7 minutes with 10 processes.
  - Constant, collinear and dependent predictors are left out before each fit (scale-free).
  - Options: imputation within the levels of a variable (`by`); for a Cox analysis, the event indicator and
    the Nelson-Aalen H(T) replace the follow-up time as predictors (White & Royston 2009). Chain traces come
    with an indicative R-hat. `suggest_m` gives the percentage of incomplete cases (White, Royston & Wood
    2011).
  - A `slow` test suite (200 replications under MAR, `pytest -m slow`) checks that the pooled estimate is
    unbiased with 95% coverage while the complete-case estimate is biased. It is excluded from the default
    run and from CI.
- **Imputation bound to its data, saved with versions** (`biostats/mi/binding.py`, `manage/mi_store.py`,
  `ui/state.py`):
  - An imputation keeps a hash of the model columns of every row it was fitted on. A change elsewhere in the
    table (a new derived variable) leaves it valid.
  - Against the current analysis population it is fresh, an exact union of whole strata (`by`), a strict
    subset (usable only with a caveat, or run again) or stale (rows not imputed, observed values changed),
    with the reason.
  - `AppState` keeps it across changes of the same dataset (`set_mi`, `mi_status`, `analysis_mi`) and drops
    it when another dataset is loaded.
  - Saving a version saves a still-valid imputation with it (cells, row hashes, binding and diagnostics in
    the `mi/` folder). Loading the version restores it.
  - `MIData.flong()` gives the Stata "flong" layout next to the mice long layout.
- **Imputation replayed and checked** (trace step `mi_impute`):
  - Keeping an imputation records the specification, the population and a fingerprint of the imputed cells
    in the trace, with the engine and library versions.
  - The replay re-evaluates the population, re-runs the engine (it is deterministic) and checks that it
    finds exactly the same cells; a difference is reported, with any library version that changed since.
  - The adopted single imputation of the former MI panel (`multiple_imputation`) stays reported as not
    recomputable.
- **Guided multiple-imputation panel** (Enrich > Handle missing data > Multiple imputation, replaces the former
  expander that pooled only means):
  1. Population and missingness: rows, share of incomplete rows, suggested m, missingness per variable, the MAR
     assumption.
  2. Model: variables (identifiers, dates, free text and categories with more than 20 levels left out by
     default), the outcome of the planned analyses kept in, and a method per incomplete variable.
  3. Settings: m, iterations, seed, strata, a planned Cox model, several processor cores.
  4. Diagnostics: chain traces, R-hat, observed against imputed values, predictors left out.
  5. Use, save, export: status against the analysis population, saving a version with the imputation, mice long
     and Stata flong files, and a methods paragraph following Sterne et al. (2009).
  - The single imputation of "Global imputation" remains, for exploration.
  - Parallel imputation runs in a separate Python process: worker processes no longer re-run the Streamlit page
    script, which is `__main__` inside the application.
  - The event indicator of survival analyses reads text (yes, event, died) everywhere (`event_indicator`).
- **Pooled analyses on the Survival & regression page** (`ui/components/mi_mode.py`):
  - Once an imputation exists, the sidebar offers "Analysis data: Observed | Imputed (m = ...)" under the
    population, with the status of the imputation.
  - With imputed data every analysis of the page runs on the m datasets of the analysis population and is
    pooled: Kaplan-Meier curves (cloglog scale, median read from the pooled curve, log-rank by D2), and Cox,
    logistic and count models (ratio, 95% CI, df, FMI and RIV per term, a D1 test per factor). The
    complete-case analysis is shown for comparison. The proportional-hazards test runs on the first 5 imputed
    datasets.
  - An imputation that is out of date, or fitted on more rows than the current population, is never used
    silently: the page says why and analyses the observed data. The second case can be used with a caveat.
- Fixed: a negative follow-up time made the log-rank test and the Cox model fail with a length error when a
  group or covariates were given. Such rows are now left out of every survival analysis, with a warning.
- **Pooled analyses on the Analyze page** (`app_pages/analysis_pooled.py`):
  - Group comparison with imputed data, per variable:
    - numeric, two groups: Welch t-test with the mean difference pooled;
    - numeric, three or more groups: a linear model with pooled group means and a D1 test of the groups;
    - binary variable: odds ratio, risk ratio and risk difference, with a Wald test on the pooled log odds
      ratio and the chi-square by D2;
    - other categorical variables: the chi-square test of independence by D2 (`chi_square_independence`).
  - The multiple-testing correction applies to the pooled p-values. A pooled test card shows m, df, FMI and
    RIV.
  - ANCOVA with imputed data: Type II tests by D1, pooled adjusted means and Harel's R2.
  - Rank tests and Fisher's exact test are listed as not poolable, with the reason and the alternative.
    Transformations are not offered with imputed data: z-scores and min-max scaling differ between imputed
    datasets.
  - The other tabs say that they analyse the observed data.
- The observed ANCOVA runs on `biostats.linear`: the same Type II table and partial eta squared as before, plus
  the adjusted (least-squares) means with their CI. Its result stays until the data, the population or the
  settings change.
- t-tests can skip their standardised effect sizes (`effect_sizes=False`). Their bootstrap took about 2 s per
  call, too slow to repeat on every imputed dataset.
- **Paired and one-sample t-tests pooled** (Analyze > Paired, rates & stratified): with imputed data, the mean
  paired difference and the difference from a reference value are pooled. The Wilcoxon tests say why they cannot
  be pooled.
  - McNemar, Mantel-Haenszel, diagnostic accuracy, Bland-Altman, kappa and the ICC say that they analyse the
    observed data.
  - Diagnostic accuracy and kappa are not pooled yet. Bland-Altman limits and the ICC have no pooling rule.
- **Values filled by single imputation are flagged** (`core/session.py`, `ui/results.py`):
  - The session remembers which cells the global and formula imputations filled (row ids per column). Saved
    versions keep them in their lineage file, and a full replay finds them again.
  - Every result that reads such values shows a warning under it: the analysis treats them as observed, so its
    standard errors and p-values are too small.
  - The multiple-imputation panel flags them too, since it would treat them as observed. The analysis-data
    selector then reads "Observed + single-imputed".
- **Documentation**: the README, the About page, the workflow map, the documentation site and the handbook
  describe multiple imputation as it now works:
  - chapter 3: the multiple-imputation panel;
  - chapter 5: what each page pools, how to read df, FMI, RIV, D1 and D2, and what is not pooled;
  - chapter 8: the replay of a multiple imputation;
  - the tutorial: multiple imputation in step 3.3 and pooled analyses in step 6.1.
  - The audit report closes S4 and marks M2 done, pending the owner's review.
- The Word report gives the methods paragraph of a multiple imputation (Sterne et al. 2009), written from its trace
  step, which now records the share of incomplete rows.

### Repository moved to Cohort-Studio

- The code is now at https://github.com/KarimELMERNISSI/Cohort-Studio and the documentation site at
  https://karimelmernissi.github.io/Cohort-Studio/. The former `Cohort-Monitoring-Package` addresses no longer
  exist; clone the repository again from the new address.
- The Google API key exposed in the former repository's history (owner action of 2.3.0) was revoked and replaced
  on 2026-10-15. The history had been purged and the former repository deleted on 2026-10-14.

### Fixed: replay of Windows traces on Linux, trace files read as UTF-8, CI failures readable

- A trace recorded on Windows names its source with backslashes; on Linux (the Docker image) the replay now finds
  that file by its name in the data folders, as it already did on Windows.
- The Word report reads a trace file as UTF-8, like the rest of the application (accented descriptions were
  garbled on Windows).
- CI: the failed tests and the coverage verdict are published as an annotation of the run.

### One analysis population, one source of truth (follow-up programme M1)

**Behaviour changes**

- **Inclusion and exclusion criteria are filters, not deletions.** Validate's buttons now tick the criteria
  ("Exclude these anomalies from the analysis population", "Restrict the analysis population to these criteria");
  the sidebar lists every criterion with what it excludes, and every page analyses the same population. The dataset
  keeps all its rows: *Keep only these rows* (Validate > Analysis population) is the one action that deletes rows,
  after a confirmation, as a traced step.
- **Enrich works in a draft.** Merges, derived variables, transformations and imputations reach the analyses when
  the draft is applied (banner at the top of the page, or "Use This Dataset for Further Analysis"). A draft built on
  data that changed since, or whose imputation was fitted on another population, cannot be applied as it is: it can
  be redone or refitted, or discarded. The Enrich page no longer loads files of its own (use Data > Import).
- **Imputation is fitted on the analysis population.** Excluded patients neither inform nor receive imputed values;
  one-hot encoding, scaling and dropping remainder columns need every row.
- **Dates are no longer parsed silently.** Validate previews the text columns that look like dates and converts them
  on request, as a traced and replayable step.
- **Outliers**: excluding them adds a criterion of the *outliers* family (the flagged rows, frozen); clipping is a
  traced step with its bounds.
- **Results are not shown as current after a change.** A result computed on other data or another population
  disappears with a notice; the analysis is run again. Every result says what it was computed on.

**Under the hood**

- `core/session.py` (`DataSession`) owns the data: committed table with row ids (`__row_id`), fingerprint and
  revision; the Enrich draft; the ticked criteria. `ui/state.py` (`AppState`) is its Streamlit facade and the only
  writer of the session data (quality gate); the legacy keys are views reinstalled every run, and the tests run in
  strict mode (modifying them in place raises).
- `core/population.py`, `core/criteria.py` (strict evaluation, one-sided bounds fixed), `core/lineage.py` (SHA-256
  chain, `verify_trace`), `core/fingerprint.py` (row ids, ranges); `ui/results.py` (stamped results);
  `ui/components/population.py` and `ui/components/staging.py`.
- Trace schema 2: data block (rows, columns, fingerprints in and out) and hash chain on every step, population
  block for population-scoped steps, `applied_at` for applied drafts; loaded traces are read-only.
- Replay (`manage/reproduction_manager.py`): every step checked against its fingerprint; new handlers (targeted and
  population-scoped imputation, population definition, freeze, clipping, merges with distinct identifiers);
  non-recomputable steps listed, their snapshot used. The formula evaluator and the imputer moved to
  `enrich/formulas.py` and `enrich/imputation_service.py`.
- Saved versions keep row ids and dtypes (pyarrow) and get a lineage file; loading one links the trace to its parent,
  restores the ticked criteria and detects a file modified outside the app.
- Merges keep the row ids of the main table and record join counts; they no longer print the merged tables
  (patient data) to stdout.
- Workflow marks: a revision is a step that changed the data fingerprint; an analysis made on another population
  shows as "run again".
- Tests: end-to-end acceptance test (tutorial cohort 1,206 -> 1,190 -> 928, imputation on the population, replay to
  the analysed table, chain verified), core suites for the population, criteria, lineage and session, AppTest suites
  for the population selector, the Enrich draft, result provenance and saved-version lineage.

### Truth and safety net (follow-up programme M0, after an external architecture review)

- **Claims brought back to what the code does** — The documentation site no longer claims FDA 21 CFR Part 11 audit-trail integrity, "signed" Word dossiers or certificates, "immutable" traces, STROBE/CONSORT compliance, 100 % test coverage, "0 hallucinations" or deterministic 100 % replay. It now describes a recorded, replayable trace and a Word report of it (not an electronically signed record), and names the steps that cannot be replayed yet. The code sample uses the real API (`run_trace`, `TraceDocumenter.generate_report`). README, About page, workflow map, tutorial and epidemiology chapter describe multiple imputation as what it is today: a diagnostic that pools the imputed means; analyses are not yet pooled. The README test section gives the real numbers and the install step for the test tools.
- **Audit report** — S4 (multiple imputation) is marked partially open, E8 (stale state) and E10 (replay equals the analysed table) are reopened with the findings of the review, and a follow-up programme (M0 to M4) is recorded.
- **Validate filter reaches Enrich** — Applying a filter restarts Enrich's working copy from the filtered rows; imputation can no longer be fitted on excluded rows or put them back into the analysed data.
- **Results follow the data** — Analysis results (survival, regression, paired, McNemar, Mantel-Haenszel, multiple imputation, clustering, document coverage) are cleared when another dataset is loaded; the group-comparison results are keyed by a content fingerprint (`core/fingerprint.py`), so imputation, clipping or a change of transformation recomputes them.
- **Traces stay in the workspace** — A trace saved under a relative name lands in the user's trace directory instead of the process working directory; traces are written as UTF-8 with an atomic replace.
- **Multiple-imputation advice** — The panel no longer suggests leaving the outcome out of the imputation model; "adopt" is labelled as single imputation.
- **Wheel** — The wheel now ships `biostats`, `core`, `ingest` and `ui` (an installed wheel could not import before); a quality gate keeps the wheel list equal to the first-party packages, and CI installs the built wheel in a clean environment and imports every package.
- **CI** — uv (0.10.9) and Python 3.14 are pinned: without a pin uv downloads the newest interpreter allowed by `requires-python` (3.15 since October 2026), for which duckdb, pyarrow, pydantic-core and others publish no wheel yet, so `uv sync` has to compile them. `uv lock --check` runs before the install, the workflow can be started by hand, the global coverage floor is 55 % (measured about 58 %), and per-package floors (biostats and core 90 %, graph 91 %, ingest 87 %, ui 75 %) are read from pytest-cov's report by `dev_scripts/check_coverage_floors.py`.

### Anthropic Claude connector

- **Claude as an AI provider** (`manage/rag/providers/claude.py`, official `anthropic` SDK) — Next to Ollama, Gemini, OpenAI and Mistral. Models listed from the API when a key is given, otherwise `claude-fable-5-1` (default, most capable), `claude-opus-5-5`, `claude-sonnet-5-5`, `claude-haiku-5-5`. System messages go to the `system` parameter, turns are made to alternate as the Messages API requires, stop sequences are passed, text blocks are returned with token usage. Errors come back as readable messages like the other connectors. Cloud provider: API key (`ANTHROPIC_API_KEY` or typed per session, never pre-filled) and the explicit data-sharing consent are required.
- **Separate embedding provider** — Anthropic has no embedding API, so with Claude the documents are embedded by another provider chosen in Settings: local Ollama by default (the library is indexed on the server; only the passages retrieved for a question reach Claude), or Gemini, OpenAI or Mistral with their own key. `ConnectorFactory.create_connectors(embedding_provider=...)` and `RAGManager.initialize_system(embedding_provider=..., embedding_api_key=..., embedding_base_url=...)`; the consent covers both providers and the vector index records the embedding provider, so reusing an index built with other embeddings is refused.
- **Tests** — `tests/test_claude_connector.py` (message translation, request without temperature, response, LCEL chain, errors, factory pairing, consent and key checks, embedding provider passed to the index) and `tests/ui_layer/test_settings_claude.py` (Settings offers Claude with Ollama embeddings by default).

### Tutorial pack: a richer cohort, its study configuration and a full walk-through

- **Tutorial cohort** (`assets/tutorial/`, generated by `dev_scripts/make_tutorial_cohort.py`, seed 2027) — A synthetic randomised heart-failure cohort: 1,200 patients at 4 sites, 1,206 rows (6 duplicates), 38 variables. Planted data-entry errors, patients failing the inclusion criteria, MCAR / MAR / structural missingness, inconsistent codes, two raters, paired measurements, two blood-pressure methods, a marker against an adjudicated diagnosis, binary, count and time-to-event outcomes with real effects, a stratifying site, and a 6-month laboratory file to merge.
- **Study configuration** (`tutorial_config.json`) — 8 clinical anomaly rules (expression and numeric range rules) that flag exactly the 16 planted errors, and 4 inclusion criteria (consent, age 18-85, NYHA II-IV, creatinine at most 265 umol/L).
- **Full tutorial** (handbook chapter *14 Full tutorial*) — Load, validate, filter, merge, derive (including eGFR CKD-EPI 2021), impute, explore, test, model and report, with the result to expect at each step. `tests/test_tutorial_pack.py` recomputes every number of the chapter and checks the files against the generator.
- **Start page** — A *Tutorial cohort* card: load the cohort with its configuration, open the tutorial, download the files.
- **Fixes on Validate** — *Apply Filtering Criteria for Further Analysis* now works after *Run Anomaly Detection* / *Run Inclusion Criteria Check* (the results disappeared on the next rerun, so the button could never be used); the check is reset when another dataset is loaded. The exclusion percentage reads 22.14% instead of 2.2e+01%. The help of the numeric rule *strategy* now says what the code does (include = inside the range matches, exclude = outside). The developer guide documents the real configuration format.

### Guidance in Analyze > Hypothesis tests & epidemiology

- **Guidance switch and orientation** — A *Guidance* switch (on by default) and a "Which tab do I need?" panel mapping questions to the six tabs, with a link to the *Which test?* chooser.
- **Every parameter explained** — The 38 inputs that had no help now have one (power calculator, detailed view, z-scores, plot references); the configuration panel uses the same texts. `tests/ui_layer/test_guidance.py` now checks the page, its configuration panel and the paired/rates module, so no input can lose its help again.
- **Before you run: the plan** — Live, before anything is computed: the group sizes, for each variable the test that will run and why (the engine's own rules, so the prediction matches what runs), missing data, and warnings (groups under 2 or 5, unbalanced groups, more than 6 groups, 20% or more missing, several tests without correction, transformed scale). A suggested multiple-testing correction applied in one click.
- **Detailed view and ANCOVA** — Notes on the assumptions of a hand-picked test; an ANCOVA model check (observations per parameter, rows excluded, covariates largely explained by the group).
- **Guided power calculator** — Four-step plan; effect sizes from expected means and SD (t-test, Mann-Whitney), group means and SD (ANOVA, Kruskal-Wallis) or event proportions per group (chi-square, degrees of freedom derived); results as sentences with a dropout reminder and power put in context; an effect size of 0 is refused with an explanation.
- **Code** — `ui/guidance/analysis.py` (rules and texts, pure), `app_pages/analysis_guidance_ui.py` (rendering). Tests: `tests/test_analysis_guidance.py`, `tests/ui_layer/test_analyze_guidance.py` (plan, one-click correction, switch, power from means and proportions, ANCOVA check on the sample cohort).

### Renamed: Cohort Monitoring Package becomes CohortStudio

- **Why** — "Monitoring" suggested clinical-trial monitoring or safety surveillance, and "Package" a library; the application is a workbench that takes a cohort from raw file to reproducible evidence (quality, enrichment, biostatistics, literature knowledge, traced reports). Tagline: *Clinical cohort data, from raw file to reproducible evidence.*
- **Where** — Browser tab, sign-in card, page heroes and the header wordmark (regenerated: "Cohort" in navy, "Studio" in teal); the Python project (`cohortstudio`), the conda environment, the Docker image, container and compose service (`cohortstudio`), the launch scripts, the handbook, the developer docs, the documentation site and its citation (`elmernissi2026cohortstudio`). The name lives in one place in the code (`ui.theme.APP_NAME`); `tests/ui_layer/test_branding.py` keeps the former name out of the application.
- **Unchanged on purpose** — The `CMP_` prefix of the server settings (existing deployments keep working), the GitHub repository and documentation-site addresses (until the repository itself is renamed), internal identifiers, and historical documents (audit report, legacy docs, earlier changelog entries).
- **Also** — `requirements.txt` and `requirements-yfiles.txt` re-exported from the lockfile: they now list LanceDB and Qdrant instead of ChromaDB.

### Vector database connector: LanceDB or Qdrant (ChromaDB removed)

- **Engine-agnostic RAG** — The document index goes through `manage/vectorstores/`: the RAG code only sees `DocumentIndex` (similarity search, maximal-marginal-relevance search, retriever, build), whatever the engine. Filtering by document is a list of sources that each engine translates (SQL `IN` for LanceDB, a payload filter for Qdrant).
- **LanceDB** (default) — embedded and serverless, vectors on disk in the Lance format: low memory, exact search below 50 000 chunks and an IVF index above. **Qdrant** — embedded (in memory, small and medium libraries) or a server via `CMP_QDRANT_URL` / `CMP_QDRANT_API_KEY` (one collection per user; for several users or very large libraries). ChromaDB and `langchain-chroma` are removed.
- **Choice for the machine** — Settings > AI engine > Vector database: LanceDB or Qdrant, with the recommended engine and its reasons from the machine (CPU cores, memory, free disk, probed without extra dependencies), the library size and the presence of a Qdrant server; Qdrant server URL, key and connection test; status of the current index. Server default: `CMP_VECTOR_STORE` (`auto`, `lancedb`, `qdrant`).
- **Safer indexes** — Each index has a manifest (engine, embedding model, dimensions, chunks, documents, build time); reusing an index built with another embedding model is refused instead of returning meaningless neighbours; embedding failures are reported after three attempts instead of silently skipping chunks. Indexes live in `<workspace>/vectors/<engine>/`; an old ChromaDB index is detected and can be removed (the PDFs are kept, the index is rebuilt).
- **Fix** — The Start-page evidence for Documents no longer creates an empty `chroma_db` folder; an empty one left behind is removed.
- **Tests** — `tests/test_vectorstores.py` (the same contract on both engines: build, search, document filter with quotes, legacy filter, MMR, retriever, reopen, rebuild, embedding mismatch, failure reporting, recommendation, settings), `tests/test_rag_vector_integration.py` (RAGManager end to end on both engines).

### Interactive global quality score (Validate)

- **Ring built from the dimensions** — The static gauge is replaced by a ring whose coloured segments are the audit dimensions, sized by their weights; the grey gap after a segment is the points it loses. Hover a segment for its score, weight, points added and lost; the ring follows the weight sliders; the centre shows the score, its band and the change since the previous check. Theme-aware (light and dark).
- **Where the score loses points** — A bar per dimension, most costly first; clicking a bar lists the columns, rows or rules behind the loss (empty cells per column, outliers per column or rows flagged by a multivariate method, numbers or dates stored as text, spacing or capitalisation issues, duplicated rows, clinical rules). The breakdown is a fragment, so a click does not rerun the audit.
- **Score history** — One point per version of the data (as loaded, then after each merge, computed variable, imputation or applied filter), recomputed with the current weights; each dimension can be added from the legend.
- **Code** — `explore/quality_score.py` (weighted decomposition, bands; pure), `DataQualityAuditor.details` (where points are lost), `ui/quality_charts.py` (ring, bars, history), `ui.progress.record_quality_check`, `core.progress.revision_labels`. Tests: `tests/test_quality_score.py`, history checked end to end in `tests/ui_layer/test_workflow_progress.py`.

### Workflow progress on the Start page

- **Ticks and "suggested next" follow what you did** — The workflow diagram and the step cards show, for the dataset currently loaded, which steps are done (green tick), done before the data changed (amber circular arrow, "run again") and which step comes next (pulsing ring). Hovering a box gives its status in a sentence, e.g. "Checked before the last 2 changes to the data: check the quality again."
- **Evidence, not clicks** — Data changes come from the transformation trace (merge, computed and transformed variables, imputation, adopted multiple imputation, applied filters), which gives the revision of the data; saved versions count as reporting. Pages record a step when they perform it on the current data: the quality dashboard on Validate, descriptive statistics on Explore, every Run of a test or model on Analyze and Survival (not the calculators that take typed-in numbers), the Word report of the session's own trace on Reproduce. Documents and Variable map are done when the workspace holds a knowledge graph or index, or a saved variable map.
- **The enrichment loop** — After an enrichment, Validate and Explore & analyse turn amber and Validate is suggested again; once both are redone, saving a new version completes the round. Loading another dataset starts afresh.
- **Code** — `core/progress.py` (pure rules: revision, done / stale / to do, next step), `ui/progress.py` (`mark_step`, `current_progress`), the map and the Start page read the same progress. Tests: `tests/test_progress.py` (rules and the loop), `tests/ui_layer/test_workflow.py` (badges, ring, hints), `tests/ui_layer/test_workflow_progress.py` (Start, Validate, Explore, a data change and a new dataset, end to end).

### Fixed: imputation after adding variables, one-hot encoding, multivariate outliers

- **Imputation column mismatch** — "Shape of passed values is (500, 20), indices imply (500, 19)": columns that were neither numerical nor categorical (dates stored as text, identifiers) were imputed and kept by the transformer but missing from the list of output names. It happened on any cohort with a date read from a CSV, with or without a new variable. Every column is now assigned to exactly one group (numerical, categorical, pass-through) and the names are built from that partition (`enrich.data_imputation.partition_columns`).
- **Pandas 3 text columns** — High-cardinality text such as patient identifiers is recognised again as a pass-through column (pandas 3 stores text as `str`, not `object`), so identifiers are no longer imputed or one-hot encoded.
- **Yes/no computed variables** — A variable such as `age > 65` (true/false) no longer fails with "Some column names are not columns of the dataframe"; the column reordering step no longer drops any column.
- **One-hot encoding** — The "Apply one-hot encoding" option, and the multivariate outlier detectors (LOF, Isolation Forest, DBSCAN) on data with missing values, failed because the encoder returned a sparse matrix; the output is now a dense, named table.
- **Clearer error** — An imputation failure shows a readable message with the technical details folded away. Unused copies of the imputer factory were removed.
- **Tests** — `tests/test_imputation_columns.py`: the sample cohort with a computed numeric and a computed yes/no variable, every imputation method, one-hot encoding, the drop strategy and MissForest; names are checked against the values.

### Interactive workflow map, incremental enrichment

- **Clickable map** — Every box of the Start-page workflow opens its page (mouse, or Enter/Space from the keyboard). Hovering or focusing a box highlights it and the links that connect to it, dims the rest and explains the step in a line under the map, including when it needs the AI engine. Built as a Streamlit custom component (`ui/components/workflow_map.py`) that follows the light and dark themes.
- **Enrichment is a loop** — Validate ↔ Enrich ↔ Explore & analyse are drawn as two-way links ("iterate"): derive or impute a variable, validate and analyse again, come back for the next one. Descriptions, the optional-path card and the handbook say so.
- **Clean layout** — Each link has its own lane: no two links cross or share an anchor, labels sit beside their line, "suggested next" sits above its box, the pulse ring is drawn under the arrows and a legend explains boxes and arrows inside the drawing. `tests/ui_layer/test_workflow.py` checks the geometry (no crossings, no link or label over a box, a label, the legend or "suggested next").

### Workflow with optional paths

- The Start page shows the workflow as a diagram: the core path (load, validate, explore and analyse, report) and the optional branches (enrich after validation; map your variables to your field and learn from your documents, both with the AI engine), with where each joins. Cards for each optional path give its purpose, when to use it, its status (AI engine on/off, PDFs in the library) and links (`ui/workflow.py`). The handbook describes the core and optional paths.

### Visual redesign

- **Brand** — Theme built from the new logo (navy, teal, cyan, mint) with full light and dark variants, a tighter type scale, softer radii, widget borders and branded chart palettes; the top bar shows a "Cohort Monitoring" wordmark that switches with the theme (`assets/brand/`, `dev_scripts/make_brand_assets.py`); the Deploy button is hidden.
- **Hero banner** — Every page opens with a gradient banner: section name, title, one-line purpose and a Handbook pill (`ui.components.page_header`, scoped stylesheet `ui/styles.css`). The large empty band above page titles is gone.
- **Sign-in** — A centred branded card with icons in the fields and a one-line trust statement.
- **Sidebar** — Compact dataset card (name, version and origin badges, size, Change / Save with icons), AI engine badge, quick links and a user block with role badge; the oversized logo moved to the top bar.
- **Start page** — Personal welcome banner, a row of bordered figures (dataset, rows, columns, missing cells, saved versions), three workflow step cards with status badges, and cards for saved versions, the sample cohort and the AI engine.
- **Settings** — The AI configuration is grouped into cards (provider and models, data sharing, document library, retrieval, engine) in two columns; tabs carry icons; the other tabs use fixed-width cards.
- **About** — Six feature cards instead of coloured alert boxes; the LinkedIn badge no longer loads an external image.
- **Analysis page** — Analysis type and transformation controls moved from the sidebar onto the page; legacy duplicate page titles demoted to section headings.
- **Tests** — Theme tokens, WCAG AA contrast of the theme colours, brand assets and an offline-only stylesheet are checked.

## [2.3.0] - 2026-10-13

The October 2026 audit and professionalisation programme (phases P0-P8; see `documentation/13_Audit_Report_2026-10.md`).

### Statistics Tier C: survival, regression, rates, paired tests, reliability, multiple imputation (audit phase P8; findings S4, S11)

- **Survival & regression page** — Kaplan-Meier curves with pointwise Greenwood intervals and medians, the log-rank test, Cox proportional hazards (Efron ties; Grambsch-Therneau proportional-hazards test and concordance with the optional `survival` extra, `lifelines`), logistic regression (odds ratios with Wald intervals, DeLong AUC, events-per-parameter warning) and Poisson / negative-binomial regression with person-time offsets (`biostats.survival`, `biostats.regression`).
- **Paired, rates & stratified tab** — Paired t-test and Wilcoxon signed-rank with CIs and Cohen's d_z, one-sample t / Wilcoxon / exact binomial, exact McNemar, incidence rates with exact Poisson intervals and rate ratios, Mantel-Haenszel odds and risk ratios with the Breslow-Day test and a crude-vs-adjusted confounding note, standardised ratios (`biostats.paired`, `biostats.rates`).
- **Reliability** — Cohen's kappa (unweighted, linear, quadratic) with CI and Cohen's kappa interpretation, ICC(2,1) and ICC(3,1) with Shrout-Fleiss intervals (`biostats.reliability`), on the Diagnostic accuracy & agreement tab.
- **Multiple imputation** — m completed datasets by chained equations with posterior sampling, Rubin's rules with the fraction of missing information, on the Enrich page (`biostats.imputation`); adopting a completed dataset is traced.
- **Guidance** — The test chooser now routes paired, one-sample and survival questions to the pages that run them; twelve glossary entries added.
- **Tests** — Reference values: Gehan 6-MP trial (medians 23 / 8, log-rank 16.79, Cox log HR 1.572, SE 0.412), statsmodels spector logit (ORs 16.88 / 1.100 / 10.79), incidence 15/1200 py (7.00 to 20.62 per 1000), SMR 25/18.4 (0.879 to 2.006), McNemar [[59,6],[16,80]] p 0.0525, kappa [[20,5],[10,15]] 0.400 (0.151 to 0.649), Shrout-Fleiss ICC example, Rubin pooling (T 0.060, df 64, FMI 0.272).

### Quality gates (audit phase P7; finding E14)

- **CI** — `ci.yml` runs ruff with the project configuration, `ruff format --check` on the audit packages, pyright (errors only, scoped by `pyrightconfig.json`) and the offline test suite with a coverage floor; `integration.yml` runs the live-provider tests on demand only.
- **Ruff configuration** — `[tool.ruff.lint]` is now explicit (E/F/W/I/B/UP/SIM/TID + selected RUF/PL rules); the legacy pages and helpers carry per-file ignores until they are split; imports are sorted project-wide.
- **Pre-commit** — ruff check/format, gitleaks, large-file/TOML/YAML/JSON/private-key checks (`.pre-commit-config.yaml`).
- **Source gates** (`tests/test_quality_gates.py`) — no `eval` outside the whitelisted evaluator, no artificial sleeps, no hard-coded `data/` paths, raw-HTML and `print()` ratchets, version recorded in the changelog, test directories never named like packages; `tests/test_layering.py` enforces the dependency direction.
- **Docker** — uv-based image with locked dependencies, `libgomp1` instead of `build-essential`, non-root `app` user, offline graph engine and local IAM by default, `WITH_YFILES` build argument; `.dockerignore` excludes secrets, data, tests and the virtual environment; `docker-compose.yml` without the obsolete `version` key, optional `.env`, no shared `DOCUMENTS` mount (document libraries are per user).
- **Project metadata** — version 2.3.0 (aligned with this changelog); `pyright` added to the dev extra; coverage configuration.

### Structure: layering, dead code, single replay engine, traced mutations (audit phase P6; findings E4, E8, E10, E13, U4, U5, U6)

- **One replay engine** — `manage.reproduction_manager.run_trace()` replays a trace headlessly (optionally up to a step, from a pre-loaded frame, with path overrides chosen in the UI) and reports a missing file as structured data instead of rendering widgets; the Reproduce page's own replay loop and the Enrich page's duplicate "Reproduce Analysis" sidebar were removed. Supported steps: enrichment, imputation, variable transformation and computation, type conversion, missing-value handling, column renaming and row filters. The source file is read through the import engine with the parsing options recorded at import.
- **Traced mutations** — Column renaming (Explore) and "Apply filtering criteria" (Validate: anomalies, inclusion criteria, outliers) are now logged in the trace with the kept rows, so they replay; applying a filter asks for confirmation.
- **Layering** — No Streamlit in `utils/analysis_utils.py`, `utils/statistics_utils.py`, `utils/path_utils.py`, `utils/visualization_utils.py`, `utils/multipage.py`, `manage/reproduction_manager.py` (widgets moved to `app_pages/analysis_config.py`, `app_pages/statistics_guides.py` and the Visualize page); `enrich/transformation_logic.py` holds the transformation logic (the `app_pages` module is a shim); `tests/test_layering.py` enforces the dependency direction and bounds the remaining lazy page imports.
- **Dead code removed** — `app_pages/data_preparation.py`, `app_pages/statistical_tests.py`, `explore/statistics.py`, `explore/stats_examples.py`, `manage/anonymize.py` (commented-out), `utils/custom_gemini.py` and its test; the Excel export (`to_excel`) and the empty-config factory now exist once; the legacy loader delegates to `ingest.loaders`.
- **Correctness** — Statistics caches on disk are keyed by a content fingerprint of the frame (filters and renames no longer serve stale statistics); DuckDB benchmark queries quote column identifiers; the Analyze page runs its battery of tests on an explicit "Run analysis" button and keeps results until the configuration changes; artificial `time.sleep` pauses were removed; two raw-HTML snippets became native elements.
- **Logo** — New application logo (transparent background).

### Statistics Tier B, result cards and guidance (audit phase P5; findings S6-S10, S12-S16, U4, U7)

- **Welch-first inference** (`biostats.inference`) — Two groups: Welch's t-test by default with the mean difference and its CI; Mann-Whitney with the Hodges-Lehmann shift (bootstrap CI). Three or more: Welch ANOVA by default, one-way ANOVA, Kruskal-Wallis. Normality (Shapiro/D'Agostino) and Levene are reported as diagnostics with sample-size caveats; they never select the test.
- **Pairwise comparisons** (`biostats.posthoc`) — Games-Howell, Tukey HSD, Dunn (tie-corrected, Holm), pairwise Welch/Student t and Mann-Whitney with Holm; every pair is reported with raw and adjusted p-values, estimates (group 2 minus group 1), CIs where the method provides them and an effect size. Plot significance brackets now use the same family and correction as the summary table.
- **Effect sizes with intervals** (`biostats.effect_sizes`) — Hedges' g with the exact correction and a noncentral-t CI (Steiger 2004), Vargha-Delaney A and rank-biserial r with the sign of g (bootstrap CI), eta squared with a noncentral-F CI and omega squared, bias-corrected Cramer's V (Bergsma 2013) with a bootstrap CI, eta squared (H).
- **Multiplicity** (`biostats.multiplicity`) — Explicit families; Holm is the default for the variables of a run, Benjamini-Hochberg for screens; NaN-safe adjustment.
- **Normality** (`biostats.normality`) — Lilliefors replaces the self-standardised KS test; Anderson-Darling returns a real p-value; large-sample caveat in the verdict.
- **Little's MCAR** (`biostats.mcar`) — EM estimates of the mean and covariance (Dempster 1977) instead of available-case means and pairwise covariance; wording "no evidence against MCAR" and "cannot distinguish MAR from MNAR". The README/docs no longer claim a permutation test.
- **Diagnostic accuracy and agreement** (`biostats.diagnostic`, `biostats.agreement`) — New Analyze tab: sensitivity, specificity, PPV/NPV (Wilson), prevalence-adjusted predictive values, LR+/LR- (Simel log CIs), diagnostic odds ratio, AUC with the DeLong interval, Youden cut-off with a bootstrap CI and an optimism note; Bland-Altman with SD (ddof = 1), CIs for the bias and both limits, and a proportional-bias check. ROC curves on the Visualize page show the DeLong interval.
- **Mixed-type associations** (`biostats.correlation`) — Pearson/Spearman with Fisher-z / Bonett-Wright CIs, point-biserial, correlation ratio, bias-corrected Cramer's V, BH-adjusted p-values; the correlation heatmap is centred on zero and pairs with fewer than 10 complete observations are left empty.
- **Outliers** (S14) — Binary and boolean columns (any dtype, including 0/1 integers) are skipped by the Z-score and IQR detectors; nullable and 32-bit numeric columns are handled; Isolation Forest is seeded; a robust MAD-based detector was added.
- **Reproducibility** (S16) — LightGBM learners in missForest are seeded and deterministic; date-format sampling is seeded.
- **Interpretation wording** (S9) — Result quadrants are CI-first ("inconclusive: large observed effect, p >= 0.05" instead of "underpowered", "not significant, small observed effect" instead of "true negative"); ANCOVA diagnostics read "no evidence against ..."; power is "power at this sample size", never "achieved power".
- **Result cards and guidance** (U4, U7) — `ui.components.result_card` renders estimate -> CI -> p (raw/adjusted) -> n -> effect sizes -> diagnostics -> references -> "How to read this"; a declarative **Which test?** page (four questions, 60 routes, 16 available now, the rest marked planned); a 50-term glossary (popover and Handbook); `documentation/11_UI_Style_Guide.md`; a test enforces `help=` on every input of the new modules.
- **Tests** — `tests/biostats_layer/`: reference values from R's PlantGrowth (Welch F 5.181, Tukey p 0.012, Kruskal H 7.988), Hedges J(8) = 0.902703, g CI [-0.133, 2.361], V_c = 0.3977 for [[30,10],[20,40]], DeLong toy AUC 0.875 / var 0.020833, Lilliefors calibration, EM-vs-available-case bias under MAR.

### Navigation shell, typed state and onboarding (audit phase P4; findings U1-U3, U5, U6, E8, E11)

- **Information architecture** — The app runs on `st.navigation` with sectioned pages and real URLs: Start; Data (Import, Validate, Enrich); Explore (Statistics & correlations, Visualize); Analyze (Hypothesis tests & epidemiology); Knowledge (Variable map, Documents); Reproduce (Versions & traces); Help (Handbook, Settings, About); Admin (Users, AI quality). Page files live in `app_pages/views/`, the registry in `ui/nav.py`. The custom "Go to" selectbox and `MultiPageApp` are gone.
- **Sign-in screen** — Centred in the main area (the sidebar no longer hosts login/sign-up).
- **Shell sidebar** — A dataset card (name, version badge, rows x columns, origin) with **Change** (saved versions, import, sample) and **Save version** dialogs, the AI engine status, a Handbook link and the user menu. The three duplicated "Dataset History" sidebars and the automatic loading of the newest dataset on the Visualize page were removed.
- **Typed state** — `core/state.py` (`DatasetRef`, `ParseOptions`) and `ui/state.py` (`AppState`): `set_dataset()` is the single place that installs a dataset; it clears every derived result, starts the trace in the user's workspace, stores the source file as an artifact and logs the parsing options and parse report.
- **Import flow** — `Data > Import`: choose a file; check how it is read (encoding, delimiter, decimal and thousands separators, header row, skipped lines, day-first dates, missing-value tokens, Excel sheet selection/merge), with a 20-row preview, column types and a parse report (malformed lines are counted, never silently dropped); name and load, optionally saving v1. Engine in `ingest/loaders.py` (semicolon + decimal-comma exports are detected; Windows-1252 is preferred over guesses for non-UTF-8 files).
- **Onboarding** — Start page with the three-step checklist, saved versions and a bundled, documented **sample cohort** (`assets/sample_cohort.csv`, 500 synthetic patients, generated by `dev_scripts/make_sample_cohort.py`); every page shows a consistent empty state with links to Import/Start and a one-click sample load instead of warnings pointing to pages that did not exist.
- **Handbook in the app** — `Help > Handbook` renders the researcher documentation offline with a table of contents; each page header links to its chapter.
- **Settings page** — AI provider, keys and data-sharing consent (moved out of the per-page sidebar), graph engine, import defaults and a confirmed "clear my cached statistics". The Ollama probe and the cloud model list are cached (60 s / 1 h) instead of running on every rerun; the upload hash is computed once per file.
- **Theme and components** — `.streamlit/config.toml` themes (clinical blue primary, light/dark), `ui/theme.py` tokens and a registered Plotly template (Okabe-Ito palette); `ui/components` (page header, sections, status chips, empty states, confirmation dialog, long-operation status, error card). The Main View cache reset now asks for confirmation and only clears the user's own cache.
- **Tests** — `tests/ingest_layer/` (detection, parse reports, Excel merge, sample cohort properties) and `tests/ui_layer/` (tokens vs config, navigation registry, `set_dataset` invalidation and trace, AppTest smoke over every page with and without data).

### Security hardening (audit phase P3; findings E3, E4, E5)

- **E3 Expression evaluation** — Every user-, config-, LLM- or trace-supplied expression (custom metrics, filters, enrichment formulas, naming patterns, replayed traces) now runs through `core/expressions`, an AST whitelist evaluator: only `df`, `np`, `pd` and a fixed list of pandas/numpy/Series/`.str`/`.dt` operations are reachable, builtins are reduced to `abs/bool/float/int/len/max/min/round/str`, dunder/private attributes, imports, lambdas, comprehensions and I/O helpers are rejected with a clear message. Naming patterns use `safe_format` (known fields only). The unused `enrich/conditional_transformations.py` (a second `eval` path) was removed.
- **E4 Per-user workspace** — Datasets, statistics caches, traces, artifacts, snapshots, uploads, taxonomies, knowledge graphs, the document library and the vector store now live under `<DATA_ROOT>/users/<username>/` (`core/workspace.UserScope`). `DBManager`, `TransformationManager` and `RAGManager` bind to the signed-in user's scope; dataset and trace listings are directory-scoped (the username-prefix convention is kept only outside a session, e.g. CLI/tests). Manually entered server paths (Enrichment, Validation, Reproduce Analysis) and paths recorded in trace files are only accepted inside the user's workspace; uploaded file names are sanitised; snapshots carry a random suffix; DuckDB file paths are bound as parameters or escaped. Logging out clears the whole session state. Existing installations: run `python dev_scripts/migrate_user_layout.py --dry-run`, then without `--dry-run`, to move the legacy flat layout into per-user workspaces.
- **E5 Login lockout** — Five consecutive failed logins lock the account for 15 minutes (`CMP_MAX_FAILED_LOGINS`, `CMP_LOCKOUT_MINUTES`); a successful login resets the counter.
- **Tests** — `tests/core_layer/`: allow/deny expression corpora, naming patterns, workspace path containment, cross-user dataset/trace isolation, DuckDB quoting, lockout behaviour.

### Statistics: wrong-number fixes (audit phase P2; findings S1-S5)

- **S1 2x2 orientation** — Binary outcomes coded No/Yes, 0/1 or False/True had their risk measures inverted (the sorted crosstab put the non-event in the "event" column). New `biostats.contingency` engine with an explicit table orientation (`TwoByTwo.from_crosstab` with reference group and event level, auto-detected or chosen in the UI), labelled tables, risk per group, Newcombe risk-difference interval, NNT interval (Altman 1998), primary test by expected counts, design-aware guidance. Example: Placebo 40/100 vs Treatment 15/100 events now reports RR 2.67 (95% CI 1.58 to 4.51) instead of RR 0.71.
- **S2 ANOVA / Kruskal-Wallis power** — statsmodels' `nobs` is the total sample size; the page treated it as per group (required sizes overstated by k, achieved power understated). `biostats.power` states total and per-group sizes explicitly; the chi-square calculator takes degrees of freedom instead of "number of categories".
- **S3 Missingness scan** — The global dependency scan now returns every performed test with family-adjusted p-values (`biostats.missing`); previously only p < 0.05 results were kept and then "corrected", so the correction controlled nothing. Wording is associational.
- **S4 Imputation** — "MICE" relabelled as single imputation by chained equations (IterativeImputer) and "MissForest" as tree-based iterative imputation with LightGBM; a warning explains that imputation uncertainty is not propagated; estimators are seeded; missForest models are fitted on originally observed rows only.
- **S5 Z-scores** — A control group coded 0/False no longer silently falls back to mean 0 / SD 1; the manual-reference mode has mean/SD inputs; computation moved to `biostats.zscores`.

### Graph rendering connector (audit phase P1; findings G1-G8)

- **New `graph/` package** — Renderer-agnostic model (`GraphSpec`), a palette/legend single source of truth, pure builders for the correlation network, the clinical knowledge graph and the document graph, and a renderer contract (`translate()` / `render()`) with three backends: Plotly + NetworkX (offline default), yFiles (opt-in extra `graph-yfiles`) and Graphviz (static). See `documentation/developer/Graph_Rendering.md`.
- **yFiles is optional** — Removed from the required dependencies (`requirements.txt`; use `requirements-yfiles.txt` or `uv sync --extra graph-yfiles`), selected with `CMP_GRAPH_RENDERER=yfiles`, refused with `CMP_OFFLINE=1`; third-party notice on the About page and under every yFiles-rendered graph (license clause 1.7).
- **Pages** — Main View, Data Insight and Documents Insight render through `graph.ui.render_graph`; graph selection now reaches the pages (the Data Insight selection panel works, a graph click selects the entity on the Documents page); a renderer failure shows an error card and falls back to the offline engine instead of blanking the remaining tabs; JSON/HTML exports and an in-page "Graph engine" selector were added.
- **Fixes** — Stable node/edge ids and ordering (no uuid per rerun, no `hash()` ids); node heat computed from the numeric correlations rather than rounded labels; one idempotent `make_formula_id()` (no `FORM_FORM_` double prefix); one variable-type vocabulary with a style for `Outcome`; document-graph relations remapped after entity merge (no spurious "Inferred" nodes); one correlation-strength threshold table (`explore/correlation_strength.py`, Schober et al. 2018) shared by the graph, the Excel export and the UI; the dead `plotly_corr_network` was removed and `explore/corr_matrix.py` no longer imports a UI library (`get_yfiles_network_data` remains as a deprecated shim).
- **Tests** — `tests/graph_layer/`: model integrity and serialisation, palette coverage, builder reference graphs, Plotly and yFiles translation (the latter replayed through the library's own mapping engine), renderer-selection policy, headless renderer contract.

### Security and stability (audit phase P0; finding IDs refer to `documentation/13_Audit_Report_2026-10.md`)

- **Audit report** — Added `documentation/13_Audit_Report_2026-10.md` with the verified findings and the remediation programme.
- **E5 Admin takeover** — Signing up with the username `admin` no longer creates an active administrator. Administrators are bootstrapped from `CMP_ADMIN_USERNAME` / `CMP_ADMIN_PASSWORD(_FILE)` on first start or with `python manage_users.py create-admin <name>`; all privilege checks now use the stored role (`manage/iam/authz.py`). Minimum password length is 12 (`CMP_MIN_PASSWORD_LENGTH`), usernames are validated. `reset_password.py` is replaced by `manage_users.py` (passwords are prompted, never passed on the command line).
- **E6 Cloud LLM** — The AI provider now defaults to local Ollama. Cloud providers require an explicit per-session confirmation listing exactly what would leave the server; `RAGManager.initialize_system` refuses cloud providers without it. Server API keys are no longer pre-filled into the sidebar and user-typed keys are no longer written to `os.environ`.
- **E7 Patient data on stdout** — Debug `print` calls that dumped data frames, rows and identifiers (`enrichment`, `monitor/changes.py`, `enrich/external_data.py`, dashboard) were replaced by `logger.debug` of shapes and column names. Logging is configured centrally (`utils/logging_config.py`, `CMP_LOG_LEVEL`); global warning suppression in `main.py` was removed.
- **E12 Tracked secrets** — `.streamlit/secrets.toml`, client study configs (`config/*.json`) and the `DOCUMENTS/` PDF library are no longer tracked (git-ignored; local copies kept). Added `.streamlit/config.toml` with usage statistics disabled.
- **E2 Visualization page** — Fixed the `ImportError` (`DataAnalyzer` now imported from `utils.data_analyzer`) that prevented the page from loading.
- **G1/G2 yFiles 2.0** — Replaced `Layout.HIERARCHIC` with `Layout.HIERARCHICAL`, moved height/directed to `show()`, removed widget attributes ignored by yfiles-graphs-for-streamlit 2.x, and isolated graph rendering so a failure no longer blanks the other tabs. The dead `app_pages/yfiles_test.py` page was removed.
- **E9 Crashes** — Fixed 16 undefined names (`compute_transformations` imports, `logger` in `utils/analysis_utils.py`, `Any` in `utils/export_utils.py`, `time` shadowing in Data Insight) and the empty-state crashes on Data Enrichment and Data Validation (clear notices instead of `KeyError` / `UnboundLocalError`).
- **U4/U8 Copy** — Fixed the comparison export always reporting failure, the duplicate dictionary key in Visualization, the empty delete-user button label, and the French label on the comparison tab.
- **E14 CI** — Added `.github/workflows/ci.yml` (ruff syntax/undefined-name check + offline pytest); `pytest` now excludes `integration` tests by default so a plain run never calls live LLM APIs.

### Owner actions still required

- Revoke and rotate the Google API key committed in `fdd29cb` (it is still the key in the local `.env`), purge it from history and enable GitHub secret scanning. Rotate the admin password that was tracked in `.streamlit/secrets.toml`.

---

## [2.2.0] - 2026-02-10

### Documentation Overhaul

- **README.md** — Added clustering, user management, RAG quality monitor, DBSCAN outlier detection, document summariser, replay modes, full configuration table, and expanded tech stack.
- **02_Data_Validation.md** — Added DBSCAN, outlier visualisation (2-D projection), handling strategy matrix, and impact analysis measures.
- **03_Data_Enrichment.md** — Added Import External Data, AI imputation modes (Free Thinking / Guided), clustering (K-Means, DBSCAN, GMM), scaling methods, and dataset versioning.
- **04_Visualization.md** — Added medical plots (Kaplan-Meier, Bland-Altman, ROC), integrated statistics table, and export formats.
- **05_Epidemiology.md** — Removed duplicate sections; consolidated into clean structure with test recommendation, post-hoc tests, and effect size interpretation.
- **06_Data_Insight.md** — Added taxonomy refinement (repair, enrich, concept nodes), versioning, graph visualisation details, and export options.
- **07_Document_Insight.md** — Added document summariser tab, concept merging, citation tracking, and coverage analysis details.
- **08_Reproduction.md** — Added Fast vs Full replay modes, cross-platform path resolution, and TraceDocumenter reporting.
- **09_Clustering.md** — Aligned with Data Enrichment naming; added cross-references and common pitfalls.
- **10_Developer_Guide.md** — Added user management architecture, trace & reproduction system, clustering utilities, RAG quality monitor, and config documentation.
- **00_Researcher_Handbook.md** — Added Knowledge & Literature phase and Reproducibility phase; aligned all phases with navigation order.
- **Tutorials.md** — Added tutorials for clustering, document analysis, and reproducing analysis sessions (9 total).

---

## [2.1.0] - 2026-01-02

### Architecture Improvements

#### RAG Manager Refactoring

- **Mixin Pattern**: Split monolithic `rag_manager.py` (2098 lines → 451 lines)
  - `rag_taxonomy.py` - Taxonomy generation
  - `rag_documents.py` - Document processing
  - `rag_computed_vars.py` - Variable suggestions

#### LLM Reliability Enhancements

- **NEW** `manage/rag_schemas.py` - Pydantic models for type-safe LLM responses
- **NEW** `utils/llm_utils.py` - Robust JSON parsing with:
- Multi-strategy parsing (markdown extraction, bracket balancing)
- Automatic repair (trailing commas, unmatched brackets)
- Schema validation with Pydantic
- Retry mechanism with LLM self-repair

### Utility Extraction

#### Home Page Modularization

- **NEW** `utils/data_analyzer.py` - Consolidated DataAnalyzer class
- **NEW** `utils/export_utils.py` - Excel export functions
- **NEW** `utils/statistics_utils.py` - Normality tests and guidelines

### Performance
- Added `@st.cache_data` for `get_statistics_dataframe()`
- Added `@functools.cache` for `generate_palette()`

### Documentation
- Updated README with architecture diagrams
- **NEW** Developer Guide (`documentation/10_Developer_Guide.md`)
- **NEW** CHANGELOG.md

---

## [2.0.0] - 2025-12-22

### Features
- RAG Quality Monitoring Dashboard
- Document Knowledge Graph
- Taxonomy Generation
- AI-powered column renaming

---

## [1.0.0] - 2025-11-01

### Initial Release
- Data loading and versioning
- Data quality checks
- Visualization suite
- Epidemiology module
