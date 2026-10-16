# Audit Report — Cohort Monitoring Package (October 2026)

**Scope.** Full read-only audit of HEAD `53ecb19` across three axes: the graph/yFiles layer, the
scientific/statistical layer, and engineering/security/UX. Each finding was verified against the code,
the installed libraries (`.venv`: yfiles-graphs-for-streamlit 2.0.0, scipy 1.18.1, statsmodels 0.15.0,
Streamlit 1.65.0), `ruff`, module imports and git history.

**How to read.** Severity **H** = wrong results, security exposure or crash; **M** = misleading or fragile;
**L** = hygiene. Finding IDs (G/S/E/U) are referenced from commit messages and `CHANGELOG.md` as items
close. Line numbers refer to HEAD `53ecb19`.

## Findings (verified)

Severity: **H** = wrong results / security / crash, **M** = misleading or fragile, **L** = hygiene.

### 1.1 Graph layer (yFiles)

| # | Sev | Finding | Evidence |
|---|-----|---------|----------|
| G1 | H | Code targets yFiles 1.3.0 but 2.0.0 is installed. `Layout.HIERARCHIC` no longer exists (`HIERARCHICAL`) → AttributeError; Data Insight graph tab crash also prevents the Formulas/Clustering/Refinement tabs from rendering. | `app_pages/data_insight.py:1267`, `app_pages/document_insight.py:383`, `app_pages/yfiles_test.py:98`; `.venv/.../yfiles_graphs_for_streamlit/layout/layout.py` |
| G2 | H | Settings silently ignored in 2.0.0: `widget.height`, `node_label_style_mapping`, `node_tooltip_mapping`, `edge_label_styles_mapping`. Height is a `show()` kwarg. `show(directed=True)` default overrides `widget.directed=False`, and `EdgeStyle.to_js()` drops falsy `directed=False` → undirected correlation graph gets arrowheads. Invalid `LabelStyle(text_position=…)` / `wrapping="word"` would raise if the hooks were live. | `app_pages/home.py:1372-1397` |
| G3 | H | Selection never reaches the app: `show()` only returns selections with `sync_selection=True`, never passed. Data Insight "Selection Details" panel is dead code. Library mapping errors go to `widget._error`, never read. | `data_insight.py:1270-1357` |
| G4 | H | License/privacy: yWorks Free License requires an internet connection at runtime (loads `https://www.yworks.com/streamlit-yfiles-core/...`), collects anonymised graph-structure analytics, requires an end-user notice (§1.7), forbids redistribution. README/docs claim "Air-Gapped · Zero Telemetry". No notice in `about.py`. | `.venv/.../yfiles_graphs_for_streamlit-2.0.0.dist-info/licenses/LICENSE.md`; `README.md:45`; `docs/index.html:160` |
| G5 | M | Domain module imports the UI library: `explore/corr_matrix.py:15` (`from yfiles_graphs_for_streamlit import Edge, Node`). All three builders return vendor objects; callers read/mutate `.id/.start/.end/.properties`. | `explore/corr_matrix.py:471-645`, `data_insight.py:111-474`, `document_insight.py:73-165` |
| G6 | M | Duplication: 4 copies of `get_edge_style`, 2 of `get_node_style`, 2 identical invalid label-style pairs, 3 focus/neighbourhood implementations, hand-written HTML legends duplicating palettes, correlation thresholds table copied 4× in `corr_matrix.py`. Dash key is `dash` in two builders and `style` in the third. | `home.py:1286-1338, 1377-1392`; `corr_matrix.py:315,505,662,764` |
| G7 | M | Data defects: edge ids are a fresh uuid4 per rerun; node order follows set iteration; heat recomputed by parsing the rounded label string; `FORM_FORM_<var>` double prefix (Refinement tab stores `FORM_<var>` then builder prefixes again) breaks selection lookup; document-graph edges not remapped after node merge → spurious "Inferred" nodes; `node_type` vocabulary drift (7 variants, `Outcome` has no style). | `corr_matrix.py:621,630`; `data_insight.py:2000 vs 274`; `manage/rag_documents.py:184-211` |
| G8 | L | `yfiles_test.py` is a dead playground; taxonomy/graph persistence hard-codes `data/` and ignores `DATA_ROOT` although `utils/data_paths.get_taxonomy_dir()/get_knowledge_graphs_dir()` exist; docs promise layout selector, click details, HTML export that do not exist. | `data_insight.py:9-44`; `documentation/06_Data_Insight.md:73-89` |

### 1.2 Scientific / statistical layer

| # | Sev | Finding | Evidence |
|---|-----|---------|----------|
| S1 | H | **2x2 orientation bug.** `pd.crosstab(group, target)` is passed straight to the 2x2 engine, which treats `[0,0]` as "exposed & event". Levels are sorted, so for No/Yes, 0/1, False/True outcomes the "Event +" column is the non-event. Example Placebo 40/100 vs Treatment 15/100: app shows OR 0.26, RR 0.71, RD −0.25, "NNT 4 (Benefit)", prevalence 72.5 %; correct RR 2.67, OR 3.78, RD +0.25, prevalence 27.5 %. No reference-group / event-level choice. Tests only check keys. | `utils/analysis_utils.py:176,199`; `utils/epidemiology_utils.py:152-160`; `app_pages/epidemiology.py:464-465` |
| S2 | H | **ANOVA / Kruskal-Wallis power off by factor k.** statsmodels `FTestAnovaPower.nobs` is total N; app labels it "per group" and multiplies by k; in power mode per-group n is passed as total (f=0.25, k=3: correct 157 total; app 158/group, 474 total; power at 30/group shown 0.195 vs 0.54). | `epidemiology.py:2339-2346, 2367-2376` |
| S3 | H | **Missingness scan "correction" is applied after pre-filtering to p<0.05**, so the family is wrong and the "out of N tests" count is the number of hits. Causal wording ("value affects missingness"). | `analysis_utils.py:623,658,630`; `data_quality_dashboard.py:624-643` |
| S4 | H | **"MICE" is a single deterministic imputation** (`IterativeImputer(random_state=0)`, no m datasets, no Rubin pooling), runs over the whole dataset (outcome used as predictor and imputed), overwrites the working dataset; "MissForest" is LightGBM trained on rows including previously imputed targets; "MICE" uses numeric predictors only. | `enrich/data_imputation.py:433,42-49,454-457`; `data_enrichment.py:2928,3010-3026`; `utils/miss_forest/missforest.py:482-504` |
| S5 | H | **Z-score reference modes return raw values**: `if ref_group_col and ref_group_val:` is false for a control group coded 0/False → μ=0, σ=1. "Manual Reference Values" has no inputs (`mu_{var}`/`sigma_{var}` never set). | `epidemiology.py:1079,1086-1088,1134,1141-1142` |
| S6 | M | Significance brackets ignore the chosen test and correction (uncorrected pairwise Mann-Whitney); Visualization "Auto-Detect" always Mann-Whitney, "Force Parametric" Student's t; no omnibus test. | `epidemiology.py:877`; `utils/visualization_utils.py:26-35` |
| S7 | M | Test routing: pre-tests at α=0.05 gate the test; Student's t is default (Welch only if Levene p<0.05); Kruskal used as the remedy for unequal variances; no Welch ANOVA / Games-Howell / Dunn although the in-app guide recommends them; groups n<3 skipped → `all([])` → "normal"; post-hoc returns only significant pairs. | `analysis_utils.py:244-363`; `epidemiology.py:1621,1635,1649` |
| S8 | M | Effect sizes are point estimates only; Cohen's d CI computed then discarded; Hedges g / ω² never displayed; rank-biserial sign opposite to d; group ordering differs between d (appearance) and 2x2 (sorted). | `analysis_utils.py:98,140,151`; `epidemiology_utils.py:311-391` |
| S9 | M | Interpretation text is statistically wrong in several places ("True negative — no effect exists", "Groups overlap → no significant difference", "Likely MCAR" on non-rejection, assumption "confirmed/validated", post-hoc "achieved power", "Statistically and Clinically Significant" from generic thresholds). | `epidemiology.py:338,348,371,821,838,847,1333,1483,1781-1783,1868,1949-2063`; `analysis_utils.py:468` |
| S10 | M | Diagnostic-accuracy engine (Wilson CIs) is never called by any page; ROC has no AUC CI and in-sample Youden cutoff; Bland-Altman uses ddof=0, no CIs, not wired. Docs promise DeLong, DOR, LR±, BA CIs. | `epidemiology_utils.py:553-620`; `visualization_utils.py:908-992` |
| S11 | M | Missing core epidemiology: Kaplan-Meier, Cox, incidence rates/person-time, Mantel-Haenszel, logistic/Poisson regression, paired tests, kappa/ICC. `about.py:65` claims survival analysis. | — |
| S12 | M | Normality: KS on self-standardised data (Lilliefors problem); Anderson-Darling branch inverted; "Not Normal" flags at n>5000 without context; category % denominators include missing. | `utils/statistics_utils.py:83-96`; `home.py:371-376,432` |
| S13 | M | Correlation: pairwise deletion with `min_periods=1`, no p-values/CIs/multiplicity, no mixed-type association (Spearman on numeric codes), Cramér's V from Yates-corrected χ² without bias correction, thresholds uncited, colour scales not centred on 0. | `explore/corr_matrix.py`; `analysis_utils.py:376-382`; `home.py:1208` |
| S14 | M | Outliers: binary exclusion only matches the string set `{'0.0','1.0','nan'}` → integer 0/1 columns get IQR=0 and **every positive case is flagged**; dtype checks skip nullable/32-bit; detection runs after imputation; IsolationForest unseeded. | `monitor/outliers.py:47,76,84-92,345` |
| S15 | M | Little's MCAR uses available-case means and pairwise covariance via `pinv` (comment claims EM); README claims a permutation test in a module that does not exist. | `analysis_utils.py:384-469` |
| S16 | L | Reproducibility: unseeded LightGBM/IsolationForest/date-sampling; `main.py:27-29` globally suppresses small-sample warnings. ~2,100 lines of dead statistics code (`explore/statistics.py`, `stats_examples.py`, `app_pages/statistical_tests.py`). | — |

### 1.3 Engineering, security, state

| # | Sev | Finding | Evidence |
|---|-----|---------|----------|
| E1 | H | **Leaked Google API key** committed in `fdd29cb` (`.env`), later deleted; SHA-256 of that key equals the key in the current untracked `.env`. Repository is public. | `git show fdd29cb:.env` (hash compared, not printed) |
| E2 | H | **Visualization page cannot load**: `from utils.visualization_utils import DataAnalyzer` → ImportError (re-export removed by the ruff cleanup commit `4d26acf`). Verified by import. | `app_pages/visualization.py:16` |
| E3 | H | **Code execution via `eval`** on expressions from free text, uploaded config JSON, LLM output and uploaded trace JSON; evaluated automatically on every dashboard render. `evaluate_formula_safely` only whitelists `ast.Name`; `naming_pattern.format()` on user strings. | `enrich/custom_metrics_and_filters.py:56-61,230-236`; `data_monitoring.py:1521-1539,2016-2028,2253-2261`; `data_enrichment.py:991-1033,1250`; `reproduce_analysis.py:620-628`; `app_pages/transformation_logic.py:78-160` |
| E4 | H | **No per-user isolation**: dataset/trace lists unfiltered, Visualization auto-loads any user's newest dataset, arbitrary server-path inputs, prefix-based isolation collides, usernames unvalidated and used in paths and SQL literals, f-string SQL with uploaded column names. | `data_enrichment.py:2608,155-164,228-241`; `visualization.py:29,46-55`; `data_monitoring.py:878-895`; `manage/db_manager.py:113,136,151,193-194`; `home.py:568-686` |
| E5 | H | **Admin takeover**: self-signup as `admin` is auto-activated with the admin role; authorisation is `username == 'admin'` string compares; min password length 4; no lockout/session persistence. The IAM facade is used only for `is_enabled`; login goes through `DBManager`, hard-wired to the local provider. | `manage/iam/providers/local_duckdb.py:173-174`; `main.py:102-118,136,151`; `db_manager.py:27` |
| E6 | H | **Patient data to a cloud LLM by default** (Gemini), first 3 full rows + top-5 values per column sent; `manage/anonymize.py` fully commented out; server API keys pre-filled into password inputs and user-typed keys written to process-wide `os.environ`. | `rag_sidebar.py:46-55,109-119,166-176,203-213`; `rag_manager.py:128,575-579` |
| E7 | H | Patient data printed to stdout (container logs); 227 `print` vs 136 logging calls; no logging config. | `data_enrichment.py:1235,1251,2284`; `monitor/changes.py:125-288`; `enrich/external_data.py:137,187` |
| E8 | H | Stale/wrong statistics: on-disk stats cache keyed by dataset name only (ignores filters); `current_dataset_name`/`enriched_df`/`working_df`/`rename_dict` never reset on new upload; three session keys hold "the current dataset"; pages mutate the shared DataFrame on render. | `home.py:297-311`; `utils/multipage.py:319-355`; `data_monitoring.py:2277-2288,1783-1798`; `visualization.py:65` |
| E9 | H | Crashes: 16 undefined names (`ruff --select F821,F823`): 11 in `compute_transformations` (a divergent copy of `transformation_logic.py`), `logger` in `analysis_utils.py:202`, `time` before assignment in `data_insight.py:1093`, `Any` ×3 in `export_utils.py`; KeyError/UnboundLocalError on empty state in Enrichment and Monitoring. | `data_enrichment.py:1884-2300,2800,2928`; `data_monitoring.py:2397` |
| E10 | H | Reproducibility claims unmet: renaming, flags, filtering, outlier handling, date re-parsing, sheet merge never logged; two divergent replay engines; sidebar "Reproduce Analysis" button is a no-op (nested buttons); replay ignores detected encoding/separator. | `manage/reproduction_manager.py:24-134,158`; `reproduce_analysis.py:546-699`; `data_enrichment.py:2682-2683` |
| E11 | H | Per-rerun expensive work: MD5 of upload, Gemini `models.list` network call, 3 s Ollama probe, DataAnalyzer date scans (unseeded), full quality audit incl. eval, all univariate tests on any widget change. 3 `cache_data`, 0 `cache_resource`, 0 `fragment`, 40 `rerun`, 7 `sleep`. | `multipage.py:311-316`; `rag_sidebar.py:73-75,134-142`; `epidemiology.py:173-191` |
| E12 | M | Tracked secrets/artefacts: `.streamlit/secrets.toml` (plaintext admin password, unused), `config/config*.json` (client study metadata, loaded by no code), `DOCUMENTS/*.pdf` (third-party papers). No `.streamlit/config.toml` (usage stats on). | `git ls-files` |
| E13 | M | Layering: `utils → app_pages`, `manage → app_pages`, page → page imports; Streamlit imported in 5 domain/util modules; 59 functions >100 lines (largest `visualization.app` 1,556 lines); ~3,200 lines dead code; `to_excel` ×4, loaders ×2 with different algorithms, dataset-history sidebar ×3. | `multipage.py:18`; `reproduction_manager.py:17` |
| E14 | M | Quality gates: no test/lint CI (only CodeQL + Pages); no page/AppTest tests; coverage 34 % (stale); `integration` marker not excluded → plain `pytest` hits live Gemini; minimal ruff; no type checking; Dockerfile runs as root and copies `.venv`/secrets; version 0.2.0 vs CHANGELOG 2.2.0; `tabulate` missing. | `pyproject.toml`; `.github/workflows/` |

### 1.4 UX / ergonomics / guidance

| # | Sev | Finding | Evidence |
|---|-----|---------|----------|
| U1 | H | Login/signup lives entirely in the sidebar; main area blank. Custom `selectbox` navigation (no `st.navigation`, no URLs/back button); nav control is third in the sidebar under two expanders; page controls mixed into the sidebar. | `main.py:75-118`; `multipage.py:434-652` |
| U2 | H | No onboarding: no sample dataset, no guided flow; with no data every tab shows warnings; empty-state messages point to pages that do not exist ("Data Preparation", "Home page"). | `home.py:821,841`; `document_insight.py:468`; `data_insight.py:866` |
| U3 | H | Import UX: encoding/delimiter auto-detected with no override, no preview, no decimal-comma option (FR/LU semicolon CSVs mis-parse), malformed rows dropped silently; uploader label vs accepted types mismatch. | `multipage.py:98-146,436` |
| U4 | M | Analyses rerun on every widget change (no Run button/form); long ops without feedback (MICE/KNN); artificial sleeps; comparison always reports "Failed" after success. | `epidemiology.py:173-191`; `data_enrichment.py:3005-3021`; `data_monitoring.py:2487-2502` |
| U5 | M | Destructive actions without confirmation: Reset Cache (all users), Reindex (shared vector store), Apply Filtering Criteria. Delete-user button has an empty label. | `home.py:850-858`; `rag_manager.py:295-309`; `users_management.py:112` |
| U6 | M | Design: no theme file; 39 `unsafe_allow_html` blocks, ~270 hard-coded hex colours, light boxes unreadable in dark mode, colour-only semantics (red/green), unescaped data values inside HTML (injection), mixed heading hierarchy, double H1s. | `epidemiology.py:274,337,339-370,1496`; `home.py:817,848` |
| U7 | M | Guidance: the rich `documentation/` handbook is unreachable from the app; static decision tree recommends tests the engine cannot run; no glossary; "AI-Driven Recommendations" header over rule-based advice; interpretation labels overclaim. | `data_quality_dashboard.py:138`; `epidemiology.py:1487-1892` |
| U8 | L | Copy defects: "symetrical", "Upload de Data Files", qualitative picker labelled "quantitative", debug text in UI, duplicate dict key, 22 `icon=""` remnants. | `home.py:1094,1186`; `data_monitoring.py:2406,2428`; `visualization.py:290-291` |

---

---

## Remediation programme

| Phase | Goal | Status |
|---|---|---|
| P0 | Stop the bleeding: key rotation, crashes, admin takeover, secrets, cloud-LLM consent, PHI on stdout, lite CI | done (commit `ea6bd09`); the key was revoked and replaced by the owner on 2026-10-15 |
| P1 | Library-agnostic graph connector (model, builders, yFiles + Plotly adapters, license compliance) | done (commit `3c9b55e`) |
| P2 | Science Tier A: the five wrong-number bugs | done (S4 stage 1: honest labelling, seeds, observed-rows fit; multiple imputation with pooling is P5) |
| P3 | Security hardening: safe expression evaluator, per-user workspace, SQL quoting | done (`core/expressions`, `core/workspace`, lockout; migration script `dev_scripts/migrate_user_layout.py`) |
| P4 | State + UX foundation: typed AppState, st.navigation shell, components, sample cohort, import flow, Settings/consent | done (`ui/`, `ingest/`, `core/state.py`, `app_pages/views/`) |
| P5 | Science Tier B + UI-free biostats package + result cards + guidance | done (multiple imputation with Rubin pooling deferred to P8) |
| P6 | Structure: layering rules, dead code removal, de-duplication, single replay engine | done (function splitting and the remaining `unsafe_allow_html` blocks deferred to P7/P8 clean-up) |
| P7 | Quality gates: full CI, ruff/pyright, pre-commit, AppTest smoke suite, Docker hardening | done (Docker image not built in the audit environment: Docker unavailable) |
| P8 | Science Tier C: survival, regression, Mantel-Haenszel, paired tests, agreement, SMR, multiple imputation | done (`biostats/survival`, `regression`, `rates`, `paired`, `reliability`); multiple imputation delivered as a diagnostic only (`biostats/imputation`: pooled means), pooled analyses moved to follow-up M2 |

## Owner actions required (cannot be done from the repository)

1. Revoke the Google API key that was committed in `fdd29cb`; create a restricted replacement and keep it only in untracked `.env`. **Done 2026-10-15**: the key is revoked and replaced.
2. Purge the key from history (`git filter-repo --replace-text`), force-push, ask collaborators to re-clone, and enable GitHub secret scanning with push protection. Treat the old key as compromised regardless. **Purge done 2026-10-14**: the history was rewritten with `git filter-repo --sensitive-data-removal`, the code moved to `Cohort-Studio` and the former repository was deleted, so its commits, including `fdd29cb`, no longer exist on GitHub. Secret scanning with push protection is enabled on the new repository.
3. Rotate the admin password that was tracked in `.streamlit/secrets.toml`.

## Item status

| Item | Status | Closed by |
|---|---|---|
| G1 yFiles 2.0 crash (`Layout.HIERARCHIC`) | closed | P0 hotfix, P1 connector |
| G2 ignored widget settings / directed edges | closed | P0 hotfix, P1 |
| G3 selection never returned | closed | P1 (`sync_selection` in the adapter; selection drives the panels) |
| G4 license vs air-gapped claims | closed | P1 (opt-in extra, About notice, docs/README wording) |
| G5–G8 coupling, duplication, data defects | closed | P1 (`graph/` builders, palette, deterministic ids) |
| S1–S3, S5 wrong numbers | closed | P2 (`biostats/`) |
| S4 single imputation presented as MICE | closed (2026-10-15, M2, awaiting the owner's review): chained-equations engine with m independent chains (pmm, norm, logreg, polyreg, rf); the Survival & regression and Analyze pages pool their analyses with Rubin's rules (Barnard-Rubin df, D1/D2), with parity against R mice 3.19 and mitml 0.4.5; the imputation is bound to the analysis population, saved with versions, replayed and checked; values filled by single imputation are flagged under the results that use them | P2, P8; M2 |
| S6–S10, S12–S16 soundness | closed (S12 category denominators and S13 UI table of associations remain) | P5 (`biostats/` Tier B) |
| S11 missing methods (survival, regression, MH, paired, agreement) | closed | P8 (Survival & regression page; Paired, rates & stratified tab; reliability section) |
| E1 leaked key | closed (2026-10-15): key revoked and replaced; history purged and the former repository deleted (2026-10-14) | owner action; history purge |
| E2 Visualization import | closed | P0 |
| E3 eval code execution | closed | P3 (`core/expressions` AST whitelist evaluator, `safe_format`) |
| E4 per-user isolation | closed | P3 (`UserScope` workspaces, contained path inputs, parameterised DuckDB paths); P6 (quoted identifiers in the DuckDB benchmark) |
| E5 admin takeover | closed | P0 (role-based checks, env/CLI bootstrap, password policy); P3 (login lockout) |
| E6 cloud LLM by default / key exposure | closed for consent and keys | P0; Settings-page UX in P4 |
| E7 PHI on stdout | closed | P0 |
| E8 stale state / cache | closed (2026-10-14): one write path (`AppState` over `core/session.py`), dataset views reinstalled every run (strict mode in the tests), the Enrich draft cannot be applied over newer data or another population, results stamped with the data, population and settings (`ui/results.py`), no render-time mutation, inclusion and exclusion criteria as a shared, reversible analysis population | P4, P6; M0; M1 |
| E9 undefined names / empty-state crashes | closed | P0 |
| E10 reproducibility claims | closed (2026-10-14): trace schema 2 (data block per step, SHA-256 chain, `verify_trace`), replay checked step by step against the recorded fingerprints (divergence and non-recomputable steps reported), row ids kept through merges, saves and replays, lineage file for saved versions, dates converted by an explicit traced step, outlier exclusion as a criterion and clipping as a traced step; an adopted multiple imputation is reported as not recomputable; since M2 a multiple imputation (`mi_impute`) is re-run by the replay and its imputed cells checked | P6; M1; M2 |
| E11 per-rerun expensive work | partially closed (upload hash once per file, AI probes cached and moved to Settings) | P4; remaining DataAnalyzer/audit recomputation in P6 |
| E12 tracked secrets/config | closed | P0 |
| E13 layering / dead code / duplication | closed for layering (test-enforced), dead code and duplicates; giant functions (59 > 100 lines) remain | P6; splitting with P7/P8 |
| E14 quality gates | closed | P0 (lite CI); P7 (ruff configuration, pyright, coverage floor, pre-commit, source gates, hardened Docker) |
| U1–U3 login, navigation, onboarding, import | closed | P4 (`st.navigation`, Start page, sample cohort, import flow) |
| U5 destructive actions | closed (cache reset, cached statistics, apply filter); reindex confirmation pending | P4, P6 |
| U6 design | partially closed (theme tokens, components, interpretation quadrants); 30 `unsafe_allow_html` blocks remain, mostly the educational HTML tables | P4, P6; P7/P8 clean-up |
| U4, U7 feedback and guidance | closed (result cards, test chooser, glossary, handbook, Run button on the Analyze page, no artificial sleeps); `long_op` adoption on the legacy pages pending | P5, P6 |
| U8 copy | open | P6 |

## Follow-up programme (2026-10-14)

An external architecture review (October 2026) judged the engine strong and the orchestration between
enrichment, models and the UI weak. Each of its claims was checked against the code. The checks confirmed
and extended the orchestration findings: S4, E8 and E10 above are reopened. The owner approved the following
order, with a review after each milestone.

| Milestone | Goal | Status |
|---|---|---|
| M0 | Truth and safety net: false regulatory and reproducibility claims removed (21 CFR Part 11, "signed", "immutable", STROBE/CONSORT "compliant", 100 % coverage), MI advice corrected, result caches cleared on a dataset swap, Validate filter reaches Enrich, traces written inside the user workspace, wheel contents fixed, per-package coverage floors in CI | done |
| M1 | One analysis population and one source of truth: inclusion and exclusion criteria become a shared, reversible selection (only an explicit, traced "freeze" deletes rows); single mutation path; data and population fingerprints; results stamped with the data they were computed on; hash-chained (tamper-evident) trace v2 with attrition counts | done |
| M2 | Multiple imputation end to end: chained-equations engine (PMM, logistic, polytomous), pooled analyses with Rubin's rules (Barnard-Rubin degrees of freedom, D1/D2 multi-parameter tests, scale transformations), parity tests against R `mice` | done (2026-10-15), awaiting the owner's review |
| M3 | Participant flow (STROBE item 13) and study dossier (Word) built from the trace | done (2026-10-16), awaiting the owner's review |
| M4 | Packaging and deployment: optional dependency groups, engine-only CI job, Docker image built in CI, first-run administrator | planned |

Owner decisions on the review: Python stays at 3.14; no repository split; staleness is handled with
fingerprints and stamped results, not a workflow-orchestration framework.
