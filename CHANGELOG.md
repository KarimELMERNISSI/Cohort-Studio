# Changelog

All notable changes to the Cohort Monitoring Package are documented here.

---

## [Unreleased]

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
