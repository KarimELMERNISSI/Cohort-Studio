"""
What the architecture explorer says of CohortStudio: the bands (layers), the bricks (nodes) with their words, the
features that highlight a path through them, the step-by-step journeys, the rules, the recipes and the matrix.

Only words are written here. Everything that can be counted or read from the code is computed by
``build_explorer.py``: which modules a brick holds, their sizes, the import arrows between bricks, the libraries each
brick and each row of the matrix imports with their locked versions, the catalogues' kinds and the connector
families' names.

A brick's ``modules`` are exact module names, or ``package.*`` for a package and everything under it. Every module
of the code base belongs to exactly one brick, and a brick sits on the layer of its modules: the build stops
otherwise. A brick's ``slug`` is its name in an address (``#/architecture/bricks/<slug>``): documents link to it, so
it is never renamed.
"""

from __future__ import annotations

# ----------------------------------------------------------------------------- the bands, top to bottom

BANDS = [
    {
        "id": "E",
        "label": "ENTRY POINTS",
        "title": "Above every layer",
        "path": "2 files",
        "group": "entry",
    },
    {"id": "L9", "label": "LAYER 9", "title": "Pages", "path": "app_pages/", "group": "interface"},
    {
        "id": "L8",
        "label": "LAYER 8",
        "title": "Interface kit",
        "path": "ui/",
        "group": "interface",
        "boundary_after": True,
    },
    {"id": "L7", "label": "LAYER 7", "title": "Services", "path": "manage/", "group": "services"},
    {"id": "L6", "label": "LAYER 6", "title": "Graphs", "path": "graph/", "group": "services"},
    {
        "id": "L5",
        "label": "LAYER 5",
        "title": "Audit & changes",
        "path": "explore/ · monitor/",
        "group": "data",
    },
    {"id": "L4", "label": "LAYER 4", "title": "Enrichment", "path": "enrich/", "group": "data"},
    {"id": "L3", "label": "LAYER 3", "title": "Reading files", "path": "ingest/", "group": "data"},
    {"id": "L2", "label": "LAYER 2", "title": "Domain kernel", "path": "core/", "group": "engine"},
    {
        "id": "L1",
        "label": "LAYER 1",
        "title": "Statistics · prompts",
        "path": "biostats/ · prompts/",
        "group": "engine",
    },
    {"id": "L0", "label": "LAYER 0", "title": "Platform", "path": "utils/", "group": "foundation"},
    {
        "id": "T",
        "label": "ACROSS",
        "title": "Tests & gates",
        "path": "tests/ · CI",
        "group": "tests",
    },
]

GROUPS = {
    "entry": "Entry points",
    "interface": "Interface",
    "services": "Services",
    "data": "Data work",
    "engine": "Data & statistics engine",
    "foundation": "Platform",
    "tests": "Tests & gates",
}

# what is said of an older module of utils/ shown in the brick it serves
SERVES = "lives in utils/ but serves this brick: the layering test places it on this layer until it moves."

# ----------------------------------------------------------------------------- the bricks

NODES = [
    # --- entry points
    {
        "id": "entry_app",
        "slug": "web-application",
        "band": "E",
        "short": "Web application",
        "sub": "streamlit run main.py",
        "title": "Web application (main.py)",
        "modules": ["main"],
        "role": (
            "What the browser reaches. main.py first loads .env and checks every setting, and checks that each "
            "connector setting names a connector of this version. Then it configures the logging, sets one BLAS "
            "thread for the process, turns third-party tracing off, makes the first administrator when one is "
            "configured, installs the expression workers, asks for sign-in, and builds the navigation."
        ),
        "rule": (
            "A wrong setting, an unknown connector or an administrator's password that cannot be used stops the "
            "start. The message names the setting, never its value."
        ),
        "methods": [
            (
                "<code>load_settings()</code> runs first: <code>.env</code> is read once per server process, and a "
                "variable already set wins."
            ),
            (
                "Start-up work runs once per process (<code>st.cache_resource</code>): the BLAS thread, the first "
                "administrator, the workers."
            ),
            (
                "<code>ui.nav</code> holds the one list of pages; <code>st.navigation</code> is built from it, and "
                "Streamlit runs each page by its path: no module imports a page."
            ),
        ],
    },
    {
        "id": "entry_cli",
        "slug": "accounts-command-line",
        "band": "E",
        "short": "Command line",
        "sub": "python manage_users.py",
        "title": "Command line of the accounts (manage_users.py)",
        "modules": ["manage_users"],
        "role": (
            "The accounts without a browser: <code>python manage_users.py</code> creates an administrator, "
            "activates or deactivates an account, resets a password, lists the accounts and writes a setup token "
            "(create-admin, activate, deactivate, reset-password, list, setup-token, bootstrap). The other "
            "command, <code>python -m manage.replay</code>, enters at the services: its modules are in the brick "
            "Replay & package."
        ),
        "rule": (
            "R5: a headless entry point reaches neither Streamlit nor the AI packages. A password is never an "
            "argument of a command."
        ),
        "methods": [
            (
                "A password is typed unseen on a terminal. On Windows the console itself is asked, since the null "
                "device says it is a terminal."
            ),
            (
                "It reaches the accounts through the same facade as the app (<code>manage.iam</code>), so one "
                "password rule holds for both."
            ),
        ],
    },
    # --- L9 pages
    {
        "id": "pages_data",
        "slug": "pages-data",
        "band": "L9",
        "short": "Data pages",
        "sub": "Import · Validate · Enrich",
        "title": "Data pages: Import, Validate, Clean, Enrich",
        "modules": [
            "app_pages.data_monitoring",
            "app_pages.data_quality_dashboard",
            "app_pages.validate_dates",
            "app_pages.data_cleaning",
            "app_pages.clean_categories",
            "app_pages.clean_dialog",
            "app_pages.data_enrichment",
            "app_pages.enrichment_derived",
            "app_pages.enrichment_mi",
            "app_pages.views.import_data",
            "app_pages.views.validate",
            "app_pages.views.clean",
            "app_pages.views.enrich",
        ],
        "role": (
            "The four pages of the Data section. Import proposes how a file is read and lets the user override "
            "it. Validate holds the quality dashboard, the plausibility rules, the inclusion criteria and the "
            "outliers. Clean recodes categories and reads text as dates. Enrich merges files, computes variables, "
            "imputes and transforms, on a draft that is applied or discarded."
        ),
        "rule": "A page changes the data only through AppState (one trace step per change) and builds no service.",
        "methods": [
            (
                'Validate runs only its open tab (<code>st.tabs(..., on_change="rerun")</code>, then '
                "<code>if tab.open:</code>)."
            ),
            (
                "Enrich works on a draft (<code>AppState.stage</code>): each prepared change writes a snapshot, "
                "deleted with a discarded draft."
            ),
            "The multiple-imputation panel is a guided flow in five steps (<code>enrichment_mi.py</code>).",
        ],
    },
    {
        "id": "pages_explore",
        "slug": "pages-explore",
        "band": "L9",
        "short": "Explore & knowledge",
        "sub": "statistics · charts · maps",
        "title": "Explore & Knowledge pages",
        "modules": [
            "app_pages.home",
            "app_pages.visualization",
            "app_pages.data_insight",
            "app_pages.document_insight",
            "app_pages.views.explore",
            "app_pages.views.visualize",
            "app_pages.views.variable_map",
            "app_pages.views.documents",
        ],
        "role": (
            "Describing and looking at the data: Statistics & correlations, Visualize (distributions, categories, "
            "time series, relationships), the Variable map (a taxonomy of the variables with their formulas, as a "
            "knowledge graph) and Documents (a literature graph and a chat over the user's documents)."
        ),
        "rule": (
            "Statistics are cached per session, keyed by the table's fingerprint: a Reset clears that session's "
            "only."
        ),
        "methods": [
            "Statistics & correlations runs only its open tab.",
            (
                "A graph is described once (<code>graph.model.GraphSpec</code>) and drawn by the graph engine the "
                "user chose."
            ),
            (
                "About a dataset, an AI feature sends a summary of each column, never a row "
                "(<code>manage.ai_payload</code>). The Documents page sends passages of the indexed documents and the "
                "chat messages; for a document's graph, the text of the PDF, indexed or not (to Gemini, the file "
                "itself)."
            ),
        ],
        "note": (
            "Visualize computes its group tests itself, and Statistics & correlations its correlations with "
            "pandas: neither goes through biostats."
        ),
    },
    {
        "id": "pages_analyze",
        "slug": "pages-analyze",
        "band": "L9",
        "short": "Analysis pages",
        "sub": "tests · measures · models",
        "title": "Analyze pages: Which test?, Hypothesis tests & epidemiology, Survival & regression",
        "modules": [
            "app_pages.epidemiology",
            "app_pages.analysis_config",
            "app_pages.analysis_diagnostic",
            "app_pages.analysis_extras",
            "app_pages.analysis_guidance_ui",
            "app_pages.analysis_layout",
            "app_pages.analysis_pooled",
            "app_pages.analysis_survival",
            "app_pages.analysis_survival_results",
            "app_pages.group_figures",
            "app_pages.views.analyze",
            "app_pages.views.survival",
            "app_pages.views.which_test",
        ],
        "role": (
            "Where an analysis is chosen, run and read: group comparisons, epidemiological measures, diagnostic "
            "accuracy and agreement, paired tests, rates and stratified measures, calculators from numbers typed, "
            "survival and regression. Group comparisons, ANCOVA, paired and one-sample t-tests, 2x2 and r x c "
            "tables and the models also run on multiply imputed data and are pooled; the rank tests, Fisher's "
            "exact test and the other analyses read the observed rows, and the page says why."
        ),
        "rule": (
            "The Analyze pages take their numbers from biostats, from two older helpers of utils/ (the "
            "categorical comparison, the ANCOVA diagnostics), and for the figure's error bars from the page "
            "itself. A result is kept with what it was computed on, so it is shown again only while its data, "
            "population and settings still hold."
        ),
        "methods": [
            (
                "The model page is <code>analysis_survival.py</code>: every control has a key; on the button, "
                "<code>specify(...)</code>, <code>run(spec, df, mi=mi)</code>, <code>remember(..., spec=done.spec)"
                '</code>, <code>mark_step("analyse")</code>.'
            ),
            "<code>provenance(key)</code> says what a result was computed on and offers to add it to the dossier.",
            (
                "With multiply imputed data, <code>analysis_pooled.py</code> pools the group comparisons and ANCOVA; "
                "the models are pooled through <code>biostats.plan</code>."
            ),
        ],
    },
    {
        "id": "pages_reproduce",
        "slug": "pages-reproduce",
        "band": "L9",
        "short": "Reproduce pages",
        "sub": "traces · replay · dossier",
        "title": "Reproduce pages: Versions & traces, Study dossier",
        "modules": [
            "app_pages.reproduce_analysis",
            "app_pages.study_dossier",
            "app_pages.views.reproduce",
            "app_pages.views.dossier",
        ],
        "role": (
            "What was done, and done again: the saved versions with their traces, the replay of a trace on its "
            "files, and the study dossier (the participant flow, the results added to the report, the Word file)."
        ),
        "rule": (
            "The replay never reads the session: it runs the recorded steps on the files the trace names, found "
            "through a resolver limited to the user's workspace."
        ),
        "methods": [
            "A large imputation is drawn again in the worker processes that are free.",
            "A result that fails writes one line in its place in the dossier, never the whole download.",
            "A cut-off found on the data that calls too few participants is written without its value.",
        ],
    },
    {
        "id": "pages_shell",
        "slug": "pages-start-admin",
        "band": "L9",
        "short": "Start, help & admin",
        "sub": "Start · Settings · Users",
        "title": "Start, Help & Admin pages",
        "modules": [
            "app_pages.about",
            "app_pages.rag_sidebar",
            "app_pages.rag_monitoring",
            "app_pages.users_management",
            "app_pages.views",
            "app_pages.views.start",
            "app_pages.views.handbook",
            "app_pages.views.settings",
            "app_pages.views.about",
            "app_pages.views.users",
            "app_pages.views.ai_quality",
        ],
        "role": (
            "The Start page (the dataset's figures, the workflow map, saved versions, the sample cohort), the "
            "handbook shown in the app, Settings (the AI engine and its confirmation, the graph engine, import "
            "defaults), About, and for administrators Users and AI quality."
        ),
        "rule": (
            "Sending data outside the private network (a cloud AI provider, an Ollama cloud model, an Ollama or a "
            "Qdrant server outside it) needs a confirmation in each browser tab. A key, a password or a "
            "confirmation is never kept in the preferences."
        ),
        "methods": [
            (
                "Settings shows each connector choice with its reasons through one control "
                "(<code>ui.components.connector_selector</code>)."
            ),
            (
                "The confirmation names each recipient and what it receives, and holds for one browser tab; "
                "unticking it stops the engine."
            ),
            "The Users page manages the accounts of the configured identity provider.",
            "The handbook works offline: it is the repository's own documentation.",
        ],
    },
    # --- L8 ui
    {
        "id": "ui_shell",
        "slug": "ui-shell",
        "band": "L8",
        "short": "Shell & sign-in",
        "sub": "nav · theme · auth · workflow",
        "title": "Shell, navigation & sign-in",
        "modules": [
            "ui",
            "ui.nav",
            "ui.shell",
            "ui.theme",
            "ui.auth",
            "ui.workflow",
            "ui.progress",
            "ui.waiting",
            "ui.session_scope",
            "ui.import_flow",
        ],
        "role": (
            "What surrounds every page: the single list of pages, the sidebar with the dataset card and the user "
            "menu, the design tokens and the Plotly template, the sign-in card, the research workflow of the "
            "Start page with its progress, the three-step import flow, and the line shown while a formula waits "
            "on a worker."
        ),
        "rule": (
            "One registry of pages: a link to a page, or to one of its tabs, is refused when the page does not "
            "declare it."
        ),
        "methods": [
            "<code>ui.nav.PageDef</code> lists each page's file, title, URL path, section and tabs.",
            "The workflow's ticks come from the trace (<code>core.progress</code>), not from the pages' memory.",
            "<code>ui.waiting.computing</code> shows one line after a second, with Stop.",
        ],
    },
    {
        "id": "ui_state",
        "slug": "ui-state",
        "band": "L8",
        "short": "AppState & results",
        "sub": "state · keys · results · services",
        "title": "AppState, results & the session's services",
        "modules": ["ui.state", "ui.keys", "ui.results", "ui.services", "ui.preferences"],
        "role": (
            "The session's side of the engine. AppState is the only code that changes the data, through "
            "core.session.DataSession. ui.results keeps each analysis result with what it was computed on. "
            "ui.services builds the services once and hands them to the pages. ui.keys declares the session keys "
            "that one module writes and another reads."
        ),
        "rule": "Only AppState writes the dataset and the trace steps; pages never build a service (quality gates).",
        "methods": [
            (
                "<code>commit_change()</code> for the dataset, <code>stage()</code> for the Enrich draft, "
                "<code>note()</code> for a step that changes nothing: each is one trace step."
            ),
            "The analysis population is a view reinstalled at the start of every run (<code>begin_run</code>).",
            (
                "A declared key has an owner and a lifetime (session, page, dataset, data change) that decides what is "
                "cleared."
            ),
        ],
    },
    {
        "id": "ui_components",
        "slug": "ui-components",
        "band": "L8",
        "short": "Components & guidance",
        "sub": "components · guidance",
        "title": "Components & guidance",
        "modules": [
            "ui.components.*",
            "ui.guidance.*",
            "ui.graph_view",
            "ui.graph_legend",
            "ui.quality_charts",
        ],
        "role": (
            "The reusable pieces of the pages (the page header, empty states, the population selector and banner, "
            "the participant flow diagram, the connector selector, the study strip, the tabs registry) and the "
            "guidance: the test chooser, the plan shown before a run, the glossary, how a number is written."
        ),
        "rule": (
            "Text that comes from outside (a model's answer, an imported file) is shown as it is written, never "
            "as markup."
        ),
        "methods": [
            "<code>ui.components.tabs</code>: a page's tabs as a registry, of which only the open one runs.",
            (
                "<code>ui.guidance.test_chooser</code>: four questions lead to the analysis the application runs, and "
                "why, or say that it is not offered yet."
            ),
            (
                "The result card's reading (<code>ui.guidance.interpretation</code>) starts from the confidence "
                'interval: never "proves", never "no effect exists".'
            ),
        ],
    },
    # --- L7 manage
    {
        "id": "svc_accounts",
        "slug": "manage-accounts",
        "band": "L7",
        "short": "Accounts",
        "sub": "manage/iam · 3 providers",
        "title": "Accounts: the identity providers",
        "modules": ["manage.iam.*"],
        "role": (
            "Sign-in and accounts behind one facade (IAMManager), on a connector family of identity providers: "
            "local accounts in DuckDB with bcrypt hashes (the default), a Better Auth service through its admin "
            "API, or no sign-in, for a researcher's own machine: every visitor is then one administrator, with "
            "one workspace."
        ),
        "rule": (
            "The role the provider stores grants access, never the username. One rule for every password set in "
            "CohortStudio, applied by the facade itself."
        ),
        "methods": [
            "An unknown <code>CMP_IAM_PROVIDER</code> stops the start and lists the valid names.",
            (
                "Without a configured password, the first start writes a one-time setup token to the server log; "
                "the data root keeps only its hash, and the token stops working once used."
            ),
            "The local provider locks an account for a time after consecutive failed sign-ins.",
            (
                "Better Auth is reached at <code>CMP_BETTER_AUTH_URL</code> and receives each sign-in's name and "
                "password, and the passwords set for its accounts; the code neither lists nor places that address."
            ),
        ],
    },
    {
        "id": "svc_ai",
        "slug": "manage-ai",
        "band": "L7",
        "short": "AI engine & vectors",
        "sub": "rag · vectorstores",
        "title": "AI engine & vector databases",
        "modules": [
            "manage.rag.*",
            "manage.rag_computed_vars",
            "manage.rag_documents",
            "manage.rag_evaluator",
            "manage.rag_manager",
            "manage.rag_schemas",
            "manage.rag_taxonomy",
            "manage.vectorstores.*",
            "manage.ai_payload",
            "manage.server_addresses",
            "utils.llm_utils",
        ],
        "role": (
            "The optional AI features: chat and embedding connectors for five providers (Ollama by default; "
            "Gemini, OpenAI, Mistral, Claude), the document index on a vector database (LanceDB embedded, or "
            "Qdrant), and the engine that proposes taxonomies, formulas and document graphs."
        ),
        "rule": (
            "R4: a provider's SDK is imported only in manage/rag/providers. About a dataset, what may be sent is "
            "a summary of each column and, in Data > Clean, the values of one variable without their counts (all "
            "of them to an Ollama on this machine, to any other only those held by at least 5 rows); never a row; "
            "values are counted by rows, not by people. A user chooses a server's address from the server's list "
            "and never types one; only an administrator adds one."
        ),
        "methods": [
            (
                "Providers are declared once (<code>manage/rag/providers/declarations.py</code>): what Settings, the "
                "factory and the confirmation read."
            ),
            (
                "A provider answers, embeds the documents, or both. Claude has no embedding model: another provider "
                "embeds (Ollama by default), receives the text of the indexed PDFs and every search text, and the "
                "confirmation names it apart."
            ),
            "The server log keeps each confirmation's account, recipients and time, never what is sent.",
            "A model's structured answer is parsed against a Pydantic schema.",
        ],
        "note": "utils/llm_utils.py " + SERVES,
    },
    {
        "id": "svc_repro",
        "slug": "manage-replay",
        "band": "L7",
        "short": "Replay & package",
        "sub": "replay · package · inspect",
        "title": "Trace, replay & study package",
        "modules": [
            "manage.replay.*",
            "manage.transformation_manager",
            "manage.trace_documenter",
            "manage.paths",
            "manage.snapshots",
            "manage.package_zip",
            "manage.protocol_store",
            "manage.mi_store",
        ],
        "role": (
            "What lets a study be done again. The transformation manager records each change as a step of a "
            "hash-chained trace. The replay engine runs a trace without Streamlit on the files it names. The "
            "study definition is saved with numbered revisions. The package's zip has one writer, and a safe "
            "reader for a received file. <code>python -m manage.replay inspect</code> says what a received "
            "package holds, its hash before any password, without replaying it."
        ),
        "rule": (
            "A trace is user-supplied JSON: the replay reads only files found by its resolver. A received zip is "
            "read by a reader that refuses what the writer would not have written."
        ),
        "methods": [
            (
                "One engine for every caller: it runs without Streamlit and reads only the files its resolver gives "
                "it."
            ),
            (
                "A step is recomputed, read from its snapshot or only documented, as its kind says "
                "(<code>core.steps</code>)."
            ),
            (
                "In a package that holds the data, every member but its notice and its manifest is encrypted "
                "(AES-256) with a generated password; a package without the data has no password."
            ),
            (
                "<code>inspect</code> loads no setting and no engine, and writes back nothing typed. Its exit codes: "
                "0 read, 1 not the package its sender made, 2 not read, 64 not understood, 70 an error of the "
                "program."
            ),
        ],
    },
    {
        "id": "svc_report",
        "slug": "manage-dossier",
        "band": "L7",
        "short": "Dossier & reports",
        "sub": "dossier · report_tables",
        "title": "Study dossier & report tables",
        "modules": ["manage.dossier", "manage.report_tables", "manage.flow_figure"],
        "role": (
            "The Word dossier of a study: what a methods section and its appendix need. report_tables turns each "
            "result into report blocks (tables, lines of text, references, a Kaplan-Meier figure). flow_figure "
            "draws the participant flow as a picture."
        ),
        "rule": (
            "A result is written from its record when its kind has a record block (a share, rates, a "
            "standardised ratio, every calculation from numbers typed), and otherwise from the stored result. "
            "Its numbers are written by the pages' own rule (utils.numbers)."
        ),
        "methods": [
            "<code>result_blocks</code> gives each type of result its tables; an unknown type gets a short text.",
            "A calculation from numbers typed has its own section, with its typed source.",
            "The flow picture has the same boxes and geometry as the app's diagram.",
        ],
    },
    {
        "id": "svc_runtime",
        "slug": "manage-services",
        "band": "L7",
        "short": "Services & workers",
        "sub": "services · workers · stores",
        "title": "Composition root, workers & stores",
        "modules": [
            "manage.services",
            "manage.connectors",
            "manage.compute",
            "manage.workers.*",
            "manage.preferences",
            "manage.db_manager",
        ],
        "role": (
            "Where the services are built, once: build_shared per server process (the settings, the identity "
            "provider, the graph engines, the imputation's worker budget) and build_services per session (the "
            "user's scope, the path resolver, the dataset store, the trace). Also the pool of worker processes "
            "that compute formulas and criteria written as expressions, and the user's preferences."
        ),
        "rule": (
            "A service takes the user's scope as a parameter and never asks the session for the user. An "
            "expression is computed in a worker process, under a time limit and a memory limit (the time limit "
            "alone on macOS)."
        ),
        "methods": [
            (
                "A worker is an isolated interpreter (<code>-I -B</code>) in an empty folder, with no setting or key "
                "in its environment."
            ),
            (
                "One worker budget per process for the multiple imputation: a run leases what is free and never "
                "waits."
            ),
            "Preferences are written whole and atomically, per user.",
        ],
        "note": (
            "A development switch, CMP_FORMULA_IN_PROCESS, keeps expressions in the server's own process, "
            "without a limit; the log warns of it."
        ),
    },
    # --- L6 graph
    {
        "id": "graph",
        "slug": "graph",
        "band": "L6",
        "short": "Graph model, builders & engines",
        "sub": "GraphSpec · taxonomy, formula, correlation, documents · Plotly, Graphviz, yFiles",
        "title": "Graphs: model, builders & engines",
        "modules": ["graph.*"],
        "role": (
            "A graph is described once, free of any library (GraphSpec: nodes and edges with semantic "
            "categories), by pure builders (the variable taxonomy with its formulas, a formula's diagram, a "
            "correlation network, the documents' knowledge graph), then drawn by one of three graph engines (the "
            "renderers of graph/renderers/)."
        ),
        "rule": (
            "R3: a renderer imports Streamlit inside its drawing methods only. Plotly and Graphviz draw with no "
            "network; yFiles is never in the automatic order."
        ),
        "methods": [
            "The graph engines are a connector family (<code>GRAPH_RENDERERS</code>), each imported on first use.",
            "The Plotly renderer takes its positions from NetworkX layouts.",
            (
                "With yFiles, the user's browser loads its runtime from yworks.com and hands it the whole graph; "
                "<code>CMP_OFFLINE</code> turns it off."
            ),
            "Colours, styles and legend entries have one source (<code>graph.palette</code>).",
        ],
    },
    # --- L5
    {
        "id": "explore",
        "slug": "explore-audit",
        "band": "L5",
        "short": "Quality audit & correlations",
        "sub": "audit · quality index · correlation matrix",
        "title": "Quality audit, quality index & correlations",
        "modules": [
            "explore.*",
            "utils.analysis_utils",
            "utils.epidemiology_utils",
            "utils.statistics_utils",
            "utils.visualization_utils",
        ],
        "role": (
            "The audit of a dataset by dimension, the data-quality index (0 to 100) as the weighted mean of those "
            "dimensions with its decomposition, and descriptive correlation matrices labelled by conventional "
            "strength bands."
        ),
        "rule": "The strength thresholds have one source. The index is descriptive, and explains its own parts.",
        "methods": [
            "Rules are counted through <code>core.criteria</code>, as the population counts them.",
            "The audit's outlier dimension uses the detector of the Validate page (<code>biostats.outliers</code>).",
            "Correlations are exported as a workbook (openpyxl).",
        ],
        "note": (
            "Four older helpers of utils/ (analysis, epidemiology, statistics, visualization) live in utils/ but "
            "are placed on this layer by the layering test until they move. Two of them serve the Analyze page: "
            "the categorical group comparison (chi-square, or Fisher's exact test by the expected counts) and the "
            "ANCOVA diagnostics."
        ),
    },
    {
        "id": "monitor",
        "slug": "monitor-changes",
        "band": "L5",
        "short": "Changes between versions",
        "sub": "monitor/changes.py",
        "title": "Changes between two versions",
        "modules": ["monitor.*"],
        "role": (
            "Compares two tables by an identifier, exactly and on the columns both hold: the rows only one of "
            "them holds, the rows that changed, and a workbook that shows them (Validate's Compare tab)."
        ),
        "rule": "It reads its files with the same loader as the import (ingest.loaders).",
        "methods": [
            "<code>compare_dataframes(df1, df2, id_column, ...)</code> lists what differs, cell by cell.",
            "explore and monitor share a layer and do not import each other (R1).",
        ],
    },
    # --- L4 enrich
    {
        "id": "enrich_formulas",
        "slug": "enrich-formulas",
        "band": "L4",
        "short": "Formulas & derived variables",
        "sub": "formulas · derived · transformations",
        "title": "Formulas, derived variables & transformations",
        "modules": [
            "enrich.formulas",
            "enrich.derived",
            "enrich.transformation_logic",
            "utils.clustering_utils",
        ],
        "role": (
            "New variables from existing ones: a formula evaluated row by row through the safe evaluator, a "
            "declared derived variable checked against the recorded values (a BMI from the weight and the "
            "height), and the transformations (statistics, scaling, dimensionality reduction, clustering, "
            "encoding)."
        ),
        "rule": (
            "A formula is never run as code: it goes through core.expressions. A derived variable's check records "
            "counts and a tolerance, never a row's value."
        ),
        "methods": [
            (
                "A row agrees when the recorded value is within a tolerance of the computed one, taken from each "
                "column's recording resolution."
            ),
            (
                "Under multiple imputation, a derived variable is passively imputed (impute, then transform): it is "
                "neither imputed nor used as a predictor, and is computed again from its imputed inputs "
                "(<code>impute_and_derive</code>)."
            ),
            "A transformation applied twice gives a new, numbered column, on the page and on replay alike.",
        ],
        "note": "utils/clustering_utils.py " + SERVES,
    },
    {
        "id": "enrich_impute",
        "slug": "enrich-imputation",
        "band": "L4",
        "short": "Imputation, recoding & merges",
        "sub": "imputation · recode · external data",
        "title": "Single imputation, recoding & merges",
        "modules": [
            "enrich.data_imputation",
            "enrich.imputation_service",
            "enrich.recode",
            "enrich.patterns",
            "enrich.external_data",
            "utils.miss_forest.*",
        ],
        "role": (
            "Filling, recoding and merging: single imputation of the session data (mean, median, most frequent, "
            "k nearest neighbours, iterative regression, MissForest), the values of a categorical variable "
            "recoded into a new column, and an external file merged on a key."
        ),
        "rule": (
            "Single imputation is for exploration: the pages say that its uncertainty is not carried into the "
            "results. The application logs no value of the data."
        ),
        "methods": [
            (
                "Iterative regression is scikit-learn's IterativeImputer: one value per cell and no draws. It is not "
                "the multiple imputation's chained equations."
            ),
            (
                "MissForest's learners are LightGBM gradient-boosted trees, run in their deterministic mode with a "
                "recorded seed."
            ),
            "Recoding keeps the original column; near-matches are suggestions only.",
            "A merge records its file's checksum, so that a replay knows it reads the same file.",
        ],
        "note": "utils/miss_forest/ " + SERVES,
    },
    # --- L3 ingest
    {
        "id": "ingest",
        "slug": "ingest",
        "band": "L3",
        "short": "Loaders, dates & column profile",
        "sub": "CSV, Excel, Parquet with a parse report · dates read with an explicit format · DataAnalyzer",
        "title": "Loaders, dates & column profile",
        "modules": ["ingest.*", "utils.multipage"],
        "role": (
            "Reads CSV, Excel and Parquet files with explicit options the user can override, and reports what it "
            "did (the encoding, the delimiter, the lines that could not be read by count and line number). Reads "
            "text columns as dates with an explicit format. Profiles and categorises the columns."
        ),
        "rule": (
            "One loader for every entry point: the import, an external file, the comparison and the replay read "
            "a file the same way."
        ),
        "methods": [
            (
                "The import flow proposes the encoding, the delimiter and the decimal mark; nothing is guessed "
                "silently."
            ),
            "A date column is read with a declared format (ISO, day first, month first, compact, Excel serial).",
            "<code>DataAnalyzer</code> categorises the columns once per data fingerprint.",
        ],
        "note": "utils/multipage.py " + SERVES,
    },
    # --- L2 core
    {
        "id": "core_data",
        "slug": "core-session",
        "band": "L2",
        "short": "Data & population",
        "sub": "session · criteria · flow",
        "title": "Data session, analysis population & criteria",
        "modules": [
            "core",
            "core.session",
            "core.state",
            "core.population",
            "core.criteria",
            "core.attrition",
            "core.derived",
            "core.progress",
        ],
        "role": (
            "The session's data without Streamlit: the committed table, the Enrich draft and the analysis "
            "population. Inclusion and exclusion criteria are filters the user ticks. The participant flow gives, "
            "stage by stage, the rows each criterion is the first to exclude and the numbers analysed, for "
            "reporting STROBE item 13 (rows are participants when there is one row per participant)."
        ),
        "rule": (
            "A criterion never deletes a row: Keep only these rows is the one action that deletes the rows the "
            "criteria exclude. A missing value meets no condition unless an expression asks for it (isna()): an "
            "exclusion rule keeps the participant, an inclusion criterion excludes them, and the flow counts them "
            "apart."
        ),
        "methods": [
            (
                "One function decides the rows a criterion fires on, for the population, the flow and the audit "
                "(<code>criterion_mask</code>); the criterion's kind decides how."
            ),
            "Criterion kinds are a catalogue (<code>CRITERION_KINDS</code>).",
            "<code>DataSession.memo</code> keeps a computation until the data fingerprint or its key changes.",
        ],
    },
    {
        "id": "core_expr",
        "slug": "core-expressions",
        "band": "L2",
        "short": "Safe expressions",
        "sub": "core/expressions",
        "title": "Safe expressions",
        "modules": ["core.expressions.*"],
        "role": (
            "Evaluates the expressions that reach the application from users, configuration files and language "
            "models: a whitelisting evaluator for pandas and numpy expressions, read as a criterion in "
            "three-valued logic, as SQL's WHERE reads it."
        ),
        "rule": (
            "No eval anywhere else in the code (a quality gate). The evaluator offers no function that reads "
            "memory it was not given, or a file."
        ),
        "methods": [
            "An expression is parsed to a tree; only whitelisted nodes and names are evaluated.",
            (
                "<code>evaluate_tree</code> hands the tree to the runner installed in the process: the worker pool in "
                "the app, this process in a notebook."
            ),
            (
                "A stop at the time or memory limit is the expression's own failure, remembered and said in its own "
                "words."
            ),
        ],
    },
    {
        "id": "core_trace",
        "slug": "core-lineage",
        "band": "L2",
        "short": "Lineage & steps",
        "sub": "lineage · steps · profiles",
        "title": "Lineage, fingerprints & trace steps",
        "modules": ["core.lineage", "core.fingerprint", "core.steps", "core.profiles"],
        "role": (
            "What each change did, kept as a hash-chained sequence of trace steps: an edit, a removal or a "
            "reordering breaks the chain, and it is not a signature. Also content fingerprints of tables and of "
            "row identities, the catalogue of the kinds of trace step, and column profiles written so that two "
            "environments can compare them."
        ),
        "rule": (
            "An id stored in a trace is never renamed: a kind that nothing records any more stays registered as "
            "legacy, so that what was stored still reads."
        ),
        "methods": [
            "Each step's hash covers the step and the hash before it (SHA-256).",
            (
                "A kind declares its title, its category, its replay mode (recompute, snapshot, document) and "
                "whether it changes the data."
            ),
            "Row-id fingerprints are the same on every platform.",
        ],
    },
    {
        "id": "core_study",
        "slug": "core-package",
        "band": "L2",
        "short": "Package & privacy",
        "sub": "protocol · package · checks",
        "title": "Study definition, package formats & privacy",
        "modules": [
            "core.protocol",
            "core.package",
            "core.privacy",
            "core.disclosure",
            "core.verification",
        ],
        "role": (
            "What a study decides, apart from the data it is applied to (the definition, with a fingerprint and "
            "revisions); the formats of a study package (.cstudio.zip) with their one writer and strict reader; "
            "the study's privacy choices; the small-cell rule of a package shared without the data; and the "
            "catalogue of the checks of a received package."
        ),
        "rule": (
            'In a package shared without the data, a count of participants is written exactly, or as "fewer '
            'than k": k is 10 by default, and never below 5.'
        ),
        "methods": [
            "The package hash is the SHA-256 of its manifest, compared before any password is typed.",
            (
                "A check reads identical, within tolerance, within Monte Carlo error (a pooled result whose "
                "imputations were drawn again), not comparable, not run or failed."
            ),
            (
                "A text that is not the format's own is shown through <code>shown()</code>, which escapes what a "
                "terminal would interpret."
            ),
        ],
        "note": "The checks are declared here; the command that runs them on a received package is not built yet.",
    },
    {
        "id": "core_kit",
        "slug": "core-kit",
        "band": "L2",
        "short": "Kit & workspace",
        "sub": "registry · workspace",
        "title": "Extension kit & per-user workspace",
        "modules": ["core.registry", "core.workspace"],
        "role": (
            "The two mechanisms that carry the extension points, on the standard library only: a connector "
            "family, where one implementation is chosen per deployment or session, and a catalogue, where kinds "
            "live side by side and the data names which applies. And the per-user storage scope: a directory "
            "tree per user under the data root."
        ),
        "rule": (
            "A path is confined to the user's folder. An unknown connector name in a setting stops the start; an "
            "unknown id raises UnknownIdError with the valid ids."
        ),
        "methods": [
            (
                "<code>ConnectorRegistry.select</code> returns the choice with its source, reasons, warnings and "
                "alternatives."
            ),
            (
                "<code>availability()</code> answers without raising or connecting, and names the optional group to "
                "install."
            ),
            'A connector is registered as <code>"module:Class"</code> and imported on first use.',
        ],
    },
    # --- L1
    {
        "id": "stats_tests",
        "slug": "biostats-tests",
        "band": "L1",
        "short": "Tests & measures",
        "sub": "tests · rates · diagnostic accuracy",
        "title": "Tests, effect sizes & epidemiological measures",
        "modules": [
            "biostats.inference",
            "biostats.paired",
            "biostats.posthoc",
            "biostats.effect_sizes",
            "biostats.contingency",
            "biostats.rates",
            "biostats.diagnostic",
            "biostats.agreement",
            "biostats.reliability",
            "biostats.correlation",
            "biostats.normality",
            "biostats.assumptions",
            "biostats.multiplicity",
            "biostats.power",
            "biostats.zscores",
            "biostats.rank_intervals",
            "biostats.study_designs",
            "biostats.outliers",
            "biostats.missing",
            "biostats.mcar",
            "biostats.reduction",
        ],
        "role": (
            "Group comparisons with a Welch-first policy, paired and one-sample tests, pairwise comparisons, "
            "effect sizes, 2x2 tables with an explicit orientation, incidence rates and stratified measures, "
            "diagnostic accuracy, agreement and reliability, power, missing-data diagnostics, outliers."
        ),
        "rule": (
            "biostats imports nothing else of the project: plain values in, a typed result out. It reads one "
            "setting itself, the random seed (CMP_RANDOM_SEED), until the seed moves into the analysis specs. The "
            "normality and equal-variance checks of an analysis are reported and never choose its test; three "
            "declared rules do: a rank test for small (n < 15), strongly skewed groups; Fisher's exact test for a "
            "2x2 table with an expected count below 5; and the missing-data scan, in its automatic mode, chooses "
            "Welch's t-test or the Mann-Whitney test by a Shapiro-Wilk test."
        ),
        "methods": [
            (
                "A two-group estimate is oriented comparison against reference (a difference: comparison minus "
                "reference; a ratio: comparison over reference), so the signs of the mean difference, Hedges' g and "
                "the rank-biserial r agree. Each comes with its confidence interval when the method gives one."
            ),
            "Rank-based intervals (Hodges-Lehmann) are computed without random draws.",
            (
                "The tests, measures and models cite their reference next to their result "
                "(<code>biostats.references</code>)."
            ),
        ],
        "note": "No page calls biostats.correlation: the app's correlation matrices come from pandas (explore).",
    },
    {
        "id": "stats_models",
        "slug": "biostats-models",
        "band": "L1",
        "short": "Models & survival",
        "sub": "linear · logistic · Cox",
        "title": "Regression models & survival",
        "modules": [
            "biostats.linear",
            "biostats.regression",
            "biostats.survival",
            "biostats.design",
        ],
        "role": (
            "Linear regression and ANCOVA, logistic and count regression with effect tables, Kaplan-Meier curves, "
            "the log-rank test and Cox proportional hazards, on design matrices whose factor levels can be "
            "frozen."
        ),
        "rule": (
            "A logistic model's area under the curve is labelled apparent (in-sample). An event indicator "
            "outside 0 and 1 is refused."
        ),
        "methods": [
            "Factor levels are frozen so that every imputed dataset fits the same model.",
            (
                "The proportional-hazards test is reported as a diagnostic: it never changes the model, and does not "
                "depend on the order of the rows."
            ),
            (
                "statsmodels by default; lifelines (the <code>survival</code> optional group) adds the "
                "proportional-hazards test and the concordance."
            ),
        ],
    },
    {
        "id": "stats_mi",
        "slug": "biostats-mi",
        "band": "L1",
        "short": "Multiple imputation",
        "sub": "biostats/mi · pooling",
        "title": "Multiple imputation & pooling",
        "modules": ["biostats.mi.*", "biostats.imputation", "biostats.processes"],
        "role": (
            "The chained-equations engine (fully conditional specification): m imputed datasets stored compactly, "
            "a method per incomplete variable (by default pmm for a number, logreg for two categories, polyreg "
            "for more; norm and rf on request), pooling by Rubin's rules with multi-parameter tests (D1, D2), and "
            "what cannot be pooled, with the reason."
        ),
        "rule": (
            "Each chain draws from its own seeded stream on one BLAS thread, in the session's process and in the "
            "workers alike, so a run gives the same values in one process or in several. Another machine can "
            "draw other values: the step records the libraries and the processor it drew with."
        ),
        "methods": [
            (
                "Pooling follows R's <code>mice::pool</code> (Rubin's rules, Barnard-Rubin degrees of freedom). "
                "Ratios are pooled on the log scale, proportions and areas on the logit, survival on the "
                "complementary log-log; a chi-square statistic by the D2 rule."
            ),
            (
                "A parallel run starts its worker processes by the module's name (<code>biostats.mi.worker</code>), "
                "never by an import."
            ),
            "An imputation knows what it was fitted on, and whether it still fits the data being analysed.",
        ],
        "note": "No page calls biostats.imputation.",
    },
    {
        "id": "stats_plan",
        "slug": "biostats-plan",
        "band": "L1",
        "short": "Analysis plan",
        "sub": "plan · records · references",
        "title": "Analysis plan, records & references",
        "modules": ["biostats", "biostats.plan", "biostats.references", "biostats._types"],
        "role": (
            "What a stored analysis was asked to compute, and the record of what it gave. Each kind of analysis "
            "is registered once with its title. Eight have their computation here (the survival, Cox, logistic "
            "and count models, and four calculations from numbers typed), so that they can be computed again; "
            "the other kinds have titles only in this version. Also the shared result types and the method "
            "references."
        ),
        "rule": "A spec or a record that holds an unknown key is refused. A kind's inputs are never renamed.",
        "methods": [
            (
                "<code>specify(kind, ...)</code>, then <code>run(spec, frame, mi=...)</code>: the call a page makes, "
                "kept with the result so that it can be computed again."
            ),
            (
                "A calculation from numbers typed is a function of its numbers, kept as JSON and computed again "
                "wherever it is read."
            ),
            "A record holds every number and every word its report block needs.",
        ],
    },
    {
        "id": "prompts",
        "slug": "prompts",
        "band": "L1",
        "short": "Prompt texts",
        "sub": "prompts/",
        "title": "Prompt texts",
        "modules": ["prompts.*"],
        "role": (
            "The texts sent to a language model, one module per task: column renaming, the taxonomy, formulas, "
            "derived variables, anomaly criteria, category harmonisation, document graphs, evaluation."
        ),
        "rule": "Texts only. What a prompt may say of a dataset is decided in one place (manage.ai_payload).",
        "methods": [
            "biostats and prompts share a layer and do not import each other (R1).",
            "A model's answer is parsed against a schema before anything is offered to the user.",
        ],
    },
    # --- L0
    {
        "id": "utils_platform",
        "slug": "utils",
        "band": "L0",
        "short": "Settings, logging, paths & optional groups",
        "sub": "one settings object, checked at start-up · the application logs no value of the data · one data root",
        "title": "Platform: settings, logging, paths",
        "modules": [
            "utils",
            "utils.config_loader",
            "utils.data_paths",
            "utils.environment",
            "utils.export_utils",
            "utils.logging_config",
            "utils.numbers",
            "utils.optional",
            "utils.settings",
        ],
        "role": (
            "What everything stands on: one settings object read from the environment and checked at start-up, "
            "the central logging configuration, the data paths under one root, the optional dependency groups "
            "and what to tell the user when one is missing, the software a dataset was analysed with, and how a "
            "number is written."
        ),
        "rule": (
            "Only utils.settings reads the environment, apart from the three modules the layering test lists "
            "with their reasons. The application writes no value of the data to the log: an exception in any "
            "record is written by its kind."
        ),
        "methods": [
            "A wrong setting stops the start with the variable's name, never its value.",
            "<code>records_without_values</code> writes every exception that reaches a log record by its kind.",
            (
                "The log does keep account names (each AI confirmation with its account and recipients, each change "
                "of an address list) and, once, the setup token."
            ),
            "<code>.env.example</code> lists exactly the settings (a test keeps them equal).",
        ],
        "note": (
            "Eight older modules of utils/ serve a package above: the layering test places each on that "
            "package's layer, and they are shown in those bricks (explore, enrich, ingest, manage)."
        ),
    },
]

# the tests are not part of the import graph of the application: their numbers are counted by the build
TESTS = {
    "id": "tests",
    "band": "T",
    "short": "Tests & gates",
    "sub": "pytest · AppTest · layering · quality gates · R parity · type baseline · coverage floors · CodeQL",
    "title": "Tests & gates",
    "role": (
        "What keeps the rest true as the code grows: the unit and page tests (pytest, Streamlit's AppTest), the "
        "layering test that reads the import graph from the source, the quality gates on the source, R parity for "
        "the pooling of imputed analyses (fixtures from R's mice), and in CI the type baseline, a coverage floor "
        "per package and CodeQL. The parity tests of the rank-based intervals, the rates and survival are written, "
        "and skipped until the R workflow writes their fixtures."
    ),
    "rule": (
        "The exception lists only shrink: a new violation fails, and so does a listed one that no longer occurs, "
        "until it is deleted."
    ),
    "methods": [
        "Two suites: the full environment, and the core-only one with no optional group installed.",
        "<code>tests/connector_contract.py</code> runs the same checks on every connector family.",
        (
            "The pooling is compared with R through fixtures (<code>tests/reference/</code>); three more parity "
            "tests wait for theirs."
        ),
    ],
    "files": [
        "tests/test_layering.py",
        "tests/test_quality_gates.py",
        "tests/connector_contract.py",
        "dev_scripts/check_type_baseline.py",
        "dev_scripts/check_coverage_floors.py",
        ".github/workflows/ci.yml",
        ".github/workflows/codeql.yml",
    ],
    "libs": ["pytest", "ruff", "pyright"],
}

# ----------------------------------------------------------------------------- a feature highlights a path

# ``focus``: the brick the feature turns on, put in focus when the feature is highlighted
FEATURES = [
    {"id": "all", "label": "None", "focus": "", "nodes": []},
    {
        "id": "import",
        "label": "Load a file",
        "focus": "ingest",
        "nodes": [
            "pages_data",
            "ui_shell",
            "ui_state",
            "svc_runtime",
            "ingest",
            "core_data",
            "core_trace",
            "utils_platform",
        ],
    },
    {
        "id": "population",
        "label": "Criteria & population",
        "focus": "core_data",
        "nodes": [
            "pages_data",
            "ui_components",
            "ui_state",
            "svc_runtime",
            "core_data",
            "core_expr",
            "explore",
        ],
    },
    {
        "id": "analysis",
        "label": "An analysis and its result",
        "focus": "pages_analyze",
        "nodes": [
            "pages_analyze",
            "ui_components",
            "ui_state",
            "stats_tests",
            "stats_models",
            "stats_plan",
            "svc_report",
            "core_data",
        ],
    },
    {
        "id": "mi",
        "label": "Multiple imputation",
        "focus": "stats_mi",
        "nodes": [
            "pages_data",
            "pages_analyze",
            "ui_state",
            "svc_runtime",
            "svc_repro",
            "enrich_formulas",
            "core_data",
            "core_trace",
            "stats_mi",
            "stats_models",
            "stats_plan",
        ],
    },
    {
        "id": "replay",
        "label": "Trace & replay",
        "focus": "svc_repro",
        "nodes": [
            "pages_reproduce",
            "ui_state",
            "svc_repro",
            "core_trace",
            "core_data",
            "ingest",
            "enrich_formulas",
            "enrich_impute",
        ],
    },
    {
        "id": "package",
        "label": "Study package",
        "focus": "core_study",
        "nodes": ["pages_reproduce", "svc_repro", "core_study", "core_trace", "stats_plan"],
    },
    {
        "id": "ai",
        "label": "AI engine",
        "focus": "svc_ai",
        "nodes": [
            "pages_shell",
            "pages_explore",
            "ui_state",
            "ui_components",
            "svc_ai",
            "prompts",
            "graph",
            "core_kit",
        ],
    },
    {
        "id": "accounts",
        "label": "Accounts",
        "focus": "svc_accounts",
        "nodes": [
            "entry_app",
            "entry_cli",
            "pages_shell",
            "ui_shell",
            "svc_accounts",
            "core_kit",
            "utils_platform",
        ],
    },
]

# ----------------------------------------------------------------------------- journeys, step by step

ACTORS = [
    {"id": "user", "label": "Browser or terminal", "sub": "the researcher"},
    {"id": "ui", "label": "Pages & interface", "sub": "app_pages · ui"},
    {"id": "svc", "label": "Services", "sub": "manage"},
    {"id": "eng", "label": "Engine", "sub": "core · biostats · enrich"},
    {"id": "out", "label": "Workers & files", "sub": "processes · workspace"},
]

SCENARIOS = [
    {
        "id": "criterion",
        "label": "Tick a criterion",
        "steps": [
            {
                "actor": "user",
                "short": "Tick in the sidebar",
                "title": "1. A criterion is ticked",
                "fn": "ui/components/population.py",
                "where": "Browser",
                "desc": "The user ticks an inclusion criterion in the sidebar's Population box.",
            },
            {
                "actor": "ui",
                "short": "The tick is recorded",
                "title": "2. The tick is recorded, nothing more",
                "fn": "AppState.tick(..., install=False)",
                "where": "Server process",
                "desc": "A widget's callback runs before the script, outside the waiting line: it records the tick "
                "only.",
            },
            {
                "actor": "ui",
                "short": "begin_run()",
                "title": "3. The run installs the population",
                "fn": "AppState.begin_run() inside ui.waiting.computing()",
                "where": "Server process",
                "desc": "At the start of the run the population is computed. After a second, one line says what is "
                "being computed, with Stop.",
            },
            {
                "actor": "eng",
                "short": "criterion_mask",
                "title": "4. The criterion's kind decides",
                "fn": "core.criteria.criterion_mask",
                "where": "Server process",
                "desc": "One function decides the rows a criterion fires on. A range, a bound or a list of rows is "
                "computed here; an expression, the expressions inside a threshold rule included, is parsed to "
                "a tree of whitelisted nodes (core.expressions.evaluate_tree). A comparison with a missing "
                "value is unknown, as in SQL's WHERE, unless the expression asks about missing values.",
            },
            {
                "actor": "out",
                "short": "A worker computes an expression",
                "title": "5. An expression goes to a worker",
                "fn": "manage/workers/pool.py · wire.py",
                "where": "Worker process, time and memory limit",
                "desc": "The tree and only the columns it reads travel to an isolated worker process. At the time "
                "limit the worker is ended and replaced.",
            },
            {
                "actor": "ui",
                "short": "The view is reinstalled",
                "title": "6. The analysis population is a view",
                "fn": "AppState: working_df",
                "where": "Server process",
                "desc": "Rows are filtered, never deleted: only Keep only these rows deletes. A result computed on "
                "another population is no longer shown.",
            },
        ],
    },
    {
        "id": "analysis",
        "label": "Run an analysis, keep its result",
        "steps": [
            {
                "actor": "user",
                "short": "Choose and run",
                "title": "1. The analysis is chosen",
                "fn": "Analyze > Survival & regression",
                "where": "Browser",
                "desc": "Variables and options are chosen; the plan says what will be computed before the run.",
            },
            {
                "actor": "ui",
                "short": "specify, then run",
                "title": "2. The page states the analysis, then asks for it",
                "fn": "app_pages/analysis_survival.py: biostats.plan.specify -> run(spec, df, mi=mi)",
                "where": "Server process",
                "desc": "The analysis is written as a spec, then computed from it: plain values in, a typed result "
                "out. An exception goes to an error card.",
            },
            {
                "actor": "eng",
                "short": "biostats computes",
                "title": "3. The estimate, with its interval",
                "fn": "biostats.survival",
                "where": "Server process",
                "desc": "The proportional-hazards test is reported as a diagnostic and never changes the model. The "
                "method's reference is cited next to the result.",
            },
            {
                "actor": "ui",
                "short": "remember(...)",
                "title": "4. The result is stamped",
                "fn": 'ui.results.remember(key, done.result, spec=done.spec, config=...) · mark_step("analyse")',
                "where": "Server process",
                "desc": "Kept with its spec, the data fingerprint, the population and the settings it was computed "
                "on.",
            },
            {
                "actor": "ui",
                "short": "recall(...) · provenance",
                "title": "5. Shown only while it still holds",
                "fn": "ui.results.recall(key, config=config) · provenance(key)",
                "where": "Server process",
                "desc": "On every run the result comes back only while its data, population and settings still hold. "
                "The provenance line offers to add it to the dossier.",
            },
            {
                "actor": "svc",
                "short": "Report blocks",
                "title": "6. Written into the dossier",
                "fn": "manage.report_tables.result_blocks -> manage.dossier.build_dossier",
                "where": "Server process",
                "desc": "Each result becomes report blocks, from its record when its kind has a record block and "
                "otherwise from the stored result; the Word file adds the participant flow.",
            },
        ],
    },
    {
        "id": "mi",
        "label": "Draw a multiple imputation",
        "steps": [
            {
                "actor": "user",
                "short": "Five guided steps",
                "title": "1. The panel, in five steps",
                "fn": "app_pages/enrichment_mi.py",
                "where": "Browser",
                "desc": "The population and its missingness, the model, the settings, the run with its diagnostics "
                "(chain traces, R-hat flagged above 1.1, observed against imputed values), then use and "
                "save.",
            },
            {
                "actor": "svc",
                "short": "lease(wanted)",
                "title": "2. Workers are leased",
                "fn": "manage.compute.mi_workers().lease(wanted)",
                "where": "Server process",
                "desc": "One budget per server process: a run takes what is free, never waits, and gives it back "
                "however it ends.",
            },
            {
                "actor": "eng",
                "short": "Chained equations",
                "title": "3. The chains are drawn",
                "fn": "enrich.derived.impute_and_derive -> biostats.mi.engine.impute",
                "where": "Server process",
                "desc": "Each incomplete variable is drawn from its conditional model. A derived variable is left "
                "out of the chains, neither imputed nor used as a predictor, and computed again from its "
                "imputed inputs.",
            },
            {
                "actor": "out",
                "short": "python -m biostats.mi.worker",
                "title": "4. A large run uses processes",
                "fn": "biostats/mi/parallel.py -> biostats.mi.worker",
                "where": "Worker processes, one BLAS thread",
                "desc": "Separate interpreters are started by the module's name. Each chain has its own seeded "
                "stream and every process draws on one BLAS thread, so the parallel run gives the values "
                "of a sequential one.",
            },
            {
                "actor": "svc",
                "short": "Saved with the version",
                "title": "5. Only the imputed cells are stored",
                "fn": "manage/mi_store.py · trace step mi_impute",
                "where": "Workspace files",
                "desc": "The imputed cells are saved beside the dataset version; the trace gets one step with the "
                "model and the seed.",
            },
            {
                "actor": "eng",
                "short": "Pooled analyses",
                "title": "6. Analyses are pooled",
                "fn": "biostats.mi.analyze.analyze_mi · biostats.mi.pooling",
                "where": "Server process",
                "desc": "An analysis runs on each completed dataset and is pooled by Rubin's rules (a chi-square "
                "statistic by the D2 rule), or the page says why it cannot be.",
            },
        ],
    },
    {
        "id": "inspect",
        "label": "Check a received package",
        "steps": [
            {
                "actor": "user",
                "short": "inspect <file>",
                "title": "1. The command is run",
                "fn": 'python -m manage.replay inspect "<file>" --package-hash ...',
                "where": "Terminal",
                "desc": "The recipient checks a package before typing its password. The password is never an "
                "argument.",
            },
            {
                "actor": "svc",
                "short": "The command line",
                "title": "2. Only this command is loaded",
                "fn": "manage/replay/__main__.py · commandline.py",
                "where": "Command's process",
                "desc": "A command's module is imported when it runs: no setting, no engine. inspect writes back "
                "nothing typed.",
            },
            {
                "actor": "svc",
                "short": "open_package",
                "title": "3. The clear part",
                "fn": "manage.package_zip.open_package",
                "where": "Command's process",
                "desc": "The file, the kind of package, its hash and its files are read without a password, by the "
                "safe reader.",
            },
            {
                "actor": "eng",
                "short": "The hash is compared",
                "title": "4. The hash, before any password",
                "fn": "core.package (the manifest's SHA-256)",
                "where": "Command's process",
                "desc": "A hash that differs is shown under the package's own, with where they part, and the run "
                "ends there (exit 1).",
            },
            {
                "actor": "svc",
                "short": "read_package",
                "title": "5. With the password",
                "fn": "manage.package_zip.read_package",
                "where": "Command's process",
                "desc": "Each file is checked against the manifest. In a package that holds the data, every member "
                "but its notice and its manifest is encrypted (AES-256).",
            },
            {
                "actor": "user",
                "short": "What the index says",
                "title": "6. What the package is",
                "fn": "manage/replay/inspection.py",
                "where": "Terminal",
                "desc": "The software, the study, to whom and under what it was sent, the rows and columns of each "
                "file, and the two texts its exporter wrote. Nothing of the data.",
            },
        ],
    },
]

# ----------------------------------------------------------------------------- the rules, as a small lab

# ``held``: what holds the rule (a test's file, "a quality gate"); the lab says it under each answer
RULES = [
    {
        "id": "R1",
        "held": "tests/test_layering.py",
        "text": "Imports go strictly down, and two packages on the same layer do not import each other.",
    },
    {
        "id": "R2",
        "held": "tests/test_layering.py",
        "text": "No import cycles; no exception is listed. The parallel imputation starts its worker by the "
        "module's name.",
    },
    {
        "id": "R3",
        "held": "tests/test_layering.py",
        "text": "Nothing below ui reaches Streamlit, through any chain of imports. A graph renderer imports it "
        "inside its render method only, and in translate a Streamlit component package.",
    },
    {
        "id": "R4",
        "held": "tests/test_layering.py",
        "text": "The AI providers' SDKs are imported only in manage/rag/providers.",
    },
    {
        "id": "R5",
        "held": "tests/test_layering.py",
        "text": "The headless code (manage_users, manage.package_zip, manage.services, and every module of "
        "manage.replay and manage.workers) reaches neither Streamlit nor the AI packages.",
    },
]
MORE_RULES = [
    "biostats imports only itself.",
    "Importing core, biostats or ingest loads neither Streamlit, nor Plotly, nor an AI package.",
    (
        "Only utils.settings reads the environment, apart from the three modules the test lists with their reasons; "
        "that list can only shrink."
    ),
    (
        "No module of core or biostats imports utils.settings: biostats reads its seed itself until the seed moves "
        "into the analysis specs, and core.workspace asks utils.data_paths for the data root."
    ),
]

# what the lab offers besides the project's own packages
OUTSIDE = [
    {"id": "streamlit", "label": "Streamlit"},
    {"id": "ai_sdk", "label": "an AI provider's SDK (anthropic, openai, google.genai, ...)"},
    {"id": "plotting", "label": "a plotting library (plotly, matplotlib)"},
    {"id": "numeric", "label": "pandas, numpy, scipy"},
    {"id": "environment", "label": "the environment (os.environ)"},
    {"id": "settings", "label": "the settings (utils.settings.get_settings)"},
]

# ----------------------------------------------------------------------------- recipes: where a new thing goes

RECIPES = [
    {
        "id": "connector",
        "label": "A connector",
        "nodes": ["core_kit", "svc_accounts", "svc_ai", "graph", "utils_platform", "pages_shell"],
        "steps": [
            (
                "Write the class in its family's package, on the family's base class, with <code>name</code>, "
                "<code>display_name</code> and <code>availability()</code>, which never raises, never connects, and "
                "names the optional group its package comes from."
            ),
            (
                'Register it in the family\'s registry as <code>"module:Class"</code>, with <code>extra=</code> '
                "when it needs an optional group. Its name is stored in preferences and settings: never rename it; "
                "an old spelling is an alias."
            ),
            (
                "A package it needs goes into an optional group of <code>pyproject.toml</code> and into "
                "<code>utils.optional.EXTRAS</code>. An AI provider imports its SDK in "
                "<code>manage/rag/providers</code> only (R4); a renderer imports Streamlit in its drawing methods "
                "only (R3)."
            ),
            (
                "A graph engine needs nothing else. A vector database also needs its member in "
                "<code>VectorEngine</code>, and with a server its list of addresses and its recipient for the "
                "confirmation. An AI provider also needs its declaration (cloud or not, its SDK, the setting of its "
                "key, its chat and embedding connectors in <code>manage/rag/providers</code>) and its member in "
                "<code>ProviderType</code>; an identity provider, its name in <code>IAMProviderType</code> and its "
                "settings."
            ),
            "Say where it runs in the architecture explorer's words: its build stops without that.",
        ],
        "test": "Add the name to its family's test of the stored names; the family's tests run the checks of "
        "<code>tests/connector_contract.py</code> on it.",
    },
    {
        "id": "step",
        "label": "A kind of trace step",
        "nodes": ["core_trace", "ui_state", "svc_repro"],
        "steps": [
            (
                "Register a <code>StepKind</code> in <code>core/steps.py</code>: its title, its category, its replay "
                "mode and whether it changes the data."
            ),
            (
                "Record it through <code>AppState</code> (<code>commit_change</code>, <code>stage</code> or "
                "<code>note</code>), never with <code>add_step</code>."
            ),
            "A step the replay recomputes gets its handler in <code>manage/replay/engine.py</code>.",
            "Never rename an id: a kind nothing records any more is registered as legacy.",
        ],
        "test": "<code>tests/core_layer/test_steps.py</code>: every step written is registered, titled and "
        "handled by its mode.",
    },
    {
        "id": "criterion",
        "label": "A criterion kind",
        "nodes": ["core_data", "ui_components", "pages_data"],
        "steps": [
            "Its functions go into a module of their own.",
            "One registration in <code>core/criteria.py</code>, after the existing kinds.",
            "A writer, so that rules are written through their kind.",
        ],
        "test": "Its examples in <code>tests/core_layer/test_criterion_kinds.py</code>.",
    },
    {
        "id": "analysis",
        "label": "An analysis",
        "nodes": [
            "stats_tests",
            "stats_models",
            "stats_plan",
            "pages_analyze",
            "ui_state",
            "svc_report",
        ],
        "steps": [
            (
                "The computation goes into <code>biostats</code>: plain values in, a result dataclass out, no import "
                "of the application."
            ),
            (
                "Register its kind in <code>biostats/plan.py</code> (<code>KINDS</code>: its title, and when it has "
                "them its runner and its record); the titles of the results are read from there."
            ),
            (
                "On the page, every control has a key. On the button: compute (through <code>specify(...)</code> and "
                "<code>run(...)</code> when the kind has a runner), <code>remember(...)</code>, "
                '<code>mark_step("analyse")</code>.'
            ),
            "On every run, <code>recall(key, config=config)</code>, then <code>provenance(key)</code>.",
            "A new type of result gets its tables in <code>manage.report_tables.result_blocks</code>.",
            (
                "Compare an estimator's numbers with R: a script in <code>tests/reference/r/</code>, its values as "
                "fixtures, a parity test."
            ),
        ],
        "test": "An AppTest of the page, and the module in the list of the result-store quality gate.",
    },
    {
        "id": "typed",
        "label": "A calculation from numbers typed",
        "nodes": ["stats_plan", "svc_report", "pages_analyze", "ui_state"],
        "steps": [
            "The computation goes into <code>biostats</code>, as for an analysis.",
            (
                "In <code>biostats/plan.py</code>, give its kind its typed inputs (named once, never renamed), its "
                "options and its record."
            ),
            (
                "Its words and its block go into <code>manage/report_tables.py</code>, written from the record "
                "alone."
            ),
            (
                "On the page: the source field, <code>specify(kind, typed=..., options=...)</code>, "
                "<code>run(spec)</code> for the headline, then the line of its numbers and the button. Never "
                "<code>remember</code>, which refuses a typed spec."
            ),
            "An error names a number, never its value; nothing typed is logged.",
        ],
        "test": "The round trip of the spec and of the record (<code>tests/biostats_layer/test_plan_typed.py"
        "</code>); the button, the dossier page and the Word section "
        "(<code>tests/ui_layer/test_typed_dossier.py</code>).",
    },
    {
        "id": "panel",
        "label": "A page panel",
        "nodes": ["pages_data", "ui_shell", "ui_state", "ui_components"],
        "steps": [
            (
                "Write it in a module of its own, with one rendering function called in its tab, as "
                "<code>app_pages.enrichment_derived.render_derived_variables</code> is in Enrich."
            ),
            (
                "On a page that runs only its open tab, the panel goes inside <code>if tab.open:</code>. For a link "
                "to open its tab, add the tab to the page's <code>PageDef.tabs</code> in <code>ui/nav.py</code>."
            ),
            'Every control has a <code>key</code> and <code>persist_state="page"</code>.',
            "A key it shares with another tab or page is declared in <code>ui/keys.py</code>.",
            (
                "It changes the data through <code>AppState</code> and takes its services from "
                "<code>ui/services.py</code>."
            ),
        ],
        "test": "An AppTest in <code>tests/ui_layer/</code>, its module in <code>LAZY</code> "
        "(<code>tests/ui_layer/test_lazy_tabs.py</code>) when its page runs only the open tab, and the "
        "session-keys test.",
    },
]

# ----------------------------------------------------------------------------- the matrix

# each row's libraries are read by the build from the files of its ``comps`` (a folder: every module under it; a
# bare file name: in the folder of the file before it)
MATRIX = [
    {
        "feature": "Import a file",
        "node": "ingest",
        "comps": "ui/import_flow.py, ingest/loaders.py, ingest/analyzer.py",
        "methods": "Encoding, delimiter and decimal mark proposed and overridable; a parse report by count and line "
        "number; one loader for the import, a merge, the comparison and the replay",
    },
    {
        "feature": "Dates",
        "node": "ingest",
        "comps": "ingest/dates.py, app_pages/validate_dates.py",
        "methods": "A declared format per column: ISO, day first, month first, compact, Excel serial; heaped dates "
        "flagged (binomial tail)",
    },
    {
        "feature": "Criteria & analysis population",
        "node": "core_data",
        "comps": "core/criteria.py, core/population.py, core/attrition.py, ui/components/population.py",
        "methods": "Reversible filters; five criterion kinds; three-valued logic; the participant flow, stage by "
        "stage, for reporting STROBE item 13",
    },
    {
        "feature": "Safe expressions in workers",
        "node": "core_expr",
        "comps": "core/expressions/, manage/workers/",
        "methods": "A whitelisting evaluator over the parsed tree; an isolated worker with a time limit and a memory "
        "limit (a Job Object on Windows, RLIMIT_DATA on Linux)",
    },
    {
        "feature": "Derived variables",
        "node": "enrich_formulas",
        "comps": "core/derived.py, enrich/derived.py, app_pages/enrichment_derived.py",
        "methods": "A formula checked against the recorded values, with a tolerance from each column's recording "
        "resolution; under multiple imputation, passive imputation (impute, then transform)",
    },
    {
        "feature": "Single imputation",
        "node": "enrich_impute",
        "comps": "enrich/imputation_service.py, enrich/data_imputation.py, utils/miss_forest/",
        "methods": "Mean, median, most frequent, k nearest neighbours, iterative regression (scikit-learn's "
        "IterativeImputer, one value per cell); MissForest with LightGBM learners, seeded and "
        "deterministic",
    },
    {
        "feature": "Multiple imputation & pooling",
        "node": "stats_mi",
        "comps": "biostats/mi/",
        "methods": "Chained equations as R's mice (by default pmm for a number, logreg for two categories, polyreg "
        "for more; norm and rf on request), one seeded stream per chain; Rubin's rules with Barnard-Rubin "
        "degrees of freedom, D1 and D2 tests; one BLAS thread; a worker budget per process",
    },
    {
        "feature": "Missing data",
        "node": "stats_tests",
        "comps": "biostats/mcar.py, missing.py",
        "methods": "Little's test of MCAR on the numeric columns (EM estimates); a scan of what the missingness of a "
        "variable depends on, every test reported, adjusted over the whole family (Benjamini-Hochberg by "
        "default)",
    },
    {
        "feature": "Group comparisons",
        "node": "stats_tests",
        "comps": "biostats/inference.py, posthoc.py, effect_sizes.py, rank_intervals.py",
        "methods": "Welch first; Games-Howell pairwise comparisons; Hodges-Lehmann intervals without random draws; "
        "explicit multiplicity families",
    },
    {
        "feature": "Paired & one-sample tests",
        "node": "stats_tests",
        "comps": "biostats/paired.py",
        "methods": "Paired and one-sample t-tests; Wilcoxon signed-rank test with its pseudo-median interval; exact "
        "McNemar test; exact binomial test with a Wilson interval",
    },
    {
        "feature": "Epidemiological measures & rates",
        "node": "stats_tests",
        "comps": "biostats/contingency.py, rates.py, study_designs.py",
        "methods": "2x2 tables with an explicit orientation (Katz, Woolf and Newcombe intervals); Mantel-Haenszel "
        "stratified measures with the Breslow-Day and Woolf tests; the measures each sampling supports; "
        "incidence rates and standardised ratios as calculators from numbers typed",
    },
    {
        "feature": "Diagnostic accuracy & agreement",
        "node": "stats_tests",
        "comps": "biostats/diagnostic.py, agreement.py, reliability.py",
        "methods": "Sensitivity and specificity (Wilson); predictive values for a consecutive or random series, or "
        "at a stated prevalence; likelihood ratios, diagnostic odds ratio, Youden's J; ROC area with "
        "DeLong's interval; a Youden cut-off with its bootstrap range; Bland-Altman bias and limits of "
        "agreement with their intervals; Cohen's kappa (weighted or not), ICC(2,1) or ICC(3,1)",
    },
    {
        "feature": "Power & sample size",
        "node": "stats_tests",
        "comps": "biostats/power.py",
        "methods": "Power and sample size for a one-way ANOVA (Cohen's f), the Kruskal-Wallis test (through the "
        "asymptotic relative efficiency) and a chi-square test (Cohen's w), each saying whether a size is "
        "the total or per group; none yet for Welch's t-test",
    },
    {
        "feature": "Outliers",
        "node": "stats_tests",
        "comps": "biostats/outliers.py",
        "methods": "One detector for Validate, the dashboard and the quality index: z-score, Tukey's fences, "
        "quantiles; Local Outlier Factor, Isolation Forest, DBSCAN. Flags for review, not errors: the "
        "quantile rule flags the outer 1% at each end by construction, and the multivariate methods read "
        "complete rows only",
    },
    {
        "feature": "Survival & regression",
        "node": "stats_models",
        "comps": "biostats/survival.py, linear.py, regression.py, design.py",
        "methods": "Kaplan-Meier, log-rank, Cox proportional hazards; linear regression and ANCOVA; logistic and "
        "count regression",
    },
    {
        "feature": "Quality audit & correlations",
        "node": "explore",
        "comps": "explore/",
        "methods": "An audit by dimension; a data-quality index (0 to 100) as a weighted mean with its "
        "decomposition; descriptive correlation matrices (Pearson, Spearman, Kendall) with strength "
        "bands",
    },
    {
        "feature": "Changes between versions",
        "node": "monitor",
        "comps": "monitor/changes.py",
        "methods": "Two tables compared by an identifier, exactly, on the columns both hold; a workbook of what "
        "differs",
    },
    {
        "feature": "Trace & replay",
        "node": "svc_repro",
        "comps": "core/lineage.py, core/steps.py, manage/replay/engine.py, manage/paths.py",
        "methods": "Hash-chained steps (SHA-256); content fingerprints; a step recomputed, read from its snapshot "
        "or documented",
    },
    {
        "feature": "Study package & inspect",
        "node": "core_study",
        "comps": "core/package.py, manage/package_zip.py, manage/replay/inspection.py",
        "methods": "The manifest's hash before any password; with the data, members encrypted with AES-256; a safe "
        "reader of a received zip; exit codes a script can read",
    },
    {
        "feature": "Study dossier (Word)",
        "node": "svc_report",
        "comps": "manage/dossier.py, manage/report_tables.py, manage/flow_figure.py",
        "methods": "Report blocks from a record or from the stored result; the participant flow as a picture; one "
        "rule for how a number is written",
    },
    {
        "feature": "AI engine & documents",
        "node": "svc_ai",
        "comps": "manage/rag/, manage/vectorstores/, manage/ai_payload.py, manage/rag_manager.py",
        "methods": "Providers as connectors, Ollama by default; about a dataset, column summaries, never rows; a "
        "document index on LanceDB or Qdrant; a confirmation that names each recipient",
    },
    {
        "feature": "Knowledge graphs",
        "node": "graph",
        "comps": "graph/model.py, graph/builders/, graph/renderers/",
        "methods": "One description of a graph; NetworkX layouts; three graph engines, two of them with no network",
    },
    {
        "feature": "Accounts",
        "node": "svc_accounts",
        "comps": "manage/iam/, ui/auth.py, manage_users.py",
        "methods": "bcrypt hashes in DuckDB; access by the stored role; a one-time setup token written to the "
        "server log, of which the data root keeps only the hash; a lock after failed sign-ins",
    },
    {
        "feature": "Settings & logging",
        "node": "utils_platform",
        "comps": "utils/settings.py, utils/logging_config.py, utils/optional.py",
        "methods": "One settings object, checked at start-up; a wrong setting stops the start, named without its "
        "value; the application logs no value of the data",
    },
]
