"""
The levels of detail of the architecture explorer, and its view of the connectors and the catalogues.

Each level matches a document, so that a reader moves from one to the other without changing words:

1. **Overview**: the README's scheme. Its cards, its connector sockets and its storage are read from the scheme's own
   script (``dev_scripts/make_architecture_diagram.py``); the groups' captions are copied here, and the build checks
   that the script still holds them.
2. **Layers & packages**: the ten layers and the packages of the developer guide's layer table
   (``documentation/10_Developer_Guide.md``) and of ``tests/test_layering.py``; the build checks the three agree.
3. **Bricks**: the explorer's own level (``explorer_content.NODES``).
4. **Modules**: every module, read from the code.

The connectors and the catalogues are read from their registries. What is written here is what no registry says:
where each connector runs, what it receives and what is asked first. The build stops when a registered connector has
no place, and checks each place against what the connector's class declares.

A ``slug`` is a name in an address (``#/connectors/<slug>``): documents link to it, so it is never renamed.
"""

from __future__ import annotations

REPOSITORY = "https://github.com/KarimELMERNISSI/Cohort-Studio"

# ``home``: where a level's own description lives (shown on its tile); ``also``: the link under the diagram's title
LEVELS = [
    {
        "id": "overview",
        "label": "Overview",
        "home": "README",
        "title": "Five groups: between them, imports go down only",
        "says": (
            "The README's scheme: what each group of packages is for, the two ways in, the connectors, the "
            "catalogues and the storage."
        ),
        "also": "Also in the README, Architecture",
        "href": REPOSITORY + "#architecture",
    },
    {
        "id": "packages",
        "label": "Layers & packages",
        "home": "developer guide",
        "title": "Ten layers, one package or two on each",
        "says": "The developer guide's layer table, and the rules the layering test holds.",
        "also": "Also in the developer guide, The layers",
        "href": REPOSITORY + "/blob/main/documentation/10_Developer_Guide.md#the-layers",
    },
    {
        "id": "bricks",
        "label": "Bricks",
        "home": "only here",
        "title": "The bricks of each layer, and what each imports",
        "says": "A brick is a part of a package that does one job.",
        "also": "",
        "href": "",
    },
    {
        "id": "modules",
        "label": "Modules",
        "home": "here, then the code",
        "title": "Every module of the code base",
        "says": "Each module: its layer, its brick, its size, what it imports and what imports it.",
        "also": "Also in the code on GitHub",
        "href": REPOSITORY + "/tree/main",
    },
]

# ----------------------------------------------------------------------------- level 1: the README's scheme

# The labels and captions are the scheme's own (the build looks each ``label`` and ``title`` up in its script, but
# for the bands the scheme leaves unnamed: ``own`` marks those). ``path`` (the layers of a band) is written by the
# build from the layering test.
OVERVIEW_BANDS = [
    {"id": "E", "label": "ENTRY POINTS", "title": "two ways in", "group": "entry", "own": True},
    {
        "id": "GI",
        "label": "INTERFACE",
        "title": "the only layer that needs Streamlit",
        "group": "interface",
        "boundary_after": True,
    },
    {
        "id": "GS",
        "label": "SERVICES",
        "title": "run without Streamlit, like every layer below",
        "group": "services",
    },
    {"id": "GD", "label": "DATA WORK", "title": "", "group": "data"},
    {
        "id": "GE",
        "label": "DATA AND STATISTICS ENGINE",
        "title": "plain Python, used and tested on its own",
        "group": "engine",
    },
    {
        "id": "GP",
        "label": "PLATFORM",
        "title": "every layer may use it",
        "group": "foundation",
        "own": True,
    },
    {"id": "S", "label": "STORAGE", "title": "One workspace per user", "group": "storage"},
    {"id": "T", "label": "ACROSS", "title": "tests & gates", "group": "tests", "own": True},
]
# the band of each card of the README's scheme (by the scheme's own tuples), and the two entry points
OVERVIEW_PLACES = {
    "INTERFACE": "GI",
    "SERVICES": "GS",
    "DATA_WORK": "GD",
    "ENGINE": "GE",
    "FOUNDATION": "GP",
}
OVERVIEW_ENTRIES = [
    {
        "id": "o_web",
        "packages": ["main"],
        "short": "Browser",
        "sub": "the Streamlit application (main.py)",
        "role": (
            "main.py: the settings checked first, the connector settings checked, then the pages. Streamlit runs "
            "each page by its path; the interface is the only layer that needs Streamlit."
        ),
        "rule": "A wrong setting stops the start, named without its value.",
    },
    {
        "id": "o_cli",
        "packages": ["manage_users"],
        "short": "Command line and scripts",
        "sub": "manage_users.py · headless replay (run_trace)",
        "role": (
            "What enters below the interface: <code>manage_users.py</code> for the accounts, "
            "<code>python -m manage.replay</code> for a received study package, and a script that replays a "
            "trace (<code>run_trace</code>). The last two are modules of the services."
        ),
        "rule": "R5: headless code reaches neither Streamlit nor the AI packages.",
    },
]
# what the scheme draws beside its cards: the connector sockets in the services, the catalogues in the engine. Each
# is a link to the Connectors & catalogues view (``says`` is the scheme's caption).
OVERVIEW_SOCKETS = {
    "band": "GS",
    "label": "CONNECTORS",
    "says": "one implementation at a time, loaded when it is used",
}
OVERVIEW_CATALOGUES = {
    "band": "GE",
    "label": "CATALOGUES",
    "says": "kinds declared once, named in the data",
}
# the storage panel of the scheme, as a box of its own; its items are read from the scheme's script
OVERVIEW_STORAGE = {
    "short": "One workspace per user",
    "title": "Storage: one workspace per user",
    "role": (
        "What a user makes is kept in their own workspace, under <code>&lt;DATA_ROOT&gt;/users/&lt;username&gt;/"
        "</code>: the versions of their datasets (Parquet, with lineage files), their hash-chained traces, the "
        "study definition and its revisions, their documents and the vector index built from them, their "
        "taxonomies and knowledge graphs, their preferences. An index on a Qdrant server is kept on that server, "
        "and the workspace keeps its manifest. The cache of dataset statistics is Parquet files in the workspace too; "
        "DuckDB holds the accounts. "
        "With no sign-in, every visitor shares one workspace."
    ),
    "rule": "A path is confined to the user's folder (core.workspace). The services write; the pages ask them.",
    "files": [
        "core/workspace.py",
        "manage/snapshots.py",
        "manage/protocol_store.py",
        "manage/mi_store.py",
        "manage/preferences.py",
        "manage/db_manager.py",
    ],
}
# the longer words of each card (the card's own name and lines come from the README's scheme)
OVERVIEW_WORDS = {
    "app_pages": {
        "role": "The pages: their inputs, one call to a service, a result card.",
        "rule": "A page builds no service, and changes the data only through AppState.",
    },
    "ui": {
        "role": (
            "The shell and the navigation, the components, and AppState: the one writer of the data, which records "
            "each change as a step of the trace."
        ),
        "rule": "Only AppState writes the dataset and the trace steps (a quality gate).",
    },
    "manage": {
        "role": (
            "The application's services: traces and replay, the study definition and its package, the Word dossier, "
            "the accounts, the AI engine, the stores of each user's workspace, and the worker processes. Three "
            "connector families live here: identity providers, AI providers, vector databases."
        ),
        "rule": "A service takes the user's scope as a parameter, never from the session.",
    },
    "graph": {
        "role": (
            "The knowledge graphs: one model, the builders that fill it, and the renderers that draw it, the fourth "
            "connector family (the graph engines)."
        ),
        "rule": "R3: a renderer imports Streamlit inside its drawing methods only.",
    },
    "ingest": {
        "role": "Reading CSV, Excel and Parquet files with explicit options and a parse report; dates; the column profile.",
        "rule": "One loader for every entry point, the replay's included.",
    },
    "enrich": {
        "role": "Formulas and derived variables, transformations, single imputation, recoding and merging.",
        "rule": "A formula is evaluated by the safe evaluator, never run as code.",
    },
    "explore · monitor": {
        "role": (
            "The quality audit and its index, descriptive correlations, and the exact comparison of two versions "
            "of a table."
        ),
        "rule": "explore and monitor share a layer and do not import each other.",
    },
    "core": {
        "role": (
            "The domain kernel: the data session, the analysis population and its criteria, lineage and "
            "fingerprints, the trace steps, the study definition and its package's formats, and the extension kit "
            "that the connector families and four of the five catalogues are built on."
        ),
        "rule": (
            "Importing it loads no Streamlit, and no module of it imports utils.settings: values come in as "
            "parameters, apart from the data root, which core.workspace asks of utils.data_paths."
        ),
    },
    "biostats": {
        "role": "The statistics library: tests and measures, models, multiple imputation and pooling, the analysis plan.",
        "rule": "Imports no other package of the project, so it runs and is tested on its own.",
    },
    "prompts": {
        "role": "The texts sent to a language model, one module per task.",
        "rule": "Texts only.",
    },
    "utils": {
        "role": "One settings object checked at start-up, the logging, the optional groups, the data paths.",
        "rule": "utils.settings reads the environment; three other modules do, listed in the layering test with their reasons.",
    },
}

# ----------------------------------------------------------------------------- level 2: the layer table

PACKAGES = {
    "main": {
        "short": "main.py",
        "sub": "the web application",
        "role": "The web application's entry point: the settings, the start-up work once per process, sign-in, the navigation.",
        "rule": "Above every layer: it may import any package.",
    },
    "manage_users": {
        "short": "manage_users.py",
        "sub": "the accounts, on a terminal",
        "role": "The command line of the accounts: create an administrator, activate, deactivate, reset a password, list.",
        "rule": "R5: it reaches neither Streamlit nor the AI packages.",
    },
    "app_pages": {
        "short": "app_pages",
        "sub": "the pages",
        "role": (
            "The pages: their inputs, a service call, a result card. The scripts that <code>st.navigation</code> "
            "runs are in <code>app_pages/views/</code>; each delegates to a page module."
        ),
        "rule": "A page builds no service and changes the data only through AppState.",
    },
    "ui": {
        "short": "ui",
        "sub": "shell, AppState, components",
        "role": "The shell, AppState, the components, the guidance, and <code>ui/services.py</code>, which keeps the session's services.",
        "rule": (
            "With app_pages and main.py, the code that may import Streamlit; below it, only a graph renderer's "
            "drawing methods do (R3)."
        ),
    },
    "manage": {
        "short": "manage",
        "sub": "application services and connectors",
        "role": (
            "The application services, their stores and three connector families, in groups: accounts, AI, "
            "reproduction, storage; and the worker processes."
        ),
        "rule": (
            "manage.package_zip, manage.services and every module of manage.replay and manage.workers run "
            "without Streamlit or the AI packages (R5)."
        ),
    },
    "graph": {
        "short": "graph",
        "sub": "graph model, builders, renderers",
        "role": "The graph model, its builders, and its renderers: the fourth connector family, the graph engines.",
        "rule": "A renderer imports Streamlit inside its drawing methods only (R3).",
    },
    "explore": {
        "short": "explore",
        "sub": "quality audit, correlations",
        "role": "The quality audit and its index, and descriptive correlations.",
        "rule": "It does not import monitor, which shares its layer (R1).",
    },
    "monitor": {
        "short": "monitor",
        "sub": "changes between versions",
        "role": "The exact comparison of two versions of a table, by an identifier.",
        "rule": "It does not import explore, which shares its layer (R1).",
    },
    "enrich": {
        "short": "enrich",
        "sub": "formulas, imputation, merging",
        "role": "Formulas and derived variables, transformations, single imputation, recoding and merging.",
        "rule": "Every formula goes through core.expressions.",
    },
    "ingest": {
        "short": "ingest",
        "sub": "reading files",
        "role": "Reading files with one loader and explicit options, dates read with a declared format, the column profile.",
        "rule": "Importing it loads neither Streamlit, nor Plotly, nor an AI package.",
    },
    "core": {
        "short": "core",
        "sub": "domain kernel, extension kit",
        "role": "The domain kernel, and the extension kit (<code>core/registry.py</code>) of the connector families and catalogues.",
        "rule": (
            "Importing it loads neither Streamlit, nor Plotly, nor an AI package; no module of it imports "
            "utils.settings."
        ),
    },
    "biostats": {
        "short": "biostats",
        "sub": "the statistics library",
        "role": "The statistics library, which imports nothing else of the project.",
        "rule": (
            "Imports only itself. It reads one setting itself, its random seed, until the seed moves into the "
            "analysis specs."
        ),
    },
    "prompts": {
        "short": "prompts",
        "sub": "prompt texts",
        "role": "The texts sent to a language model.",
        "rule": "Texts only.",
    },
    "utils": {
        "short": "utils",
        "sub": "settings, logging, paths",
        "role": (
            "Settings, logging, optional groups, paths and export helpers. Its older modules count with the "
            "package they serve, until a milestone moves them."
        ),
        "rule": "utils.settings reads the environment; three other modules do, listed in the layering test with their reasons.",
    },
}

# ----------------------------------------------------------------------------- the connectors

# what a connector is, for a reader who has never met one (the Connectors view opens on it)
CONNECTOR_LEAD = [
    (
        "Some jobs can be done in more than one way: signing users in, answering with a language model, indexing "
        "documents, drawing a graph. Each job is a connector family, and each way of doing it is a connector. One "
        "connector per family is used at a time, loaded only when it is used; each says whether it can run here and, if "
        "not, which optional group to install. Changing connector is a setting or a choice in Settings, never a change "
        "of code."
    ),
    (
        "An example: with <code>CMP_AI_PROVIDER</code> at its default, <code>auto</code>, the AI engine uses Ollama on "
        "your network; a user who chooses Mistral in Settings is asked for a confirmation in each browser tab before "
        "it is used."
    ),
]

# the two mechanisms side by side, in the developer guide's words
MECHANISMS = [
    (
        "What it holds",
        "the implementations of one service, one used at a time",
        "kinds side by side, each named by an id",
    ),
    (
        "Who chooses",
        (
            "the session, then the saved choice, the server's setting, then the family's order; for the "
            "identity provider, the server's setting alone"
        ),
        "the data: a trace or a rule names its kind",
    ),
    (
        "Registered as",
        '<code>"module:Class"</code>, imported on first use',
        "one frozen dataclass per kind",
    ),
    (
        "An unknown name",
        "in a setting, stops the start and lists the valid names",
        "<code>UnknownIdError</code>, listing the valid ids (the analysis kinds, a plain dict, raise a ValueError)",
    ),
    (
        "A name stored",
        "never renamed: an old spelling stays an alias",
        "never renamed: a kind nothing records any more stays, marked legacy",
    ),
    (
        "Tested by",
        "<code>tests/connector_contract.py</code>, run on every family; each family's test pins its names",
        "one consistency test per catalogue",
    ),
]

# Where a connector runs, which no registry says. The line between the second zone and the third is the code's own
# (manage.ai_payload.outside_private_network): this machine and the private network on one side, everything else on
# the other.
PLACES = {
    "server": {
        "label": "On your server",
        "says": "in the server's own process, with its own files",
    },
    "network": {
        "label": "On your network",
        "says": (
            "a service on this machine or the private network, at an address an administrator chose: it receives "
            "what it is sent, with no confirmation"
        ),
    },
    "outside": {
        "label": "Outside your network",
        "says": (
            "a cloud service, or a server outside the private network: a confirmation in each browser tab first, "
            "except for yFiles (a notice follows each drawing)"
        ),
    },
}
PRIVATE_NETWORK = (
    "The private network, as the code reads an address without looking it up: this machine, a private or link-local "
    "address, or a name with no dot or ending in .local, .lan, .internal, .home.arpa or .localdomain. Any other "
    "name, the organisation's own domain included, is outside it."
)
OUTSIDE_THE_NETWORK = "when its address is outside the private network"
CONFIRMATION = "a confirmation in each browser tab"
NO_CONFIRMATION = "no confirmation"


def place(
    zone: str,
    label: str,
    *,
    when: str = "",
    receives: str = "",
    asked: str = "",
    unplaced: bool = False,
) -> dict[str, str | bool]:
    """A connector in a zone: ``when`` it is there, what it ``receives`` there, what is ``asked`` before.
    ``unplaced``: the code does not check where its address is; the diagram marks it and says so."""
    return {
        "place": zone,
        "label": label,
        "when": when,
        "receives": receives,
        "asked": asked,
        "unplaced": unplaced,
    }


# a provider that embeds the documents itself receives all their text; Claude, which only answers, the passages a
# question retrieves (another provider embeds for it)
AI_RECEIVES = (
    "every prompt: the column summaries, a variable's values (Data > Clean), what the user types; the text of the "
    "indexed PDFs (each passage with its PDF's name) and every search text; the text of a PDF whose graph is drawn"
)
CHAT_RECEIVES = (
    "every prompt: the column summaries, a variable's values (Data > Clean), what the user types, the passages of "
    "the indexed documents a question retrieves, the text of a PDF whose graph is drawn"
)
QDRANT_RECEIVES = (
    "the indexed passages with their PDFs' names, their vectors, and the vector of each search"
)
FAMILIES = [
    {
        "registry": "manage.iam.providers:IAM_PROVIDERS",
        "slug": "identity-providers",
        "base": "manage.iam.base:BaseIAMProvider",
        "brick": "svc_accounts",
        "places": {
            "local": [place("server", "local accounts")],
            "better_auth": [
                place(
                    "network",
                    "Better Auth",
                    when="at the address CMP_BETTER_AUTH_URL names, wherever it is: the code neither lists nor places it",
                    receives="each sign-in's name and password, and the passwords set for its accounts",
                    asked=NO_CONFIRMATION,
                    unplaced=True,
                )
            ],
            "disabled": [
                place(
                    "server",
                    "no sign-in",
                    when="every visitor is one administrator, local_researcher, with one workspace: for a researcher's own machine",
                )
            ],
        },
        "note": "",
    },
    {
        "registry": "manage.rag.providers:AI_PROVIDERS",
        "slug": "ai-providers",
        "base": "manage.rag.providers.declarations:BaseAIProvider",
        "brick": "svc_ai",
        # a provider the declaration marks cloud is outside: every session confirms before data are sent
        "places": {
            "ollama": [
                place("network", "Ollama", receives=AI_RECEIVES, asked=NO_CONFIRMATION),
                place(
                    "outside",
                    "remote Ollama · cloud models",
                    when=OUTSIDE_THE_NETWORK
                    + ", or its model runs elsewhere: a cloud model runs on ollama.com "
                    "under the account the server's Ollama is logged into, the same for every user",
                    receives=AI_RECEIVES,
                    asked=CONFIRMATION,
                ),
            ],
            "gemini": [
                place(
                    "outside",
                    "Google Gemini",
                    receives=AI_RECEIVES + ", and in Document insight the PDF file itself",
                    asked=CONFIRMATION,
                )
            ],
            "openai": [place("outside", "OpenAI", receives=AI_RECEIVES, asked=CONFIRMATION)],
            "mistral": [place("outside", "Mistral AI", receives=AI_RECEIVES, asked=CONFIRMATION)],
            "anthropic": [
                place("outside", "Anthropic Claude", receives=CHAT_RECEIVES, asked=CONFIRMATION)
            ],
        },
        "note": (
            "A provider answers, embeds the documents, or both. Claude has no embedding model: another provider "
            "embeds (Ollama by default), receives the text of the indexed PDFs and every search text, and the "
            "confirmation names it apart. The confirmation lists all that each recipient receives, and holds in one "
            "browser tab for the recipients it names, until it is unticked, the user signs out, or the tab is "
            "closed or reloaded; another recipient asks again. Unticking it stops the engine; what was sent stays "
            "with the recipient, under its own terms. The server log keeps each confirmation's account, recipients "
            "and time, never what is sent. A user chooses an Ollama address from the server's list and never types "
            "one; only an administrator adds one."
        ),
    },
    {
        "registry": "manage.vectorstores:VECTOR_STORES",
        "slug": "vector-databases",
        "base": "manage.vectorstores.base:BaseVectorIndex",
        "brick": "svc_ai",
        "places": {
            "lancedb": [place("server", "LanceDB", when="in the user's workspace")],
            "qdrant": [
                place("server", "Qdrant, embedded", when="in the user's workspace"),
                place(
                    "network", "a Qdrant server", receives=QDRANT_RECEIVES, asked=NO_CONFIRMATION
                ),
                place(
                    "outside",
                    "a remote Qdrant server",
                    when=OUTSIDE_THE_NETWORK,
                    receives=QDRANT_RECEIVES,
                    asked=CONFIRMATION,
                ),
            ],
        },
        "note": (
            "When the setting says auto, the store recommended for this machine comes first, with its reasons "
            "(Qdrant once a server is configured), then the family's order. Every Qdrant server keeps what it "
            "received until the index is rebuilt or deleted there. A user chooses a server from the server's list "
            "and never types one; only an administrator adds one."
        ),
    },
    {
        "registry": "graph.renderers:GRAPH_RENDERERS",
        "slug": "graph-engines",
        "base": "graph.base:BaseGraphRenderer",
        "brick": "graph",
        "places": {
            "plotly": [place("server", "Plotly")],
            "graphviz": [place("server", "Graphviz")],
            "yfiles": [
                place(
                    "outside",
                    "yFiles",
                    when="the user's browser loads its runtime from yworks.com at each drawing",
                    receives="the whole graph, names and labels included, handed to that runtime in the browser",
                    asked="no confirmation: a notice follows each drawing",
                )
            ],
        },
        "note": (
            "yFiles is never in the automatic order. Once installed, a user's choice or the server's setting uses "
            "it, and the offline mode (CMP_OFFLINE) refuses it. yWorks states that its runtime reports anonymised "
            "usage data on the graph's structure."
        ),
    },
]
CONTRACT = [
    (
        "A class with <code>name</code>, <code>display_name</code> and <code>availability()</code>, which answers "
        "without raising or connecting and names the optional group to install when it cannot run."
    ),
    (
        'Registered as <code>"module:Class"</code> and imported only when used: a missing library never breaks the '
        "others, and listing a family imports none of its connectors."
    ),
    (
        "<code>select(...)</code> gives the choice with its source, its reasons, its warnings and the alternatives; "
        "Settings shows them with one control (<code>ui.components.connector_selector</code>). A name that cannot run "
        "here falls through to the next source, with a warning."
    ),
    (
        "An unknown name in a <code>CMP_*</code> setting stops the start and lists the valid names, never the value "
        "(<code>manage/connectors.py</code>)."
    ),
    (
        "A name is stored in settings and preferences: it is never renamed (an old spelling is an alias), and each "
        "family's test pins its names."
    ),
    (
        "A new graph engine is one module and one registration. A vector database also needs its member in "
        "<code>VectorEngine</code>, and, when it has a server, its list of addresses and its recipient for the "
        "confirmation. An AI provider also needs its declaration (cloud or not, its SDK, the setting of its key, its "
        "chat and embedding connectors) and its member in <code>ProviderType</code>, through which the factory "
        "builds every engine; an identity provider, its name in <code>IAMProviderType</code> and its settings."
    ),
]

# ----------------------------------------------------------------------------- the catalogues

CATALOGUES = [
    {
        "holder": "core.steps:STEPS",
        "slug": "trace-steps",
        "what": "Kinds of trace step",
        "brick": "core_trace",
        "label": "title",
        "note": "Each declares its replay mode and whether it changes the data.",
    },
    {
        "holder": "core.criteria:CRITERION_KINDS",
        "slug": "criterion-kinds",
        "what": "Criterion kinds",
        "brick": "core_data",
        "label": "label",
        "note": "One function decides the rows a kind fires on, for every screen.",
    },
    {
        "holder": "biostats.plan:KINDS",
        "slug": "analysis-kinds",
        "what": "Analysis kinds",
        "brick": "stats_plan",
        "label": "title",
        "note": (
            "biostats imports nothing of the project, not even the kit: an unknown kind raises a ValueError, and no "
            "kind is legacy. Eight kinds can be computed again; the others have titles only in this version."
        ),
    },
    {
        "holder": "core.verification:CHECKS",
        "slug": "package-checks",
        "what": "Checks of a received package",
        "brick": "core_study",
        "label": "title",
        "note": (
            "Each check reads identical, within tolerance, within Monte Carlo error, not comparable, not run or "
            "failed. The command that runs them is not built yet."
        ),
    },
    {
        "holder": "ingest.dates:DATE_FORMATS",
        "slug": "date-formats",
        "what": "Date formats",
        "brick": "ingest",
        "label": "label",
        "note": "A column is read with a declared format, never guessed silently.",
    },
]
