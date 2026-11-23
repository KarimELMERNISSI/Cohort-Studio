# Developer Guide

This guide covers the architecture, design patterns, and key subsystems of CohortStudio for developers who want to understand, extend, or maintain the application.

---

## Architecture

CohortStudio is a set of Python packages in layers. Each layer imports only the layers below it: the statistics engine
runs without the web interface, and the pages hold no logic of their own. Two mechanisms carry every extension point,
one place builds the services, and a fixed set of tests keeps all of this true as the code grows.

![The architecture of CohortStudio in layers: the interface (app_pages, ui), the services (manage, graph) with their four connector families, the data work (ingest, enrich, explore, monitor), the data and statistics engine (core, biostats, prompts) with its catalogues, and utils; imports go down only; the browser enters through the interface and the command line at the services, which store one workspace per user](../assets/readme/architecture.svg)

The scheme is drawn by `dev_scripts/make_architecture_diagram.py` from a description of the layers; a test fails when it
no longer names every package of the table below.

### The layers

| Layer | Packages | Role |
| --- | --- | --- |
| L9 | `app_pages` (the page files in `app_pages/views/`) | the pages: inputs, a service call, a result card |
| L8 | `ui` | the shell, `AppState`, the components, `ui/services.py` |
| L7 | `manage` | application services, stores and connectors, in groups: iam, ai, repro, storage |
| L6 | `graph` | the graph model, its builders and its renderers |
| L5 | `explore`, `monitor` | the quality audit and correlations; outliers and comparisons |
| L4 | `enrich` | formulas, transformations, imputation, merging |
| L3 | `ingest` | reading files, with one loader |
| L2 | `core` | the domain kernel, and the extension kit (`core/registry.py`) |
| L1 | `biostats`, `prompts` | the statistics library, which imports nothing else of the project; prompt texts |
| L0 | `utils` | settings, logging, optional extras, paths, export helpers |

`main.py` and `manage_users.py` sit above every layer. The platform modules of `utils` (`utils.settings`,
`utils.optional`, `utils.data_paths`, ...) are L0; the older ones count with the package they serve (`UTILS_GROUPS` in
the test) until a milestone moves them. The groups of `manage` are not ordered among themselves yet.

**Rules:**
- **R1:** imports go strictly down, and two groups on the same layer do not import each other.
- **R2:** no import cycles. The multiple-imputation engine starts its worker process by the module's name
  (`biostats.mi.worker`, which imports the engine), never by an import.
- **R3:** nothing below `ui` reaches Streamlit, through any chain of imports. A graph renderer imports it inside its
  `render` method only (and yFiles' component package inside `translate`).
- **R4:** the AI providers' SDKs are imported only in `manage/rag/providers`.
- **R5:** the headless entry points (`manage_users`, `manage.replay` and `manage.replay.engine`, `manage.services`)
  reach neither Streamlit nor the packages of the `ai` extra. Each name that runs outside the app is listed, since
  `python -m manage.replay` runs a `__main__` no import of the package reaches; a listed name must exist.

**How `tests/test_layering.py` holds them.** It reads the import graph from the source: an import inside a function
counts, one under `if TYPE_CHECKING:` does not. R1 and R2 use the imports as written; R3 and R5 follow what runs,
parent packages and modules named in strings (the lazy registries) included. Today's exceptions are listed at the top
of the test (`KNOWN_UPWARD`, `CYCLES_BY_DESIGN`, `KNOWN_STREAMLIT`, `KNOWN_SDK`, `KNOWN_ENVIRONMENT`) and only shrink:
a new violation fails, and so does a listed one that no longer occurs, until it is deleted. The same file checks that
every package of the wheel has a layer, that `core`, `biostats` and `ingest` load no UI, plotting or AI library, that
`biostats` imports only itself, that only `utils.settings` reads the environment, and that `core` and `biostats`
never read the settings.

### Two extension mechanisms

Every extension point takes one of two shapes, both in `core/registry.py` (standard library only):

| | Connector family (`ConnectorRegistry`) | Catalogue (`Catalogue`) |
| --- | --- | --- |
| What it holds | the implementations of one service, one used at a time | kinds side by side, each named by an id |
| Who chooses | the session, then the preference, the setting, `auto` | the data: a trace or a rule names its kind |
| Registered as | `"module:Class"`, imported on first use | one frozen dataclass per kind |
| An unknown name | in a setting, stops start-up with the valid names | `UnknownIdError`, listing the valid ids |
| Used by | identity providers, AI providers, vector databases, graph renderers | trace steps, criterion kinds |
| Tested by | `tests/connector_contract.py`, called by each family's tests | one consistency test per catalogue |

**A connector family.** Each connector is a class with `name`, `display_name` and a class method `availability()`
that answers without raising or connecting: an `Availability(ok, reason, extra)`, where `extra` is the optional group
to install (`Availability.of_module("lancedb", extra="ai")` finds a package without importing it). The registry
imports a connector only when it is used, so a missing library never breaks the others. `select(...)` returns a
`Selection` (`chosen`, `source`, `reasons`, `warnings`, `alternatives`): the session's choice, then the saved
preference, the server's setting and the family's own order; a name that is unknown or cannot run here falls through,
with a warning that shows it as code. An unknown `CMP_*` value stops start-up instead (`main.py` calls
`manage.connectors.check_connector_settings` after loading the settings), naming the variable and the valid names,
never the value, which may be a pasted key. An old spelling stays accepted as an alias.

| Family | Registry | Setting | Automatic order |
| --- | --- | --- | --- |
| Identity providers | `manage.iam.providers.IAM_PROVIDERS` | `CMP_IAM_PROVIDER` | none: the setting only (`local`) |
| AI providers | `manage.rag.providers.AI_PROVIDERS` | `CMP_AI_PROVIDER` | `ollama` |
| Vector databases | `manage.vectorstores.VECTOR_STORES` | `CMP_VECTOR_STORE` | a recommendation, `lancedb`, `qdrant` |
| Graph renderers | `graph.renderers.GRAPH_RENDERERS` | `CMP_GRAPH_RENDERER` | `plotly`, `graphviz`; never `yfiles` |

Settings shows each choice and its reasons with one control, `ui.components.connector_selector`.

**A catalogue.** `register(kind, legacy=False, aliases=())` adds a kind, and the catalogue lists its kinds in the
order they were registered. `get(id)` finds a kind by its id or an alias, and raises `UnknownIdError` (a
`LookupError`) for any other; `ids(legacy=False)` lists what is still offered. An id stored in a trace, a
configuration or a result is never renamed: a kind that nothing writes any more stays registered with `legacy=True`,
so that what was stored still reads. A test that needs a kind of its own registers it in `copy()`, patched over the
module's catalogue for that test only. The catalogues are `core.steps.STEPS`, the kinds of trace step, and
`core.criteria.CRITERION_KINDS`, the criterion kinds (M6's date order and M9's concepts will be criterion kinds).

### The composition root

- **`manage/services.py`**, headless (rule R5); the AI engine, which needs the AI packages, is built in `ui`:
  - `build_shared(settings=None)` returns `Shared`, one per server process: the settings, the identity provider
    (`iam`), the graph renderers (`graphs`) and the multiple-imputation worker budget (`mi_workers`), each read from
    its process-wide instance when used. A command line passes the settings it loaded.
  - `build_services(shared, scope, paths)` returns `Services`, one per session: the user's scope
    (`core.workspace.UserScope`), the resolver of the files a trace names (`manage.paths.PathResolver`), and the
    dataset store (`datasets`) and the trace (`trace`), built on first use with that scope.
- **`ui/services.py`** keeps them: `shared()` for the process (`st.cache_resource`); `services()` in the session
  state (`_services`; the stores also under `db_manager` and `transformation_manager`, which pages read); and
  `ai_engine()`, built on first use, under `rag_manager`. A session holding anything of another user is ended
  (`end_session`); a dropped engine is stopped first (`drop_ai_engine`). Not `st.cache_resource(scope="session")`:
  its entries go when the browser's connection drops, while the session state survives.

**Why pages build no service.** A service takes the user's scope as a parameter and never asks the session for the
user: its files land in that user's workspace, and the command line of M8 builds the same services with a resolver
limited to the files it names. One AI engine per session means that a withdrawn consent stops the only engine that
could send data. `tests/test_quality_gates.py` holds this (`test_pages_never_build_services`,
`test_the_session_builds_its_services_with_the_users_workspace`, `test_only_the_services_write_their_keys`), and
`tests/test_services.py` builds the services without Streamlit or the AI packages.

### Settings and preferences

- **`utils/settings.py`** holds one `Settings` object, in sections (`paths`, `logging`, `iam`, `ai`, `vector_store`,
  `graph`, `statistics`, `development`), with keys and passwords as `SecretStr`. `main.py` and `manage_users.py` call
  `load_settings()` first: it loads `.env` (a variable already set wins) and checks every value, and a wrong one
  stops start-up with the variable's name, never its value. Elsewhere, read them with `get_settings()`.
- **One source.** `utils.settings` is the only module that reads the environment (`tests/test_layering.py`), and
  `.env.example` lists exactly its variables (`tests/test_settings.py`). `.streamlit/secrets.toml` is not read; its
  presence is logged. `core` and `biostats` never read the settings: they receive values as parameters.
- **Preferences** are the non-secret choices made in Settings: the day-first default, the AI provider and its models,
  the vector database, the Ollama address and the Qdrant server, the graph renderer. `manage/preferences.py` keeps
  them per user in `<workspace>/preferences.json` (`load_preferences`, `update_preferences`; written whole and
  atomically), and `ui/preferences.py` seeds the session after sign-in (`seed_session`). A preference never holds a
  key, a password or a consent, and an address only as chosen among those the server offers
  (`tests/test_preferences.py`; `tests/ui_layer/test_settings_persist.py` and
  `tests/ui_layer/test_ollama_address_choice.py` for the addresses). Details: *Settings, preferences and API keys*,
  below.

### Session state

**Declared keys.** A session key that one module writes and another reads is declared in `ui/keys.py` as a `Key`:
its `name`, its `owner` (the module that writes it), its `lifetime`, its `readers` and a `note`. The lifetime decides
what `AppState` clears:

| Lifetime | Kept | Cleared by `AppState` |
| --- | --- | --- |
| `session` | for the whole session | never |
| `page` | while the user stays on its page | with another dataset too, when `reset_on_dataset` is set |
| `dataset` | while the same dataset changes | when another dataset is loaded |
| `data_change` | until the data change | on any change of the data, and with another dataset |

`ui.state.DERIVED_KEYS`, which `AppState.set_dataset` and `clear_dataset` clear, and `KEPT_ON_DATA_CHANGE` are derived
from the declarations; after a change of the data, `AppState` clears the others (`DATA_DERIVED_KEYS`).
`tests/ui_layer/test_session_keys.py` checks each declaration against `AppState` and against the code of its owner
and readers, and a quality gate counts the keys written without a declaration: that count can only fall.

**The tables.** `AppState` (`ui/state.py`) is the only code that changes the data, through
`core.session.DataSession`: `commit_change()` for the dataset, `stage()` for the Enrich draft, `note()` for a step
that changes nothing. Each change is one trace step; with `CMP_STRICT_STATE=1` (the test suite), changing a view
below in place raises `UntracedMutationError`.
- `working_df` (and `data`) is the analysis population: the rows the ticked criteria keep, as a view reinstalled at
  the start of every run (`AppState.begin_run`). While a ticked criterion cannot be evaluated, it holds every row and
  a banner stops the analyses; `AppState.analysis_frame()` raises `PopulationError` instead.
- `AppState.draft_frame()` is what Enrich works on: the draft when there is one, otherwise every row of the dataset,
  as a copy. The `enriched_df` key shows the same table.

### Runtime

- **Only the open tab runs** on Validate (`app_pages/data_monitoring.py`), Statistics & correlations
  (`app_pages/home.py`) and Hypothesis tests & epidemiology (`app_pages/epidemiology.py`):
  `st.tabs(list(TABS), key=TAB_KEY, on_change="rerun")`, then `if tab.open:` around each tab's body. Validate's
  Compare tab always runs: its file uploaders cannot keep a file while it is closed. The tab keys are declared in
  `ui/keys.py`, and every control of these pages has a `key` and, but for the exceptions the test lists,
  `persist_state="page"`, so a closed tab keeps its values (`tests/ui_layer/test_lazy_tabs.py`).
- **Computations are kept by fingerprint.** `DataSession.memo(name, key, compute)` keeps a value until the data
  fingerprint or the caller's key changes. The population is kept by the data fingerprint and the fingerprints of the
  ticked criteria; `AppState.analyzer()` keeps the column profile, and `AppState.quality_audit()` the audit, keyed by
  the criteria, the derived variables and the method. The statistics of Statistics & correlations are cached per
  session (`st.cache_data(scope="session")`), keyed by the table's fingerprint, and their Reset clears that session's
  only; the correlation workbooks are cached eight at most, for an hour.
- **One worker budget per process for the multiple imputation.** `manage.compute.mi_workers()` returns the
  `WorkerBudget`: a run leases what is free (`lease(wanted)`), never waits, and gives it back however it ends. With
  fewer than two workers free, the run takes place in the session's own process. `CMP_MI_MAX_WORKERS` sets the total
  (the processors minus one by default; a container's CPU quota is not visible there, so set it).
- **A discarded Enrich draft deletes its snapshot files**, but never one that a saved trace uses
  (`manage.snapshots.delete_unreferenced`).
- **Expressions are computed in worker processes, with a time limit and a memory limit.** Every formula and criterion
  goes through `core.expressions.evaluate_tree`, which hands the tree to the runner installed in the process
  (`core/expressions/runner.py`); none means this process (a notebook, the worker itself). `main.py` installs the pool
  of `manage/workers/` once per server process (`install_for`), unless `CMP_FORMULA_IN_PROCESS` keeps expressions in
  the process (the test suite sets it; the log warns; never on a shared server). Every expression goes there, the
  user's own included: one rule, so that no page computes one in the server's process.
  - **A worker** is the real interpreter, started isolated (`-I -B`), in an empty folder, with no setting or key in its
    environment; it imports `core.expressions` only. Its memory limit is a Job Object on Windows and its own
    `RLIMIT_DATA` on Linux (`manage/workers/limits.py`); macOS has the time limit only. The pool ends a worker at the
    time limit, the only way to stop Python's pattern engine, and starts its replacement in the background. A worker
    runs at the normal priority: the time limit is the server's clock, and a starved worker missed it.
  - **A worker serves every user in turn:** what one job leaves in its memory is out of the next job's reach only
    because the evaluator offers no function that reads memory it was not given, or a file
    (`test_no_expression_reads_memory_it_was_not_given_or_a_file`). Adding a numpy or pandas name to the evaluator
    must keep that test passing.
  - **What travels:** the tree, the table cut to the columns it reads, the values among its names, and the functions it
    calls by name, which must be registered in `runner.HELPERS`. An answer is read by a reader that knows the names
    values and errors need and nothing else (`manage/workers/wire.py`), its size capped.
  - **A stop** is the expression's own failure (`ExpressionStopped`, an `UnsafeExpressionError` with its reason and
    limit): remembered, shown in its own words, and said "stopped earlier" when it is refused again. **No worker**
    (`WorkerUnavailable`) is nobody's: the handlers that catch everything let it through and nothing is remembered.
    The sidebar's population box and an AI suggestion's trial say it in place, so the page goes on; a page that needs
    an expression stops, and `main.py` says why in the waiting line's place, with Try again.
  - **A page that waits** more than a second shows a line (`ui/waiting.computing`): under the control that started the
    computation when the page opens its own (Compute & Preview, Validate's criterion form), at the top of the page
    otherwise; Stop or a click elsewhere stops the run there, and the worker computing is ended. A widget's callback
    runs before the script, outside that line, so the criteria's callbacks record the ticks only
    (`AppState.tick(..., install=False)`) and the run computes the population (`begin_run`, inside `computing()`).
  - **Tests:** the suite computes every expression through the wire in memory (`manage.workers.wire.round_trip`,
    installed by `tests/conftest.py`), so a value or a class the reader of answers does not know fails there first; the
    processes have their own tests (`tests/test_expression_workers.py`).
  - **What a worker does not stop:** it can open a file the server's account can read, and the network. The evaluator's
    whitelist remains the barrier against running code; the worker is the one against holding the server.

### Keeping it so

| Check | Where | What may only move one way |
| --- | --- | --- |
| Layers, R1–R5 | `tests/test_layering.py` | the exception lists shrink |
| Lint | `ruff check` (CI, pre-commit) | |
| Formatting | `ruff format --check` (CI, pre-commit) | `[tool.ruff.format] exclude` in `pyproject.toml` shrinks |
| Types | `dev_scripts/check_type_baseline.py` (pyright, CI) | the errors per file of `dev_scripts/pyright_baseline.json` fall |
| Coverage | `dev_scripts/check_coverage_floors.py` (CI) | a floor per package, raised as coverage grows |
| Source rules | `tests/test_quality_gates.py` | ceilings that only fall |

- **Formatting and types.** A file leaves the exclude list in a commit that only reformats it, listed in
  `.git-blame-ignore-revs`. A type error in a file outside the baseline fails; a count that falls is written down
  with `--update`, which never raises one.
- **Coverage.** CI fails below 70 % in all, and below each package's floor (`FLOORS`); a package more than 4 points
  above its floor (`MARGIN`) fails too, until the floor is raised.
- **Source rules.** No `eval` outside the expression evaluator; only `AppState` writes the dataset and the trace
  steps; pages build no service; rules are written through their kinds. Raw HTML, `print()` calls, undeclared
  session keys and unformatted files each have a ceiling that only falls.

### Recipes

#### Add a connector

1. Write the class in its family's package, on the family's base class (`BaseIAMProvider`, `BaseAIProvider`,
   `BaseVectorIndex`, `BaseGraphRenderer`), with `name`, `display_name` and `availability()`, which never raises,
   never connects, and names the optional group its package comes from.
2. Register it in the family's registry, for example `GRAPH_RENDERERS.register("name", "graph.renderers.name:Class",
   extra="group")`. The name is stored in preferences and settings: never rename it; an old spelling is an alias.
3. A package it needs goes into an optional group of `pyproject.toml` and into `utils.optional.EXTRAS`
   (`tests/test_ai_provider_sdks.py` keeps them equal). An AI provider imports its SDK in `manage/rag/providers`
   only (R4); a renderer imports Streamlit in `render` only (R3).
4. Nothing else changes: the setting accepts the name, and Settings offers it where it can run (not an identity
   provider, which the setting alone chooses).

Test: add the name to the family's `test_the_stored_names_never_change`; the family's tests run the checks of
`tests/connector_contract.py` on it (availability, light listing, a missing package, precedence, unknown names).

#### Add a kind of trace step

1. Register a `StepKind` in `core/steps.py` with `_register(StepKind(id, title, category, replay, changes_data=...))`:
   `category` is one of `CATEGORIES`, `replay` is `RECOMPUTE`, `SNAPSHOT` or `DOCUMENT`, and `flow` and `fills` are
   given when they apply. `title` is its heading in the trace report (`step_title`, read by
   `manage/trace_documenter.py`).
2. Record it through `AppState` (`commit_change`, `stage` or `note`), never with `add_step` (a quality gate).
3. A step that the replay recomputes gets its handler in `_HANDLED`, in `manage/replay/engine.py`.
4. Never rename an id: a kind that nothing records any more is registered with `legacy=True`.

Test: `tests/core_layer/test_steps.py` (every step written is registered, the report titles every kind, the replay
handles each kind by its mode); with `CMP_STRICT_STATE=1`, an unregistered step raises `UnknownIdError`.

#### Add a criterion kind

See *Adding a criterion kind*, below: its functions in a module of its own, one registration in `core/criteria.py`
after the five kinds of M5, a writer, and its examples in `tests/core_layer/test_criterion_kinds.py`.

#### Add an analysis

Follow `app_pages/analysis_survival.py`:
1. The computation goes into `biostats`: plain values in, a result dataclass out, no import of the application.
2. In the page module, every control has a key. On the button: compute, then `remember(key, result, config=...,
   n_analysed=..., columns=...)` (`ui/results.py`) and `mark_step("analyse", ...)`; an exception goes to
   `error_card(...)`.
3. On every run, `recall(key, config=config)` returns the result only while its data, population and settings still
   hold. Show it, then call `provenance(key)`, which says what it was computed on and offers to add it to the dossier.
4. Give the key a title in `RESULT_TITLES` (`ui/results.py`; without one, the dossier shows the key in words). A new
   type of result gets its tables in `manage/report_tables.result_blocks`, which gives an unknown type a short text.
5. Compare its numbers with R: the reference computation goes into a script of `tests/reference/r/`, the manual
   workflow `.github/workflows/r-fixtures.yml` runs it, its values are copied into `tests/reference/fixtures/`, and a
   parity test in `tests/biostats_layer/` (as `test_rates_r_parity.py`) compares, skipped until they are there.

Test: an AppTest of the page (as `tests/ui_layer/test_survival_page.py`), and the module added to the list of
`test_analysis_pages_keep_results_in_the_result_store` in `tests/test_quality_gates.py`.

#### Add a calculation from numbers typed

A calculator's result is no result of the dataset: it is a function of its numbers, kept as JSON and computed again
wherever it is read. Follow the three calculators of `app_pages/analysis_extras.py`, or the typed 2x2 table of
`app_pages/analysis_diagnostic.py`:
1. The computation goes into `biostats`, as for an analysis.
2. In `biostats/plan.py`, give its kind what it needs to be computed from numbers typed: `typed_title`,
   `typed_inputs` (each number by a name that is never renamed, as a `COUNT` or a `QUANTITY`), `typed_options` with
   their defaults (their closed values in `choices`), `typed_run`, and a `record` (the counts behind each part in
   its support: a count typed is counted as a participant's). The record holds every number and every word its
   block needs (a rate's unit and multiplier, what the events are, each interval's level): the block is written
   from the record alone. A change of the numbers, of the options or of what `typed_run` computes moves
   `typed_version`, never `version`, which is the dataset's.
3. Its words and its block go into `manage/report_tables.py`: `typed_numbers` (the numbers with their names),
   `typed_title`, and a block written from the record in `RECORD_BLOCKS`, with the numbers written by
   `utils/numbers.py`, the pages' rule. `calculation_blocks` gives a calculation its block. A kind whose numbers
   the dossier holds back under a rule of the dataset's results goes through that result's own block, never a copy
   of its rule: the 2x2 table typed is written by the accuracy block (`CALCULATION_KINDS` lists the kinds).
4. On the page: `layout.numbers_source(key)` for the source field, `spec = specify(kind, typed=..., options=...)`,
   `run_spec(spec).result` for the headline, then `layout.typed_close(spec, source, key)`, which writes the line of
   the numbers and offers the button (`ui.results.typed_provenance`; `key` is the source field's: a button's
   callback runs with the arguments of the run before, so the source is read from its field when the button is
   clicked). Never `remember`: it refuses a typed spec.
5. Privacy: an error names a number, never its value. Nothing typed is logged (a guard logs the exception's type).
   A calculation is never named outside the session by a hash of its numbers: a small count is found back from
   one. A block the dossier holds back says so (`ResultBlocks.held_numbers`), and the line of its numbers is not
   written either.

Test: the round trip of the spec and of the record, and the function the calculator shows
(`tests/biostats_layer/test_plan_typed.py`); the button, the dossier page and the Word section
(`tests/ui_layer/test_typed_dossier.py`).

#### Add a page panel

1. Write it in a module of its own, with one `render_...()` function called in its tab, as
   `app_pages/enrichment_derived.py` (`render_derived_variables`) is in Enrich: a long page function does not grow.
2. On a page that runs only its open tab, the panel goes inside `if tab.open:`. For a link to open its tab, add the
   tab's id and labels to the page's `PageDef.tabs` in `ui/nav.py`, build the row with `linked_tabs(...)`, and link
   with `page_link(..., tab=...)` or `switch_to(..., tab=...)`, which refuse a tab the page does not declare.
3. Every control has a `key` and `persist_state="page"`, except a confirmation and an input whose value comes from a
   session key that already keeps it. A tab's label never changes (a count in it would reset the open tab).
4. A key it shares with another tab or page is declared in `ui/keys.py`, with the lifetime it needs.
5. It changes the data through `AppState` and takes its services from `ui/services.py`.

Test: an AppTest in `tests/ui_layer/`, its module in `LAZY` (`tests/ui_layer/test_lazy_tabs.py`) when its page runs
only the open tab, and `tests/ui_layer/test_session_keys.py`.

---

## RAG Manager (Mixin Pattern)

The RAG system is structured using the **Mixin composition pattern** to keep each concern in its own module:

```text
RAGManager(TaxonomyMixin, DocumentsMixin, ComputedVarsMixin)
├── rag_taxonomy.py      # Taxonomy generation & refinement
├── rag_documents.py     # PDF processing, summarisation, knowledge graphs
├── rag_computed_vars.py # Variable suggestion & imputation formulas
└── rag_manager.py       # Orchestrator: initialisation, vector index, shared utilities
```

### Key Design Decisions

- **The vector database is a connector** in `manage/vectorstores/`: **LanceDB** (embedded, on disk; the
  default) or **Qdrant** (embedded in memory, or a server). The RAG code only sees `DocumentIndex`
  (`similarity_search`, `max_marginal_relevance_search`, `as_retriever`, `build`), so it is identical whatever the
  engine; filtering by document is a list of sources that each engine translates (SQL `IN` for LanceDB, a `MatchAny`
  payload filter for Qdrant). See "Vector databases" below.
- **AI providers are pluggable connectors** in `manage/rag/` (local Ollama by default; Google Gemini, OpenAI, Mistral and Anthropic Claude as cloud providers that require an explicit data-sharing consent).
- **Anthropic Claude** (`manage/rag/providers/claude.py`, official `anthropic` SDK) answers but has no embedding
  API. `ConnectorFactory.create_connectors(provider="anthropic", embedding_provider=...)` pairs it with another
  provider's embeddings (local Ollama by default, so the library is indexed on the server and only the retrieved
  passages reach Claude); the index manifest records that embedding provider. Claude models choose their own
  sampling, so the app's temperature is not sent. Model IDs: `claude-fable-5-1` (default, most capable),
  `claude-opus-5-5`, `claude-sonnet-5-5`, `claude-haiku-5-5`.
- Each mixin accesses shared resources (`self.vectorstore`, `self.llm`) from the base `RAGManager`.

---

### Vector databases

| Engine            | Mode     | Where vectors live                                                                          | Suits                                                                                                       |
| ----------------- | -------- | ------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| LanceDB (default) | embedded | `<workspace>/vectors/lancedb/`, columnar Lance files on disk                                | one machine, any library size; low memory; exact search below 50 000 chunks, an IVF index above             |
| Qdrant            | embedded | `<workspace>/vectors/qdrant/storage/`, every vector in memory                               | small and medium libraries (up to ~20 000 chunks); one process per folder (the client is shared per folder) |
| Qdrant            | server   | a Qdrant service (`CMP_QDRANT_URL`, optional `CMP_QDRANT_API_KEY`), one collection per user | several users, very large libraries, concurrent sessions; HNSW search and payload indexes                   |

- **Choice** — Settings > AI engine > _Vector database_. The default comes from `CMP_VECTOR_STORE` (`auto`, `lancedb`
  or `qdrant`); `auto` follows `manage.vectorstores.recommend()`, which weighs the machine (CPU cores, memory, free
  disk, probed without extra dependencies), the size of the library (from the current index, else estimated from the
  PDF sizes) and whether a Qdrant server is configured, and explains its choice.
- **Manifest** — every index has a `manifest.json` (engine, embedding model, dimensions, chunks, documents, location,
  build time). Reusing an index built with another embedding model is refused, because the vectors would not be
  comparable. Embedding failures are reported after three attempts instead of leaving silent gaps in the index.
- **Qdrant rebuilds** write a new collection (`<base>_<id>`, recorded in `collection.txt`) and then drop the previous
  one; embedded Qdrant on Windows can reload a deleted collection's files when the same name is recreated.
- **Adding an engine** — one module implementing `BaseVectorIndex` (`availability`, `exists`, `reset`, `add`, `query`,
  `count`, optional `optimize`) and one branch in `open_index()`; `tests/test_vectorstores.py` runs the same contract
  tests on every engine.

## Authentication & User Management

### Account Lifecycle

```text
Sign Up → Pending → Admin Activates → Active → (Admin can Deactivate/Delete)
```

| State           | Description                                           |
| --------------- | ----------------------------------------------------- |
| **Pending**     | Account created but not yet activated. Cannot log in. |
| **Active**      | Admin-approved. Full access to the application.       |
| **Deactivated** | Admin-disabled. Cannot log in until reactivated.      |
| **Deleted**     | Permanently removed from the database.                |

### Implementation

- Passwords are hashed with **bcrypt** before storage.
- User records are stored in `users.duckdb` (shared data folder) by the local IAM provider (`manage/iam/providers/local_duckdb.py`), which `manage.iam` opens.
- The `is_active` field controls login eligibility.
- The **Admin > Users** page goes through `get_iam_manager()`, so it manages the accounts of the configured provider
  (local or Better Auth), never the local database in its place.
- Administrators are created from `CMP_ADMIN_USERNAME` / `CMP_ADMIN_PASSWORD(_FILE)` on first start or with `python manage_users.py create-admin <name>`; the role (not the username) grants access to the **Admin > Users** page.
- One rule for every way a password is set, `manage.iam.authz.validate_password`: at least `CMP_MIN_PASSWORD_LENGTH` characters (12 by default; the setting is refused outside 8 to 64, `utils/settings.py`), not spaces only, at most 72 bytes (the hash's limit). The forms and `manage_users.py` call it for their message, and `IAMManager` applies it itself in `create_user`, `update_user_password` and `create_first_admin`, so that no caller can forget it; the providers stay thin.
- `IAMManager.ensure_bootstrap_admin` raises `IAMConfigurationError` for a `CMP_ADMIN_PASSWORD(_FILE)` that is set and cannot be used (too short; a file that is missing, empty or not UTF-8 text): the start stops, and no setup token is issued in its place, since whoever sets a password keeps every credential out of the log. Its words name the setting and give a reading error by its kind: a decoding error's own text quotes a byte of the password.
- `manage_users.py` asks for a password on the terminal (and says so where nobody can type); only `create-admin` of the account named by `CMP_ADMIN_USERNAME` takes `CMP_ADMIN_PASSWORD`.
- **First-run setup token** — Without a bootstrap password, the first start (or `python manage_users.py bootstrap`, which the Docker image runs before the server) writes a one-time setup token to the server log, and the login page creates the first administrator with it. `python manage_users.py setup-token` issues a new token (the previous one stops working). Only a SHA-256 hash is stored (`setup_token.json` in the data root, `manage/iam/setup_token.py`); the token is cleared once an administrator exists.

### Data Isolation

Per-user isolation is enforced by `core/workspace.py`:

- **Workspace directories** — `UserScope.for_user(username)` maps every storage kind (`datasets`, `stats`, `traces` with `artifacts/` and `snapshots/`, `uploads`, `taxonomy`, `knowledge_graphs`, `documents`, `vectors`, `rag_eval_logs`) to `<DATA_ROOT>/users/<folder>/<kind>/`. The folder is the username when it is a folder name already (ASCII letters and digits, with `_`, `.` or `-` inside, at most 64 characters, not a Windows device name), and otherwise `_` and a readable code of it that no other name has (`core.workspace.safe_dirname`: `j.smith@chl.lu` gets `_j.smith_at_chl.lu`); a code longer than 64 characters gets `_` and 32 hexadecimal characters of the name's SHA-256. The folder an earlier version gave a local account is moved at its sign-in when it is clearly its own (`move_earlier_workspace`); `dev_scripts/move_workspaces.py` plans and moves them for another identity provider. `DBManager`, `TransformationManager` and `RAGManager` take the scope they are given (`scope=`) and never ask the session: `AppState` and `ui/services.ai_engine()` pass the signed-in user's (`ui/session_scope.get_current_scope()`). Without one (command line, tests) they use the shared directories, where a `username=` argument selects the `<user>_` files.
- **Path containment** — `UserScope.resolve_within()` rejects any manually entered or trace-recorded path that resolves outside the workspace; uploaded file names go through `safe_filename()`.
- **Migration** — `python dev_scripts/migrate_user_layout.py --dry-run` previews, and without the flag moves, the legacy flat layout (`data/datasets/<user>_*.parquet`, `data/traces/<user>_session_*.json`, `data/taxonomy/<user>/`, `data/knowledge_graphs/<user>_*.json`, `DOCUMENTS/`) into the per-user workspaces. Files without a username prefix are assigned to the administrator.
- **Login lockout** — `CMP_MAX_FAILED_LOGINS` consecutive failures lock an account for `CMP_LOCKOUT_MINUTES` (defaults 5 and 15).

---

## Pydantic Response Schemas

`manage/rag_schemas.py` defines Pydantic models for structured LLM responses:

| Schema                                             | Purpose                                                                   |
| -------------------------------------------------- | ------------------------------------------------------------------------- |
| `TheoreticalConcept`, `TheoreticalConceptList`     | A medical concept or score with its standard formula and required inputs. |
| `ComputedVariableSuggestion`, `SuggestionResponse` | AI-suggested computed variables mapped to dataset columns.                |
| `ProxyVariable`                                    | A proxy for a missing variable.                                           |
| `AlternativeFormula`                               | An alternative formula that avoids a missing variable.                    |
| `ExpandedTerm`                                     | The full standard name of an abbreviated term.                            |
| `FormulaCorrection`                                | A corrected formula and the changes made.                                 |
| `MarkdownFormulas`                                 | LaTeX versions of formulas.                                               |

Only the proxy-variable and alternative-formula suggestions are validated against their schema
(`validate_and_parse()` in `manage/rag_computed_vars.py`). The other responses (taxonomy, documents, suggestions)
are parsed as plain JSON with `parse_json_safe()` or `json.loads()` (for example in `manage/rag_taxonomy.py`),
without schema validation.

### Why Schemas Matter

- **Type safety** — Catch malformed LLM output before it reaches the UI.
- **Repair** — If parsing fails, `utils/llm_utils.py` repairs common JSON errors. Re-prompting the LLM with the error
  is implemented in `StructuredOutputHelper.invoke_with_schema()`, which no feature calls yet.
- **Consistency** — Downstream code can rely on well-typed objects instead of raw dictionaries.

---

## LLM Utilities (`utils/llm_utils.py`)

Robust JSON parsing pipeline for LLM responses:

1. **Markdown Extraction** — Strips ```json fences from the response.
2. **Bracket Balancing** — Fixes unmatched `[`, `{`, `]`, `}`.
3. **Trailing Comma Removal** — Cleans common JSON syntax errors.
4. **Pydantic Validation** — Validates the parsed object against the expected schema (`validate_and_parse()`, where a schema is used).
5. **LLM Self-Repair** — `StructuredOutputHelper` can send the malformed output back to the LLM with an error message and ask for a corrected version; no feature uses it yet.

---

## Data Analysis Utilities

### `ingest/analyzer.py` — DataAnalyzer

Classifies DataFrame columns into semantic types:

| Type                           | Description                                                         |
| ------------------------------ | ------------------------------------------------------------------- |
| `numeric_cols`                 | Continuous numerical columns (float/int, high cardinality).         |
| `categorical_cols`             | String/object columns.                                              |
| `binary_cols`                  | Columns with exactly 2 unique values.                               |
| `date_cols`                    | Datetime columns.                                                   |
| `low_cardinality_numeric_cols` | Numerical columns with few unique values (may be coded categories). |

Pages do not build one themselves: `AppState.analyzer()` (the analysis rows) or `AppState.analyzer("base")` (every
row) returns a new analyzer each call, built from a column profile that is computed once per dataset and population
(`DataSession.memo`). The Validate dashboard's quality audit is kept the same way (`AppState.quality_audit`).

### `utils/statistics_utils.py`

- Normality tests (Shapiro-Wilk, D'Agostino).
- Test selection guidelines based on sample size and distribution.

### `utils/visualization_utils.py`

- Plotly chart generators with statistical overlays.
- Palette generation (`generate_palette()`, cached with `@functools.cache`).

### `utils/clustering_utils.py`

- `prepare_data_for_clustering()` — Standardises data and handles missing values.
- `fit_kmeans()`, `fit_dbscan()`, `fit_gaussian_mixture()` — Algorithm wrappers returning labels and model objects.

---

## Trace & Reproduction System

### TransformationManager

Manages the recording of analysis steps:

- `initialize_session(dataset_name, *, parent=None)` — Starts a new trace session, in the user's workspace (`parent`: the trace of the saved version it starts from).
- `add_step(function_name, params, description, output_dataset_path=None, ..., *, data=None, population=None, replayable=None)` — Records a transformation step. Pages do not call it directly: changes of the data go through `AppState` and `DataSession`, which record the step.
- `get_available_datasets_from_traces()` — Lists datasets referenced in past traces.

### TraceDocumenter

Generates `.docx` reports from trace JSON files using `python-docx`:

- Session metadata header.
- Step-by-step table with function, parameters, and description.
- Downloadable via Streamlit's file download button.

### Replay (`manage/replay/engine.py`)

`run_trace()` is the single replay engine: it replays a trace up to a step and checks each step against its
recorded fingerprint. It reads the files a trace names through the resolver it is given (`manage/paths.PathResolver`):
the user's workspace in the app, only the named files for a command line, only a study package's members
(`PathResolver.for_package`, each by the step that reads it and the string the trace records, never by a name);
given none, it reads no file. The engine asks for each file with its step (numbered from 1, as the package's index
numbers it), so that two merges that record one string (two sheets of one workbook) read their own members. A member
the resolver marks as a table (`reads_as_table`: a merged file cut to the rows its merge took, a workbook's table) is
read as that table, by pandas, without the step's reading options and without comparing it with the checksum the
step recorded of the original file; no member of a package is ever read by DuckDB (`read_version(...,
duckdb_fallback=False)`, a keyword every caller gives), nor looked up by a name. `package_paths(trace, files,
folder)` builds that resolver from the package's index (each `DataFile` matched with the file `files_needed`
lists by its role and step); `for_package` refuses a table that is no member.
The package `manage/replay/` imports nothing in its `__init__`, so that the study package's command line starts
without loading the engine.

- `trace_problems(trace)` says what keeps a file from being a trace (its steps a list of objects, each naming its
  kind, every record the replay reads an object), before anything reads it; `run_trace` refuses such a file with the
  first reason, and so does the Reproduce page.
- `files_needed(trace)` lists every file a replay may read, as `NeededFile`s: the role (a stored id: `source`,
  `saved_version`, `replayed_state`, `enrichment`, `snapshot`; `ROLE_WORDS` says them in a sentence), the step
  (from 1; None for the source), the
  paths the trace records in the order the replay tries them, the saved versions it then looks up (taken from what the
  load recorded: the version it was loaded from, or the dataset's name and version), and the checksum the step
  recorded. `locate(needed, paths)` finds one as the replay does. `load_source` reads the source through them, the
  Reproduce page asks for the missing files from them, and the study package carries those files; a test checks that
  a replay asks its resolver for nothing else (`tests/test_replay_files.py`).
- `on_step`, given to `run_trace` or `apply_steps`, receives a `StepSeen` after each step the replay passes: the
  tables before and after, how the step was passed (`recomputed`, `documented`, `snapshot`, `in_part`, `skipped`,
  `not_drawn`),
  the fingerprint computed here and the one recorded. A step taken from its recorded table is listed in
  `ReplayResult.from_snapshot` and leaves the replay not reproducible.
- `options`, a `ReplayOptions`, says how a replay redoes what the trace does not hold. `process_seed`: the seed a
  MissForest step recorded without its own is replayed with (a study package's, the exporter's, which its
  pre-flight used); it matters only for a column fitted on more than 200,000 rows, and the step's recorded
  fingerprint decides. `draw`: the hashes of the steps whose multiple imputation is drawn again (None: every one). A
  step not drawn is checked against the counts it recorded (its rows, the cells it imputed per column, its m; a
  difference is in `mi_divergence`), its rows are in `ReplayResult.mi_rows`, it is passed in the mode `not_drawn`
  and listed in `not_drawn`, which leaves the replay not `reproducible`. Every imputation step's inputs are checked
  so, drawn or not (`mi_inputs_differ`). `matches` says whether everything checked held: no error, no divergence of
  a table or an imputation, no step skipped or redone in part. A draw reports its `progress`, and runs in the
  replay's own process, or in worker processes when `workers` is the server's budget
  (`manage.compute.mi_workers()`): the engine leases what is free around each draw large enough to pay for their
  start (`manage.compute.pays_to_parallel`), and gives it back when that draw ends. A parallel draw writes its copy
  of the table under `scratch`, a private folder of the caller's, or under the system's temporary folder.
- **One BLAS thread.** The cells of an imputation are the same wherever it is drawn: every chain runs on one BLAS
  thread (`biostats.mi.engine.one_blas_thread`), in the session's process, in a worker and in a replay alike. The
  setting is the process's and is never given back: a limit set for the time of a draw is undone by other code that
  saves and restores it meanwhile (scikit-learn's KMeans does). The server sets it when it starts, a replay at its
  first step and the test suite at its start (`manage.compute.one_blas_thread_for_this_process`); a process that
  only draws gets it at its first draw. A draw reads the threads as it begins, after each chain and as it ends
  (`ImputationRun.blas_threads`), which the step records (`platform.blas[].threads`: 1), and warns if other code
  moved them. A step recorded before that, and drawn in the session's process, can give other last digits for a
  variable observed on more than 10,000 rows (or on fewer with very many predictor columns): its warning then
  names the threads it recorded ("BLAS threads 14 -> 1"), and `ReplayResult.mi_changes` holds what changed, by
  step, as data: any change listed, the two draws are not comparable; none, the same software gave other cells.
- **One rule for what a replay compares.** Whatever a fingerprint is taken of is computed on one BLAS thread, when
  it is recorded (the server's process, from its start) and when it is replayed (a replay sets it at its first
  step): `manage.compute.one_blas_thread_for_this_process`. The scores of a PCA or a FAMD of a large table, and the
  values of a single imputation by chained equations, change their last digits with the threads, as a multiple
  imputation's cells do. Such a step recorded before does not record its threads: replayed now, it can read as not
  reproduced without the replay naming why.

### Kinds of trace step (`core/steps.py`)

Every kind of step is registered once, in the `STEPS` catalogue (built on `core/registry.py`). Each kind declares:

| Field          | Meaning                                                                                                                       |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `id`           | the name stored in traces: never renamed                                                                                      |
| `title`        | the heading of the step in the trace report                                                                                   |
| `category`     | the part of the workflow whose page records it (`load`, `validate`, `enrich`, `population`, `analyse`, `report`, `other`)     |
| `replay`       | `recompute` (redone from its parameters), `snapshot` (the recorded table is used) or `document` (nothing to redo)             |
| `changes_data` | whether it changes the values or the rows (renaming or retyping columns does not); a step with fingerprints is judged by them |
| `flow`         | the noun of its box in the participant flow ("After the merge")                                                               |
| `fills`        | `fitted` or `formula` for a step that fills missing values with one value each                                                |

The replay, the trace report, the workflow progress, the participant flow, the study dossier and the imputation
bookkeeping read the catalogue.

To add a kind of step:

1. Register a `StepKind` in `core/steps.py`.
2. If the replay recomputes it, add its handler to `_HANDLED` in `manage/replay/engine.py`.
3. Never rename an id; register a kind that nothing records any more with `legacy=True`, so old traces still read.

`TransformationManager.add_step` logs a step recorded under an unregistered name. With `CMP_STRICT_STATE=1` (the test
suite) it raises `UnknownIdError` instead. `tests/core_layer/test_steps.py` checks that every step the application
writes is registered, that the report titles every kind, and that the replay handles every recomputed kind.

---

## RAG Quality Monitor

The `rag_monitoring.py` module renders the **Admin > AI quality** page. Scores come from an LLM acting as judge
(`manage/rag_evaluator.py`), on a 0-10 scale:

| Tab                    | Description                                                                                                                                                                                                                                                                                            |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Quality Overview**   | Average context relevance, faithfulness and answer relevance, JSON success rate, a 7-day trend, scores per function, and a warning when the hallucination rate exceeds 10%.                                                                                                                            |
| **Evaluation Details** | The recent evaluations with their scores, the judge's reasoning and detected hallucinations.                                                                                                                                                                                                           |
| **Embedding Health**   | The retrieval test suite (queries and expected concepts) and a form to add test cases. It shows no embedding statistics; what the index holds (chunks, documents, embedding model and dimensions) is shown in Settings > AI engine > _Vector database_.                                                |
| **Run Evaluation**     | Evaluate one query against a chosen function, or run the test suite (a test passes with an overall score of at least 5 and at least half of the expected concepts in the answer); export the results as JSON. Without an initialised AI engine, only a demo evaluation with fixed scores is available. |

This is primarily a developer and admin tool for diagnosing RAG performance issues.

---

## Configuration

### Study configuration (JSON)

The study configuration lives in the session (`st.session_state.config`). It starts from
`utils.config_loader.create_empty_config()`. A file uploaded on **Data > Validate** (sidebar, _Load Existing Config_)
adds the families and rules the session lacks, each whole: a rule the session holds keeps its definition, and the page
names those whose definition in the file differs (`core.population.merge_families`).

The criteria are `config["mask_families"]`. Each family maps rule names to definitions, and its `operator` says how it
combines them. Operators are read without regard to case, and a Validate tab gives one only to a family it creates.

| Operator           | The family is                                          | A row is excluded when        |
| ------------------ | ------------------------------------------------------ | ----------------------------- |
| `"any"` (or none)  | an exclusion family (`clinical_anomalies`, `outliers`) | a ticked rule fires           |
| `"all"`            | an inclusion family (`study_inclusion`)                | it fails a ticked criterion   |
| `{"threshold": k}` | one exclusion rule, "at least k of these rules"        | at least k of its rules fire  |

Do not add keys at family level other than `operator`. `description` in a rule is shown as the help text of its
criterion in the sidebar, and in its box on Validate.

A family's id is stored (configuration files, criterion ids such as `clinical_anomalies::Implausible age`, traces,
the population fingerprint) and never changes; the pages, the participant flow and the dossier name it in words,
`core.population.family_words`: `clinical_anomalies` is shown as *implausible values*, the family of the
plausibility rules, and any other family as its id with spaces.

#### Criterion kinds

Each kind of rule is one `CriterionKind` in the catalogue `core.criteria.CRITERION_KINDS` (a `core.registry.Catalogue`).
The five kinds of M5 are read from their keys, in this order:

| Kind         | Label            | Definition                                                                     | Fires when                                   |
| ------------ | ---------------- | ------------------------------------------------------------------------------ | -------------------------------------------- |
| `threshold`  | At least k rules | `{"threshold": k, "members": {name: definition}}`                              | at least k of its rules fire                 |
| `row_ids`    | Listed rows      | `{"row_ids": [[first, last]], "method", "params", "columns", "description"}`   | the row is listed (outliers frozen as rows)  |
| `bounds`     | Frozen bounds    | `{"bounds": {"column": [lower, upper]}}`                                       | a value of any column is outside its range   |
| `numeric`    | Numeric range    | `{"numeric": true, "lower_bound": a, "upper_bound": b, "strategy": "include"}` | inside the range, or outside with _exclude_  |
| `expression` | Expression       | `{"numeric": false, "expression": "df['age'] > 110"}`                          | the expression is true                       |

A numeric range reads the column named like the rule, or the one its `"column"` names.

An expression uses `df['column']` (bare names are rejected), `& | ~` with parenthesised comparisons, and the
allow-listed `pd.*` / `np.*` functions and Series methods of `core.expressions.safe_eval` (`isin`, `between`,
`.str.*`, `.dt.*`, `pd.to_datetime(..., dayfirst=True)`, ...). A missing value meets no condition of the other kinds.
In an expression, `.isna()` finds it. An expression is read in three-valued logic (Kleene's, as SQL's `WHERE`;
`core.expressions.kleene`): a comparison or a test of a value at the level of the expression's logic (the expression
itself, the operands of `& | ^` and of `np.logical_and/or/xor`, the operand of `~` and of `np.logical_not`) is unknown
where an operand is missing, `~unknown` is unknown, and a rule fires where its expression is True only. One reading
however the test is spelt: the methods `eq ne lt le gt ge` are the comparisons, a test compared with True or False is
the test or its negation, `between` is its two comparisons (unknown where the value or a bound is missing, unless the
other is False), and `isin` and the tests of `.str` are unknown where the value is missing. An expression that gives a
missing value a value itself decides it, whatever the column's dtype: a test's `fillna(True)` or `fillna(False)` fills
the test's unknowns, and an `isin` list holding a missing value (None, `np.nan`, `pd.NA`, `pd.NaT`) finds the missing
values, read as `isin(the others) | isna()` (pandas matches these markers by dtype). Inside any other call
(`np.where`, `.astype`, a sum), a comparison method with `fill_value=` and a text test with `na=` are read as Python
reads them. This reading is `RULES_VERSION` 2: a change to it is a new number, which changes the fingerprint of every
study with criteria.

`core.criteria.rule_kind` is the only place that decides a rule's kind:

1. the five kinds above, from their keys and in their order, whatever `"kind"` says: a definition that reads today
   reads the same once a kind it names is registered, so that its recorded population still replays;
2. otherwise a `"kind"` that names a kind registered without a recogniser (from M6: `date_order`, then M9's
   `concept`): that kind;
3. nothing read: refused, with the kind it states when it states one. An older version refuses a newer file, and says
   which kind it does not know.

A rule holding both a numeric range and an expression is refused rather than read one way or the other. `"kind"`
inside a definition is the catalogue's key; a criterion's `"kind"` in a population block of a trace is its role,
exclusion or inclusion.

**The interface.** Pages and services never call a kind's functions. They call the free functions of `core.criteria`,
which read the kind:

- `criterion_mask(df, name, definition)`: the rows where it fires. It raises `CriterionError` and nothing else.
- `criterion_columns(name, definition)`: the columns it reads, those of the rules it combines included; `()` when they
  cannot be read.
- `rule_checks(df, name, definition)`: its Domain Validity checks, each a `RuleCheck(rule, column, checked, fired)`.
  A check is one rule on one row holding the values it reads; a row where it fires is always checked. By default, the
  rows holding every column the kind reads and, for a kind that combines rules, where each of them is settled
  (`settled`): a missing value is never counted as a pass. It raises `CriterionError` and nothing else.
- `describe_definition(name, definition)`: what makes it fire, in the words of the participant flow, the dossier and
  Validate's boxes ("18 <= age <= 85", "one of the 3 listed rows (flagged by |z| > 3 on sbp)"). A rule that the
  checks refuse reads "cannot be evaluated" and why. It never raises.
- `criterion_problems(name, definition)`: what makes it unusable on any table. `core.population.configuration_problems`
  refuses an uploaded file holding such a rule, an unknown operator, a threshold that is not a whole number, or a
  family entry that is not a rule. What evaluates (a bound written `"18"`, a threshold of 2.5) still loads.
- `holds_row_list` and `shared_definition`: the copy for sharing (`core.population.configuration_for_sharing`, the
  **Download for Sharing** button) keeps a rule only when it lists no row of the dataset, can be read and passes the
  checks, rebuilt from its kind's keys with values of the types declared. It names what it leaves out.
- The writers `numeric_rule`, `expression_rule`, `listed_rows_rule` and `threshold_rule`: every rule the application
  writes comes from one of them (a gate in `tests/test_quality_gates.py`).

A malformed rule never stops a page: the sidebar counts every criterion before each page. Listed rows are found on
the table's sorted row ids, whatever the length of the ranges they list.

**A stored definition is never rewritten.** A criterion's fingerprint (`core.population.Criterion.fingerprint`)
hashes its definition: it keys the population memos, and every population block of a trace holds it, which the trace
chain hashes. No reader adds, removes or reorders a key. So the five kinds keep their keys, and only the kinds
registered from M6 on carry `"kind"`. A kind's id is stored with every rule of its kind: what a kind selects never
changes under the same id. A changed selection is a new kind, and the old id stays registered, marked legacy.

#### Adding a criterion kind

M6's `date_order` and M9's `concept` are one registration each.

1. Write its functions in a module of its own (`core/temporal_checks.py`), which must not import `core.criteria` (rule
   R2): `mask(df, name, definition)`, `columns(name, definition)`, `describe(name, definition)` and
   `problems(name, definition)`. Raise `ValueError` for what cannot be evaluated: `criterion_mask` names the criterion.
   When the rows it checks are not the rows holding every column it reads, write `plausibility_checks(df, name,
   definition, members)` too: per check, `(column or "", the rows holding the values it reads, the rows where it
   fires)`, plain boolean Series, with `members` the (fires, settled) of the rules it combines. A numeric range reads
   only the numbers, bounds make one check per column, listed rows count the rows their method assessed.
2. Register it in `core/criteria.py`, after the five, without a recogniser:
   `CRITERION_KINDS.register(CriterionKind("date_order", "Date order", mask=..., columns=..., describe=...,
   problems=..., keys={"first": is_text, "then": is_text, "tolerance": is_scalar, "description": is_text}))`. A kind
   that combines rules names the key it keeps them under (`members_key`): the free functions add their checks and
   columns. Say how it treats a missing value (`missing`), and give `plausibility_checks=` when the default is not
   its checks. `tests/core_layer/test_criterion_kinds.py` blanks each example's columns: no blank row may count as a
   pass.
3. Add a writer that states the kind (`{"kind": "date_order", ...}`), used by every page that writes the rule.
4. Give it examples, the rows each selects (`ROWS`), a read-back of its words and malformed examples in
   `tests/core_layer/test_criterion_kinds.py`, which fails until they are there. Its definitions hold none of the keys
   of the five. A test that registers a kind of its own does so in `CRITERION_KINDS.copy()`, patched over the
   catalogue for that test.

#### The population

Ticked criteria define the analysis population (`core/population.py`). A ticked criterion that cannot be evaluated
(unknown column, syntax error) stops the analyses: the population is not `ok`, and the pages show the error until the
criterion is unticked or corrected. The **Run the Plausibility Check** and **Run Inclusion Criteria Check** buttons of the
Validate tabs and the quality dashboard evaluate the rules with the same functions (`core.population.family_masks`),
so they count what the population counts. A rule that cannot be evaluated is shown with its reason, and the family
result is left out until it is corrected.

`assets/tutorial/tutorial_config.json` is a complete, tested example (8 plausibility rules, 4 inclusion criteria).
`ui.state.tutorial_config()` merges it into the default configuration (`folder_names` stay machine-specific).

#### The study's privacy choices

A study definition may hold a section `privacy` (`core/privacy.py`): what the study asks for the documents made from
it. No screen sets it in this version. It is a file format with one writer and a closed reader, written now so that
the Data privacy settings to come change no study file.

```json
"privacy": {"k": 20,
            "followed_to_the_event": {"death": {}, "visits": {"beyond_the_cap": true}},
            "rows_not_people": "animal_data",
            "rows_word": "animals"}
```

- **What it holds:**
  - `k`: the smallest count the study's documents give, a whole number never below 5 (`FLOOR_K`);
  - `followed_to_the_event`: the outcomes declared followed to the event, each with `beyond_the_cap` only when the
    override of the cap was asked for;
  - `rows_not_people`: the reason of the declaration that the study's rows are not people, an id of
    `NOT_PEOPLE_REASONS` (animal data, simulated data, the tutorial cohort, a teaching set);
  - `rows_word`: the word the documents use for the rows, `animals` or `subjects`.

  Nothing in it is typed by a person: no reason in words, no date, no account. Its keys and its closed values are
  stored ids, never renamed.
- **One writer.** `StudyPrivacy.to_section()` writes the keys in one order and writes no default (no `k` of 10, no
  `participants`, no `false`), so that equal choices have one checksum. Never write the key by hand.
- **A closed reader.** `StudyPrivacy.from_section` gives the choices, or why the section is refused: a key or a
  closed value it does not know reads "written by a later version"; a value of the wrong kind, a `k` below the
  floor, or a section the writer would write otherwise (a default written out, its keys or its outcomes in another
  order) is refused too. Nothing is ignored (a `k` of 20 written as text and skipped would apply 10), and a reason
  quotes nothing of the file. `StudyProtocol.from_file` then refuses the whole file, and a `StudyProtocol` is never
  built with a section its own reader would refuse.
- **In the session, the choices are a study's.** `AppState.set_privacy_choices` keeps them under a session key of
  their own, with the id of the study they are of; `privacy_choices()` and `protocol_body()` read them for that
  study only, and loading a dataset of another study drops them. They are never in `st.session_state.config`,
  which stays for the next dataset loaded and is written to files: a declaration that one study's rows are animals
  must not be saved with the next study's first revision. Loading a revision makes its choices the session's.
- **Outside the fingerprint, inside the checksum.** The choices select no row and compute no number: they are left
  out of `protocol_hash`, as the labels are, so no result goes stale for one. They are in the body's checksum: a
  change of them alone is the next revision, and the study's strip says it apart (`ui/components/study.py`, the
  state `privacy`).
- **It never travels.** `protocol_for_sharing` and `core.population.configuration_for_sharing` leave the section out
  (`SharedConfiguration.privacy` says that it was there), and so does the whole configuration's file
  (`app_pages.data_monitoring.whole_configuration`). An uploaded configuration's `privacy` key is dropped, with a
  notice, and the session's own choices stay as they were: whoever opens a study chooses for their own data.
- **What applies is decided where an output is made,** never in the section, which holds what was asked:
  `core.disclosure.policy_of(choices, floor, floor_set_by=...)` gives the policy of a study's choices on an
  installation (the larger of the `k` asked, or the default, and the floor; who set that floor, as a level; the
  declarations), and is the one function a package and the dossier's sharing mode ask, so that the two cannot
  disagree. `rows_not_people` and `rows_word` are read and kept, and applied by nothing in this version: the rules
  stay on and the documents say "participants".

#### The study package's formats

`core/package.py` holds what a study package (`.cstudio.zip`) is made of, and `manage/package_zip.py` puts its
members in a zip and takes them out of a received one (below): nothing builds a package from a study yet.
`core/package.py` imports the standard library and `core.privacy` only, so that a package can be read without the
engine, numpy or pandas (a test loads it in a fresh interpreter).

- **The members**, in the zip's order (`MEMBERS`): `package.json`, `protocol.json`, `results.json`, `README.md`,
  `uv.lock` when the installation has one, `MANIFEST.sha256`. Each tier has its own file name (`package_name`).
- **One writer.** `dumps(member, payload)` is the only writer of a JSON member: an envelope `format` + `schema`,
  keys sorted, two spaces, UTF-8 without a byte order mark, line feeds. The body of `protocol.json` keeps the
  order it is given, since the order of the families and of their rules is part of the study. Two builds of one
  study are the same bytes.
- **A strict reader.** `loads(data, cap)` refuses what the writer never writes (a byte order mark, a key written
  twice, NaN or an infinite number, a nest of more than `DEPTH` levels, a member over its size), and
  `read_envelope` another member's format or a later schema. A reason quotes nothing of a file: it counts.
- **The text rule.** `core.privacy.hidden(text)` counts the characters that no name of a study and no text of a
  package may hold: control characters other than a line feed, format characters (the marks that change the
  direction of a text), line and paragraph separators, a lone surrogate, and the characters that show nothing
  (`INVISIBLE`). One rule for the study's section and for the written members of a package: a study never
  declares a name that its package would refuse. The writer refuses to write such a character in a written member
  and the reader refuses one that holds it (the carried members of a package that holds the data are the
  exception, below). `shown(text)` is for what does not come from a member, and for every key and text of a
  carried member. A source file never holds such a character itself: a test that needs one builds it with
  `chr()`.
- **The manifest and the hash.** `manifest_of` lists the SHA-256 of every other member; **the package's hash is
  the SHA-256 of the manifest's bytes** (`package_hash`).
- **The policy block.** `Policy` is what `package.json` records of the privacy rules a package was made under:
  one closed shape with every key always written, for `rules` `on` (what this version writes), `off` and `none`.
  It records what applied, and of the study's k what was asked: `k_asked` is null when the study asked none, with
  `k_default` beside it; the cap of "followed to the event" once, never above the code's (it is lowered, never
  raised). `DisclosurePolicy.block()` writes it, from what the policy itself holds; nothing else does. An override
  names its place by the format's own names and by positions (`is_place`), never by a text.

**The package that holds the data** (the tier `reproduction`, `<study>_r<rev>_with_data.cstudio.zip`) has its own
members, and the functions above take the tier: `members_of(tier)`, `clear_members`, `formats_of`, `cap_of`,
`dumps(member, payload, tier)`, `read_envelope(value, member, tier)`, `manifest_of(members, tier)` (and
`manifest_from_checksums` for a writer that streams), `read_manifest(data, tier)`, `in_zip_order(names, tier)`.
Six of them read `shareable` when no tier is given: pass it for every member of this tier. Read a member with
`read_member(member, data, tier)`, which knows its writing, rather than with `loads` and `read_envelope` by hand.

- **Its members**, in the zip's order: `NOTICE.txt` (clear), then, encrypted, `package.json`, `protocol.json` (the
  saved revision whole, its notes and privacy section included, so that its checksum can be checked; nothing
  applies that section), `results.json`, `trace.json`, `profiles.json`, `README.md`, `uv.lock`, the data, and
  `MANIFEST.sha256` (clear, last). The data members are named by the format, never by a file: `source.<ext>` or
  `version.parquet`, then `merge_NN.parquet` (one per merge step, the table as its merge read it, cut to the rows
  it took) and `snapshot_NN.parquet` (`data_role`, `data_members_problem`). A workbook that can hold code (`.xls`,
  `.xlsm`) never travels as its bytes: its table does, as `source.parquet`. A recorded table stands only for a step
  of `SNAPSHOT_STEPS`; a trace that holds a `multiple_imputation` step (an imputed dataset adopted as the data)
  does not travel. The package's hash is still the SHA-256 of the manifest, which lists the checksums of the
  members before encryption. A member and the whole package stay below the bound at which a zip writer adds a
  ZIP64 field (`zipfile.ZIP64_LIMIT`, for a member's size and for where its header starts).
- **Two writings.** `package.json` and `README.md` are *written*, under the text rule. The members that hold the
  study's own texts are *carried* (`is_carried`): the same writer in ASCII, every character outside ASCII as an
  escape, and no text rule (a sealed trace can hold a tab or a direction mark in a recoded value; under the rule
  such a study could never travel). `loads(data, cap, carried=True)` reads one: it refuses a byte that is not ASCII,
  a lone surrogate (which no sealed trace can hold), and all that the strict reader refuses but the text rule, so
  a carried key or text may hold any other character. Show one only through `shown()`, on a terminal with
  `terminal=True` (ASCII within the limit: `?` for a hidden character, the stream's escape for another character
  outside ASCII, a backslash doubled, `...` for a cut): as it is, a direction mark or a terminal's escape sequence
  would reach the screen.
- **The notice** is one of two fixed texts (`NOTICES`: personal data; rows declared not people, which M8b writes),
  read by its exact bytes (`read_notice`): a clear member is never a place for anyone's words. Its first line
  names the schema, so that a later one reads as a later version's, not as a changed file. After the password,
  `notice_problem` checks it against the policy block. **Its words belong to its schema**: a reader takes any other
  bytes for a changed file, so never edit a notice in place; a new wording comes with a new schema, and the readers
  keep the earlier notices' bytes. A test pins each notice's SHA-256.
- **The index** (`DataPackageIndex`, `package.json` in this tier) holds no text the user or the data gave but the
  study's id and the declaration's two (the recipient and the reference, typed for the export, one line each,
  composed): the study's id, revision, checksum and fingerprint; the software (ASCII versions, the platform's
  words for the system and the machine, never a host's name; with the process's seed); the trace's schema, what it starts from and every step's kind and hash; the data members by role, step,
  rows and columns, a merged file with its cut (`rows_taken` or `every_row`, the rows of the original and its
  recorded checksum); the policy (`rules` `none`); the declaration (`BASES`); the anchor. **The anchor**
  (`anchor_of`) is the SHA-256 of a fixed prefix and of the bytes of the member the trace starts from: the clear
  manifest gives each member's SHA-256, and a member whose every value could be guessed would be given away by
  trying them; nobody computes the anchor without the data. The roles of a file (`FileRole`, `ROLE_WORDS`) live
  here too: the replay engine reads them from this module.
- **One format name per shape**, and one table of schemas (`FORMAT_SCHEMAS`): the index, the revision and the
  results are not the same in the two tiers and have a name in each, so that one tier's schema moves alone and a
  member taken out of a package of one tier never reads as the other's.

A key or a closed value that a reader does not know reads as written by a later version, and every id here (the
tiers, the members, the values of `rules`, the levels of `*_set_by`, the roles of a file, the rules of a cut, the
bases, the notices) is a stored id: never renamed.

**The zip** (`manage/package_zip.py`, a headless module: rule R5) has one writer and one reader for both tiers.

- **`write(out, members, tier, password=None, expected=None)`** takes every member but the manifest, each as its
  bytes or as a file it streams, and returns the package's hash. It makes the manifest from the checksums it takes
  while it streams, puts the members in `in_zip_order`, one after the other, with fixed fields (the date
  1980-01-01, the system, version 20, the mode of the tier; no comment), and reads the finished zip back with the
  reader's own clear part: a zip its reader would refuse is a `RuntimeError`, never a package. A package without
  the data is stored and clear, so one study gives the same bytes twice (a test pins their SHA-256). One that holds
  the data is encrypted member by member with pyzipper (WinZip AES-256, AE-2, deflate inside; a test decrypts a
  member by the specification, without pyzipper), its notice and its manifest clear and stored, its members of
  mode 0600: a zip program that restores a Unix mode (Info-ZIP's unzip, the usual tools of Linux and macOS) writes
  them readable by their owner only, while on Windows, or through Python's `zipfile`, they take the rights of the
  folder they go into. Two exports are other bytes (a salt per member) and one package hash. Before writing a byte
  it refuses a name of another tier, a member over its cap, a password missing or not needed, an `out` it could not
  read back, an index its reader would refuse or that says another tier, and, with the data, a notice that is not
  one of the fixed texts or does not say what the index's policy says, or an index that does not name the data
  members given (no other bytes ever go in clear). While writing, it checks the anchor against the member the trace
  starts from, and each member against the checksum `expected` gives (the pre-flight's). After an exception, `out`
  holds a zip to discard.
- **The password** is made by `new_password()` (`secrets`, 20 symbols of Crockford's base32: 100 bits, shown in
  five groups of four), and the string shown, hyphens included, is the password. `password_of(typed)` reads any
  spelling a person types (case, spaces, O for 0, I or L for 1). It is never stored, logged or put in a reason (a
  test plants one).
- **`open_package(file, max_size=MOST_FILE_BYTES)`** reads the clear part of a received file and returns `Opened`
  (the tier, the members, the checksums the manifest lists, the package's hash, the notice) or `Refused`, a closed
  id that the caller puts in its words. Nothing is decrypted, pyzipper is not imported (a test checks it in a fresh
  interpreter), and no zip library reads the file: the module reads the end record itself, bounds the directory
  before a byte of it is read (a crafted directory would otherwise take about nine times its size in memory), and
  walks each record. The zip must be laid out as the writer lays one out: the members one after the other from the
  first byte, each local header saying what the directory says, the records filling the directory exactly, the end
  record last. No byte lies outside the members (no ZIP64 record, no comment), and a tool that reads the local
  headers in turn sees the members the reader checked.
- **The container is frozen** across the format's schemas (`core/package.py` says so beside `FORMAT_SCHEMAS`):
  that layout, at most `MOST_ENTRIES` (1,024) entries named in ASCII by at most `NAME_MOST` (64) characters, no
  folder, no name twice, and the member that tells a later version (`NOTICE.txt` with the data, `package.json`
  without) stored and clear, within this version's cap of it, `package.json` as JSON its strict reader reads. The
  reader checks these before any schema; a later schema changes the members and what they hold, never these.
- **`read_package(file, opened, password, into=None)`** returns a copy of `opened` with the JSON members read by
  `read_member`, the index checked against the zip (its data members, its notice, its anchor), and the data
  written into `into`. `opened` is left as it was, so a password can be typed again. The file must still be the
  one opened, and pyzipper's reading of its directory the one checked. A wrong password is two members failing the
  password's check before any member decrypted; no password, or one of another shape, is `not_a_password`, and
  nothing is tried. A member whose bytes are not those its manifest lists does not stop the reading: it is named in
  `not_matching`, and only what altered bytes raise counts so: an `OSError` (a full disk, a file that cannot be
  read) reaches the caller. `into` is an empty folder the caller made, on a system with modes its user's only
  (`tempfile.mkdtemp`); on Windows the files take the folder's access list. The data are written there once the
  index is read, each file only if it does not exist, under the format's own name, and every one is deleted when
  its bytes are not the sender's, when the reading is refused or when anything stops it. A file the system will not
  delete is said, by its count and never its path: an `OSError` in place of the refusal, or a note on the error
  that stopped the reading. Exit codes: 0 read as sent, 1 read but not what the sender made
  (`not_matching`), 2 not read (`Refused`); a caller that parses arguments gives a usage error a code of its own.

**The command line** (`python -m manage.replay`, `manage/replay/__main__.py`) has one command so far, `inspect
PACKAGE`. `__main__` names each command's arguments and its module in `_COMMANDS`, and imports the module only when
the command runs: reading a package's hash never loads what a replay needs. What every command shares is
`manage/replay/commandline.py` (the exit codes, the parser, the password, the lines of 78 columns); `inspect` itself
is `manage/replay/inspection.py` (not named `inspect`, which a file run from that folder would import in place of
the standard library's). It says what a received package holds without replaying it:

- **The clear part first:** the file, the kind of package, its hash, its files with their sizes, from
  `open_package`. `--package-hash` (16 to 64 hexadecimal characters) is compared here: a hash that differs is
  written under the package's own with the first character at which they part, and ends the run (exit 1) before
  any password is read, asked for or tried.
- **Nothing typed on the command line is written back.** The password is never an argument: `--password-file FILE`
  (its first line, read as the file's byte order mark says, after the clear part) or `--ask-password` (each key
  shown as `*`, three tries; exit 64 where nobody can type one unseen: on Windows the null device says it is a
  terminal, so the console itself is asked). Without either it stops after the clear part and says how to go on:
  it never waits for an input. The parsers take no abbreviation (`--password X` would read as `--password-file`),
  and of argparse's messages only the forms that name options, never values, are written (`usage_words`): "invalid
  choice: '...'" would repeat a password typed in the command's place. A file that is not found is not named.
  A misspelt command or option is not repeated either, so each such error shows the way back: what a command does
  not take is refused by that command's own parser (`Parser.parse_known_args`), so that the usage above the error
  is the command's, and the error ends with the command that lists the options, on a line of its own (`way_back`).
- **With the password:** `read_package`, then what the index says (the software, the study, the declaration, the
  data by role, the trace, the policy). Nothing of a carried member, of the README or of the data is written out.
  Every text that is not the module's own goes through `core.package.shown(..., terminal=True)`: the file's name
  and the two texts the exporter typed, as themselves where the stream writes UTF-8 (`utf8=True`), as escapes
  elsewhere, cut between two characters when wider than a line (`wrapped(..., breakable=True)`).
- **Its output** is lines of 78 columns at most (ASCII, the UTF-8 case aside), the report on stdout, the refusals
  and the prompts on stderr; `--json` gives the same facts as one object. `refusal_words` holds the words of each
  refusal id (they quote nothing of the file), and a test keeps the word tables equal to the format's closed
  values.
- **Exit codes:** 0, 1 and 2 as the reader's; 64 for a command that was not understood or cannot run here
  (argparse's own 2 would read "not read"); 70 for an error of the program, named by its kind, with an object on
  that path too under `--json`.
- **What it loads:** the standard library, `core.package`, `manage.package_zip`, `manage.replay.commandline` and
  `utils.environment`; pyzipper once a password is given; never the engine, numpy, pandas, pydantic or Streamlit,
  no `.env` and no setting (a test compares the application's modules a run has loaded with a list). It therefore
  does not call `configure_logging()`, which reads the settings: it writes no log, and `__main__` turns the
  process's logging off when it runs as that command (a command that logs, as `verify` will, configures its log
  instead).

### Settings, preferences and API keys

**One settings object.** `utils/settings.py` holds every setting the application reads, one field per environment
variable (the names of `.env.example`, which a test keeps equal to the fields), in sections: `paths`, `logging`,
`iam`, `ai`, `vector_store`, `graph`, `statistics` and `development`. Keys and passwords are `SecretStr`. Read them
with `get_settings()`, which validates again only when a variable has changed; it is the only module that reads the
environment (`tests/test_layering.py` checks every module, with a short list of exceptions that can only shrink).
`core` and `biostats` never read settings: they receive values as parameters.

**When `.env` is read.** `main.py` and `manage_users.py` call `load_settings()` first: it loads the `.env` next to
them (once per process; a variable already set in the environment wins, which is how Docker passes them), then
validates every setting. A wrong one stops start-up with the variable's name and what is expected, never its value.
`main.py` then checks the connector settings (`manage.connectors.check_connector_settings`, below). The tests never
read `.env`: `tests/conftest.py` turns `load_dotenv` into a no-op, and `tests/test_startup_config.py`
checks the loading in a subprocess. `st.secrets` is not read; a `.streamlit/secrets.toml` is logged at start-up.

**API keys.** Cloud providers read the server's key from the settings: `manage.rag.providers.server_key(provider)`
reads `get_settings().ai.key(variable)` for the variable the provider declares (`GOOGLE_API_KEY`, `OPENAI_API_KEY`,
`MISTRAL_API_KEY`, `ANTHROPIC_API_KEY`). A key typed in Settings > AI engine is kept for the session only.

**Preferences.** The user's choices in Settings (the AI provider and models, the vector database, the Ollama address
and the Qdrant server, the graph engine, the day-first default) are kept for the next session in
`<workspace>/preferences.json` (`manage/preferences.py`, written whole and atomically). `ui/preferences.py` seeds the
session after sign-in, once per user, only for the keys the session has not set, and the controls save them from their
`on_change` callbacks. A preference never holds a key, a password or a consent; it holds the Ollama address and the
Qdrant server only as chosen among those the server offers (`manage.server_addresses`, never typed: `_PREFERENCES` in
`app_pages/rag_sidebar.py`); a choice the server no longer offers is not used, and the page names it. The file is read
only if it is small (64 KiB) and its names short; the lock serialises the writes of one server process (two servers on
one `DATA_ROOT`: the last write wins).

**Which connector is used.** The AI provider, the vector database and the graph engine are chosen with one control,
`ui.components.connector_selector`, on the kit (`core/registry.py` `ConnectorRegistry.select`): the session's choice
(for the AI provider, first the engine that runs), then the saved preference, then the server's setting
(`CMP_AI_PROVIDER`, `CMP_VECTOR_STORE`, `CMP_GRAPH_RENDERER`), then `auto` (Ollama at `OLLAMA_BASE_URL`; the
recommendation for this machine; Plotly, then Graphviz). It offers only what can run here, and a name that cannot run
here falls through. On Settings it says why the connector used was chosen, which saved choice or setting was set
aside, and what cannot run here with the group to install (`explain_choice`). A cloud provider preselected so
receives none of the user's data until the session confirms it, and an engine started without a provider uses the
server's (`manage.rag.providers.default_provider`).

**The Ollama address and the Qdrant server are chosen from the server's lists.** `manage.server_addresses` holds one
`AddressList` per kind of server. `QDRANT_ADDRESSES` offers `CMP_QDRANT_URL` (read as the client reads it: https when
written without a scheme and the server's key is set; `utils.settings.qdrant_address`), then those added, after the
embedded mode; `key_for` gives the server's key to that address only, however it is written, and the connection test
names a failure by its kind and status, never by the reply. `OLLAMA_ADDRESSES` offers the deployment's addresses
(`OLLAMA_BASE_URL` first, then `CMP_OLLAMA_ADDRESSES`), then those an administrator added
(`<DATA_ROOT>/server/ollama_addresses.json`, and `qdrant_addresses.json` beside it: two stores of account names, which
outlive a deleted account; written atomically under a lock, each with the account that added it and when, and an
optional name (free text every user sees: at most 40 characters, no "@" nor "://", no two alike; changed with
`AddressList.rename`, which never changes a page's choice; gone with its address); the account that adds, renames or
removes one is also logged, never a name, since the file forgets an address removed). A user never types an address,
so that the server sends no request to an address someone wrote (CodeQL `py/full-ssrf`): the Settings list cannot be
typed in, an administrator's typed address only goes into that file (Add's callback), and every request uses an
address read back from it or from the settings. The value of the list is checked against it again (a forged one is
replaced by the first address), the engine refuses to start on an address not offered
(`RAGManager.initialize_system`), and an engine whose address was removed stops at the next run of its session
(`ui.services.stop_engine_off_list`, from `main.py`). `utils.settings.server_address` is the one rule for an address
(ASCII, http or https, a host, no user name, password, query or fragment; its messages never quote the value), applied
at start-up to the three variables and to each address added. A file that cannot be read offers the deployment's
addresses only, says so to administrators, and is not overwritten by an Add.

**An unknown connector name stops start-up.** `CMP_IAM_PROVIDER` accepts `local`, `better_auth` (or `betterauth`,
`better-auth`, `remote`) and `disabled` (or `none`, `off`, `bypass`); `CMP_AI_PROVIDER`, `CMP_VECTOR_STORE` and
`CMP_GRAPH_RENDERER` accept `auto` and their registered names; all in any case, an empty value meaning the default.
`main.py` checks the four right after loading the settings (`manage.connectors.check_connector_settings`, which asks
each registry for its names only): any other value stops start-up with the variable and the valid names, never the
value, instead of an identity provider or an engine other than the one meant. `manage_users.py` does not import that
module (its lazy names lead to the AI packages: rule R5); there the identity manager refuses an unknown provider
(`IAMConfigurationError`, printed as one line).
The identity providers are a connector family chosen by this setting only (`manage.iam.providers.IAM_PROVIDERS`, no
automatic choice, no user's choice): each declares its `name` and `display_name` and builds itself from the
configuration (`from_config`), and is loaded when used. A custom provider is installed with `IAMManager.set_provider`.

---

## Contributing

1. Follow the existing Mixin pattern when adding new RAG capabilities.
2. Validate all LLM responses with Pydantic schemas.
3. Change the data only through `AppState` (`ui/state.py`), which writes through `DataSession` (`core/session.py`):
   `commit_change()` for a change of the dataset, `stage()` for the Enrich draft (traced when applied), `note()` for a
   trace step that does not change the data. Each change becomes one trace step; with `CMP_STRICT_STATE=1` (the test
   suite), an in-place change of the session data raises `UntracedMutationError`.
4. A tab reads its own inputs: it never uses a value that another tab of the page computed in the same run. A new
   session key that one tab or page writes and another reads is declared in `ui/keys.py`, with its owner, its
   lifetime and its readers; a test checks the declarations against the code and against what `AppState` clears. On
   a page whose tabs run only when open, every control has a key and `persist_state="page"`, except a confirmation
   and an input whose `value=` comes from a session key that already keeps it (a kept value would come back stale).
5. Put new code in the layer it belongs to (*Architecture*, above), never in a page: a computation in `biostats`
   or `core`, a service in `manage`, a component in `ui`. `utils` holds platform code (settings, logging, paths,
   optional extras); its older modules are being moved out, so do not add to them.
6. Say things through the log, never `print` (ruff's `T20`), and keep values and paths out of it: a log line holds
   counts, column names and settings, never a cell, a file's path or what a user typed. Of a file, at most the name
   of one the application named (a saved version's), never an uploaded file's name, which is the user's.
   - **A caught exception** is named by its kind where the failure comes from the data, whose text can quote a
     cell: `type(exc).__name__`, or `utils.logging_config.exception_kind(exc)`, which adds the system's own words
     for a system error. A service's error (the identity provider, an AI provider, a store) goes through
     `exception_said(exc)`: the kind and the text without quoted strings and paths. When unsure whose error it is,
     the kind alone: pyarrow's write error looked like a store's and quoted the cell.
   - **Never** the exception itself, its `str` or an f-string of it, nor a message built from it beforehand or the
     exception kept under another name (`err = f"... {exc}"`, then `logger.error(err)`).
   - **A quality gate** (`tests/test_quality_gates.py`) reads the logging calls for it: a logger held in a name, in
     an attribute or obtained by `getLogger`, and the names a function gives a caught exception or its text. It does
     not follow a text through a call or an attribute.
   - **The net under it** is `records_without_values`: an exception that reaches a log record as itself, and each
     exception of a traceback (`exc_info`, `logger.exception`), is written by its kind alone, the traceback's frames
     kept, whichever logger makes the record (Streamlit's own included). It cannot undo a text already put into a
     message. Each entry point calls `configure_logging()` first, which installs it: `main.py`, every command of
     `manage_users.py`, and a new command line or script of yours. The one exception is `python -m manage.replay
     inspect`, which loads no setting and so configures no log: it writes none.
7. Test with the scripts in `tests/` (e.g. `test_rag_full_integration.py`, `test_trace_documenter.py`).
