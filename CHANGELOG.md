# Changelog

All notable changes to CohortStudio (named Cohort Monitoring Package until 2.3) are documented here.

---

## [Unreleased]

### Statistics: results corrected (follow-up programme M5)

- **Rate ratio** (Analyze > Hypothesis tests > Paired, rates & stratified).
  - The 95% interval is now the exact conditional (Clopper-Pearson) interval, the one of R's `poisson.test`.
  - The p-value is the matching exact conditional one (twice the smaller tail), so the interval excludes 1 exactly
    when p < 0.05.
  - Before, a score interval was paired with the minimum-likelihood exact test, two methods that can disagree. With
    no events in one group, the score interval collapsed to a single value.
  - Example: 15 events in 1,200 person-years against 4 in 1,000 give a rate ratio of 3.1, now with 95% CI 0.996 to
    12.9 and p = 0.051; before, 1.09 to 8.96 and p = 0.038.
  - The score interval remains available (`ci_method="score"`), paired with the score test.
- **Mantel-Haenszel.** The note no longer calls the stratum variable "a confounder" whenever the crude and adjusted
  odds ratios differ by more than 10%: odds ratios are non-collapsible, so they can differ without any confounding.
  - The crude risk ratio is now reported next to the crude odds ratio, and the note reads the risk ratios first.
  - When the risk ratios differ too, the stratum variable "may be a confounder".
  - When only the odds ratios differ, the note says that this need not mean confounding.
- **Survival: the event indicator.**
  - **Codes:** only 0 and 1 (or words such as Yes/No and Dead/Alive) are accepted.
  - **A code of 2:** before, it passed through, and the Kaplan-Meier estimate counted such an event twice. A 1/2
    coding (R's survival package) is now refused with how to recode it, and a 0/1/2 coding is refused as a
    competing event.
  - **Unknown words:** before, a word that was neither an event nor a censoring word (a misspelling, "relapse")
    silently became censored. It is now refused too.
  - "Death", "deceased" and "survived" are now recognised.
- **Multiple imputation: zero cells in a 2x2 table.**
  - **Before:** the +0.5 (Haldane-Anscombe) correction was decided table by table. When a cell was zero in some
    imputed datasets only, corrected and uncorrected odds and risk ratios were pooled together.
  - **Now:** the ratios are then not pooled. The risk difference still is, and a warning says why.
  - **Sensitivity analysis:** a new option adds 0.5 to every cell of every table, so that the ratios can be pooled.
    The page and the dossier label that run as a sensitivity analysis.
- **Multiple imputation: three groups or more are compared without assuming equal variances.**
  - **Before:** on the observed data, three groups or more are compared with Welch's ANOVA, which gives each group
    its own variance. Pooled over the imputed datasets, they were compared with a linear model that assumes equal
    variances (a one-way ANOVA), because Welch's ANOVA has no pooling rule.
  - **Now:** the pooled comparison compares the group means as Welch's ANOVA does. Each group keeps its own variance
    (s²/n, the HC2 variance of the linear model on the groups), and each pooled quantity has Welch-type degrees of
    freedom: n − 1 for a group mean, Satterthwaite's for a difference, Welch's (1951) for the D1 test of the group
    effect, which also takes Welch's second-order correction. When the imputations agree, the pooled statistic is
    Welch's F. For two groups it is Welch's t-test, as before. Below the pooled result, the complete cases show their
    group means and Welch's ANOVA.
  - **When the variances differ, the results change.** Simulations where the means were equal and the smallest
    groups varied most, rejections at the 5% level. Complete data: 2,000 runs, the F test against Welch's ANOVA.
    Imputed: 30% of the values missing, 5 imputations within each group, 1,000 runs, both tests by D1.

    | Group sizes (standard deviations) | Complete, equal variances | Complete, now | Imputed, equal variances | Imputed, now |
    | --- | --- | --- | --- | --- |
    | 10, 20, 40 (4, 2, 1) | 22% | 5.0% | 12.5% | 4.4% |
    | 5, 20, 40 (4, 2, 1) | 34% | 6.2% | 13.7% | 4.0% |
    | 8, 12, 20, 30, 40 (4, 3, 2, 1.5, 1) | 26% | 5.3% | 12.8% | 5.3% |

    The biostatistician's review of a first version, with HC3 variances and the residual df, found it rejected too
    often on complete data in the same settings (6.2% and 8.4%), and much more with 2 to 4 values per group.
  - **Small groups:** a group with a single value has no variance of its own: the page asks to deselect or merge
    it, or to choose the equal-variance model. Below 5 values, a warning says that the test can reject too often.
  - **Imputation:** the pooled test only sees the differences in variance that the imputations keep, so the page
    advises imputing within each group when the variances may differ.
  - **Choose the method** keeps the equal-variance model, with a warning; the summary and the dossier mark it as
    chosen.
  - The R reference script checks the variances (`sandwich::vcovHC`), Welch's ANOVA (`oneway.test`), the pooled
    means and differences (`mice::pool.scalar`) and the test (`mitml`); the check runs once the reference values
    are regenerated.
- **D1 test with no variance between the imputed datasets.** When the imputations of the variables of a model did
  not differ (a fully observed variable), the D1 test used F(k, ∞) instead of the model's complete-data df
  (mitml's rule). Whether that happened depended on rounding. Such tests now use the complete-data df, so their
  p-values are slightly larger.
- **D1 test with a small complete-data df.** Reiter's small-sample df, which the D1 test uses when the model has a
  complete-data df, needs that df to be large enough. With Welch's df of a group of three values it gave a
  negative df (no p-value) or one larger than the data allow (too many rejections). A Barnard-Rubin-type df is then
  used instead, and the test says so. With too few imputations for Reiter's df (two or three), the large-sample df
  ignored the complete-data df and could reach hundreds where the data allow four: the same combination now applies.
  mitml uses Reiter's formula throughout.
- **Transformations (Enrich > Create new variables) replay to the same table.** The page computed the new
  columns with its own copy of the transformation engine; the replay of the trace used the shared one.
  - Applying a transformation twice gave `age_Z-Score_Standardization_1` on the page, but a second column with
    the same name on replay.
  - PCA was not seeded, so every replayed PCA step differed slightly and was flagged.
  - The page now calls the shared engine. The renaming and the refusal of missing values moved into it, and PCA
    is seeded like the other methods.
  - Older traces now replay with the names the page gave.
- **Criteria: one reading of the rules on every screen.** The Validate checks and the quality dashboard read the
  rules with older helpers, the population with `core.criteria`. They could disagree:
  - a rule holding both a range and an expression counted 3 rows in Validate but 6 in the population;
  - a threshold family counted 5 rows on the dashboard but 1 in the population;
  - an inclusion rule that could not be evaluated still showed rows "to include", while the population refused it;
  - an outlier rule (frozen bounds or rows) was not evaluated by the checks at all.

  Now:
  - Validate and the dashboard evaluate the rules as the population does, with the family's operator, and show a
    rule that cannot be evaluated with its reason.
  - A rule with both shapes is refused.
  - The participant flow and the dossier describe a one-sided *exclude* rule the right way round: "age < 18", not
    "age >= 18".
  - Operators are read without regard to case: `"All"` made an inclusion family an exclusion family.
  - An uploaded configuration with an unknown operator or an ambiguous rule is refused, with the list of its problems.
  - The legacy helpers (`enrich/custom_metrics_and_filters.py`) are removed.
- **Files read for an enrichment (and for Compare data files) go through the import engine.** A file given by its
  path went through a legacy loader, which the replay did not use:
  - a semicolon file with decimal commas came out as one column;
  - "-" and "?" stayed text on the page, but were missing values on replay;
  - a chosen sheet replayed as the first sheet;
  - date keys were parsed on the page only, so the replay skipped the merge;
  - the merged columns changed order with Python's hash seed.

  Now every entry point reads with the import engine, the enrichment step records how the file was read, the
  replay reads it the same way and parses the date keys the same way, and the merged columns keep the file's order.
  The legacy loader (`manage/file_handling.py`) is removed.
- **Outliers: one detector for Validate, the quality dashboard and the health score.** The three screens had their
  own detectors and flagged different rows for the same method:
  - quantile: 370 vs 400 rows on the tutorial cohort;
  - Isolation Forest: 140 vs 133 (different seeds);
  - LOF: 30 vs 27 (different numbers of neighbours);
  - Validate skipped the nullable integer columns.

  Validate's clip preview used the last column's outliers for every column: it showed a maximum NT-proBNP of
  28,017 where the recorded clip gives 9,055.

  Now all three use `biostats/outliers.py`, with one table of defaults:
  - the bounds that flag a value are the bounds that clip it, so the preview is the clip that is recorded and
    replayed;
  - a 0/1 column is never clipped;
  - the multivariate methods assess the complete rows only and say how many they left out. Validate used to
    impute those rows first (with MissForest, from the optional ML extra), so it now flags only among the
    complete rows.
- **One name, one statistic.**
  - **Cramér's V:** the missingness scan reported the uncorrected V (0.500 on [[3,1],[1,3]]), while every other
    screen reports the bias-corrected one (0.354). It now reports the bias-corrected V and says so; a 2x2
    co-occurrence keeps its phi.
  - **Normality:** above 5,000 values, Describe's "Shapiro-Wilk p-value" was Shapiro-Wilk's, which is not reliable
    there, while the analyses use D'Agostino's K². Describe now uses the analyses' test, and the column's help says
    so.
- **Logistic regression: the apparent AUC.** The AUC of a logistic model is computed on the data the model was
  fitted to, so it overstates how well the model would discriminate in new patients. The page and the dossier now
  call it the "apparent AUC (in-sample)", with a note saying so. Validated performance comes with the prediction
  analysis (M11).
- **FAMD** (Enrich > Create new variables, Validate's outlier projection, Data insight's clustering).
  - An integer column, such as the age in years, was analysed as one category per value: prince takes only
    decimal columns as numbers. Numbers are now quantitative and other columns categories, as in FactoMineR. A
    FAMD with integer columns gives other components, and a recorded FAMD step replays with the corrected method:
    the replay reports that its table differs from the recorded one.
  - Enrich offered every text column, so a date stored as text became one category per day. Dates, text with mostly
    distinct values (identifiers, free text) and single-value columns are no longer offered; a caption says which
    columns are left out and why. A date column is refused with what to do.
  - The explained-variance plot showed percentages on an axis that read as fractions, over the first components
    only. It now shows the cumulative percentage over every dimension of the data (one per numeric column, and per
    categorical column its categories minus one), reaching 100%, with "k components explain x% of the variance".
    On the tutorial cohort, the 11 columns of the example gave "0.79" for 3 components (0.79%, out of 810
    dimensions); without the date column there are 25 dimensions, and 3 components explain 16.4%.
  - The contributions table gives each variable's share of a component, rather than each category's.
- **PCA projection on Validate** (outliers): the columns are standardised, as in Enrich's PCA and Data insight. On
  raw values, the variable with the largest variance (a weight in grams) made the projection on its own.

### Privacy: what leaves the app

- **Traces hold no value of the data and no server path.** A trace is shared with the people who check or repeat
  the work.
  - Malformed lines skipped at import were copied into the trace (and its Word report); now only their count and
    line numbers are recorded, while the Import page still shows them.
  - Paths are written relative to the user's workspace instead of the server's absolute paths, which named the
    account.
  - The source file's checksum is SHA-256 (older traces keep their MD5, shown as such).
  - The software of the session (CohortStudio version and commit, Python, key libraries) is recorded when a
    dataset is loaded.
- **What the AI features send about the data**: one summary per column, never a row.
  - The column renaming sent the minimum and maximum of every numeric column, which can single out one participant,
    and the exact first and last dates. The suggested criteria sent the first 3 rows of the table.
  - Now the renaming, the variable map, the unit hints and the suggested criteria all send the 5th to 95th
    percentiles, the frequent values shared by at least 5 rows, and the years covered by dates.
  - The consent card of Settings lists exactly that.
- **The variable map no longer sends the extremes of a column.** Refining it sent the minimum, maximum and mean of
  every numeric column, though the notice says the minimum and maximum are never sent. Later prompts then sent the
  whole map with them, and a map imported or edited could carry anything. The prompts that carry the variable map
  now keep only the percentiles of its statistics (an allow-list) and none of its keys that hold values of the data;
  the refinement sends the summary the notice lists.
- **Withdrawing the confirmation stops the AI engine.** The data-sharing confirmation was checked only when the
  engine started, so an engine kept answering, and sending data to its cloud provider, after the box was unticked.
  - Unticking *I confirm this data may be sent to …* now stops an engine that uses one of the providers it names:
    its connections are dropped, so nothing more is sent from this tab. Settings shows the engine as stopped, and
    why.
  - **Withdraw the confirmation and stop**, at the top of the Data sharing card, does the same whatever provider is
    selected. The card says which providers the running engine sends data to: with Ollama selected, it read "no
    data leaves this server" while a cloud engine ran.
  - **Every recipient needs it.** With Claude, a cloud embedding provider receives the documents and every search
    text; the confirmation of Claude covered it. The box now names both, and confirms them together: a provider
    confirmed as Claude's embedding provider is not confirmed to answer on its own. A confirmation given for one
    provider was shown as given for the next one chosen in the list. Ollama is exempt on this machine and the
    private network; an Ollama at a public address needs the confirmation too.
  - Only ticking or unticking the box counts: looking at another configuration changes nothing.
  - A refused, failed or interrupted start keeps no connection, and no longer relabels the engine that runs, which
    a later withdrawal could then miss.
  - The documents stay indexed on disk: once the box is ticked again, **Initialise / reindex** reuses the index.
- **Smaller leaks closed.**
  - Listing a cloud provider's models could send the typed key to the Ollama address of an earlier engine.
  - Google Gemini's graph extraction uploads the PDF file itself. The notice now says so, and Google is asked to
    delete the file even when the extraction fails.
  - With Claude, the graph extraction uploaded each PDF to Anthropic (whose client has a files API too) and left it
    there. The file API is now used with Gemini only; the other providers receive the document's text.
  - LangChain's tracing service (LangSmith) is switched off at start-up, its four environment switches and its own
    global switch: a tracing switch left in `.env` would have sent every prompt there.
  - An Ollama address on another machine is shown as such, not as "Local: no data leaves this server", with a
    warning when it uses http. The sidebar and the Start page say so too.
  - The notice says that the confirmation is the user's, for one browser tab, and does not replace the approvals of
    a data-protection framework, and that what was sent stays with the provider. "Mistral AI (cloud, EU)" claimed
    a location the app cannot check: it reads "Mistral AI (cloud)".
- **Sharing a study configuration**: a switch leaves out the outlier rules that list row ids, which only mean
  something for the user's dataset.

### Configuration and identity

- **`.env` is read at start-up.**
  - Before, it was read only when the Settings page's AI section was first imported. Outside Docker, the identity
    provider, the first administrator (`CMP_ADMIN_*`), `DATA_ROOT` and the log level ignored it. The AI keys were
    found only after someone had opened Settings.
  - `main.py` and `manage_users.py` now load it first. A variable already set in the environment still wins, as
    with Docker.
  - A setting written in `.env` now applies outside Docker.
- **An unknown identity provider stops start-up.** A misspelt `CMP_IAM_PROVIDER` (for example `ldap` or
  `better_authh`) silently gave local accounts. The app now shows the error and the valid names instead.
  `better-auth` is accepted, and the names are read without regard to case.
- **Admin > Users manages the accounts of the configured identity provider.** It always managed the local accounts,
  so under Better Auth its activations, deletions and password resets reached nobody who signs in. The page now names
  the provider, and says there is nothing to manage when sign-in is disabled.

### Usability

- **The OpenAI and Mistral providers work.** Both use the `openai` package, which the `ai` group did not install:
  choosing either in Settings gave "'openai' package is not installed". The group now installs it, and `httpx`, which
  the Ollama connector uses, is declared rather than arriving through other packages. A test checks that each
  provider's packages belong to the `ai` group.
- **Settings keeps its choices for the whole session.** Streamlit forgets a choice when its page is not shown:
  - the day-first default never reached Import, the page that reads it;
  - a return to Settings showed Ollama, the first model and an empty consent box, and withdrew the consent of the
    engine that runs.

  The provider, models, typed keys, consent, chosen documents, vector database and import default are now kept. When
  an engine runs, Settings starts from its provider, model and consent.
- **Reset Cache on Home clears your statistics only.** It emptied every cache of every user of the server. The
  statistics are now kept per session, and freed when the session ends. The cache of correlation workbooks (Excel)
  keeps at most 8, for an hour.
- **Versions & traces: Download the trace (JSON).** The selected trace downloads as recorded, so its integrity
  check passes on another machine. Before, the page offered only the Word report.
- **Text files with an unusual delimiter.**
  - When none of the usual delimiters (comma, semicolon, tab, pipe) fits, the import detects rarer ones from
    laboratory, HL7 or older exports: colon, caret, tilde, hash, exclamation mark, broken bar, and the ASCII unit
    separator. Files read correctly before are read the same way.
  - The Import page offers **space** and **other…** (any single character). Enrich's external file and Validate's
    Compare get the same choice in **How the file is read**, which opens by itself when only one column is found;
    before, they had no way to change the delimiter.
  - A delimiter that cannot work (a letter, a digit, a double quote, or the decimal or thousands separator) is
    refused with the reason.
  - The delimiter chosen is recorded with the step, and the replay reads the file the same way.
  - Enrich and Compare used to read files given by their path with the `detect-delimiter` package, which found
    colons and tildes but read semicolon files with decimal commas as one column. They now use the import engine,
    and the package is no longer a dependency.
- **Visualize: Select columns to display on hover works again.**
  - Choosing a column raised an error since Plotly 7, whose traces no longer have `.get`.
  - Before that, every trace received the values of the whole table, so with a colour group a point showed another
    row's values, and bars or boxes showed the first rows' values.
  - The columns now go to Plotly Express, which matches them to the points of each trace. The choice is offered where
    each point is one row: scatter plots (points or regression), and line plots without aggregation or interval band.
- **Analyze > Hypothesis tests: the result card explains its assumption checks.**
  - The card showed "normality (No)" and "normality (Yes)" side by side, with neither the group variable, the
    finding, nor whether the check matters for the test.
  - Each check is now one line: the group it concerns (*Normality, diabetes = No (n = 506)*), the finding with its
    test and p-value, and what it means for the test that ran. For example, unequal variances are "Not required:
    Welch's test allows the groups different variances", and non-normality with 30 or more values is not critical
    for a test of means.
  - The same reading feeds "How to read this".
- **Analyze > Hypothesis tests: the figures of the detailed view.**
  - They stretched to the full width at fixed heights, the histograms overlaid opaque counts, and one Q-Q plot of
    the pooled residuals sat beside per-group normality checks.
  - Four figures now form a two-column grid at one height, one colour per group throughout (the colour-blind-safe
    palette): box plot with brackets and n, means with 95% CI, histograms as the percentage of each group, and a
    normal Q-Q plot of each group.
  - Significance brackets give "p < 0.001" rather than scientific notation, here and on Visualize.
- **Group comparison pooled over the imputed datasets: Test selection.** *Recommended* keeps the automatic choice.
  *Choose the method* lets a numeric variable in two groups be compared with Student's t-test (equal variances)
  instead of Welch's, pooled the same way, and three groups or more with the equal-variance linear model (see
  Statistics). A warning says why the recommended analysis is recommended, and the summary and the dossier mark
  the chosen one. The page says which analyses have a single pooled analysis (categorical variables) and which
  have no pooling rule (rank tests, Fisher's exact test).
- **Enrich > Create new variables: the columns chosen for a transformation stay chosen.** Applying a transformation
  adds columns to the draft, and the column list then started again empty, so a second transformation of the same
  columns meant choosing them again. The three column lists (encoding, PCA and FAMD, the other methods) now keep
  the choice.
- **Groups to include: groups that differ only by a space are told apart.** "Luxembourg" and "Luxembourg " are two
  groups, which the list showed identically, so the tutorial cohort's sites seemed to be listed twice.
  - A space at either end, or a run of spaces, now shows as ␣ (*Luxembourg␣*).
  - A warning names the groups that differ only by spaces or capitals, and gives the formula of a cleaned copy to
    group by, to add in Enrich > Create New Variables: for example `site_clean` = `site.str.strip()`.
  - Nothing is merged for you: the analysis uses the values as they are.
  - The same applies to the pooled comparison over the imputed datasets and to the power analysis.
  - Another grouping variable now starts with all its groups. The list kept the previous variable's choice, none of
    which is a group of the new variable, so it was empty (Analyze and the power analysis).
- **The conda environment installs what the application uses.** `environments/environment.yml` kept a list of its
  own, which had drifted: it installed ChromaDB and three LangChain packages that the application no longer uses,
  and none of the vector databases (LanceDB, Qdrant), the OpenAI, Anthropic and Ollama connectors, pyarrow, xlrd or
  lifelines. It now installs Python 3.14 from conda-forge only (never the user's `defaults` channel), then the
  locked packages of `requirements.txt`, which follow `uv.lock`; a test keeps it so. The README shows how to use it,
  how to update it (create it again), and that on macOS the full install needs Apple silicon.

### Internal structure: one catalogue of trace steps

- **`core/steps.py` registers every kind of trace step once**, on the new shared kit `core/registry.py`. Each kind
  declares:
  - its title in the trace report;
  - the part of the workflow that records it;
  - how the replay treats it;
  - whether it changes the data.

  Before, seven files kept their own lists of step names: the replay, the trace report, the workflow progress, the
  participant flow, the study dossier and the imputation bookkeeping. They now all read the catalogue.
- **The trace report titles every step.** Five appeared as their name in words: renamed columns, clipped outliers,
  the import's parse report, saved versions, and data replaced by a page.
- **Old traces still read and replay.** Four kinds that no page records any more are kept as legacy:
  `apply_filter`, `missing_value_handling`, `multiple_imputation` and `outlier_handling`. A trace holding
  `outlier_handling` now replays from its snapshot instead of skipping the step as unknown.
- **A step recorded under an unregistered name is logged**, and the test suite refuses it.

### Internal structure: the safety net for the architecture work (M5b)

- **The layers are checked.** `tests/test_layering.py` classifies every module by layer, from `utils` up to the
  pages, and checks five rules:
  - imports go down the layers;
  - no import cycles;
  - nothing below the user interface reaches Streamlit;
  - the AI providers' packages are imported only by their connectors;
  - the command-line entry point loads neither Streamlit nor the AI packages.

  It follows what actually runs: imports inside functions, parent packages, and modules named in strings. Today's
  exceptions are listed, each with the step that removes it, and the lists can only shrink.
- **Formatting covers every file.** CI and pre-commit now check the whole repository with `ruff format`, except a
  list of the 166 files not yet formatted, which can only shrink: a test fails when a listed file is formatted and
  still listed. Each file is reformatted in its own commit, recorded in `.git-blame-ignore-revs`.
- **Type checking covers every package.** pyright now checks all twelve packages and both entry points, where it
  checked six. The 24 errors of today's code in 11 files are counted in `dev_scripts/pyright_baseline.json`. CI fails
  when a file has more errors than its count, and when it has fewer until the count is lowered, so the count can
  only fall. A setting that pyright did not recognise (`reportPossiblyUnbound`) now has its real name.
- **Every package has a coverage floor**, about 2 points under its measured coverage: the pages, `utils`,
  `prompts` and `monitor` had none (`monitor`, at 8 %, needs tests). Floors only rise: a package more than 4 points
  above its floor fails until the floor is raised. The overall minimum rises from 63 % to 70 % (72 % measured).
- The pre-commit hook and CI format with the same version of ruff, which a test checks against `uv.lock`.

### Runtime (M5b)

- **An interrupted imputation stops all its processes and deletes its copy of the data.** A click during a parallel
  multiple imputation reruns the page, which skipped the clean-up. The run's interpreter and its worker processes
  ran on, unseen, on cores already given back to other runs, and the temporary folder with a copy of the data stayed.
  The workers now exit with the interpreter that started them, and the folder is deleted however the run ends. On
  Windows, the run starts the base interpreter of the virtual environment, as Python's multiprocessing does:
  stopping the environment's launcher left the interpreter it started running. The interruption itself reaches the
  page unchanged; it no longer turns into a failed run that was redone in one process.
- **Imputations share the processor cores of the server.** Each session started as many worker processes as the
  server has cores minus one, so two people imputing at once ran twice that number. The worker processes are now
  shared by every session of the server: a run gets the ones that are free and never waits. When fewer than two
  are free, it runs in one process, and the page says so. `CMP_MI_MAX_WORKERS` sets the total (default: the cores
  minus one); set it on a container with a CPU quota (`cpus:`), which the app cannot see.
- **Changes that never reach the trace no longer leave copies of the data behind.** Each merge, transformation
  and imputation prepared on the Enrich page saves a copy of the whole table (a snapshot) in the workspace before it
  joins the draft. The copy stayed when the change never reached the trace: a draft discarded, redone on newer data
  or refitted on the current population, or dropped when another dataset was loaded, the dataset was cleared or the user signed out; a change that changed nothing or failed; and a
  copy whose writing failed part-way. These copies are now deleted. Only files in the workspace's snapshots folder
  are deleted, and never one that a saved trace names (the replay and the fast mode of Reproduce use them).
  - Limits: a draft still pending when a session ends without signing out (a closed tab, a server restart) keeps
    its snapshots, and copies left by earlier versions stay until they are deleted by hand.
  - A deletion is a plain file deletion: copies kept by backups or synchronised folders stay (README, "The data
    folder").
- **The column types are worked out once per dataset.** Validate, Statistics & correlations, Visualize, Hypothesis
  tests & epidemiology and Survival & regression sorted the columns into numeric, categorical, date and other types
  on every click, trying a dozen date formats on each text column; Statistics & correlations did it four times per
  click. The types are now kept until the data or the ticked criteria change.
- **The quality audit on Validate runs once per dataset and detection settings.** The dashboard audited every row
  again on every click, a score weight included. The audit is now kept until the data, the criteria definitions or
  the outlier detection settings change.

### AI engine: Gemini Flash-Lite

- **The Gemini chat models now include Flash-Lite**, cheaper and faster for the AI features: `gemini-flash-lite-latest`,
  `gemini-3.5-flash-lite` and `gemini-2.5-flash-lite`.
- **2.0 Flash, 1.5 Flash and 1.5 Pro are no longer offered**, because Google no longer serves them.
- **One list:** the Settings page reads the provider's list instead of keeping its own copy.
- **A typed API key** may still add the other chat models it can use; the image, speech and robotics models are left
  out.

### Documentation: a DOI to cite

- Releases are archived on Zenodo from 2.4.1 on. The concept DOI
  [10.5281/zenodo.23436094](https://doi.org/10.5281/zenodo.23436094) resolves to the latest version; each version
  also has its own (2.4.1: 10.5281/zenodo.23436095).
- The DOI is in `CITATION.cff` (GitHub's *Cite this repository*), the README (a badge and the BibTeX entry) and the
  site's APA and BibTeX citations. A test keeps it the same everywhere.

## [2.4.1] - 2026-10-17

Security work, fixes found by code scanning and in use, and one addition to imputation. The first release archived
on Zenodo, which gives it a DOI.

**Highlights**

- **Code scanning**: every CodeQL security alert is fixed in the code (paths typed by the user stay in their
  workspace, log entries cannot be forged, no regular expression backtracks on long input). API keys no longer
  appear in the log or in the errors of the AI engine.
- **AI engine**: a chat model that cannot answer is reported as such instead of "Suggestions received!" over an
  empty list; the default Gemini model is one Google still serves to new keys.
- **Exact imputation by formula**: a missing value the other variables determine (a weight from the BMI and the
  height) is imputed exactly and counts as observed in a later multiple imputation.
- **Start page**: "Your documents" is ticked only when the library can be used; a document can be removed.
- **Documentation**: the variables the AI engine suggests for a concept, shown on the site and in the README.

### Security and fixes: the code-scanning alerts resolved

CodeQL (the `security-extended` and `security-and-quality` suites of `.github/workflows/codeql.yml`) reported 195
results on 2.4.0. Every security alert is resolved in the code, not dismissed, and the analysis was run again to
confirm it.

- **Paths chosen by the user** (20 alerts). `core.workspace.resolve_within`, which keeps a typed or recorded path
  inside the user's workspace, is now written in the form code scanning recognises (`os.path.realpath`, then a
  prefix check that stops at a separator); a test covers a sibling folder sharing the prefix. The page helpers
  that accept a typed path refuse it when no user is signed in, instead of using it as typed. A trace is
  user-supplied JSON: the existence of its recorded paths is now checked inside the workspace only, and a recorded
  file name is looked up inside each search folder, never above it.
- **Log entries** (17 alerts). A file path, formula or user name written to the log is kept on one line
  (`utils.logging_config.single_line`), so typed text cannot forge another log entry.
- **Regular expressions** (2 alerts). The parser of the anomaly expressions no longer backtracks on long input;
  its output is unchanged on the expressions tested.
- **Bugs found on the way**:
  - a nested loop on Enrich reused the index of the suggestion cards, whose button keys could then collide;
  - external data enrichment returned nothing when no enrichment was requested, and its caller then failed;
  - "Session outdated" on Enrich dropped the dataset for the rest of the run;
  - the Gemini connector returned nothing when no attempt was allowed;
  - dead code removed: an outlier summary that read an undefined column and a mask-family form that called a
    helper with a missing argument, both unreachable, and a commented-out copy of an old page.
- Variables assigned in only some branches now start with a value or close their branches with an `else`, so no
  path can reach them unset.
- Code-quality notes: a bare `except:` is now `except Exception:` (or narrower), so an interruption is no longer
  swallowed; each remaining silent `except` says why it is safe. Three silent failures in statistics code could
  hide a missing result and are now logged: the AUC of a logistic regression, a predictor left out of the MAR
  check, and a p-value left out of the MCAR heuristic. No computed value changes. Unused code and commented-out
  code are removed, and the generated developer documentation is regenerated.
- 6 of the 195 results remain, all false positives to dismiss:
  - `biostats.mi.analyze.analyze_mi` passes the imputation index only to the functions that take it (2);
  - `biostats.mi.engine` and `biostats.mi.parallel` import each other inside functions, for the subprocess entry
    point (2);
  - the `_CONFIGURED` flag of `utils.logging_config` is read through a `global` statement;
  - `enable_iterative_imputer` is the import scikit-learn requires before `IterativeImputer`.
- CodeQL Action v4 (v3 is deprecated in December 2026).

### Security: API keys no longer appear in the log or in the AI engine's errors

- A provider can quote the key it refuses: Google answers a suspended key with "Consumer 'api_key:...' has been
  suspended". The providers logged that answer as it was, and Settings > AI engine showed it when the
  initialisation failed.
- Every log record, its traceback included, is now written without keys, and so is the initialisation error. A key
  is recognised by its shape (Google, OpenAI, Anthropic), as the value of an `api_key`, `x-api-key` or
  `Authorization` field, or as the value of a configured `*API_KEY`, `*_SECRET` or `*_PASSWORD` variable, which
  covers keys without a recognisable shape (Mistral). The one-time setup token of the first start stays in the log.
- Tests cover the redaction, the log handlers and the initialisation error; a browser check confirmed both with a
  refused key.

### Documentation: the variables the AI engine suggests, shown

- A short animation, on the site (AI section) and in the README (*AI Assistant*), follows a request on the
  tutorial cohort with a review of the body-surface-area formulae indexed (Redlarski et al., *Sci Rep* 2016,
  CC BY 4.0). It shows the request for *BSA*; a suggestion, the Du Bois formula mapped onto `height_cm` and
  `weight_kg`, citing the review; and its preview, where the five values made from entry errors stand out.
- Data Enrichment (chapter 03) shows the suggestion card in a worked example. A test checks that every image of the
  documentation exists.
- `dev_scripts/ai_screenshots.py` captures the screens; it needs a provider key and the review in the library. The
  screenshot library keeps them, and `dev_scripts/make_readme_assets.py` builds the images. The five formulas the
  engine proposed were each checked against their published reference.

### Fixed: a chat model that could not answer was reported as a success

- Google no longer serves gemini-2.5-flash, the default Gemini chat model, to new API keys. Every call failed,
  yet the AI engine said it was initialised, and AI Variable Suggestions showed "Suggestions received!" over an
  empty list. The connectors answer a failure with the error as their message, and the suggestion pipeline read
  it as an answer without suggestions.
- The default Gemini chat model is now gemini-3.8-flash, as Google recommends; the 2.5 models stay in the list
  for older keys.
- An error answer is now recognised (`utils.llm_utils.provider_error`). The initialisation stops with "The chat
  model ... did not answer: ...", the suggestions show the error, and a failed answer is no longer cached. When
  the model answers but no suggestion fits, the page says so instead of "Suggestions received!".
- A failed initialisation leaves the AI engine off; it used to stay marked active with replaced connectors.
- The connectors' error answers no longer quote an API key either (the chat shows them).

### Fixed: "Your documents" was ticked on the Start page workflow although no document could be used

- The documents path of the workflow map counted as done whenever the workspace's library held an indexed PDF or a
  saved knowledge graph. That held for every dataset and even with the AI engine off, and for good, since a document
  could not be removed from the library.
- It is now done when the AI engine is on and the library holds documents that are indexed and still there. Without
  the engine, the box shows *AI off*. The hover names the documents.
- Settings > AI engine > Document library: **Remove a document** deletes a PDF from the library, after a
  confirmation (never a file outside it).
- With *Reuse the existing index* on, the index is reused only when it holds the documents being indexed, and is
  rebuilt after a document is added or removed. Before, an added document was not indexed until the option was
  turned off.
- Tests cover the workspace evidence, the removal, and the reuse or rebuild of the index on LanceDB and Qdrant; a
  browser check covers the map and the removal.

### Exact imputation by formula (Data > Enrich > Handle Missing Data)

- Targeted imputation has an **Exact formula** option, for formulas that give the true value because other
  variables determine it: a weight from the BMI and the height, or a total from its parts.
- Such values are imputed exactly and recorded apart from the single-imputed ones. Nothing was estimated, so:
  - the analyses treat them as observed, without the single-imputation warning;
  - the multiple-imputation panel keeps them as observed values with a short note, instead of advising to undo
    them;
  - the study dossier lists them as imputed by an exact formula, next to its formula under *Data handling*.
- This lets an exact imputation come before the statistical ones, which then have fewer cells to estimate.
- Unticked (the default), and for steps recorded before this option, nothing changes: the values are flagged as a
  single imputation.
- The choice is recorded in the trace step (`exact`). It is kept with saved versions (the lineage file) and restored
  by a full replay.
- Tests cover:
  - the session's bookkeeping and the replay, including steps recorded before the option;
  - the dossier paragraph;
  - an application test: no warning, and the record kept across saving and reloading.

### Documentation: missing values, imputed exactly first, then statistically

- The documentation now presents the imputation tools as one sequence:
  1. impute by an exact formula what other variables determine (targeted imputation);
  2. impute the rest statistically, by single imputation for exploration (KNN, missForest, chained equations, mean
     or median) or by multiple imputation for inference.
- Imputing exactly first leaves the statistical imputations fewer cells to estimate and more information to estimate
  them from.
- Chapter 03:
  - opens *Handle Missing Data* with a table, *which approach for which missing value*;
  - describes targeted imputation precisely: it fills the missing cells only, fits nothing, and fills every row
    whose inputs are known;
  - describes the *Exact formula* option and how the filled cells are flagged;
  - corrects two claims: the AI imputation suggestions use the related columns, not the documents; and the editor's
    example (a BMI computation) did not fill a missing variable.
- Site: the imputation feature, now labelled *Imputation*, shows the three steps.
- The README, the handbook and the short tutorial follow the same order.

---

## [2.4.0] - 2026-10-17

The follow-up programme after an external architecture review (M0 to M4), the rename to CohortStudio, and a pass over
the documentation and the site.

**Highlights**

- **One analysis population** (M1): inclusion and exclusion criteria are filters ticked in the sidebar and used by
  every page; each result states the data and the population it was computed on; the trace is hash-chained and its
  replay verified.
- **Multiple imputation end to end** (M2): a chained-equations engine fitted on the analysis population, Rubin's
  rules with the D1 and D2 tests, pooled survival and regression models, checked against R `mice` and `mitml`.
- **Participant flow and study dossier** (M3): the participant flow (STROBE item 13), and a Word dossier of the
  results with their provenance.
- **Packaging and deployment** (M4): optional dependency groups, a Docker image checked in CI, and a first
  administrator created from a one-time setup token.
- **Connectors**: Anthropic Claude, and LanceDB or Qdrant for the document library.
- **Documentation**: a redesigned site, a README with an animated tour and a plot gallery, each claim linked to the
  test that checks it, and a `CITATION.cff` for GitHub's *Cite this repository*.
- **Behaviour changes**: filters no longer delete rows, Enrich changes reach the analyses when the draft is applied,
  and dates are no longer parsed silently (see *One analysis population, one source of truth* below).

### Documentation: each claim links to the test that checks it

- README, *Verification & Test Suite*:
  - A table, *What is checked, and where*, links each strong claim to its test or workflow: the parity with R
    `mice` and `mitml`, the weekly bias and coverage simulation, the hash-chained trace, the pipeline integrity, the
    layering, the workspaces and the sandbox, and the coverage floors.
  - It replaces the table of test modules, whose static *PASSED* column said nothing the CI badge does not.
  - The test count no longer drifts ("more than 800" instead of "776 passed").
  - "Multi-user isolation" is now "Per-user workspaces": the workspaces separate accounts in the application; they
    are not an operating-system sandbox.
- Site: each fact of the facts strip links to its proof: the parity tests, the trace code, the CI runs, the licence.
- A test checks that every relative link of the README points to an existing file.

### Documentation: creating variables from formulas, and the variables the AI engine suggests

- The documentation never described two features of Data > Enrich > Create New Variables:
  - the assisted formula editor: column suggestions as you type, a check of the name and the formula, and a preview
    with the statistics of the new variable on the analysis population;
  - the variables the AI engine suggests from the columns of the dataset: a concept search with three strategies,
    or an exploratory scan for standard indices and scores. Each formula is checked against the columns, cites its
    source, can look for a proxy for a missing input, and goes to the editor before anything is added.
- Updated:
  - chapter 03, which also loses its old "AI Variable Discovery" paragraph and its claim about units;
  - the README (overview, enrichment and AI assistant);
  - the handbook steps and both tutorials;
  - a cross-reference in chapters 06 and 07;
  - the Enrich step and the Optional AI section of the site.

### Documentation: the site shows the plots and the tour

- A new feature row, "Describe and plot", presents the data description of Explore and the figures of
  Visualize. Its image shows four figures from the tutorial cohort (`docs/assets/shots/plots.webp`, built by
  `dev_scripts/make_site_assets.py` from the screenshot library).
- The Workflow section ends with the animated tour of the README. Visitors who ask for reduced motion see its
  first screen instead.
- The tour now lives in `docs/assets/shots`, the folder GitHub Pages publishes, and the README shows it from
  there, so it is stored once.
- The facts strip now says "more than 800 automated tests".

### Documentation: a README header with an animated tour, the workflow and a plot gallery

- The README opens with the CohortStudio wordmark, light or dark to follow the GitHub theme. Below it come the
  tagline, links to the main sections, and an animated tour of nine screens on the tutorial cohort, from the Start
  page workflow to the Word study dossier.
- The overview shows the workflow map of the Start page.
- A new section, "Describe and Visualise the Data", has two parts:
  - a short animation of the data description on Explore: column types, descriptive statistics and correlations;
  - six Visualize figures: a box plot with p-values between groups, a clustermap that tags each variable by its
    p-value, violin plots, a clustered correlation matrix, a scatter plot with regression lines, and a sunburst.
- The screenshots are kept in a library, `assets/screenshots`, as lossless WebP, so the README and site images
  rebuild without running the application. `dev_scripts/keep_screenshots.py` adds a capture run to the library.
- Scripts:
  - `dev_scripts/site_screenshots.py` now also captures Explore, the Visualize figures and the workflow map.
  - `dev_scripts/make_readme_assets.py` builds the README images.
- The animations are animated WebP. Between two screens only a band of the window changes, which keeps each file
  under 2 MB, the limit of the large-file hook.
- Tests check that:
  - the README images exist;
  - the animations loop and stay under 2 MB;
  - the library holds every screenshot the builders use.
- The Docker image leaves out the README images and the screenshot library.

### Fix: the colour scale of the Visualize correlation matrix

- On Explore > Visualize, the correlation matrix fitted its colour scale to the values shown. With a diverging
  map such as RdBu, a correlation of zero could therefore be drawn dark red. The scale is now fixed from −1 to 1
  and centred on zero, as on the Explore correlation heatmap (audit S13). A test covers both heatmaps.

### Documentation: the data and statistics engine

- The README architecture diagram and the site call the plain-Python layer the **data and statistics engine**,
  next to the AI engine. Its three parts are named by what they do: statistics (`biostats`), file import
  (`ingest`) and the data session (`core`). The layer used to be called "Engine", which was easy to confuse with
  the AI engine.

### Dependency updates: pyarrow 26 and Streamlit 1.66, Google libraries, ruff 0.17

- Updated in `uv.lock`:
  - google-genai 2.30.0 and google-auth 2.62.0;
  - pyarrow 26.0.0, which brings Streamlit 1.66.0: Streamlit 1.65 required a pyarrow below 26, and it no longer
    needs `toml` and `httptools`;
  - jsonpatch 1.35, pycparser 3.11 and truststore 0.10.5;
  - for development, ruff 0.17.0; the pre-commit hook follows it.
- Not taken:
  - websockets 17.2 and tenacity 9.2.1, because google-genai requires a websockets below 17 and a tenacity below
    9.2;
  - aiohttp, which is no longer a dependency.
- Checks:
  - ruff 0.17 reports nothing and changes no formatting, and the whole suite passes.
  - A browser walk-through on Streamlit 1.66 covered the tutorial, the multiple imputation, the pooled survival and
    Cox models, the comparison, the dossier, the Enrich draft and the Variable map.
  - Streamlit 1.66's breaking changes do not touch the application's stylesheet: the main area is a `<main>` element
    and the sidebar state moved to `data-collapsed`.
- GitHub Actions, the versions Dependabot had proposed:
  - `actions/checkout` v7, `astral-sh/setup-uv` v7 and `actions/upload-artifact` v7;
  - for the documentation site, `actions/configure-pages` v6, `actions/upload-pages-artifact` v5 and
    `actions/deploy-pages` v5.

### Fixed: a Streamlit warning on Analyze > Power & Sample Size

- A group comparison and **Load Parameters into Calculator** prefill twelve inputs of the power calculator
  (effect size, group sizes, ratio, number of groups) through the session state. The inputs also passed a default
  value, so Streamlit logged a warning about a widget with both.
- The default is now passed only while an input is unset. The prefilled values are the same, and the help text is
  required by the helper.

### Dependency updates (2026-10-16)

- Updated in `uv.lock`:
  - numpy 2.5.4;
  - pydantic 2.14.1 with pydantic-core 2.50.1;
  - starlette 1.8.0;
  - pypdf 6.20.1;
  - langchain-core 1.6.10 and langsmith 0.14.8;
  - narwhals 2.27.1 and jsonpointer 3.2.1.
- Checks: the whole suite passes, a browser walk-through of the tutorial ran, and the pooled Cox model of the
  tutorial gives the same numbers.
- Not updated:
  - uuid-utils 1.0, because langchain-core requires a version below 1.0;
  - portalocker 4, because every qdrant-client requires a version below 4.0.

  Dependabot had proposed both in the requirements files only, where they would have broken pip installations.
- Dependabot now proposes updates of direct dependencies only. For an indirect dependency, its `uv` support edits
  the requirements files without touching `uv.lock`, so CI never installed the proposed version. Indirect updates
  are made with `uv lock --upgrade-package` and tested.
- The requirements files are also left out of Dependabot's scan (`exclude-paths`). They list every package, so
  Dependabot took each one for a direct dependency and kept proposing requirements-only updates.
- The requirements workflow also runs when the requirements files alone change on `main`, and brings them back to
  `uv.lock`.

### Dependency updates through `uv.lock`

- Dependabot updated only `requirements.txt` and `requirements-yfiles.txt` (its `pip` ecosystem). Those files are
  exported from `uv.lock`, which CI and the Docker image install from, so merging its pull requests would have
  made pip and uv install different versions.
- Dependabot now updates `uv.lock` (its `uv` ecosystem). The GitHub Actions updates are unchanged.
- A new workflow (`.github/workflows/sync-requirements.yml`) exports the two requirements files from `uv.lock`
  whenever `uv.lock` changes on `main`, and commits them when they differ. It can also be run by hand.

### Licence inventory generated from the lock file

- `documentation/developer/License_Dependencies.md` listed the packages of a whole development machine (Sphinx,
  Scrapy, aiobotocore and others the application does not use). It also recommended a licence for a project
  already released under Apache 2.0.
- `dev_scripts/license_inventory.py` now writes it:
  - The packages and versions come from `uv.lock`, per group: core, `ai`, `ml`, `survival`, `graph-yfiles`, and the
    development tools, which are not shipped.
  - The licences come from the installed package metadata.
  - Packages installed only on other platforms (the browser) are left out.
- The copyleft licences among the shipped packages are named. They are three weak, file-level MPL-2.0 licences:
  `certifi`, `orjson` and `tqdm`. The analysis of the proprietary yFiles engine is kept.
- `tests/test_license_inventory.py` fails when the lock file lists a package that the inventory does not, so a
  new dependency gets its licence reviewed. A version bump alone, such as a Dependabot update, does not make it
  fail.

### Fixed: the Enrich draft banner was at the bottom of the page (Data > Enrich)

- The banner with **Apply to the dataset** was drawn after the tabs, so it appeared under all their content,
  while the messages said to apply the draft "at the top of the page".
- It is now above the tabs. It is filled at the end of each run, so it also lists the change just added.

### Fixed: a saved Variable map version needed two loads to show its formulas (Knowledge > Variable map)

- **Load vN** on the page without a taxonomy read only the taxonomy file. The formulas, the links between the
  variables and their formulas, and the AI-engine sync came only with a second load from the sidebar.
- Both buttons now use the same loader, so one click brings the taxonomy, its formulas and the repaired links.
  The loaded version is also remembered for **Overwrite**.
- The files are read as UTF-8.

### Legacy Excel workbooks (`.xls`)

- `xlrd` joins the core dependencies: `.xls` files import like `.xlsx` (sheets, merges, decimals, accented text).
  They used to be accepted, then fail on the missing package.
- Every upload of the application (Import, Enrich, the comparison on Validate, the missing files of a replay)
  accepts the same file types.

### Documentation site redesigned (`docs/`)

- The site at karimelmernissi.github.io/Cohort-Studio now uses the application's logo and wordmarks
  (`assets/brand`) and its colours, in place of the former illustration.
- The content describes what the application does today, with screenshots of the real application on the
  tutorial cohort:
  - the analysis population and its participant flow, data quality, multiple imputation, the analysis plan, the
    pooled survival and Cox models, and the study dossier;
  - the methods with their references and the R parity figures;
  - the optional AI, the architecture, the installation and the citation (version 2.3.0).
- Removed:
  - the decision tree, which contradicted the Welch-first defaults;
  - the invented use cases;
  - the version 0.2.0 badge and citation;
  - Google Fonts.
- The page now loads nothing from third parties and sets no cookies.
- `tests/test_docs_site.py` checks the page:
  - No regulatory or marketing claims: 21 CFR Part 11, compliant, immutable, certified, or "signed" outside "not
    an electronically signed record".
  - Nothing is loaded from other sites, and every local file exists.
  - Every image has a text alternative and its real size.
  - The citation names the current version.
- `dev_scripts/site_screenshots.py` (Playwright) and `dev_scripts/make_site_assets.py` regenerate the
  screenshots, the WebP page images, the icons and the social preview.

### Packaging, deployment and CI (follow-up programme M4)

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
- **CI** (`.github/workflows`):
  - A job without the optional groups runs the whole suite, with the AI tests skipped.
  - A job builds the Docker image, starts a fresh container, waits for its health check, finds the first-run setup
    token in its log and issues a new one from the command line inside the container.
  - A weekly workflow runs the statistical simulation suites (`pytest -m slow`).
  - The integration workflow is pinned like CI (uv 0.10.9, Python 3.14), installs every optional group and gets
    `ANTHROPIC_API_KEY`.
  - Coverage floors are raised to the measured values minus about 2 points: 63 % overall; per package, biostats and
    core 92, graph 91, ingest 87, ui 81, explore 74, manage 61, enrich 56.
- **Docker**:
  - `run_docker.bat` mounts `./data` and `./config`, reads `.env` and shows the log with the setup token.
  - The start scripts use Docker Compose v2.
  - The optional Qdrant server is pinned to `qdrant/qdrant:v1.19.2`, the client's release.
- Documentation: the README and the site describe the optional groups, Docker Compose with the setup token and
  the CI jobs, with the current test and coverage numbers. The audit report marks M4 done.

### Fixed: wording in the application and a Student's t in the MAR check

Found while correcting the documentation:

- **Validate > Data quality dashboard**: the MAR check of one variable used Student's t-test under "Force
  Parametric". It now uses Welch's t-test, like the dependency scan and the rest of the application.
- **Visualize**: the test choices for the significance brackets name what runs, pairwise Welch t-tests or pairwise
  Mann-Whitney U tests. The former labels named ANOVA and Kruskal-Wallis, which never ran there.
- **Explore > Statistics & correlations**: the "Optimization Settings" of the benchmark are removed. Nothing used
  them, and their "Preferences saved" message was misleading. The benchmark itself stays, for information.
- **Enrich > Clustering**: the methods note says that the selected columns must be complete. It used to say that
  incomplete rows were excluded, while the page stops on a missing value.
- Two more messages, on Start and Enrich, sent users to the sidebar or "the Home page" to set up the AI engine.
  They now point to Settings > AI engine. Module docstrings use the current page names.

### Fixed: documentation claims that no longer matched the application

A read-only comparison of the README, the handbook chapters and the About page with the code found claims that
were stale or never true. They are corrected:

- **Storage**:
  - Each version is a complete Parquet file written with pyarrow. DuckDB holds the accounts and the statistics
    cache. There is no "zero-copy" versioning and no DuckDB query acceleration.
  - ChromaDB is gone; the document index is LanceDB or Qdrant.
- **Statistics**:
  - Holm is the default correction, and normality checks are diagnostics.
  - ANCOVA has one outcome.
  - Significance brackets use pairwise Welch t or Mann-Whitney tests.
  - Enrich offers min-max, z-score and log(x+1) scaling and three encodings. The elbow and silhouette chart only
    helps choose K.
- **AI**: column-name suggestions come from the language model and are not checked against UMLS, LOINC or SNOMED
  CT. Claude is among the cloud providers that need a consent.
- **Replay**: Fast replay loads the recorded snapshots; Full replay recomputes from the source file.
- **Accounts**:
  - Providers implement `BaseIAMProvider`: local, Better Auth sidecar or disabled.
  - The roles are administrator and researcher.
  - There is no Keycloak or LDAP adapter.
- **Handbook chapters**:
  - page and button names;
  - features that do not exist;
  - the Fast and Full replay descriptions;
  - the transformation lists;
  - the MCAR wording ("no evidence against MCAR").
- **README**:
  - The marketing wording is gone (enterprise-grade, production-ready, zero-lock-in).
  - The unsourced comparison with commercial EDC systems is replaced by a factual summary.
  - The architecture diagram is redrawn (Mermaid).
  - The chapter index lists the existing chapters.
- `.env.example` documents `CMP_GRAPH_NO_FALLBACK`, `CMP_RANDOM_SEED` and `CMP_STRICT_STATE`.
- `tests/test_docs_site.py` also checks the README and the chapters for the regulatory and marketing claims removed
  in M0.

### Fixed: PCA without the `ml` group, offline HTML figures, page names in two messages

- PCA on Enrich runs with `prince`, from the `ml` group, yet it was offered without that group and then failed. It is
  now offered only when `prince` is installed, like FAMD.
- The interactive HTML download of Visualize was described as self-contained but loaded Plotly.js from the internet.
  It now embeds Plotly.js, so the file opens offline (about 4.5 MB). The file is built only when the button is
  clicked, no longer at every rerun of the page.
- Two messages named pages that no longer exist ("RAG & AI Configuration", "Data Insight"). They now point to
  Settings > AI engine.

### Fixed: the study dossier failed with a group comparison in it (Reproduce > Study dossier)

- With a group comparison of Analyze > Hypothesis tests & epidemiology added to the dossier, the page and the Word
  download stopped with `AttributeError: 'list' object has no attribute 'values'`. Counting the imputations of a
  result assumed every comparison was pooled, while the comparison on observed data keeps a list of rows.
- The dossier tests now include an observed group comparison.

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

### Participant flow and study dossier (follow-up programme M3)

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
  (step 8 builds the dossier), the documentation site and the audit report (M3 done).

### Multiple imputation end-to-end (follow-up programme M2)

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
  - The audit report closes S4 and marks M2 done.
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

---

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
