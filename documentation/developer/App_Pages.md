# App Pages Documentation

## about.py

_About page body: what the platform does, third-party notices and contact._

**Imports**:
`streamlit`

### def `app` (about.py)

- **Arguments**: ``

- **Returns**: `None`

No description available.

---

## about.py

_About page._

**Imports**:
`app_pages.about.app`, `ui.components.page_header`

---

## ai_quality.py

_AI quality page (administrators): RAG evaluation and embedding health._

**Imports**:
`app_pages.rag_monitoring.render_rag_monitoring`, `ui.components.page_header`

---

## analysis_config.py

_Analysis configuration widgets of the Analyze page (moved out of utils so the engine side stays UI-free)._

**Imports**:
`streamlit`, `app_pages.clean_dialog.group_warning`, `biostats.multiplicity.UI_OPTIONS`, `ui.components.levels.show_level`, `ui.guidance.analysis.HELP`

### def `grouping_options` (analysis_config.py)

- **Arguments**: `df, numeric_cols, categorical_cols, binary_cols, max_levels`

- **Returns**: `None`

"None" and the columns with fewer than ``max_levels`` values, each once and sorted. A two-level text column is
both categorical and binary, so the column lists overlap; joined without removing the repeats, it was offered
twice.

### def `render_analysis_configuration` (analysis_config.py)

- **Arguments**: `df, numeric_cols, categorical_cols, binary_cols, group_col_options, default_group_col, target_col_options, key_prefix`

- **Returns**: `None`

Renders the Analysis Configuration expander.
Returns a dictionary with the selected configuration.

---

## analysis_diagnostic.py

_Diagnostic accuracy & agreement, a tab of Analyze > Hypothesis tests & epidemiology: three questions, one open at a
time (ui.guidance.measures): the accuracy of a test against a reference standard (sensitivity, specificity, the ROC
curve and its AUC), the agreement of two measurements (Bland-Altman limits and the ICC, on one set of columns), and the
agreement of two raters on categories (Cohen's kappa).

Each follows the page pattern of Survival & regression (app_pages.analysis_layout): nothing is preselected, the plan
says what Run will do, one click runs it, and the result reads the estimate first. What each analysis computes is a
function of its own here (``accuracy_of``, ``kappa_of``), so that the numbers are checked without the page._

**Imports**:
`__future__.annotations`, `collections.abc.Sequence`, `typing.Any`, `numpy`, `pandas`, `plotly.graph_objects`, `streamlit`, `app_pages.analysis_layout`, `app_pages.analysis_guidance_ui.guidance_on`, `app_pages.analysis_layout.Metric`, `biostats._types.CI`, `biostats._types.YOUDEN_BOOTSTRAP`, `biostats.agreement.SLOPE_READING`, `biostats.agreement.SLOPE_WARNING`, `biostats.agreement.bland_altman`, `biostats.diagnostic.YOUDEN_MIN_CLASS`, `biostats.diagnostic.confusion_at`, `biostats.diagnostic.delong_auc`, `biostats.diagnostic.diagnostic_accuracy`, `biostats.diagnostic.youden_cutoff`, `biostats.reliability.NEAR_MISSES`, `biostats.reliability.cohens_kappa`, `biostats.reliability.icc`, `biostats.reliability.koo_li`, `biostats.reliability.landis_koch`, `biostats.reliability.rater_table`, `core.registry.as_code`, `ui.components.empty_state`, `ui.components.error_card`, `ui.components.help_popover`, `ui.guidance.measures`, `ui.guidance.measures.HELP`, `ui.progress.mark_step`, `ui.results.provenance`, `ui.results.remember`, `ui.theme.OKABE_ITO`

### def `_rgba` (analysis_diagnostic.py)

- **Arguments**: `colour: str, alpha: float`

- **Returns**: `str`

No description available.

### def `_as_measured` (analysis_diagnostic.py)

- **Arguments**: `found: dict`

- **Returns**: `dict`

A cut-off found on the reversed marker (lower values pointing to the condition), on the marker's own scale.

### def `accuracy_of` (analysis_diagnostic.py)

- **Arguments**: `df: pd.DataFrame, outcome: str, present: Any, score: str, direction: str, mode: str, cutoff: ComplexType, prevalence: ComplexType`

- **Returns**: `dict[ComplexType]`

The accuracy of ``score`` against ``outcome`` = ``present`` on the rows where both are recorded: the AUC always;
with a cut-off (fixed beforehand, or found on these data by Youden's J), the 2x2 table and its measures.

``direction`` says which way the test points. For "lower", the same functions read the reversed marker, and the
cut-off is given back on the marker's own scale (positive when the test is at most the cut-off): a marker that
falls with the condition had no result (the cut-off found called everyone positive).

### def `_pairs` (analysis_diagnostic.py)

- **Arguments**: `with_it: np.ndarray, values: np.ndarray`

- **Returns**: `tuple[ComplexType]`

Of every pair of one participant with the condition and one without: the shares in which the one with it
has the value pointing further to the condition, the same value, the other. The AUC is the first plus half
the second: with ties, "in 66 of 100 pairs the higher value" was said of 51.

### def `_in_100` (analysis_diagnostic.py)

- **Arguments**: `share: float`

- **Returns**: `str`

A share as "51 of 100 pairs" is said: never "100" for less than all, never "0" for some.

### def `range_drawn` (analysis_diagnostic.py)

- **Arguments**: `clean: pd.DataFrame, outcome: str, present: Any`

- **Returns**: `bool`

Whether the range of a cut-off found on the data is drawn at random: only when each group holds enough rows.
The draws are then part of the result's settings: a result drawn with another seed is another result.

### def `_bound` (analysis_diagnostic.py)

- **Arguments**: `lower: bool`

- **Returns**: `str`

No description available.

### def `_roc_figure` (analysis_diagnostic.py)

- **Arguments**: `truth: np.ndarray, marker: np.ndarray, auc: dict, point: ComplexType, label: ComplexType`

- **Returns**: `go.Figure`

The ROC curve, square, with the point of the 2x2 table when a cut-off is set (the table's own sensitivity and
specificity: the nearest threshold of the curve was drawn, which is another point for a typed cut-off).

### def `_accuracy` (analysis_diagnostic.py)

- **Arguments**: `df: pd.DataFrame`

- **Returns**: `None`

No description available.

### def `_share` (analysis_diagnostic.py)

- **Arguments**: `value: Any, ci: Any`

- **Returns**: `str`

A share with its interval, as a sentence writes it: "40.8% (95% CI 34.5% to 47.6%)", or the share alone when
it has no interval ("n/a (95% CI n/a% to n/a%)" stood for a predictive value without a value).

### def `_measure` (analysis_diagnostic.py)

- **Arguments**: `name: str, value: Any, ci: Any`

- **Returns**: `tuple[ComplexType]`

A row of the table of every measure: a share as a percentage, as the headline writes it (the headline read
86.1% and the table 0.861), a ratio to three significant figures; its interval with the same decimals.

### def `_show_accuracy` (analysis_diagnostic.py)

- **Arguments**: `res: dict[ComplexType]`

- **Returns**: `None`

No description available.

### def `_swap_measurements` (analysis_diagnostic.py)

- **Arguments**: ``

- **Returns**: `None`

No description available.

### def `_limits` (analysis_diagnostic.py)

- **Arguments**: `ba: Any`

- **Returns**: `list[str]`

The two limits of agreement, then the bounds of the lower limit's interval and of the upper limit's, written
with the same decimals: the headline, its caption, its sentence and the figure's legend give a number one way
(the upper limit read "20.17" over "95% CI 19.4 to 21.0", and "20.2" in the legend).

### def `_bland_altman_figure` (analysis_diagnostic.py)

- **Arguments**: `ba: Any, first: str, second: str`

- **Returns**: `go.Figure`

No description available.

### def `_measurements` (analysis_diagnostic.py)

- **Arguments**: `df: pd.DataFrame`

- **Returns**: `None`

No description available.

### def `_show_bland_altman` (analysis_diagnostic.py)

- **Arguments**: `ba: Any, first: str, second: str, delta: ComplexType`

- **Returns**: `str`

The limits of agreement; gives back how to read them (said once, after the last result of the Run).

### def `_show_icc` (analysis_diagnostic.py)

- **Arguments**: `res: Any, rows: int`

- **Returns**: `str`

The intraclass correlation; gives back how to read it (said once, after the last result of the Run).

### def `kappa_of` (analysis_diagnostic.py)

- **Arguments**: `df: pd.DataFrame, first: str, second: str`

- **Returns**: `dict[ComplexType]`

Cohen's kappa of two raters' categories on the participants rated by both, with the cross-table it is computed
on. With ``weights`` (ordered grades), ``order`` gives the grades from the lowest to the highest: the weights
read distances in that order.

### def `_raters` (analysis_diagnostic.py)

- **Arguments**: `df: pd.DataFrame`

- **Returns**: `None`

No description available.

### def `_show_kappa` (analysis_diagnostic.py)

- **Arguments**: `result: dict[ComplexType], first: str, second: str`

- **Returns**: `None`

No description available.

### def `run_diagnostic_accuracy` (analysis_diagnostic.py)

- **Arguments**: `df: pd.DataFrame, analyzer: Any`

- **Returns**: `None`

The tab: its Question selector, then the analysis it names. ``analyzer`` is the page's column profile (the
entries read the columns themselves: a whole-number score is a measurement here).

---

## analysis_extras.py

_Paired, rates & stratified, a tab of Analyze > Hypothesis tests & epidemiology: four questions, one open at a time
(ui.guidance.measures): two measurements of the same participants (a number: paired t-test or Wilcoxon signed-rank; yes
or no: McNemar), one group against a reference (a mean, or a share), an association within strata
(Mantel-Haenszel), and rates from counts (a calculator).

The analyses of the dataset follow the page pattern of Survival & regression (app_pages.analysis_layout): nothing is
preselected, the plan says what Run will do, one click runs it, the result reads the estimate first and is kept with
what it was computed on. The calculators read the numbers typed: their fields start empty, they have no button, and
their result says where its numbers came from._

**Imports**:
`__future__.annotations`, `typing.Any`, `numpy`, `pandas`, `streamlit`, `app_pages.analysis_layout`, `app_pages.analysis_guidance_ui.guidance_on`, `app_pages.analysis_layout.Metric`, `biostats._types.TestResult`, `biostats.paired.mcnemar_test`, `biostats.paired.one_sample_proportion`, `biostats.paired.one_sample_t`, `biostats.paired.one_sample_wilcoxon`, `biostats.paired.paired_t`, `biostats.paired.paired_table`, `biostats.paired.wilcoxon_signed_rank`, `biostats.rates.incidence_rate`, `biostats.rates.mantel_haenszel`, `biostats.rates.rate_ratio`, `biostats.rates.standardised_ratio`, `biostats.reliability.category`, `biostats.reliability.category_order`, `core.registry.as_code`, `ui.components.empty_state`, `ui.components.error_card`, `ui.components.help_popover`, `ui.components.mi_mode.mi_token`, `ui.components.mi_mode.show_mi_notes`, `ui.guidance.measures`, `ui.guidance.measures.HELP`, `ui.progress.mark_step`, `ui.results.provenance`, `ui.results.remember`, `ui.state.get_state`

### def `_not_pooled` (analysis_extras.py)

- **Arguments**: `analysis: str, instead: ComplexType`

- **Returns**: `None`

The reason a test has no pooling rule, and what to run instead (biostats.mi.not_poolable); ``instead``
where the catalogue's alternative is another analysis's (it names the paired test for one group too).

### def `_held` (analysis_extras.py)

- **Arguments**: `columns: list[str], mi: Any`

- **Returns**: `list[str]`

With imputed data, a pooled analysis offers the columns of the imputed datasets only: one created after the
imputation is not in them.

### def `_rows_read` (analysis_extras.py)

- **Arguments**: `result: Any, mi: Any`

- **Returns**: `int`

The rows a pooled test read: those of its result (every row of the population was stamped as analysed, where
a column left out of the imputation lacks 150 values).

### def `_pooled` (analysis_extras.py)

- **Arguments**: `stored: Any`

- **Returns**: `tuple[ComplexType]`

A stored test as it is shown: the pooled test with its notes, or the observed one.

### def `_paired_number` (analysis_extras.py)

- **Arguments**: `df: pd.DataFrame`

- **Returns**: `None`

No description available.

### def `_named` (analysis_extras.py)

- **Arguments**: `name: str`

- **Returns**: `str`

A test's name inside brackets: "Paired t-test (pooled over 3 imputations)" reads "paired t-test, pooled over 3
imputations", without brackets within brackets.

### def `_show_change` (analysis_extras.py)

- **Arguments**: `result: TestResult, df: pd.DataFrame, before: str, after: str, kind: str`

- **Returns**: `None`

No description available.

### def `_paired_yes_no` (analysis_extras.py)

- **Arguments**: `df: pd.DataFrame`

- **Returns**: `None`

No description available.

### def `_show_yes_no` (analysis_extras.py)

- **Arguments**: `result: TestResult, table: pd.DataFrame, before: str, after: str`

- **Returns**: `None`

No description available.

### def `_one_mean` (analysis_extras.py)

- **Arguments**: `df: pd.DataFrame`

- **Returns**: `None`

No description available.

### def `_show_mean` (analysis_extras.py)

- **Arguments**: `result: TestResult, column: str, reference: float, kind: str, rows: int, lacking: int`

- **Returns**: `None`

No description available.

### def `_share_metrics` (analysis_extras.py)

- **Arguments**: `result: TestResult, with_it: int, counted: int, reference: float, what: str, where: str`

- **Returns**: `None`

No description available.

### def `_one_share` (analysis_extras.py)

- **Arguments**: `df: pd.DataFrame`

- **Returns**: `None`

No description available.

### def `_levels_as_text` (analysis_extras.py)

- **Arguments**: `df: pd.DataFrame, column: ComplexType`

- **Returns**: `list[str]`

No description available.

### def `_strata` (analysis_extras.py)

- **Arguments**: `df: pd.DataFrame`

- **Returns**: `None`

No description available.

### def `_ratio` (analysis_extras.py)

- **Arguments**: `value: Any`

- **Returns**: `str`

A ratio to three significant figures: at two decimals, a lower bound of 0.996 read "1.00" beside p = 0.051.

### def `_ratio_ci` (analysis_extras.py)

- **Arguments**: `ci: Any`

- **Returns**: `str`

No description available.

### def `_show_strata` (analysis_extras.py)

- **Arguments**: `res: Any, outcome: str, event: str, compare: str, exposed: str, reference: str, stratum: str, total: int`

- **Returns**: `None`

No description available.

### def `_count` (analysis_extras.py)

- **Arguments**: `label: str, key: str, column: Any, example: str`

- **Returns**: `ComplexType`

``shown``: the label is drawn (the second row of a table of fields keeps it for screen readers only: its row
names the group and the first row's labels head the columns).

### def `_time` (analysis_extras.py)

- **Arguments**: `label: str, key: str, column: Any, example: str`

- **Returns**: `ComplexType`

No description available.

### def `_rate_metric` (analysis_extras.py)

- **Arguments**: `label: str, rate: Any, per: int, unit: str`

- **Returns**: `Metric`

A rate with its exact interval; the numbers typed stand in the fields and in the line under the result.

### def `_more_or_fewer` (analysis_extras.py)

- **Arguments**: `ratio: float`

- **Returns**: `str`

No description available.

### def `_rates` (analysis_extras.py)

- **Arguments**: `df: pd.DataFrame`

- **Returns**: `None`

No description available.

### def `_ratio_metric` (analysis_extras.py)

- **Arguments**: `ratio: Any`

- **Returns**: `Metric`

The rate ratio with its exact interval and test; without an event in the reference group it has no upper
bound, and its lower end is what the data say.

### def `_ratio_words` (analysis_extras.py)

- **Arguments**: `ratio: Any`

- **Returns**: `str`

The rate ratio in a sentence, as the standardised ratio's: which group has the higher rate, and by how
much.

### def `_expected` (analysis_extras.py)

- **Arguments**: ``

- **Returns**: `None`

No description available.

### def `_paired` (analysis_extras.py)

- **Arguments**: `df: pd.DataFrame`

- **Returns**: `None`

No description available.

### def `_one_group` (analysis_extras.py)

- **Arguments**: `df: pd.DataFrame`

- **Returns**: `None`

No description available.

### def `run_paired_and_rates` (analysis_extras.py)

- **Arguments**: `df: pd.DataFrame, analyzer: Any`

- **Returns**: `None`

The tab: its Question selector, then the analysis it names. With imputed data (``mi``), the paired and the
one-sample t-tests are pooled over the imputed datasets; every other analysis says which rows it reads.

---

## analysis_guidance_ui.py

_Rendering of the guidance of Analyze > Hypothesis tests & epidemiology (rules in ``ui.guidance.analysis``).

Everything here is shown only while the page's Guidance switch is on, so experienced users can keep
the page compact._

**Imports**:
`__future__.annotations`, `collections.abc.Callable`, `collections.abc.Mapping`, `pandas`, `streamlit`, `ui.components.page_link`, `ui.components.status_chip`, `ui.guidance.analysis.HELP`, `ui.guidance.analysis.TAB_GUIDE`, `ui.guidance.analysis.TEST_NOTES`, `ui.guidance.analysis.AnalysisPlan`, `ui.guidance.analysis.AncovaCheck`

### def `guidance_toggle` (analysis_guidance_ui.py)

- **Arguments**: ``

- **Returns**: `bool`

No description available.

### def `guidance_on` (analysis_guidance_ui.py)

- **Arguments**: ``

- **Returns**: `bool`

No description available.

### def `render_orientation` (analysis_guidance_ui.py)

- **Arguments**: `open_analysis: Callable[ComplexType], selectors: Mapping[ComplexType]`

- **Returns**: `None`

Which tab answers which question, as a map that opens what it names: each button opens its tab, and the analysis
on it where the tab holds several (``selectors``: a tab's label, the key of its Question selector and the
entries, id to label). ``open_analysis`` is the page's callback: a tab or a selector is set before it is drawn.

### def `_advice` (analysis_guidance_ui.py)

- **Arguments**: `level: str, text: str`

- **Returns**: `None`

No description available.

### def `render_plan` (analysis_guidance_ui.py)

- **Arguments**: `plan: AnalysisPlan`

- **Returns**: `None`

What Run will do, checked against the data, before anything is computed.

### def `render_ancova_check` (analysis_guidance_ui.py)

- **Arguments**: `check: AncovaCheck`

- **Returns**: `None`

No description available.

### def `render_test_note` (analysis_guidance_ui.py)

- **Arguments**: `test: str`

- **Returns**: `None`

No description available.

---

## analysis_layout.py

_What the analysis tabs share on the screen (Hypothesis tests & epidemiology: Diagnostic accuracy & agreement; Paired,
rates & stratified), in the pattern of Survival & regression: the selector of one analysis among a tab's entries, the
plan shown before a run, one Run, the headline of a result (the estimate with its interval first, then the test, then
the counts), the place of a result that is not there yet, the line of an analysis under imputed data and the line of
a calculator's result.

One analysis is open at a time; each keeps its choices (``persist_state="page"``) and its result (ui.results) while
another is open. A link can open an entry: the page declares its tabs and entries (``ui.nav.PageDef.tabs``), and the
selectors read the link's labels as their default._

**Imports**:
`__future__.annotations`, `math`, `collections.abc.Callable`, `collections.abc.Mapping`, `collections.abc.Sequence`, `dataclasses.dataclass`, `typing.Any`, `streamlit`, `app_pages.clean_categories.take_done`, `app_pages.clean_dialog.clean_actions`, `biostats._types.TestResult`, `biostats.assumptions.read_assumptions`, `ui.nav`, `ui.components.glossary_popover`, `ui.components.help_popover`, `ui.components.status_chip`, `ui.components.text.plain`, `ui.guidance.interpretation.how_to_read`, `ui.guidance.measures.HELP`, `ui.guidance.measures.Entry`, `ui.guidance.measures.number`, `ui.guidance.measures.decimals`, `ui.guidance.outcomes.ModelPlan`, `ui.results.MISSING`, `ui.results.STALE_CONFIG`, `ui.results.recall`, `ui.results.status`, `ui.results.stored_settings`, `ui.state.get_state`

### class `Metric` (analysis_layout.py)

No description available.

### def `p_text` (analysis_layout.py)

- **Arguments**: `p: Any`

- **Returns**: `str`

A p-value as a sentence writes it after "p": "< 0.001", "= 0.017"; "n/a" when it was not computed. Next to
0.05 a fourth decimal tells 0.0499 from 0.0501, which both read "= 0.050".

### def `num` (analysis_layout.py)

- **Arguments**: `value: Any, digits: int`

- **Returns**: `str`

A number of a result, by the rule the plans write theirs with (ui.guidance.measures.number): ``digits``
significant figures for its decimals, thousands grouped, the whole part kept, no exponent; "n/a" without a
value.

### def `span` (analysis_layout.py)

- **Arguments**: `lower: Any, upper: Any, digits: int`

- **Returns**: `str`

"0.902 to 0.953": two bounds written with the same decimals, those the bound nearer 0 needs for ``digits``
significant figures (each rounded on its own, an interval read "-9 to -7.38"); a percentage with one decimal.

### def `interval` (analysis_layout.py)

- **Arguments**: `ci: Any, digits: int`

- **Returns**: `str`

"95% CI 0.902 to 0.953" from an interval object; "" without one, or without bounds ("95% CI n/a% to n/a%"
stood beside a predictive value that has no value).

### def `with_interval` (analysis_layout.py)

- **Arguments**: `estimate: Any, ci: Any, digits: int`

- **Returns**: `tuple[ComplexType]`

An estimate and its interval written with the same decimals ("3.125" and "95% CI 0.996 to 12.935"): rounded
each on its own, a typical change read "-5.5, 95% CI -6.00 to -4.50" and a rate ratio "3.12, 0.996 to 12.935".

### def `alike` (analysis_layout.py)

- **Arguments**: ``

- **Returns**: `list[str]`

Numbers that are read together, written with the same decimals: those the one nearest 0 needs for ``digits``
significant figures (two limits of agreement read "-8.19 to 20.17" over "95% CI 19.4 to 21.0", and "20.2" in the
figure's legend). A number without a value reads "n/a".

### def `percent` (analysis_layout.py)

- **Arguments**: `value: Any, digits: int`

- **Returns**: `str`

No description available.

### def `headline` (analysis_layout.py)

- **Arguments**: `metrics: Sequence[Metric], words: str, warnings: int`

- **Returns**: `None`

The result first: the estimate with its interval, the test and the counts, then one sentence saying what was
compared and in which direction.

### def `linked` (analysis_layout.py)

- **Arguments**: `depth: int, labels: Mapping[ComplexType]`

- **Returns**: `ComplexType`

The id a link to this page names at ``depth`` of its path (the tab is 0, its entry 1, the entry's second choice
2), among ``labels`` (id to label); None when the page was not opened by such a link.

### def `_start` (analysis_layout.py)

- **Arguments**: `key: str, value: str`

- **Returns**: `dict[ComplexType]`

A selector's ``default``, given only while it holds no value: a button of the page may have set its key (the
orientation, a suggestion), and a widget given both a default and a value set in the session is reported.

### def `choose_entry` (analysis_layout.py)

- **Arguments**: `entries: Sequence[Entry]`

- **Returns**: `Entry`

The Question selector of a tab: one entry is open at a time. Under it, what the entry answers and its terms.
(The way to the test chooser is in the map above the tabs: said again under every entry, 250 px lower, it was a
row of its own that named nothing new.)

### def `choose` (analysis_layout.py)

- **Arguments**: `label: str, options: Mapping[ComplexType]`

- **Returns**: `str`

A second choice of an entry (the kind of measurement, what a calculator computes): ids kept in the state, their
labels shown; a link names it at ``depth`` of its path.

### def `depends` (analysis_layout.py)

- **Arguments**: `key: str, on: Any, value: Any`

- **Returns**: `None`

Give the widget ``key`` the value offered for what it depends on (``on``: the column chosen above it, with its
levels), once each time that changes: the level that says "present" for a reference standard, no cut-off for
another test, no reference value for another measurement. Call it before the widget is drawn, and give the
widget no default of its own.

The value goes through the widget's state because that is what reaches both sides. A default (``index=``,
``value=``) is read by the script, not by a widget the browser already shows: the field went on reading "Choose
the value" over a result computed for Yes. A value cleared by removing its key stayed on the screen the same
way. And a value kept from another column was reported ("PPV at prevalence 0.502" for a condition 10.9% of
participants have).

### def `kept` (analysis_layout.py)

- **Arguments**: `name: str, key: tuple, compute: Callable[ComplexType]`

- **Returns**: `Any`

A plan kept until the data, the analysis population or ``key`` (the panel's choices) changes: computed at every
rerun, it would be read again at each click.

### def `plan_box` (analysis_layout.py)

- **Arguments**: `plan: ModelPlan`

- **Returns**: `None`

What Run will do, checked on the data: a status, each step with its rows, the advice. ``clean`` gives, for a
variable whose values are written several ways, the widget it was chosen in (the arguments of
app_pages.clean_dialog.clean_actions): it is then cleaned here, and replaced by its copy.

### def `checks` (analysis_layout.py)

- **Arguments**: `plan: ModelPlan`

- **Returns**: `None`

With Guidance off the plan is not drawn: what would make the result wrong, or impossible, is still said.

### def `run_button` (analysis_layout.py)

- **Arguments**: `key: str`

- **Returns**: `bool`

The one Run of an analysis of the dataset. Disabled only while nothing can be computed: advice never blocks.
Its help shows while it is disabled, and says what the plan says is missing (with Guidance off the plan is not
drawn, and nothing on the screen said why Run was greyed). On a button just clicked a help stays open as long
as the button keeps the focus, and lay over the last lines of the plan: an enabled Run has none.

### def `result_place` (analysis_layout.py)

- **Arguments**: `key: str, config: Any`

- **Returns**: `ComplexType`

The stored result of these choices, or None with a word on why none is shown: nothing was run yet (Guidance on),
or the last result belongs to other choices (it did not vanish). A change of the data or of the population is
announced by ``recall``. ``silent``: the second result of one Run, whose first has said it. ``words``: the
page's labels for the option values among the settings ("names" is "Names without an order" on the screen).

### def `imputed_line` (analysis_layout.py)

- **Arguments**: `text: str`

- **Returns**: `None`

What an analysis reads while imputed data are selected in the sidebar: pooled, or which observed rows.

### def `typed_line` (analysis_layout.py)

- **Arguments**: `numbers: str`

- **Returns**: `None`

Under a calculator's result, in place of the line "computed on": where its numbers came from.

### def `advice_lines` (analysis_layout.py)

- **Arguments**: `advice: Sequence[Any]`

- **Returns**: `None`

The checks of the numbers typed in a calculator.

### def `warn` (analysis_layout.py)

- **Arguments**: `warnings: Sequence[str]`

- **Returns**: `None`

A result's own warnings, each once (the result card never showed them: 34 pairs left out by a rank test, fewer
than 10 pairs that changed).

### def `details` (analysis_layout.py)

- **Arguments**: `result: TestResult`

- **Returns**: `None`

Folded under a test's headline: the statistic, the effect sizes, the assumption checks (diagnostics, not
gates) and the method. ``more``: what is to know of this analysis whatever the data, said once where the result
is read (in the plan it was said again at every run).

---

## analysis_pooled.py

_Analyze page with multiply imputed data (follow-up programme M2): group comparisons and ANCOVA run on the m
imputed datasets of the analysis population and pooled with Rubin's rules (biostats.mi.analyze_mi).

Group comparison, per variable:
- numeric, two groups: Welch t-test, the mean difference pooled (Barnard-Rubin df);

- numeric, three or more groups: the group means compared without assuming equal variances, as Welch's ANOVA on
  the observed data (biostats.linear.welch_means: each group's own variance, Welch-type df), the means pooled and
  the group effect tested by D1 with Welch's df; Welch's ANOVA itself has no pooling rule;

- the analysis can be chosen instead (*Choose the method*): Student's t-test for two groups, the equal-variance
  linear model (one-way ANOVA) for three or more, each with a warning;

- two categories, two groups: odds ratio and risk ratio (log scale) and risk difference pooled, the
  chi-square by D2;

- other categorical variables: the chi-square test of independence by D2.
Rank tests and Fisher's exact test have no pooling rule: the page gives the reason and the alternative._

**Imports**:
`__future__.annotations`, `dataclasses.dataclass`, `typing.Any`, `pandas`, `streamlit`, `app_pages.clean_dialog.group_warning`, `biostats.contingency.TwoByTwo`, `biostats.contingency.analyze_twobytwo`, `biostats.contingency.chi_square_independence`, `biostats.contingency.detect_event_level`, `biostats.contingency.detect_reference_level`, `biostats.inference.compare_groups`, `biostats.linear.ancova`, `biostats.linear.welch_means`, `biostats.mi.analyze_mi`, `biostats.mi.not_poolable`, `biostats.multiplicity.UI_OPTIONS`, `biostats.multiplicity.adjust`, `ui.components.error_card`, `ui.components.result_card`, `ui.components.levels.show_level`, `ui.components.mi_mode.mi_token`, `ui.components.mi_mode.show_mi_notes`, `ui.components.mi_mode.show_pooled_terms`, `ui.components.mi_mode.show_tests`, `ui.progress.mark_step`, `ui.results.provenance`, `ui.results.recall`, `ui.results.remember`, `ui.state.get_state`

### class `Pooled` (analysis_pooled.py)

One variable compared across the groups: the kind of analysis, the pooled result and its orientation.

### def `_levels` (analysis_pooled.py)

- **Arguments**: `series: pd.Series`

- **Returns**: `list[str]`

No description available.

### def `_in_groups` (analysis_pooled.py)

- **Arguments**: `data: pd.DataFrame, group: str, levels: list[str]`

- **Returns**: `pd.DataFrame`

No description available.

### def `_two_per_group` (analysis_pooled.py)

- **Arguments**: `data: pd.DataFrame, group: str, levels: list[str]`

- **Returns**: `None`

Each group keeps its own variance, which a group of one value does not have.

### def `_columns` (analysis_pooled.py)

- **Arguments**: `mi, analyzer`

- **Returns**: `tuple[ComplexType]`

Numeric and categorical columns of the imputed data, and the grouping candidates.

### def `compare_pooled` (analysis_pooled.py)

- **Arguments**: `mi, group: str, levels: list[str], target: str`

- **Returns**: `Pooled`

Compare ``target`` across the ``levels`` of ``group`` on every imputed dataset and pool the result.

``haldane="always"`` adds 0.5 to every cell of every 2x2 table (a sensitivity analysis): without it, a zero
cell in some imputed datasets leaves the odds and risk ratios unpooled. ``two_groups`` is the t-test of a
numeric variable in two groups: Welch's (recommended) or Student's (equal variances), both pooled the same way.
``more_groups`` compares a numeric variable in three groups or more: without assuming equal variances
(recommended; Welch-type) or with the equal-variance linear model (one-way ANOVA), both tested by D1.

### def `_summary_row` (analysis_pooled.py)

- **Arguments**: `target: str, item: Pooled, group: str`

- **Returns**: `dict[ComplexType]`

No description available.

### def `_show_summary` (analysis_pooled.py)

- **Arguments**: `results: dict[ComplexType], errors: dict[ComplexType], group: str, correction: str`

- **Returns**: `None`

No description available.

### def `_twobytwo_frame` (analysis_pooled.py)

- **Arguments**: `res`

- **Returns**: `pd.DataFrame`

No description available.

### def `_show_complete_twobytwo` (analysis_pooled.py)

- **Arguments**: `cc`

- **Returns**: `None`

No description available.

### def `_show_means` (analysis_pooled.py)

- **Arguments**: `means: pd.DataFrame`

- **Returns**: `None`

No description available.

### def `_show_complete_groups` (analysis_pooled.py)

- **Arguments**: `cc, target: str, group: str`

- **Returns**: `None`

The complete-case comparison: the group means, and Welch's ANOVA (or the one-way ANOVA when chosen).

### def `_show_detail` (analysis_pooled.py)

- **Arguments**: `target: str, item: Pooled, group: str`

- **Returns**: `None`

No description available.

### def `run_pooled_comparison` (analysis_pooled.py)

- **Arguments**: `mi, analyzer`

- **Returns**: `None`

No description available.

### def `run_pooled_ancova` (analysis_pooled.py)

- **Arguments**: `mi, analyzer`

- **Returns**: `None`

No description available.

---

## analysis_survival.py

_Survival & regression page body, organised by the outcome (audit S11): the time to an event (Kaplan-Meier curves,
the log-rank test and the Cox model), a binary outcome (logistic regression) or a count of events (Poisson or
negative binomial regression). For each, the user chooses what is compared and what it is adjusted for; the plan
checked on the data says what Run will do (ui.guidance.outcomes); one click runs it; the result reads effect first,
then the model and its checks (app_pages.analysis_survival_results). Nothing is decided for the user: every default
is a visible choice.

With multiply imputed data (``mi``), every analysis is run on the m imputed datasets and pooled
(biostats.mi.analyze_mi): Kaplan-Meier curves on the cloglog scale with the median read from the pooled
curve, the log-rank test by D2, hazard, odds and rate ratios with Rubin's rules and a D1 test per factor._

**Imports**:
`__future__.annotations`, `collections.abc.Sequence`, `typing.Any`, `pandas`, `streamlit`, `app_pages.analysis_survival_results.Comparison`, `app_pages.analysis_survival_results.curve_rows`, `app_pages.analysis_survival_results.show_count`, `app_pages.analysis_survival_results.show_logistic`, `app_pages.analysis_survival_results.show_time_to_event`, `app_pages.clean_categories.take_done`, `app_pages.clean_dialog.clean_actions`, `biostats.plan.run`, `biostats.plan.specify`, `biostats.survival.EVENT_WORDS`, `enrich.recode.MAX_VALUES`, `ui.components.error_card`, `ui.components.glossary_popover`, `ui.components.page_link`, `ui.components.status_chip`, `ui.components.mi_mode.mi_token`, `ui.components.text.plain`, `ui.guidance.outcomes.BINARY`, `ui.guidance.outcomes.COUNT`, `ui.guidance.outcomes.FAMILIES`, `ui.guidance.outcomes.HELP`, `ui.guidance.outcomes.NONE`, `ui.guidance.outcomes.OUTCOME`, `ui.guidance.outcomes.OUTCOME_TYPES`, `ui.guidance.outcomes.TIME_TO_EVENT`, `ui.guidance.outcomes.ModelPlan`, `ui.guidance.outcomes.chosen`, `ui.guidance.outcomes.covariates_of`, `ui.guidance.outcomes.plan_binary`, `ui.guidance.outcomes.plan_count`, `ui.guidance.outcomes.plan_time_to_event`, `ui.progress.mark_step`, `ui.results.recall`, `ui.results.remember`, `ui.state.get_state`

### def `_group_candidates` (analysis_survival.py)

- **Arguments**: `df: pd.DataFrame, analyzer, max_levels: int`

- **Returns**: `list[str]`

The variables a model can compare by or adjust for: 2 to ``max_levels`` values (a number, 3 at most). A text
variable's values are counted with spaces and capitals set aside, as Data > Clean merges its spellings: the
tutorial's `sex`, written 10 ways, holds 4 values so, and is offered, its plan then offering to clean it.

### def `_variables` (analysis_survival.py)

- **Arguments**: `df: pd.DataFrame, analyzer, mi`

- **Returns**: `tuple[ComplexType]`

The numeric, binary and grouping variables the panels offer; with imputed data, only those it holds: a variable
created after the imputation is not in it, and reading it stopped the page.

### def `_event_index` (analysis_survival.py)

- **Arguments**: `levels: list[str]`

- **Returns**: `int`

The level a binary outcome's event most likely is: the first that reads as an event (Yes, 1, Died, Event: the
words the event indicator reads), else the last sorted. The last sorted alone took "Survived" for Died/Survived.

### def `_default_index` (analysis_survival.py)

- **Arguments**: `options: list[str], hints: tuple[ComplexType]`

- **Returns**: `int`

Index of the first option whose name contains a hint (e.g. 'time', 'died'), else 0.

### def `_candidates` (analysis_survival.py)

- **Arguments**: `df: pd.DataFrame, numeric: Sequence[str], groups: Sequence[str], exclude: Sequence[str]`

- **Returns**: `list[str]`

What a model can compare by or adjust for: numeric variables, and categorical ones of 2 to 8 levels.

### def `_compare_and_adjust` (analysis_survival.py)

- **Arguments**: `candidates: list[str]`

- **Returns**: `tuple[ComplexType]`

The second row of every outcome: what is compared, and what it is adjusted for.

### def `_render_plan` (analysis_survival.py)

- **Arguments**: `plan: ModelPlan`

- **Returns**: `None`

What Run will do, checked on the data (while Guidance is on): a status, each step's rows, the advice. ``keys``:
the outcome's "Compare by" and "Adjust for" widgets, through which a variable whose values are written several
ways is cleaned here, then replaced by its copy (app_pages.clean_dialog.clean_actions).

### def `_offered` (analysis_survival.py)

- **Arguments**: `candidates: Sequence[str], mi`

- **Returns**: `dict[ComplexType]`

What a variable's clean copy can replace it with: the candidates, those of the imputed datasets under a multiple
imputation (a copy written after it is not in them).

### def `_failed` (analysis_survival.py)

- **Arguments**: `what: str, exc: Exception`

- **Returns**: `None`

An error card whose first line says why: the models' messages are written for the reader (as text).

### def `_kept` (analysis_survival.py)

- **Arguments**: `name: str, key: tuple, compute`

- **Returns**: `None`

A plan or a count of rows kept until the data, the analysis population or ``key`` (the panel's choices, the
imputation) changes: computed on every rerun, a plan took 0.5 to 0.7 s on 100,000 rows, at each click.

### def `_planned` (analysis_survival.py)

- **Arguments**: `mi, frame: pd.DataFrame, columns: Sequence[str]`

- **Returns**: `tuple[ComplexType]`

The rows the plan reads: the observed data, or the first imputed dataset when the analysis is pooled.

### def `run_survival` (analysis_survival.py)

- **Arguments**: `df: pd.DataFrame, analyzer`

- **Returns**: `None`

No description available.

### def `_comparison` (analysis_survival.py)

- **Arguments**: `mi, df: pd.DataFrame, compare: ComplexType, adjust: Sequence[str], result`

- **Returns**: `Comparison`

How a result reads: the comparison's values in the data the model read, and its reference level.

### def `left_out_of` (analysis_survival.py)

- **Arguments**: `result`

- **Returns**: `tuple[ComplexType]`

The rows a survival analysis left out for another reason than a missing value, and the words for them: the rows
of a group too small to draw, and the rows with a follow-up time below zero. They are counts of participants
with a characteristic, where the rows that lack a value are given exactly: called missing values, a shared flow
would have shown them as they are. A pooled result counts the most that an imputed dataset left out: a group
read from imputed values is too small to draw in some datasets only.

### def `analysed_by` (analysis_survival.py)

- **Arguments**: `result`

- **Returns**: `int`

The rows an analysis read: a model's, or those of the curves drawn. A pooled result's are the fewest that one of
its imputed datasets read. Every row of the data was called analysed: where the event is missing for 20 rows
of 400 and no imputation fills it, the page said 400 beside curves of 380. The rows a dataset could not read
for a value no imputation fills are missing values, and are now counted as such.

### def `_fit_curves` (analysis_survival.py)

- **Arguments**: `df, mi, key, config, time_col, event_col, group`

- **Returns**: `None`

Kaplan-Meier curves and the log-rank test, computed from their spec (biostats.plan) and stored with it.

### def `_fit_cox` (analysis_survival.py)

- **Arguments**: `df, mi, key, config, time_col, event_col, compare, adjust`

- **Returns**: `None`

The Cox model, computed from its spec and stored with it; pooled, the proportional-hazards test runs on the
first imputed datasets only.

### def `_time_to_event` (analysis_survival.py)

- **Arguments**: `df, numeric, binary, groups`

- **Returns**: `None`

No description available.

### def `_binary` (analysis_survival.py)

- **Arguments**: `df, numeric, binary, groups`

- **Returns**: `None`

No description available.

### def `_count` (analysis_survival.py)

- **Arguments**: `df, numeric, binary, groups`

- **Returns**: `None`

No description available.

---

## analysis_survival_results.py

_How the results of the Survival & regression page read: the effect first (the comparison's ratios with their
interval and p-value), then the figure (Kaplan-Meier curves with the numbers at risk, or the model's ratios on a log
scale), the warnings of the fit, then every term of the model and its checks (ui.guidance.outcomes.diagnostics), and
under each stored result what it was computed on. Observed and pooled (multiply imputed) results read the same way._

**Imports**:
`__future__.annotations`, `html`, `math`, `collections.abc.Callable`, `collections.abc.Sequence`, `dataclasses.dataclass`, `typing.Any`, `numpy`, `pandas`, `plotly.graph_objects`, `streamlit`, `biostats.plan.KINDS`, `biostats.regression.APPARENT_AUC_NOTE`, `biostats.survival.event_indicator`, `core.registry.as_code`, `ui.components.help_popover`, `ui.components.status_chip`, `ui.components.mi_mode.pooled_frame`, `ui.components.mi_mode.show_mi_notes`, `ui.components.mi_mode.show_pooled_terms`, `ui.components.mi_mode.show_tests`, `ui.components.text.plain`, `ui.guidance.outcomes.HELP`, `ui.guidance.outcomes.ComparisonTerm`, `ui.guidance.outcomes.comparison_terms`, `ui.guidance.outcomes.diagnostics`, `ui.guidance.outcomes.listing`, `ui.results.provenance`, `ui.theme.OKABE_ITO`

### class `Comparison` (analysis_survival_results.py)

What a model's result is read by: the comparison, its values in the data the model read, the adjustment.

**Methods:**

- **terms**(`self, terms: Sequence[str]`) -> `list[ComparisonTerm]`
  > No description available.

- **words**(`self`) -> `str`
  > No description available.

### class `Metric` (analysis_survival_results.py)

No description available.

### class `Curve` (analysis_survival_results.py)

No description available.

### def `_p` (analysis_survival_results.py)

- **Arguments**: `p: Any`

- **Returns**: `str`

No description available.

### def `_ratio` (analysis_survival_results.py)

- **Arguments**: `value: Any`

- **Returns**: `str`

No description available.

### def `_median` (analysis_survival_results.py)

- **Arguments**: `value: ComplexType`

- **Returns**: `str`

The median survival time, or 'not reached' when the curve stays above 0.5: in the time's unit, without an
exponent (1,119 days, not 1.12e+03), to two decimals at most (a midpoint ends in .5; 2.37 years).

### def `_table` (analysis_survival_results.py)

- **Arguments**: `df: pd.DataFrame, formats: dict`

- **Returns**: `None`

No description available.

### def `_effects` (analysis_survival_results.py)

- **Arguments**: `table: pd.DataFrame, ratio: str, comparison: Comparison, name: str`

- **Returns**: `list[Metric]`

The comparison's ratios, one metric per term (the first level is the reference).

### def `_headline` (analysis_survival_results.py)

- **Arguments**: `metrics: list[Metric], words: str, warnings: int`

- **Returns**: `None`

The result first: the comparison's effect, the test and the counts, then what the model is adjusted for.

### def `_rgba` (analysis_survival_results.py)

- **Arguments**: `colour: str, alpha: float`

- **Returns**: `str`

No description available.

### def `_ticks` (analysis_survival_results.py)

- **Arguments**: `top: float, count: int`

- **Returns**: `np.ndarray`

Round ticks from 0 to ``top``, about ``count`` of them: 1, 2, 2.5 or 5 times a power of ten apart (1,450 days:
every 250, not every 500).

### def `curve_rows` (analysis_survival_results.py)

- **Arguments**: `frame: pd.DataFrame, time_col: str, event_col: str, group: ComplexType`

- **Returns**: `ComplexType`

Per group, the times and events of the rows the curves read (a known time >= 0, a known event).

### def `_at_risk` (analysis_survival_results.py)

- **Arguments**: `follow_up: dict[ComplexType], ticks: np.ndarray`

- **Returns**: `dict[ComplexType]`

Rows still followed at each tick, per group.

### def `km_figure` (analysis_survival_results.py)

- **Arguments**: `curves: Sequence[Curve]`

- **Returns**: `go.Figure`

Kaplan-Meier curves, each with its pointwise band in its own colour (Okabe-Ito, in the order of the sorted
levels), run on to its last follow-up after its last event, with a tick where rows were censored, and the numbers
at risk under the axis, at its ticks.

### def `forest_figure` (analysis_survival_results.py)

- **Arguments**: `table: pd.DataFrame`

- **Returns**: `go.Figure`

The model's ratios on a log scale, with their 95% intervals: the comparison's terms filled and in colour, the
adjustment terms hollow and grey; a hairline at 1. Terms without a finite interval are left to the table.

### def `_decades` (analysis_survival_results.py)

- **Arguments**: `steps: Sequence[float]`

- **Returns**: `list[float]`

No description available.

### def `_ratio_ticks` (analysis_survival_results.py)

- **Arguments**: `lo: float, hi: float`

- **Returns**: `list[float]`

Round ratios from ``lo`` to ``hi`` for a log axis, 1 among them: 0.1, 0.2, 0.5, 1, 2, 5 on a wide range, halves
and doubles on a middle one (0.25, 0.5, 1, 2, 4), finer steps on a narrow one (0.8, 0.9, 1), never more than 8.
Plotly's own log ticks label 0.5 to 0.9 as "5 ... 9".

### def `_curves` (analysis_survival_results.py)

- **Arguments**: `km: Any, pooled: bool, follow_up: ComplexType`

- **Returns**: `list[Curve]`

The curves to draw; with the rows they read (``follow_up``), observed curves get their censoring ticks.

### def `_median_ci` (analysis_survival_results.py)

- **Arguments**: `lower: ComplexType, upper: ComplexType`

- **Returns**: `str`

The median's interval: 'not reached' alone when neither end is reached (the lower one is reached first).

### def `_group_table` (analysis_survival_results.py)

- **Arguments**: `km: Any, pooled: bool`

- **Returns**: `None`

No description available.

### def `show_km_observed` (analysis_survival_results.py)

- **Arguments**: `km: Any`

- **Returns**: `None`

A Kaplan-Meier result of the observed data, as the complete-case comparison under a pooled one.

### def `_logrank` (analysis_survival_results.py)

- **Arguments**: `km: Any, pooled: bool`

- **Returns**: `ComplexType`

No description available.

### def `show_time_to_event` (analysis_survival_results.py)

- **Arguments**: `km: Any, cox: Any`

- **Returns**: `None`

The curves and the Cox model of one time-to-event analysis: hazard ratios and the log-rank test first.
``follow_up`` (curve_rows, observed data): the rows the curves read, for the numbers at risk and the censoring
ticks. ``cox_chosen``: the Cox model is among the analyses; ``cox_fits``: it has a variable to read, so a missing
one has not been fitted with these variables (or could not be).

### def `_diagnostics` (analysis_survival_results.py)

- **Arguments**: `result: Any`

- **Returns**: `None`

No description available.

### def `_show_model` (analysis_survival_results.py)

- **Arguments**: `res: Any`

- **Returns**: `None`

A logistic or count model: the comparison's ratios first, the forest of every ratio, then details and checks.

### def `_reference_words` (analysis_survival_results.py)

- **Arguments**: `res: Any`

- **Returns**: `str`

No description available.

### def `show_logistic` (analysis_survival_results.py)

- **Arguments**: `res: Any`

- **Returns**: `None`

``outcome`` and ``event``: what the odds are of, said with the result.

### def `show_count` (analysis_survival_results.py)

- **Arguments**: `res: Any`

- **Returns**: `None`

``rates``: person-time was given, so the ratios compare rates; without it they compare mean counts.

---

## analyze.py

_Analyze page: hypothesis tests, epidemiological measures, ANCOVA, power, z-scores._

**Imports**:
`app_pages.epidemiology.app`, `ui.components.page_header`, `ui.components.mi_mode.mi_banner`, `ui.components.population.population_banner`, `ui.state.get_state`

---

## clean.py

_Clean page: how values are written, fixed (a variable's values recoded into a new column; text read as dates)._

**Imports**:
`app_pages.data_cleaning.app`, `ui.components.page_header`

---

## clean_categories.py

_Clean > Categories: the values of a categorical variable as written, each given a clean value or set to missing, and
written into a new column beside the original, which is kept (enrich.recode). A value written with extra spaces takes
its spelling with the spaces collapsed (the fold rule, recorded as a rule); a case variant, a one-letter code, a variant
(accents, British and American spellings), a close spelling, a code that may mean missing and what the AI
engine proposes are offered, never applied. A value set to missing says what missing means there; a grouping of
categories says why. The page shows each value's own rows, on all rows, never an outcome beside them: merging values
after seeing results is a choice the intervals ignore. Apply is one traced step, ``recode_levels``, which the replay
redoes; what it records always gives the preview (enrich.recode.recorded).

The choices live in one session key, ``clean_manual`` (ui.keys: cleared when another dataset is loaded), per column,
changed only through enrich.recode (give, tick, untick, unset_missing); the widgets that edit them carry a short id of
the dataset's trace and fold what they change into it as they change, so that no choice is lost on a page change and
none follows the user to another dataset. The table is drawn from the store: it reads each edit once, and starts
again from the store whenever the store and what the browser shows of its edits differ._

**Imports**:
`__future__.annotations`, `hashlib`, `html`, `json`, `collections.defaultdict`, `collections.abc.Mapping`, `typing.Any`, `pandas`, `plotly.graph_objects`, `streamlit`, `core.fingerprint.frame_fingerprint`, `core.registry.as_code`, `enrich.patterns.MAX_LENGTH`, `enrich.patterns.PatternError`, `enrich.patterns.matching`, `enrich.recode.AI`, `enrich.recode.CAPITALS`, `enrich.recode.CASE`, `enrich.recode.CHOSEN`, `enrich.recode.CLOSE`, `enrich.recode.FIRST_LETTER`, `enrich.recode.FOLD`, `enrich.recode.GROUPING`, `enrich.recode.MAX_VALUES`, `enrich.recode.MEANINGS`, `enrich.recode.MISSING_CODE`, `enrich.recode.OFFER_KINDS`, `enrich.recode.RESULT`, `enrich.recode.SPACES`, `enrich.recode.SPELLING`, `enrich.recode.VARIANT`, `enrich.recode.Candidate`, `enrich.recode.Offer`, `enrich.recode.ai_offers`, `enrich.recode.ai_tally`, `enrich.recode.candidates`, `enrich.recode.changes`, `enrich.recode.choices_of`, `enrich.recode.documented`, `enrich.recode.follows`, `enrich.recode.free_text`, `enrich.recode.give`, `enrich.recode.grouped_apart`, `enrich.recode.grouping_anchors`, `enrich.recode.groupings`, `enrich.recode.keep_result`, `enrich.recode.last_record`, `enrich.recode.letters_apart`, `enrich.recode.level_counts`, `enrich.recode.new_choices`, `enrich.recode.numeric_codes`, `enrich.recode.outcome`, `enrich.recode.plan`, `enrich.recode.recode_column`, `enrich.recode.recoded_columns`, `enrich.recode.recoded_records`, `enrich.recode.recorded`, `enrich.recode.replaceable`, `enrich.recode.resolve`, `enrich.recode.sort_key`, `enrich.recode.suggest`, `enrich.recode.swaps`, `enrich.recode.tick`, `enrich.recode.ticked`, `enrich.recode.unset_missing`, `enrich.recode.untick`, `enrich.recode.values_fp`, `manage.ai_payload.MIN_COUNT`, `manage.ai_payload.category_values`, `manage.ai_payload.host_of`, `manage.ai_payload.on_this_machine`, `manage.ai_payload.provider_label`, `manage.ai_payload.recipient`, `manage.ai_payload.server_recipient`, `ui.components.levels.show_level`, `ui.components.text.plain`, `ui.quality_charts.chart_config`, `ui.services.engine_model`, `ui.services.running_engine`, `ui.state.get_state`, `ui.theme.OKABE_ITO`

### def `_rows` (clean_categories.py)

- **Arguments**: `n: int`

- **Returns**: `str`

No description available.

### def `_scope` (clean_categories.py)

- **Arguments**: `state`

- **Returns**: `str`

A short id of the dataset's trace: the widget keys carry it, so a widget's state never meets another dataset.

### def `_choices` (clean_categories.py)

- **Arguments**: `column: str`

- **Returns**: `dict[ComplexType]`

No description available.

### def `_keys` (clean_categories.py)

- **Arguments**: `scope: str, column: str, revision: int, order: str`

- **Returns**: `dict[ComplexType]`

No description available.

### def `_again` (clean_categories.py)

- **Arguments**: `column: str`

- **Returns**: `None`

No description available.

### def `_counts_fp` (clean_categories.py)

- **Arguments**: `counts: dict[ComplexType]`

- **Returns**: `str`

The values as written and their rows: the choices made on them start again once they change.

### def `_choice_state` (clean_categories.py)

- **Arguments**: `choices: dict[ComplexType]`

- **Returns**: `str`

The choices a user makes in the editor, as one text: a choice made changes it.

### def `_dropped` (clean_categories.py)

- **Arguments**: `held: dict[ComplexType], fresh: dict[ComplexType]`

- **Returns**: `int`

How many values held a choice that starting again drops: a clean value, a fold put back, what missing means,
an answer "a result".

### def `_unapplied` (clean_categories.py)

- **Arguments**: `held: dict[ComplexType], fresh: dict[ComplexType]`

- **Returns**: `bool`

Whether the choices held say another thing than those the column starts again from.

### def `_cell` (clean_categories.py)

- **Arguments**: `value: Any`

- **Returns**: `Any`

A cell as the browser sends it: a cleared one comes back None or NaN.

### def `_edits` (clean_categories.py)

- **Arguments**: `key: str`

- **Returns**: `dict[ComplexType]`

No description available.

### def `_table_changed` (clean_categories.py)

- **Arguments**: `column: str, key: str, values: list[str], counts: dict[ComplexType]`

- **Returns**: `None`

The table's last edit into the store. The browser sends every edit made since the table was drawn, by row: only
the cells that changed since the last call are read, so that an earlier edit is never read again over a later one
(which, read in row order against a store the later one already changed, could undo it).

### def `_ticked` (clean_categories.py)

- **Arguments**: `column: str, name: str, offers: dict[ComplexType], counts: dict[ComplexType]`

- **Returns**: `None`

A group's pills (``name``): a ticked offer fills the table (its group's choice); unticked, its values take their
spelling again. The table and the pills keep their keys, so the keyboard's focus stays where it was: a table whose
edits no longer match the store starts again by itself, and so do pills that show another thing than the store.

### def `_tick_all` (clean_categories.py)

- **Arguments**: `column: str, offers: dict[ComplexType], counts: dict[ComplexType]`

- **Returns**: `None`

The offers of a group not yet ticked, nor answered a result, ticked.

### def `_pills_key` (clean_categories.py)

- **Arguments**: `choices: dict[ComplexType], scope: str, column: str, name: str, on: list[str]`

- **Returns**: `str`

A group's pills are drawn from the store (``default``), never kept: their key stays while they show what the store
holds ticked, so that a click on a pill keeps the keyboard's focus there; when the store holds another thing (a
table edit, a button, Apply), a new key draws them again, never a stale pill that the next click on another would
undo. Each new key is new (a counter), never one that the session still holds with another state.

### def `_buttons` (clean_categories.py)

- **Arguments**: `column: str, keys: dict[ComplexType], counts: dict[ComplexType], how: str`

- **Returns**: `None`

Map, Set to missing and Keep as written: the picked values given a clean value; the picks and the meaning are
cleared after use.

### def `_pick_by_pattern` (clean_categories.py)

- **Arguments**: `column: str, keys: dict[ComplexType], counts: dict[ComplexType]`

- **Returns**: `None`

The values a search finds, picked in Values (for Map, Set to missing or Keep as written), and its popover
closed; when nothing is found, Values stays as it was. A search that cannot be made keeps the popover open and says
why inside it. What was picked is named under the row while Values holds it.

### def `_find_values` (clean_categories.py)

- **Arguments**: `column: str, scope: str, keys: dict[ComplexType], counts: dict[ComplexType], choices: dict[ComplexType]`

- **Returns**: `None`

A messy column's values holding a text (or a pattern), picked at once in Values (the search runs apart, timed):
in a popover beside Values on the Clean page, closed by Pick them; inline in a dialog, its field and Pick them on
one row.

### def `_picked_words` (clean_categories.py)

- **Arguments**: `found: list[str], counts: Mapping[ComplexType]`

- **Returns**: `str`

What a search picked, for the line under the row while Values holds it: how many values and rows, the first six
in bold with their rows, the largest first (Values scrolls sideways once it holds a few: the line names them, to be
checked before Map or Set to missing); or that nothing was found.

### def `_shown` (clean_categories.py)

- **Arguments**: `counts: dict[ComplexType], spellings: dict[ComplexType], choices: dict[ComplexType]`

- **Returns**: `None`

Each value as written: its clean value (None: missing), what missing means there, and why.

### def `_figure` (clean_categories.py)

- **Arguments**: `counts: dict[ComplexType], full: dict[ComplexType]`

- **Returns**: `go.Figure`

One bar per clean value (after) or per value as written (before), the 15 largest, then the others in one bar;
the values set to missing apart, in grey: every bar keeps its name. The values are drawn as text: a level holding
markup is shown, never rendered.

### def `_offer_groups` (clean_categories.py)

- **Arguments**: `column: str, scope: str, choices: dict[ComplexType], counts: dict[ComplexType], offers: tuple[ComplexType], into: str`

- **Returns**: `None`

The offers in their groups, each with "Tick all"; ticking fills the table, never the data.

### def `category_editor` (clean_categories.py)

- **Arguments**: `state, df: pd.DataFrame, column: str`

- **Returns**: `None`

The editor of one variable's values: the offers, what each value becomes, a preview, and Apply. ``header``:
the container beside the variable's choice that holds the new column's name. ``host``: where it is drawn, the Clean
page or an analysis page's dialog (app_pages.clean_dialog); the choices are the variable's, wherever they were
made, but each host's widgets have keys of their own, so that the table's memory of its edits never crosses.

### def `_named` (clean_categories.py)

- **Arguments**: `values: list[str]`

- **Returns**: `str`

Values in a sentence: the first six, then how many more.

### def `_grouping_question` (clean_categories.py)

- **Arguments**: `clean: str, parts: list[str], apart: list[str], anchor: str`

- **Returns**: `str`

Why values are grouped, naming first those that are no spelling of the value the group is anchored on (Why group
Hypotension with Hypertension?), the anchor's spellings merged with them said apart, so that one answer is never
read as the reason for both. The anchor is the clean value when the column holds it (Why group Esch and Luxembourg
with Metz?); for a new name, the group's most held value, a category of its own, named with the new name and never
among the spellings (Why group Hypotension with Hypertension, as BP disorder?).

### def `_blocked` (clean_categories.py)

- **Arguments**: `df, steps, column, target, cleaned_into, counts, choices, shown, full, grouped, record`

- **Returns**: `str`

Why Apply waits ("" when it does not).

### def `_record_of` (clean_categories.py)

- **Arguments**: `record: tuple, grouped: dict[ComplexType], choices: dict[ComplexType], counts: dict[ComplexType]`

- **Returns**: `dict[ComplexType]`

The choices as the step records them: the mapping, the kinds (a merge typed by hand that groups categories is
a grouping; an offer or the AI engine's proposal keeps its kind), the folds put back, what missing means, the codes
answered a result (those still values), why each grouping was made, and who proposed each AI choice (never the
prompt or the answer).

### def `_apply` (clean_categories.py)

- **Arguments**: `state, df: pd.DataFrame, column: str, target: str, counts: dict[ComplexType], choices: dict[ComplexType], record: tuple, grouped: dict[ComplexType]`

- **Returns**: `None`

No description available.

### def `_ai_proposals` (clean_categories.py)

- **Arguments**: `column: str, counts: dict[ComplexType], offers: tuple[ComplexType]`

- **Returns**: `tuple[ComplexType]`

What the AI engine proposed for this column's values as they are now (a change of the values asks again).

### def `_ask_ai` (clean_categories.py)

- **Arguments**: `column: str, scope: str, counts: dict[ComplexType], offers: tuple[ComplexType]`

- **Returns**: `None`

"Ask the AI engine": a preview of exactly what is sent (the variable's name and its values, no count, no row;
to any recipient but an Ollama that runs its model on this machine, only the values held by 5 rows or more), then
Send, which closes it; the request is then made here, beside the heading, under a spinner, and what came back is
said here, every proposal accounted for. Never for free text: the button stays, disabled, and says why.

### def `_ai_said` (clean_categories.py)

- **Arguments**: `tally: dict[ComplexType]`

- **Returns**: `str`

What came back, every proposal accounted for: shown below, on a value the rules already offer (their groups
come after the AI engine's), or set aside.

### def `_preview` (clean_categories.py)

- **Arguments**: `engine, column: str, scope: str, counts: dict[ComplexType], popover: str`

- **Returns**: `None`

Exactly what Send gives the engine, then Send.

### def `_payload` (clean_categories.py)

- **Arguments**: `engine, counts: dict[ComplexType], model: str`

- **Returns**: `tuple[ComplexType]`

No description available.

### def `_send` (clean_categories.py)

- **Arguments**: `column: str, counts: dict[ComplexType], sent: list[str], held: int, popover: str, to: str`

- **Returns**: `None`

Send, as the click's callback: the popover closed, and the request left to the run that follows, which makes
it beside the heading under a spinner (a callback's spinner is drawn above the page's title).

### def `_request` (clean_categories.py)

- **Arguments**: `column: str, counts: dict[ComplexType], pending: dict[ComplexType]`

- **Returns**: `None`

The request Send asked for: the previewed values, to the recipient previewed, once the rule run again on the
engine that runs now gives the same values and the same recipient (an engine changed since the preview sends
nothing), and only to a recipient a box of Settings > AI engine confirmed; its answer into the store.

### def `_runs_elsewhere_now` (clean_categories.py)

- **Arguments**: `engine, model: str`

- **Returns**: `str`

An Ollama engine's server asked again, at Send, where it runs the model (3 seconds at most): why nothing is
sent when it does not say, or names another place than when the engine started (a model replaced under its
name since); "" when it says the same, or the engine is no Ollama.

### def `_recipients` (clean_categories.py)

- **Arguments**: `engine, model: str`

- **Returns**: `list[str]`

The recipients of a request that need a confirmation: the chat model's, and its Ollama server outside the
private network, which receives the prompt also when the model runs elsewhere.

### def `_receiver` (clean_categories.py)

- **Arguments**: `engine, model: str`

- **Returns**: `tuple[ComplexType]`

Who receives a request, and where it runs (as recorded and reported): the chat recipients only, the embedding
provider receiving none.

### def `_proposed_by` (clean_categories.py)

- **Arguments**: `column: str`

- **Returns**: `dict[ComplexType]`

Who proposed (provider, model, where it runs) and how many values it was sent and not sent: kept with each AI
choice, recorded with the step.

### def `_choice_label` (clean_categories.py)

- **Arguments**: `c: Candidate`

- **Returns**: `str`

A variable in the list: what it holds to check, and apart the values with extra spaces, which the fold
collapses with nothing to check.

### def `_fresh` (clean_categories.py)

- **Arguments**: `state, df: pd.DataFrame, column: str`

- **Returns**: `dict[ComplexType]`

The choices a variable starts from: its last recode's (cleaning again starts from them), else none.

### def `unapplied_choices` (clean_categories.py)

- **Arguments**: `state, df: pd.DataFrame, column: str`

- **Returns**: `int`

How many values of ``column`` hold a choice not yet applied, made on the Clean page or in a dialog: 0 for
none.

### def `discard_choices` (clean_categories.py)

- **Arguments**: `state, df: pd.DataFrame, column: str`

- **Returns**: `None`

The choices of ``column`` not yet applied dropped: it starts again from its last recode, its widgets drawn
again under new keys.

### def `take_done` (clean_categories.py)

- **Arguments**: `column: ComplexType`

- **Returns**: `list[str]`

What the last Apply did, said once after the rerun: its message for ``column`` (under that variable on an
analysis page, where another variable's warning may be drawn first), or every message (the Clean page).

### def `clean_candidates` (clean_categories.py)

- **Arguments**: `state`

- **Returns**: `list[Candidate]`

The variables Clean lists for the data as they are, with their copies and whether each copy follows its source:
one computation, kept until the data or the copies change, which Validate's line above its tabs reads too (the two
pages share the memo, so neither may fill it with less).

### def `render_categories` (clean_categories.py)

- **Arguments**: ``

- **Returns**: `None`

The Categories tab: a variable and the new column's name on one row, then its editor.

---

## clean_dialog.py

_A variable cleaned where it is analysed (Data > Clean, from the analysis warnings). Groups that differ only by spaces or
capitals are each analysed apart (ui.components.levels.level_warning); the warning offers to clean the variable in a
dialog that holds Data > Clean's own editor, since a page switch would drop the analysis page's choices (its widgets
keep their values while the user stays on the page), and, once a copy cleans it, to group by the copy._

**Imports**:
`__future__.annotations`, `collections.abc.Collection`, `collections.abc.Iterable`, `typing.Any`, `streamlit`, `app_pages.clean_categories.category_editor`, `app_pages.clean_categories.clean_candidates`, `app_pages.clean_categories.discard_choices`, `app_pages.clean_categories.take_done`, `app_pages.clean_categories.unapplied_choices`, `core.registry.as_code`, `enrich.recode.MAX_VALUES`, `enrich.recode.recoded_columns`, `ui.components.levels.CLEAN_WHERE`, `ui.components.levels.level_warning`, `ui.state.get_state`

### def `_dialog` (clean_dialog.py)

- **Arguments**: `column: str`

- **Returns**: `None`

Data > Clean's editor of ``column``: its add button writes the copy and closes the dialog (the editor's own
rerun); Close keeps the choices not yet applied, as Data > Clean does, and Discard drops them.

### def `_use` (clean_dialog.py)

- **Arguments**: `group_key: str, copy: str`

- **Returns**: `None`

No description available.

### def `_drop_from_list` (clean_dialog.py)

- **Arguments**: `list_key: str, name: str`

- **Returns**: `None`

No description available.

### def `_use_in_list` (clean_dialog.py)

- **Arguments**: `list_key: str, name: str, copy: str`

- **Returns**: `None`

No description available.

### def `_copy_of` (clean_dialog.py)

- **Arguments**: `state: Any, name: str`

- **Returns**: `tuple[ComplexType]`

The variable ``name`` stands for (itself, or the variable whose clean copy it is), that variable's copy ("" when
none is a column) and whether the copy follows it as it is (Clean's list: one computation, shared).

### def `group_warning` (clean_dialog.py)

- **Arguments**: `column: Any, values: Iterable[Any]`

- **Returns**: `None`

The warning under a list of groups (ui.components.levels.level_warning), which names the variable's clean copy
once one is written, then what can be done here (:func:`clean_actions`).

### def `clean_actions` (clean_dialog.py)

- **Arguments**: `column: Any`

- **Returns**: `None`

What can be done here about a variable whose values are written several ways: clean it in a dialog (Data > Clean's
editor, the source opened when the analysis reads a copy, or when its copy misses a later change of it), or, once a
copy follows it, group by the copy (``group_key``: the page's grouping widget, set to the copy; ``list_key``: a
list of variables, such as a model's covariates, where the copy takes the variable's place). ``groupable``: the
variables the page offers, when not every column; ``imputed``: they are the imputed datasets' (the pooled
comparison), which a copy written after the imputation is not in; ``told``: the warning above already says whether
a copy cleans the variable (the groups' warning), else whether a copy cleans it is said here; ``use``: the words of
the grouping widget's button ("Compare by" beside Survival & regression's field of that name); ``compare_key``: the
comparison's widget beside the list, where the copy may be already (then the variable leaves the list).

---

## data_cleaning.py

_Data > Clean: how values are written, fixed. Categories recodes a variable's values into a new column, the original
kept; Dates reads the text columns that look like dates (Validate's Date checks tab until 2026-10; its link id
``date_checks`` kept). Each fix is one traced step, and no row is removed here: Validate checks the data and chooses
the rows, Clean fixes how values are written, Enrich adds information._

**Imports**:
`__future__.annotations`, `app_pages.clean_categories.render_categories`, `app_pages.validate_dates.render_date_checks`, `ui.components.no_dataset`, `ui.components.tabs.PageTab`, `ui.components.tabs.lazy_tabs`, `ui.state.get_state`

### def `app` (data_cleaning.py)

- **Arguments**: ``

- **Returns**: `None`

No description available.

---

## data_enrichment.py

_No module description._

**Imports**:
`html`, `io`, `json`, `logging`, `os`, `re`, `pathlib.Path`, `numpy`, `pandas`, `streamlit`, `matplotlib.pyplot`, `rapidfuzz.fuzz`, `biostats.reduction`, `enrich.external_data`, `app_pages.enrichment_derived.render_derived_variables`, `app_pages.enrichment_mi.render_multiple_imputation`, `core.population.PopulationResult`, `core.population.catalogue`, `core.registry.as_code`, `enrich.derived.recomputed`, `enrich.formulas.CARET_IS_NOT_A_POWER`, `enrich.formulas.NOT_ROW_WISE`, `enrich.formulas.FormulaTrial`, `enrich.formulas.evaluate_formula_safely`, `enrich.formulas.extract_constant_value`, `enrich.formulas.far_outside`, `enrich.formulas.formula_from_model`, `enrich.formulas.numeric_summary`, `enrich.formulas.other_rows`, `enrich.formulas.read_formula`, `enrich.formulas.try_formula`, `enrich.formulas.validate_and_extract_columns_ast`, `enrich.formulas.validate_variable_name`, `enrich.imputation_service.describe_formula_fill`, `enrich.imputation_service.fill_with_formula`, `enrich.imputation_service.formula_inputs`, `enrich.imputation_service.left_missing`, `enrich.transformation_logic.RANDOM_STATE`, `enrich.transformation_logic.apply_variable_transformation`, `graph.builders.formula.build_formula_graph`, `graph.model.RenderOptions`, `graph.renderers.graphviz.GraphvizGraphRenderer`, `ingest.analyzer.DataAnalyzer`, `ingest.dates.detect`, `ingest.dates.recorded_formats`, `ingest.loaders.UPLOAD_TYPES`, `manage.ai_payload.column_kinds`, `manage.ai_payload.column_profile`, `manage.snapshots.folders`, `manage.snapshots.save`, `ui.components.error_card`, `ui.components.linked_tabs`, `ui.components.no_dataset`, `ui.components.csv_options.csv_options`, `ui.components.staging.render_draft_banner`, `ui.components.text.plain`, `ui.graph_view.current_theme`, `ui.graph_view.render_graph`, `ui.preferences.import_dayfirst`, `ui.session_scope.get_current_scope`, `ui.state.get_state`, `utils.data_paths.get_traces_dir`, `utils.export_utils.to_excel`, `utils.logging_config.single_line`, `utils.multipage._load_excel_with_selection`, `utils.multipage.get_file_hash`, `utils.multipage.load_bytes`, `utils.multipage.load_dataframe`, `utils.optional.available_methods`, `utils.optional.method_available`, `utils.optional.missing_methods_message`

### def `_workspace_path` (data_enrichment.py)

- **Arguments**: `path_text: str`

- **Returns**: `str`

Only files inside the signed-in user's workspace may be referenced by path (audit E4).

### def `_artifact_dir` (data_enrichment.py)

- **Arguments**: `session_id: str`

- **Returns**: `str`

No description available.

### def `_handle_multisheet_inline` (data_enrichment.py)

- **Arguments**: `uploaded_file, key_prefix: str`

- **Returns**: `ComplexType`

Handle multi-sheet Excel files inline with a selection UI.

Persists the loaded DataFrame in session state so it survives Streamlit re-runs.
Returns the loaded DataFrame, or None if the user hasn't made a selection yet.

### def `handle_main_data_upload` (data_enrichment.py)

- **Arguments**: ``

- **Returns**: `ComplexType`

The main dataset of the enrichment: the session dataset with the Enrich draft applied.

Datasets are loaded from Data > Import or the sidebar (AppState.set_dataset), so every load starts a
trace; this page no longer reads files of its own.

### def `handle_enrichment_data_upload` (data_enrichment.py)

- **Arguments**: ``

- **Returns**: `ComplexType`

Handle the upload or path input for the enrichment dataset.

### def `file_rows_choice` (data_enrichment.py)

- **Arguments**: `main_df: pd.DataFrame, enrichment_df: pd.DataFrame, left_ids, right_ids, strategy: str`

- **Returns**: `ComplexType`

How many rows of the file each row of the dataset takes (enrich.external_data.FILE_ROWS). When the file repeats
an identifier, the merge would repeat rows of the dataset, which the analyses would count as participants: the
page says how many and asks; None until the user chooses. A file with one row per identifier merges as "unique"
(checked by the merge); one that repeats only identifiers no row of the dataset holds merges as "all", which
takes one row of the file for each row of the dataset all the same: the merge's check covers every identifier of
the file, so "unique" would refuse it.

### def `_file_dates` (data_enrichment.py)

- **Arguments**: `enrichment_df: pd.DataFrame`

- **Returns**: `dict[ComplexType]`

The file's columns of dates, and how each is read: its date cells as they are; a text column whose values decide
its format and the order of day and month (Date checks' detection; none with two-digit years or Excel serials).

### def `file_rows_order` (data_enrichment.py)

- **Arguments**: `enrichment_df: pd.DataFrame, file_rows: ComplexType`

- **Returns**: `ComplexType`

Which of a participant's rows of the file is the first or the last: by its place in the file (None, as before), or
by a column of dates of the file, the earliest or the latest ({"column", "format"}, recorded with the merge).

### def `_merged` (data_enrichment.py)

- **Arguments**: `enriched_df: ComplexType, main_df: pd.DataFrame, file_df: ComplexType`

- **Returns**: `bool`

Whether the merge ran (it returns the dataset unchanged when it could not), said on the page either way.

### def `_date_formats` (data_enrichment.py)

- **Arguments**: `main_df: pd.DataFrame, enrichment_df: pd.DataFrame`

- **Returns**: `dict`

How the file's columns named as the dataset's date columns are read (enrich.external_data.date_key_formats): the
shape of their own values, the order of day and month from the format Date checks recorded for the dataset's
column, else the import's setting. Date checks commits its conversions: the trace holds them, the draft never.

### def `_record_checksum` (data_enrichment.py)

- **Arguments**: `df, data: bytes`

- **Returns**: `None`

The checksum of the bytes an enrichment table was read from (``sha256:<hex>``: an upload, or the file entered by
its path), carried by the table (its attrs, as its parse options) into the merge's params: a file found later
under the recorded name that holds other bytes (a corrected export uploaded since) is then named by the replay,
never merged silently.

### def `configure_enrichment` (data_enrichment.py)

- **Arguments**: `main_df: pd.DataFrame, enrichment_df: pd.DataFrame, enrichment_file_path: str`

- **Returns**: `ComplexType`

Configure and perform the data enrichment.

### def `display_category_box` (data_enrichment.py)

- **Arguments**: `title, columns`

- **Returns**: `None`

Display a box containing column names for a specific category.

**Parameters**:

title : str
    The title of the category box.
columns : list
    List of column names to display.

### def `display_results` (data_enrichment.py)

- **Arguments**: `enriched_df: pd.DataFrame, title: str, key_base`

- **Returns**: `None`

Display the results of the enrichment process.

### def `display_external_data_results` (data_enrichment.py)

- **Arguments**: `enriched_df: pd.DataFrame, filtered_main_data: pd.DataFrame`

- **Returns**: `None`

Display the results of the enrichment process.

### def `display_new_variables_results` (data_enrichment.py)

- **Arguments**: `enriched_df: pd.DataFrame, main_data: pd.DataFrame`

- **Returns**: `None`

Display the results of the enrichment process.

### def `display_imputation_results` (data_enrichment.py)

- **Arguments**: `imputed_data: pd.DataFrame, main_data: pd.DataFrame, cols_per_row: int`

- **Returns**: `None`

Display the results of the enrichment process.

### def `save_enrichment_config` (data_enrichment.py)

- **Arguments**: `enrichment_name: str, strategy: str, input_file_folder: str, input_file: str, identifier: str, left_identifier: str, right_identifier: str, additional_columns: str, conflict_resolution: str`

- **Returns**: `None`

Save the enrichment configuration to session state.

### def `calculate_prefix_score` (data_enrichment.py)

- **Arguments**: `input_token: str, column_name: str`

- **Returns**: `float`

Calculate a score based on how well the input matches the beginning of the column name.

**Parameters**:

input_token : str
    The user's input token
column_name : str
    The column name to compare against

**Returns**:

float
    Prefix matching score between 0 and 1

### def `calculate_similarity_score` (data_enrichment.py)

- **Arguments**: `input_token: str, column_name: str`

- **Returns**: `int`

Calculate a similarity score between input token and column name with
prefix-weighted scoring and length-aware adjustments.

**Parameters**:

input_token : str
    The user's input token
column_name : str
    The column name to compare against

**Returns**:

int
    Adjusted similarity score

### def `tokenize_formula` (data_enrichment.py)

- **Arguments**: `formula: str`

- **Returns**: `list[str]`

Tokenizes the formula, treating quoted variables as single tokens and splitting others by operators.

**Parameters**:

formula : str
    The formula input by the user.

**Returns**:

List[str]
    The list of tokens extracted from the formula.

### def `suggest_columns` (data_enrichment.py)

- **Arguments**: `computation_formula: str, column_names: list[str], threshold: int, max_suggestions: int`

- **Returns**: `list[tuple[ComplexType]]`

Suggest column names based on similarity to the input formula with enhanced
prefix matching, including handling quoted variable names as unique tokens.

**Parameters**:

computation_formula : str
    The formula input by the user.
column_names : List[str]
    List of available column names in the dataset.
threshold : int, optional
    Minimum similarity score to include in suggestions (default: 60).
max_suggestions : int, optional
    Maximum number of suggestions to return (default: 5).

**Returns**:

List[Tuple[str, int]]
    List of tuples containing (column_name, similarity_score).

### def `update_formula_callback` (data_enrichment.py)

- **Arguments**: `col, formula_key`

- **Returns**: `None`

Callback to update formula when a suggestion is clicked.

### def `display_column_suggestions` (data_enrichment.py)

- **Arguments**: `suggestions, formula_key`

- **Returns**: `None`

Displays column suggestions as buttons. Clicking a button replaces the last token
in the formula input with the selected column name.

### def `generate_stats` (data_enrichment.py)

- **Arguments**: `temp_main_data, variable_name`

- **Returns**: `None`

No description available.

### def `render_new_variable_statistics` (data_enrichment.py)

- **Arguments**: `temp_data, var_name, used_cols`

- **Returns**: `None`

Statistics of a new variable on the analysis population, robust (median, quartiles) and classical (mean, SD),
and the values far outside the others with their source values: a few entry errors (a height in metres, a
weight of 0) can move the mean and the SD far from what the median shows.

### def `define_new_variables` (data_enrichment.py)

- **Arguments**: `main_data`

- **Returns**: `None`

Enhanced version of define_new_variables with improved preview and validation.

### def `merge_transformed_columns` (data_enrichment.py)

- **Arguments**: `main_data, transformed_data, handle_duplicates`

- **Returns**: `None`

Merge transformed_data into main_data securely.

Parameters:
- main_data (pd.DataFrame): The original DataFrame.

- transformed_data (pd.DataFrame): The DataFrame with new columns.

- handle_duplicates (str): How to handle duplicate column names.
                           Options: "replace" or "ignore".

Returns:
- pd.DataFrame: The merged DataFrame.

### def `compute_transformations` (data_enrichment.py)

- **Arguments**: `dataframe`

- **Returns**: `None`

The transformation form. The columns are computed by enrich.transformation_logic.apply_variable_transformation,
the function the replay of the trace calls with the same parameters, so the replay gives the same columns.

### def `save_snapshot` (data_enrichment.py)

- **Arguments**: `df, step_name, base_dir`

- **Returns**: `None`

Saves a snapshot of the dataframe in the user's workspace and returns the path (manage.snapshots.save).

### def `_tried` (data_enrichment.py)

- **Arguments**: `sugg: dict, formula: str, data: pd.DataFrame, target: ComplexType, population: ComplexType`

- **Returns**: `FormulaTrial`

An AI suggestion's formula tried on the data (enrich.formulas.try_formula), on the rows of ``population`` when
one is given, kept on the suggestion: tried again only for another formula, target, version of the data or
population, not at every rerun. The version is the fingerprint of the Enrich data, which the page receives as a
new frame at every run.

### def `_from_missing` (data_enrichment.py)

- **Arguments**: `counts: list[tuple[ComplexType]]`

- **Returns**: `str`

The rows that got a value although an input is missing (a comparison or np.where answers "false" there).

### def `_show_trial` (data_enrichment.py)

- **Arguments**: `trial: FormulaTrial, gives: str`

- **Returns**: `None`

What a suggestion gives on the data, or why it cannot be computed there (its text as code: it quotes values).

### def `extract_column_stats` (data_enrichment.py)

- **Arguments**: `df, columns`

- **Returns**: `None`

What the AI engine is told about the selected columns for its unit hints: their percentiles, or the frequent
values that may carry a unit such as "mg/dL" (manage.ai_payload: a summary, never a row or an extreme value).

### def `app` (data_enrichment.py)

- **Arguments**: ``

- **Returns**: `None`

Main application function of the Enrich page (Data > Enrich).

Handles:
1. External Data Upload and Enrichment
2. Targeted Imputation (Manual/AI-assisted)
3. Global Imputation (MICE, KNN, etc.)
4. Dataset Versioning and Persistence

---

## data_insight.py

_No module description._

**Imports**:
`json`, `os`, `re`, `pandas`, `streamlit`, `core.registry.as_code`, `core.workspace.UserScope`, `graph.builders.taxonomy.CATEGORY_LABELS`, `graph.builders.taxonomy.RELATION_LABELS`, `graph.builders.taxonomy.build_taxonomy_graph`, `graph.builders.taxonomy.make_formula_id`, `graph.builders.taxonomy.parse_formula_id`, `graph.model.LayoutHint`, `graph.model.RenderOptions`, `manage.ai_payload.column_profile`, `ui.components.no_dataset`, `ui.components.text.answer`, `ui.components.text.plain`, `ui.graph_view.current_theme`, `ui.graph_view.render_graph`, `ui.graph_view.render_graph_exports`, `ui.graph_view.renderer_selector`, `ui.services.drop_ai_engine`, `utils.clustering_utils.compute_cluster_profiles`, `utils.clustering_utils.fit_dbscan`, `utils.clustering_utils.fit_gaussian_mixture`, `utils.clustering_utils.fit_kmeans`, `utils.clustering_utils.prepare_data_for_clustering`, `utils.clustering_utils.run_famd`, `utils.clustering_utils.run_pca`, `utils.clustering_utils.run_tsne`, `utils.clustering_utils.run_umap`, `utils.optional.available_methods`

### def `get_user_taxonomy_dir` (data_insight.py)

- **Arguments**: `username`

- **Returns**: `None`

Taxonomy directory of the user's workspace (no shared fallback; audit E4).

### def `get_taxonomy_versions` (data_insight.py)

- **Arguments**: `username`

- **Returns**: `None`

Returns list of (version_int, filepath) sorted descending.

### def `repair_taxonomy_links` (data_insight.py)

- **Arguments**: `taxonomy, formulas_registry`

- **Returns**: `None`

Ensures taxonomy variables have correct related_formula_ids based on registry.
Run this on load to fix desynchronized files.

### def `load_and_repair_taxonomy` (data_insight.py)

- **Arguments**: `target_path, rag_manager`

- **Returns**: `None`

Robust loading function:
1. Loads Taxonomy & Formulas
2. Repairs Links immediately
3. Updates Session State
4. Syncs with RAG Manager (if active)

### def `app` (data_insight.py)

- **Arguments**: ``

- **Returns**: `None`

Main application function of the Variable map page (Knowledge > Variable map).

Handles:
1. Loading and saving taxonomy versions.
2. Visualizing variables and formulas as a graph.
3. Generating new taxonomies using RAG.
4. Enriching existing taxonomies with external knowledge.

---

## data_monitoring.py

_No module description._

**Imports**:
`copy`, `json`, `os`, `collections.abc.Mapping`, `collections.abc.Sequence`, `io.BytesIO`, `typing.Any`, `pandas`, `streamlit`, `biostats.outliers`, `monitor.changes`, `utils.clustering_utils`, `app_pages.clean_categories.clean_candidates`, `app_pages.data_quality_dashboard.render_dashboard`, `app_pages.validate_dates.date_candidates`, `app_pages.validate_dates.dates_found`, `core.criteria.RULES_VERSION`, `core.criteria.criterion_columns`, `core.criteria.criterion_unknown`, `core.criteria.expression_rule`, `core.criteria.listed_rows_rule`, `core.criteria.numeric_rule`, `core.derived.CONFIG_KEY`, `core.derived.merge_definitions`, `core.fingerprint.ids_to_ranges`, `core.population.EXCLUSION`, `core.population.SharedConfiguration`, `core.population.catalogue`, `core.population.configuration_for_sharing`, `core.population.configuration_problems`, `core.population.criteria_changes`, `core.population.criterion_id`, `core.population.family_masks`, `core.population.family_words`, `core.population.merge_families`, `core.population.rules_population`, `core.registry.as_code`, `ingest.analyzer.DataAnalyzer`, `ingest.loaders.UPLOAD_TYPES`, `ui.components.no_dataset`, `ui.components.page_link`, `ui.components.criteria.criterion_cards`, `ui.components.csv_options.csv_options`, `ui.components.population.activate_family`, `ui.components.population.render_population_panel`, `ui.components.tabs.PageTab`, `ui.components.tabs.lazy_tabs`, `ui.components.text.plain`, `ui.session_scope.get_current_scope`, `ui.state.get_state`, `utils.config_loader.create_empty_config`, `utils.config_loader.transform_expression`, `utils.export_utils.to_excel`, `utils.multipage.load_dataframe`, `utils.optional.available_methods`

### class `OutlierHandler` (data_monitoring.py)

Applies a handling strategy (remove, clip, tag) to what biostats.outliers.detect_outliers flags: the detector
of the quality dashboard and the health score too, so every screen flags the same rows.

**Methods:**

- **__init__**(`self, df, config`) -> `None`
  > No description available.

- **handle_outliers**(`self, columns, method, handling_strategy`) -> `None`
  > Detect the outliers of ``columns`` and preview the handling strategy.
  >
  > Returns:
  >     DataFrame: the table with the strategy applied (a preview: the Validate actions record the change).
  >     DataFrame: the flags, one column per analysed column (univariate) or "overall" (multivariate).

- **_apply_handling_strategy**(`self, detection, strategy`) -> `None`
  > The table with ``strategy`` applied to the detected outliers.

### def `export_comparison_results` (data_monitoring.py)

- **Arguments**: `df1, df2, comparison_result_filtered, rows_only_in_df1, rows_only_in_df2, modified_common_ids, common_id, key_base`

- **Returns**: `None`

Efficiently export comparison results to an Excel file

### def `data_selection` (data_monitoring.py)

- **Arguments**: `tmp: str, label: str, load: bool`

- **Returns**: `tuple[ComplexType]`

Handle data selection from various sources.

**Parameters**:

tmp : str
    Unique identifier for the session keys.
label : str
    Label to display in the UI selector.
load : bool
    False while the Compare tab is closed: the source controls are shown, so that an uploaded file is kept, but
    nothing is read (M5b: only the open tab works).

**Returns**:

Tuple[Optional[pd.DataFrame], Optional[str]]
    The selected dataframe and its source name/path, or (None, None) if selection failed or nothing selected.

### def `add_outlier_handling_ui` (data_monitoring.py)

- **Arguments**: `df, numeric_cols`

- **Returns**: `None`

Add outlier handling UI components with improved organization and summary display.

### def `configure_method_params` (data_monitoring.py)

- **Arguments**: `detection_method`

- **Returns**: `None`

Configure parameters based on the selected outlier detection method (defaults: biostats.outliers).

### def `select_columns_for_outlier_handling` (data_monitoring.py)

- **Arguments**: `df, numeric_cols, detection_method`

- **Returns**: `None`

Select columns for outlier handling based on the detection method.

### def `format_change` (data_monitoring.py)

- **Arguments**: `change, change_measure`

- **Returns**: `None`

No description available.

### def `display_combined_information` (data_monitoring.py)

- **Arguments**: `df, df_processed, summary, selected_cols, detection_method, handling_strategy`

- **Returns**: `None`

Display outlier summary, outlier impact (percentage change), and before/after statistics side by side for each variable.

### def `calculate_vmin_vmax` (data_monitoring.py)

- **Arguments**: `df, change_measure`

- **Returns**: `None`

No description available.

### def `calculate_outlier_impact` (data_monitoring.py)

- **Arguments**: `df, df_processed, selected_cols, change_measure`

- **Returns**: `None`

Calculate the impact of outlier handling on each selected column by computing
the percentage change in mean, std, min, and max values.

### def `generate_outlier_summary_from_predictions` (data_monitoring.py)

- **Arguments**: `df, outlier_predictions, selected_cols, method`

- **Returns**: `None`

Generate outlier summary based on raw predictions.

Parameters:
    df (DataFrame): Original DataFrame.
    outlier_predictions (dict): Raw outlier predictions for each column.
    selected_cols (list): Columns analyzed for outliers.
    method (str): Outlier detection method.

Returns:
    dict: Summary of outliers for each column or overall.

### def `display_outlier_visualization` (data_monitoring.py)

- **Arguments**: `df, outlier_matrix, selected_cols, method`

- **Returns**: `None`

Display 2D visualization of outliers using PCA, FAMD, t-SNE, or UMAP.

### def `outlier_values_dialog` (data_monitoring.py)

- **Arguments**: `outlier_summary`

- **Returns**: `None`

Display outlier values in a dialog

### def `ensure_family_exists` (data_monitoring.py)

- **Arguments**: `config: dict[ComplexType], family_name: str, operator: ComplexType`

- **Returns**: `None`

Ensure the specified family exists in the configuration. A family created here gets ``operator``; one that
exists keeps how it reads, never given an operator (a family without one reads "any").

### def `validate_mask_name` (data_monitoring.py)

- **Arguments**: `config: dict[ComplexType], family_name: str, mask_name: str, st_container`

- **Returns**: `None`

Validate the mask name and handle duplicates.

### def `handle_numeric_mask_input` (data_monitoring.py)

- **Arguments**: `st_container, base_key: str`

- **Returns**: `None`

Collect numeric mask inputs from the user.

### def `handle_expression_mask_input` (data_monitoring.py)

- **Arguments**: `st_container, base_key: str`

- **Returns**: `None`

Collect expression mask inputs from the user (``base_key``: the family the rule joins).

### def `rules_column` (data_monitoring.py)

- **Arguments**: `family_name: str`

- **Returns**: `str`

The column of a check's view that lists the rules each row matches: shown, never added to the dataset.

### def `result_column` (data_monitoring.py)

- **Arguments**: `family_name: str, operator: str, threshold: Any`

- **Returns**: `str`

The column of a check's view that holds the family's result on each row: shown, never added to the dataset.

### def `apply_family_mask` (data_monitoring.py)

- **Arguments**: `df: pd.DataFrame, family_name: str, family_config: dict, st_container`

- **Returns**: `None`

Apply a single mask family from JSON configuration to the dataset.

Parameters:
- df: DataFrame to which the mask family is applied.

- family_name: Name of the mask family to apply.

- family_config: Configuration for the specific mask family.

- st_container: Streamlit container for displaying results.

Returns:
- Updated DataFrame with mask family tags and lists.

- The names of the columns related to the masks family

The rules are evaluated as the analysis population evaluates them (core.population.family_masks), so the
counts here are the counts of the population. A rule that cannot be evaluated is shown with its reason, and
the family result is then left out: the population refuses such a rule while it is ticked.

### def `_exclude_outliers` (data_monitoring.py)

- **Arguments**: `detection: dict`

- **Returns**: `None`

Callback: the flagged rows become an exclusion criterion (row ids frozen now), ticked at once.

### def `_clip_outliers` (data_monitoring.py)

- **Arguments**: `detection: dict`

- **Returns**: `None`

Callback: clip the values to the method's bounds in the dataset, as one traced, replayable step.

### def `render_outlier_actions` (data_monitoring.py)

- **Arguments**: `detection: ComplexType`

- **Returns**: `None`

What to do with the detected outliers: exclude them from the population, or clip the values.

### def `display_results` (data_monitoring.py)

- **Arguments**: `df: pd.DataFrame, bool_column: str, title: str, key_base: str, st`

- **Returns**: `None`

Display the results of the enrichment process.

### def `left_out_text` (data_monitoring.py)

- **Arguments**: `shared: SharedConfiguration`

- **Returns**: `str`

What the file for sharing leaves out of this session's configuration, and what it keeps as written.

### def `color_columns` (data_monitoring.py)

- **Arguments**: `s, columns_to_highlight`

- **Returns**: `None`

Applies a background color to specific columns.

### def `deep_merge_dicts` (data_monitoring.py)

- **Arguments**: `original, new`

- **Returns**: `None`

Deeply merges two dictionaries. Adds new keys and values from `new` to `original`
without modifying existing keys in `original`.

### def `refusal_text` (data_monitoring.py)

- **Arguments**: `problems: list[str]`

- **Returns**: `str`

Why a loaded configuration was refused: the file's text (rule names, values) as written, never as Markdown,
where an image or a link written in a rule's name would be fetched or followed.

### def `_merge_uploaded` (data_monitoring.py)

- **Arguments**: `config: dict[ComplexType], new_config: dict[ComplexType]`

- **Returns**: `list[str]`

An uploaded configuration added to the session's; returns the notices to show. The criteria and the derived
variables are added whole, each one: a rule or a definition the session holds keeps the session's, and is named
when the file's differs (core.population.merge_families, core.derived.merge_definitions). The other keys are added
when the session lacks them.

### def `read_uploaded_config` (data_monitoring.py)

- **Arguments**: `config: dict[ComplexType], file: Any`

- **Returns**: `tuple[ComplexType]`

An uploaded configuration file merged into ``config`` (:func:`_merge_uploaded`): whether it was loaded, and what to
tell. A file that cannot be read, or whose rules or operators would be read one way here and another there, is
refused with why.

### def `_version` (data_monitoring.py)

- **Arguments**: `written: Any`

- **Returns**: `int`

The rules' version a file states: 1 when it states none or no number (written before it was recorded).

### def `_as_loaded` (data_monitoring.py)

- **Arguments**: `c: Any, entry: Any`

- **Returns**: `bool`

Whether the session's criterion ``c`` is still the one loaded (its kind and definition).

### def `_taken_earlier` (data_monitoring.py)

- **Arguments**: `written: Any, uploaded: Any, config: Mapping[ComplexType]`

- **Returns**: `dict[ComplexType]`

The criteria of a file of the first rules, or stating none, that the session took: a criterion the session
held keeps its own definition when the file's differs (:func:`_merge_uploaded`), and is not the file's.

### def `_held_earlier` (data_monitoring.py)

- **Arguments**: `config: Mapping[ComplexType]`

- **Returns**: `dict[ComplexType]`

The criteria of RULES_EARLIER the session still holds as they were loaded; a criterion changed (corrected as
the notice advised) or removed since is dropped, and the key with the last one.

### def `_earlier_rules_line` (data_monitoring.py)

- **Arguments**: `written: Any, uploaded: Any, config: Mapping[ComplexType]`

- **Returns**: `ComplexType`

The sidebar's line for a configuration written under the first rules, or stating no version: the criteria the
session took from it are kept (RULES_EARLIER) for the notice above Validate's tabs (:func:`_earlier_rules_notice`),
which says what each selects now. None for a file of today's rules, none of whose criteria the session took, or
whose criteria select the same rows of the loaded data.

### def `_earlier_rules_notice` (data_monitoring.py)

- **Arguments**: ``

- **Returns**: `None`

Above Validate's tabs, the notice of the criteria loaded from a configuration of the first rules
(:func:`_rules_notice`), once data are loaded, while the session holds them as loaded, and until dismissed;
computed again when the data, the criteria or the ticks change.

### def `_rules_notice` (data_monitoring.py)

- **Arguments**: `held: Mapping[ComplexType], config: Mapping[ComplexType], ticked: Sequence[str]`

- **Returns**: `list[str]`

The criteria the session took from a file written under the first rules (a test written with != or ~ true on a
missing value), or before they were recorded, that select other rows of the loaded data now: each with how many
more or fewer rows it excludes, and what to write for it (:func:`_rules_change_said`); then the rows the ticked
criteria keep, now (as the sidebar counts them) and under the earlier reading of those criteria, since a
criterion's own count is not the study's. Nothing without data, or when none reads otherwise.

### def `_rules_change_said` (data_monitoring.py)

- **Arguments**: `df: pd.DataFrame, change: Any`

- **Returns**: `str`

One criterion of :func:`_rules_notice`: how many more or fewer rows it excludes and, for an expression of one
column changed one way only, the expression that brings back the rows whose value is missing as before: kept by
an inclusion criterion, excluded by an exclusion criterion (``isna() | (...)`` does either, by the criterion's
kind). Any other criterion is said to check.

### def `_on_config_upload` (data_monitoring.py)

- **Arguments**: ``

- **Returns**: `None`

Callback of the configuration upload: the file is merged once, before the page and the sidebar are drawn. Merged in
the page, it was merged again on every run while it stayed in the uploader (a rule changed since came back from the
file), and the sidebar, drawn before the page, showed the criteria of the run before: "every row" right after a
configuration with criteria was loaded.

### def `render_ai_criteria_assistant` (data_monitoring.py)

- **Arguments**: `st_container, config, family_name, mode`

- **Returns**: `None`

Renders the AI assistant for generating criteria.

### def `add_domain_expert_based_anomaly_mask` (data_monitoring.py)

- **Arguments**: `config: dict[ComplexType], st_container`

- **Returns**: `None`

No description available.

### def `display_results_with_anomaly` (data_monitoring.py)

- **Arguments**: `df: pd.DataFrame, anomaly_col: str, st`

- **Returns**: `None`

Display the results of the enrichment process.

### def `_share` (data_monitoring.py)

- **Arguments**: `part: int, whole: int`

- **Returns**: `str`

A share of rows as a percentage; "under 1%" rather than "0%" for rows that are there (2 of 1,206), and "over 99%"
rather than "100%" for rows that are not all (1,205 of 1,206).

### def `unknown_advice` (data_monitoring.py)

- **Arguments**: `df: pd.DataFrame, name: str, definition: dict[ComplexType]`

- **Returns**: `ComplexType`

What an inclusion criterion does where it is unknown for want of a value: it excludes those rows (a missing value
meets no condition, written with != or ~ too: "not pregnant" excludes the men whose pregnancy is not recorded).
The words, and the expression that keeps them (None for a kind other than an expression); None when the criterion
is known on every row.

### def `_say_unknown` (data_monitoring.py)

- **Arguments**: `df: pd.DataFrame, name: str, definition: dict[ComplexType], st_container`

- **Returns**: `None`

The advice of :func:`unknown_advice`, said before the criterion is added.

### def `add_study_inclusion_mask` (data_monitoring.py)

- **Arguments**: `config: dict[ComplexType], st_container`

- **Returns**: `None`

No description available.

### def `display_results_with_inclusion` (data_monitoring.py)

- **Arguments**: `df: pd.DataFrame, inclusion_col: str, st`

- **Returns**: `None`

Display the results of the enrichment process.

### def `_dashboard_tab` (data_monitoring.py)

- **Arguments**: ``

- **Returns**: `None`

No description available.

### def `_anomalies_tab` (data_monitoring.py)

- **Arguments**: ``

- **Returns**: `None`

No description available.

### def `_inclusion_tab` (data_monitoring.py)

- **Arguments**: ``

- **Returns**: `None`

No description available.

### def `_outliers_tab` (data_monitoring.py)

- **Arguments**: ``

- **Returns**: `None`

No description available.

### def `_population_tab` (data_monitoring.py)

- **Arguments**: ``

- **Returns**: `None`

No description available.

### def `_compare_tab` (data_monitoring.py)

- **Arguments**: `opened: bool`

- **Returns**: `None`

No description available.

### def `process_dataset` (data_monitoring.py)

- **Arguments**: `df: pd.DataFrame`

- **Returns**: `pd.DataFrame`

Process dataset using DataAnalyzer to fix types.

### def `app` (data_monitoring.py)

- **Arguments**: ``

- **Returns**: `None`

Main application function for the Data Monitoring page.

Handles:
1. Data Quality Dashboard (Data Validity, Completeness, etc.).
2. Implausible values: the plausibility rules (add, apply).
3. Study Inclusion Criteria Management (Add/Apply Masks).
4. Outlier Handling (Univariate & Multivariate).
5. Dataset Comparison (Changes tracking).

---

## data_quality_dashboard.py

_No module description._

**Imports**:
`html`, `numpy`, `pandas`, `plotly.express`, `plotly.figure_factory`, `plotly.graph_objects`, `scipy.cluster.hierarchy`, `statsmodels.stats.multitest`, `streamlit`, `scipy.spatial.distance.squareform`, `utils.analysis_utils`, `utils.visualization_utils`, `biostats.outliers.UNIVARIATE_METHODS`, `biostats.outliers.detect_outliers`, `core.criteria.criterion_columns`, `core.registry.as_code`, `explore.data_quality_auditor.CLEAN_AGAIN`, `explore.data_quality_auditor.CLEAN_LINK`, `explore.data_quality_auditor.COPY_KEEPS`, `explore.data_quality_auditor.DERIVED_CATEGORY`, `explore.data_quality_auditor.FORMULA`, `explore.data_quality_auditor.FORMULA_LINK`, `explore.data_quality_auditor.NO_RULE`, `explore.data_quality_auditor.TO_CLEAN`, `explore.quality_score.DEFAULT_WEIGHTS`, `explore.quality_score.DIMENSION_HELP`, `explore.quality_score.DIMENSIONS`, `explore.quality_score.interpretation`, `explore.quality_score.ordered`, `explore.quality_score.points_text`, `explore.quality_score.score_text`, `explore.quality_score.weighted_score`, `ui.nav`, `ui.components.page_link`, `ui.components.criteria.criterion_cards`, `ui.components.levels.CLEAN_WHERE`, `ui.components.text.plain`, `ui.progress.mark_step`, `ui.progress.record_quality_check`, `ui.quality_charts.chart_config`, `ui.quality_charts.dimension_color`, `ui.quality_charts.history_figure`, `ui.quality_charts.loss_figure`, `ui.quality_charts.ring_figure`, `ui.state.get_state`

### def `_selected_dimension` (data_quality_dashboard.py)

- **Arguments**: `event`

- **Returns**: `ComplexType`

Dimension of the bar clicked in the "points lost" chart, if any.

### def `_uniformity_links` (data_quality_dashboard.py)

- **Arguments**: `auditor`

- **Returns**: `None`

Where the text columns Uniformity flags are corrected: Data > Clean for those it lists (and for a variable whose
copy needs cleaning again), a formula on Enrich for free text and identifiers.

### def `_uniformity_cleaned` (data_quality_dashboard.py)

- **Arguments**: `auditor`

- **Returns**: `None`

The variables left out of the Uniformity score: each is cleaned into a copy that follows it, scored in its
place.

### def `_score_breakdown` (data_quality_dashboard.py)

- **Arguments**: `quality, details, checks, weights`

- **Returns**: `None`

Where the score loses points (click a bar to see the columns or rules) and the score across checks.

### def `_validity_control` (data_quality_dashboard.py)

- **Arguments**: `method: str, name: str`

- **Returns**: `tuple[ComplexType]`

The key of a detection parameter's control, and its start: the stored setting, or the default.

### def `_sync_validity_settings` (data_quality_dashboard.py)

- **Arguments**: ``

- **Returns**: `None`

The scorecard at the top of the dashboard is computed before the detection settings further down run, so it
used the settings of the run before: it followed a change one click late, and fell back to the defaults for a
run after each change of the data. Streamlit updates a widget's value before the rerun: the settings are read
from the widgets here.

### def `anomaly_columns` (data_quality_dashboard.py)

- **Arguments**: `rules`

- **Returns**: `dict[ComplexType]`

The columns each plausibility rule reads (core.criteria.criterion_columns), whose cells the report highlights
where the rule fired: a rule named like a column was the only one highlighted, so an expression's cells never were.

### def `missingness_linkage` (data_quality_dashboard.py)

- **Arguments**: `corr_matrix: pd.DataFrame`

- **Returns**: `np.ndarray`

The average-linkage clustering of the columns, at the distance 1 - |r| between their missingness indicators. A
column whose missingness never varies (complete, or empty) has no correlation, not even with itself: it is 1
from every other column and 0 from itself, where the 1 on the diagonal made SciPy refuse the matrix.

### def `_domain_notes` (data_quality_dashboard.py)

- **Arguments**: `auditor`

- **Returns**: `dict[ComplexType]`

Beside the Domain Validity score, the rows it pools: a rule that never fails raises the score, so the rows tell
how much of the table a failing rule touches.

### def `render_dashboard` (data_quality_dashboard.py)

- **Arguments**: `df, config`

- **Returns**: `None`

Renders the Data Quality Dashboard of ``df``, the session's dataset (every row) with its ``config``.

---

## document_insight.py

_No module description._

**Imports**:
`datetime`, `json`, `streamlit`, `core.registry.as_code`, `core.workspace.UserScope`, `core.workspace.resolve_within`, `core.workspace.safe_filename`, `graph.builders.document.build_document_graph`, `graph.builders.document.filter_by_documents`, `graph.model.LayoutHint`, `graph.model.RenderOptions`, `ui.components.no_dataset`, `ui.components.text.answer`, `ui.components.text.plain`, `ui.graph_view.current_theme`, `ui.graph_view.render_graph`, `ui.graph_view.render_graph_exports`, `ui.graph_view.renderer_selector`

### def `_graph_dir` (document_insight.py)

- **Arguments**: `username`

- **Returns**: `None`

Knowledge-graph directory of the user's workspace (audit E4).

### def `get_saved_graphs` (document_insight.py)

- **Arguments**: `username`

- **Returns**: `None`

Returns the saved graph names of the user's workspace.

### def `save_graph` (document_insight.py)

- **Arguments**: `name, graph_json, source_docs, username`

- **Returns**: `None`

Saves graph + metadata into the user's workspace.

### def `load_graph` (document_insight.py)

- **Arguments**: `name, username`

- **Returns**: `None`

Loads a graph payload from the user's workspace.

### def `app` (document_insight.py)

- **Arguments**: ``

- **Returns**: `None`

No description available.

---

## documents.py

_Documents page: literature knowledge graph and document chat._

**Imports**:
`streamlit`, `app_pages.document_insight.app`, `ui.components.page_header`, `utils.optional.extra_installed`, `utils.optional.missing_message`

---

## dossier.py

_Study dossier page: the participant flow, the results added to the report and the Word dossier._

**Imports**:
`app_pages.study_dossier.app`, `ui.components.page_header`

---

## enrich.py

_Enrich page: external data merge, computed variables, imputation, transformations._

**Imports**:
`app_pages.data_enrichment.app`, `ui.components.page_header`

---

## enrichment_derived.py

_Data > Enrich > Handle Missing Data > Derived variables: the variables computed from others in the same row (a BMI from
the weight and the height, a total from its items), declared and checked against the data (core.derived,
enrich.derived). Optional: with no definition nothing changes, and a definition is used only when the user fills with
it or ticks it in an imputation tab._

**Imports**:
`__future__.annotations`, `dataclasses.replace`, `pandas`, `streamlit`, `core.derived.KINDS`, `core.derived.NOTE_LENGTH`, `core.derived.DerivedVariable`, `core.derived.cycles`, `core.derived.note_text`, `core.registry.as_code`, `enrich.derived.AGREES`, `enrich.derived.CANNOT`, `enrich.derived.DISAGREES`, `enrich.derived.MOSTLY`, `enrich.derived.TOO_FEW`, `enrich.derived.DerivedCheck`, `enrich.formulas.evaluate_formula_safely`, `enrich.formulas.read_formula`, `ingest.analyzer.DataAnalyzer`, `manage.ai_payload.column_kinds`, `manage.ai_payload.typical_ranges`, `ui.components.help_popover`, `ui.components.text.plain`, `ui.services.running_engine`, `ui.state.get_state`

### def `_badge` (enrichment_derived.py)

- **Arguments**: `check: DerivedCheck`

- **Returns**: `None`

No description available.

### def `_check_lines` (enrichment_derived.py)

- **Arguments**: `check: DerivedCheck`

- **Returns**: `None`

The check in words: its evidence, why, and what it could not test.

### def `_inputs` (enrichment_derived.py)

- **Arguments**: `definitions: list[DerivedVariable]`

- **Returns**: `dict[ComplexType]`

No description available.

### def `_circle` (enrichment_derived.py)

- **Arguments**: `definition: DerivedVariable, definitions: tuple[ComplexType]`

- **Returns**: `ComplexType`

Why ``definition`` cannot join the others: it and another would compute each other.

### def `_changed` (enrichment_derived.py)

- **Arguments**: `notice: str`

- **Returns**: `None`

No description available.

### def `_current` (enrichment_derived.py)

- **Arguments**: `target: str`

- **Returns**: `ComplexType`

No description available.

### def `_put` (enrichment_derived.py)

- **Arguments**: `definition: DerivedVariable`

- **Returns**: `None`

``definition`` in the list, where its variable's was (at the end for a new one).

### def `_clear_form` (enrichment_derived.py)

- **Arguments**: ``

- **Returns**: `None`

No description available.

### def `_save` (enrichment_derived.py)

- **Arguments**: `definition: DerivedVariable`

- **Returns**: `None`

No description available.

### def `_edit` (enrichment_derived.py)

- **Arguments**: `target: str`

- **Returns**: `None`

No description available.

### def `_remove` (enrichment_derived.py)

- **Arguments**: `target: str`

- **Returns**: `None`

No description available.

### def `_accept` (enrichment_derived.py)

- **Arguments**: `target: str, key: str`

- **Returns**: `None`

No description available.

### def `_add_inverse` (enrichment_derived.py)

- **Arguments**: `target: str, name_key: str, formula_key: str`

- **Returns**: `None`

No description available.

### def `_remove_inverse` (enrichment_derived.py)

- **Arguments**: `target: str, name: str`

- **Returns**: `None`

No description available.

### def `_summary` (enrichment_derived.py)

- **Arguments**: `definition: DerivedVariable, check: DerivedCheck, to_fill: int`

- **Returns**: `str`

The line of a definition: its variable, its check, what it would fill, whether it is used.

### def `_disagreeing` (enrichment_derived.py)

- **Arguments**: `definition: DerivedVariable, check: DerivedCheck, data: pd.DataFrame`

- **Returns**: `None`

No description available.

### def `_inverses` (enrichment_derived.py)

- **Arguments**: `definition: DerivedVariable`

- **Returns**: `None`

Its inverses: each fills a missing variable of the formula from the others (a weight from the BMI).

### def `_new_inverse` (enrichment_derived.py)

- **Arguments**: `definition: DerivedVariable, check: DerivedCheck`

- **Returns**: `None`

No description available.

### def `_details` (enrichment_derived.py)

- **Arguments**: `definition: DerivedVariable, check: DerivedCheck, data: pd.DataFrame`

- **Returns**: `None`

No description available.

### def `_form` (enrichment_derived.py)

- **Arguments**: `data: pd.DataFrame, definitions: tuple[ComplexType]`

- **Returns**: `None`

The one form that adds a definition, or edits one (its inverses kept), checked before it is saved.

### def `_find` (enrichment_derived.py)

- **Arguments**: `data: pd.DataFrame, engine`

- **Returns**: `None`

One request for the whole dataset; its proposals are kept for this version of the data.

### def `_adopt` (enrichment_derived.py)

- **Arguments**: `indexes: list[int]`

- **Returns**: `None`

The chosen proposals join the list: never one the data disagree with, one per variable, and none that would
compute another definition's variable from its own.

### def `_dismiss` (enrichment_derived.py)

- **Arguments**: ``

- **Returns**: `None`

No description available.

### def `_proposals` (enrichment_derived.py)

- **Arguments**: `definitions: tuple[ComplexType]`

- **Returns**: `None`

The proposals of the AI engine, each checked against the data, the best evidence first, for the user to
choose: nothing is added by itself.

### def `_fill_exactly` (enrichment_derived.py)

- **Arguments**: ``

- **Returns**: `None`

Stage the fills in the Enrich draft: planned again on the current data, as the table shown was.

### def `_counted` (enrichment_derived.py)

- **Arguments**: `fill`

- **Returns**: `str`

No description available.

### def `_fill` (enrichment_derived.py)

- **Arguments**: `data: pd.DataFrame, definitions: tuple[ComplexType], scope`

- **Returns**: `None`

No description available.

### def `_ai_search` (enrichment_derived.py)

- **Arguments**: `data: pd.DataFrame, definitions: tuple[ComplexType]`

- **Returns**: `None`

No description available.

### def `render_derived_variables` (enrichment_derived.py)

- **Arguments**: ``

- **Returns**: `None`

No description available.

---

## enrichment_mi.py

_Multiple imputation on the Enrich page (follow-up programme M2): a guided panel in five steps.

1. Population and missingness: what will be imputed, on which rows, how many imputations.
2. Imputation model: the variables (the outcome of the planned analyses always in) and a method each.
3. Settings: m, iterations, seed, strata, the auxiliary of a Cox analysis, parallel processes.
4. Run and diagnose: chain traces, R-hat, observed against imputed values.
5. Use, save, export: status against the analysis population, versions, mice and Stata files, methods text.

The imputation never changes the dataset: it is kept beside it (``AppState.set_mi``), recorded in the trace
and re-computed by the replay. The panel runs as a fragment so its widgets do not rerun the whole page._

**Imports**:
`__future__.annotations`, `numpy`, `pandas`, `plotly.graph_objects`, `streamlit`, `biostats._types.RANDOM_STATE`, `biostats.mi.report.default_model_columns`, `biostats.mi.report.incomplete_share`, `biostats.mi.report.methods_paragraph`, `biostats.mi.report.missingness_summary`, `biostats.mi.spec.ImputationSpec`, `biostats.mi.spec.default_method`, `biostats.mi.spec.is_numeric`, `biostats.mi.spec.suggest_m`, `core.derived.decided_on_data`, `core.derived.record`, `core.registry.as_code`, `core.session.PopulationError`, `enrich.derived.DERIVED_METHOD`, `enrich.derived.derivation_steps`, `enrich.derived.impute_and_derive`, `enrich.derived.recomputed`, `ui.components.error_card`, `ui.components.help_popover`, `ui.components.text.plain`, `ui.state.get_state`

### def `_status_badge` (enrichment_mi.py)

- **Arguments**: `state`

- **Returns**: `None`

No description available.

### def `render_multiple_imputation` (enrichment_mi.py)

- **Arguments**: ``

- **Returns**: `None`

No description available.

### def `_trace_figure` (enrichment_mi.py)

- **Arguments**: `trace: dict, name: str`

- **Returns**: `go.Figure`

No description available.

### def `_computed_share` (enrichment_mi.py)

- **Arguments**: `mi, variable: str`

- **Returns**: `float`

The share of the rows whose derived variable was computed from imputed values (a variable of its formula was
imputed there), from the formula's variables the imputation recorded.

### def `_distribution_figure` (enrichment_mi.py)

- **Arguments**: `observed: pd.Series, imputed: pd.Series, numeric: bool`

- **Returns**: `go.Figure`

No description available.

### def `_diagnostics` (enrichment_mi.py)

- **Arguments**: `stored`

- **Returns**: `None`

No description available.

### def `_use` (enrichment_mi.py)

- **Arguments**: `state, stored`

- **Returns**: `None`

No description available.

---

## epidemiology.py

_No module description._

**Imports**:
`matplotlib.pyplot`, `numpy`, `pandas`, `plotly.express`, `plotly.graph_objects`, `streamlit`, `scipy.stats`, `scipy.stats.t`, `statsmodels.stats.power.TTestIndPower`, `app_pages.group_figures`, `utils.analysis_utils`, `utils.visualization_utils`, `app_pages.analysis_config.render_analysis_configuration`, `app_pages.analysis_diagnostic.QUESTION_KEY`, `app_pages.analysis_diagnostic.run_diagnostic_accuracy`, `app_pages.analysis_extras.QUESTION_KEY`, `app_pages.analysis_extras.run_paired_and_rates`, `app_pages.analysis_guidance_ui.MAP_KEY`, `app_pages.analysis_guidance_ui.guidance_on`, `app_pages.analysis_guidance_ui.guidance_toggle`, `app_pages.analysis_guidance_ui.render_ancova_check`, `app_pages.analysis_guidance_ui.render_orientation`, `app_pages.analysis_guidance_ui.render_plan`, `app_pages.analysis_guidance_ui.render_test_note`, `app_pages.clean_dialog.group_warning`, `biostats.multiplicity.adjust`, `biostats.power.ARE_NORMAL`, `biostats.power.ARE_WORST_CASE`, `biostats.power.anova_power`, `biostats.power.anova_sample_size`, `biostats.power.chi2_independence_power`, `biostats.power.chi2_independence_sample_size`, `biostats.power.kruskal_power`, `biostats.power.kruskal_sample_size`, `biostats.zscores.reference_zscores`, `core.fingerprint.frame_fingerprint`, `ui.components.error_card`, `ui.components.linked_tabs`, `ui.components.no_dataset`, `ui.components.result_card`, `ui.components.levels.show_level`, `ui.components.text.plain`, `ui.guidance.measures`, `ui.guidance.analysis.HELP`, `ui.guidance.analysis.ancova_check`, `ui.guidance.analysis.cohens_d_from_means`, `ui.guidance.analysis.cohens_f_from_means`, `ui.guidance.analysis.cohens_w_from_proportions`, `ui.guidance.analysis.p_superiority_from_d`, `ui.guidance.analysis.parse_numbers`, `ui.guidance.analysis.plan_group_comparison`, `ui.guidance.analysis.power_sentence`, `ui.guidance.analysis.sample_size_sentence`, `ui.progress.mark_step`, `ui.results.CURRENT`, `ui.results.provenance`, `ui.results.recall`, `ui.results.remember`, `ui.results.status`, `ui.state.get_state`, `utils.export_utils.to_excel`

### def `_effect_ok` (epidemiology.py)

- **Arguments**: `effect_size`

- **Returns**: `bool`

A planning effect must be positive: with no difference to detect, no sample is large enough.

### def `_prefilled_number_input` (epidemiology.py)

- **Arguments**: `label, default`

- **Returns**: `None`

A number input that the results above may prefill through the session state ("Load Parameters into
Calculator", the group sizes of the dataset). Streamlit warns when a widget gets both a default value and a
session-state value, so the default is passed only while the key is unset.

### def `_expected_means_inputs` (epidemiology.py)

- **Arguments**: `prefix: str`

- **Returns**: `None`

Expected means of two groups and the common SD; returns (mean1, mean2, sd).

### def `_render_twobytwo_controls` (epidemiology.py)

- **Arguments**: `df, group_col, outcome_col, default_epi`

- **Returns**: `None`

Reference-group / event-level / design selectors; recomputes the 2x2 analysis with explicit orientation.

### def `_render_twobytwo_panel` (epidemiology.py)

- **Arguments**: `epi`

- **Returns**: `None`

Labelled 2x2 table, measures of association (comparison vs reference) and guidance.

### def `open_analysis` (epidemiology.py)

- **Arguments**: `tab: str, choices: dict[ComplexType]`

- **Returns**: `None`

Open a tab of the page, and the analysis ``choices`` name on it (a selector's key and the id it takes): the
``on_click`` of a button, since a widget's key is set before the widget is drawn.

### def `_transform_of` (epidemiology.py)

- **Arguments**: `df: pd.DataFrame`

- **Returns**: `tuple[ComplexType]`

The transformation the Analysis tab applied to ``df`` and the columns it changed: part of each result's
settings, so that a result computed on other values is never shown or stored as current.

### def `_analysis_options` (epidemiology.py)

- **Arguments**: `mi`

- **Returns**: `tuple[ComplexType]`

The first row of the Analysis tab: the kind of analysis, and a transformation of the numeric variables (none
with imputed data). Only this tab reads them: they stood above the tabs, where they seemed to act on every one.

### def `_transformed` (epidemiology.py)

- **Arguments**: `df: pd.DataFrame, numeric_cols: list, transformation: str`

- **Returns**: `pd.DataFrame`

The data of the Analysis tab, its numeric variables transformed as chosen on that tab. Computed in that tab
only (M5b): the other tabs analyse the data as they are. The frame says which transformation changed which
columns (``TRANSFORM_ATTR``): Log1p leaves a column with a negative value as it is.

### def `app` (epidemiology.py)

- **Arguments**: `mi`

- **Returns**: `None`

Main application function for the Epidemiology Analysis page.

Handles:
1. Univariate Group Comparisons (T-tests, ANOVA, Chi-Square, etc.).
2. Multivariate Analysis (ANCOVA) to control for confounders.
3. Power Analysis & Sample Size Calculation.
4. Z-Score Calculation (Reference Standardization).
5. Educational Content display.

With multiply imputed data (``mi``, see ui.components.mi_mode) the group comparison and the ANCOVA are
pooled over the imputed datasets (app_pages.analysis_pooled).

### def `run_univariate_analysis` (epidemiology.py)

- **Arguments**: `df, analyzer, transformed`

- **Returns**: `None`

No description available.

### def `run_multivariate_analysis` (epidemiology.py)

- **Arguments**: `df, analyzer`

- **Returns**: `None`

No description available.

### def `show_ancova` (epidemiology.py)

- **Arguments**: `df, res, target, group, covariates`

- **Returns**: `None`

ANCOVA results (biostats.linear): Type II table, interpretation, adjusted means, coefficients, diagnostics.

### def `visualize_result` (epidemiology.py)

- **Arguments**: `df, group_col, target, analyzer`

- **Returns**: `None`

No description available.

### def `run_zscore_analysis` (epidemiology.py)

- **Arguments**: `df, analyzer`

- **Returns**: `None`

No description available.

### def `show_educational_content` (epidemiology.py)

- **Arguments**: ``

- **Returns**: `None`

No description available.

### def `run_power_analysis` (epidemiology.py)

- **Arguments**: `df, analyzer`

- **Returns**: `None`

No description available.

---

## explore.py

_Explore page: descriptive statistics, column renaming, correlations, performance benchmark._

**Imports**:
`app_pages.home.app`, `ui.components.page_header`, `ui.components.population.population_banner`

---

## group_figures.py

_The figures of a group comparison (Analyze > Hypothesis tests, the detailed view), built to be read side by side:
- the distribution of each group (box plot, with the significance brackets and n);

- the mean of each group with its 95% confidence interval;

- the histograms, as the percentage of each group, so that groups of different sizes compare;

- a normal Q-Q plot of each group, standardised within the group, which matches the normality check of each group
  in the result card.

One colour per group in every figure (the application's colour-blind-safe palette), the application's Plotly
template, and one height, so that a two-column grid stays readable on any screen. No Streamlit here._

**Imports**:
`__future__.annotations`, `numpy`, `pandas`, `plotly.express`, `plotly.graph_objects`, `scipy.stats`, `utils.visualization_utils`, `ui.theme.OKABE_ITO`, `ui.theme.PLOTLY_TEMPLATE_NAME`, `ui.theme.plotly_template`

### def `group_order` (group_figures.py)

- **Arguments**: `df: pd.DataFrame, group: str`

- **Returns**: `list[str]`

The groups in the order of the box plot (first appearance), as text.

### def `group_colors` (group_figures.py)

- **Arguments**: `levels: list[str]`

- **Returns**: `dict[ComplexType]`

No description available.

### def `_values` (group_figures.py)

- **Arguments**: `df: pd.DataFrame, group: str, target: str`

- **Returns**: `dict[ComplexType]`

No description available.

### def `_style` (group_figures.py)

- **Arguments**: `fig: go.Figure, title: str`

- **Returns**: `go.Figure`

No description available.

### def `box_figure` (group_figures.py)

- **Arguments**: `df: pd.DataFrame, group: str, target: str`

- **Returns**: `ComplexType`

Each group's distribution, its mean as a dashed line, the significance brackets and n under each box.

### def `means_figure` (group_figures.py)

- **Arguments**: `df: pd.DataFrame, group: str, target: str`

- **Returns**: `go.Figure`

The mean of each group with its 95% confidence interval (t distribution).

### def `histogram_figure` (group_figures.py)

- **Arguments**: `df: pd.DataFrame, group: str, target: str`

- **Returns**: `go.Figure`

Overlaid histograms as the percentage of each group, semi-transparent with outlines.

### def `qq_figure` (group_figures.py)

- **Arguments**: `df: pd.DataFrame, group: str, target: str`

- **Returns**: `go.Figure`

A normal Q-Q plot of each group, standardised within the group: on the line when the group is normal.

---

## handbook.py

_Handbook page: the researcher documentation rendered in the app (works offline)._

**Imports**:
`re`, `pathlib.Path`, `streamlit`, `ui.components.page_header`

### def `_chapters` (handbook.py)

- **Arguments**: ``

- **Returns**: `list[tuple[ComplexType]]`

(stem, title) for every top-level chapter, in file order.

### def `_anchor` (handbook.py)

- **Arguments**: `text: str`

- **Returns**: `str`

No description available.

---

## home.py

_No module description._

**Imports**:
`hashlib`, `html`, `json`, `time`, `io.BytesIO`, `duckdb`, `numpy`, `pandas`, `streamlit`, `explore.corr_matrix`, `core.fingerprint.frame_fingerprint`, `core.registry.as_code`, `graph.builders.correlation.build_correlation_graph`, `graph.model.LayoutHint`, `graph.model.RenderOptions`, `manage.ai_payload.column_profile`, `ui.components.confirm`, `ui.components.no_dataset`, `ui.components.text.plain`, `ui.graph_legend.render_legend`, `ui.graph_view.current_theme`, `ui.graph_view.render_graph`, `ui.graph_view.render_graph_exports`, `ui.graph_view.renderer_selector`, `ui.progress.mark_step`, `ui.state.get_state`, `utils.export_utils.to_excel`, `utils.export_utils.to_excel_sheets`, `utils.statistics_utils.normality_test`, `utils.statistics_utils.sanitize_dataframe_for_arrow`

### class `RenameColumnsComponent` (home.py)

A Streamlit component for renaming DataFrame columns with sorting, searching, type filtering, and top values display.

**Methods:**

- **__init__**(`self, df, analyzer`) -> `None`
  > No description available.

- **_get_columns_info**(`self, columns_to_process`) -> `None`
  > What the AI engine is told about the columns: a summary, never a row (manage.ai_payload).

- **display**(`self`) -> `None`
  > Displays the column renaming UI with sorting, searching, type filtering, and a compact layout.

### def `_q` (home.py)

- **Arguments**: `identifier`

- **Returns**: `str`

Quote a DuckDB identifier (column names come from uploaded files; audit E4).

### def `_frame_fingerprint` (home.py)

- **Arguments**: `df`

- **Returns**: `str`

Short content hash so cached statistics follow the frame (filters, renames), not only its name (audit E8).

### def `_table_key` (home.py)

- **Arguments**: `df: pd.DataFrame`

- **Returns**: `str`

How the caches below tell two tables apart: by every row, where Streamlit hashes a sample of 10,000 rows of a
table of 50,000 or more (two versions of a cohort that differed outside the sample could get each other's
results). Column by column: a column pandas cannot hash (lists in its cells) is hashed as text on its own; a
categorical column's categories and order, and the kinds of value an object column holds (1 and "1" hash alike),
are part of the key. Computed once per run by the caller, which passes the table itself unhashed.

### def `_owner` (home.py)

- **Arguments**: ``

- **Returns**: `ComplexType`

The workspace the cached results below belong to (the signed-in user's folder): part of their cache key, so
that a cache shared by the sessions of the server, or a session another user signs in to, never serves them.

### def `_top_correlations` (home.py)

- **Arguments**: `corr_matrix: pd.DataFrame, node_ids, limit: int`

- **Returns**: `pd.DataFrame`

Strongest correlations of the selected variables (for the graph selection panel).

### def `display_category_box` (home.py)

- **Arguments**: `title, columns`

- **Returns**: `None`

No description available.

### def `move_columns_to_front` (home.py)

- **Arguments**: `df, columns_to_move`

- **Returns**: `None`

Move specified columns to the beginning of the DataFrame.

Parameters:
df (pd.DataFrame): The original DataFrame.
columns_to_move (list): List of column names to move to the front.

Returns:
pd.DataFrame: DataFrame with specified columns moved to the front.

### def `get_statistics_dataframe` (home.py)

- **Arguments**: `_df, _analyzer, nb_top_categories, exclude_columns, qual_var_threshold, dataset_name, _db_manager`

- **Returns**: `None`

Creates new DataFrames with statistics for numerical and non-numerical columns.
Tries to load from DuckDB/Parquet cache if dataset_name is provided.

Note: Parameters starting with _ are excluded from hashing by Streamlit: the table is known by ``table_key``
(:func:`_table_key`), and ``owner`` is the user's workspace.

### def `_date_stats_for_excel` (home.py)

- **Arguments**: `stats: pd.DataFrame`

- **Returns**: `pd.DataFrame`

The date statistics as a workbook holds them: the duration in days, since Excel has no duration type (it
became an unlabelled number).

### def `_cached_corr_excel_bytes` (home.py)

- **Arguments**: `_data, targets, predictors, group_column, method`

- **Returns**: `None`

The correlation workbook of a table (known by ``table_key``, :func:`_table_key`) for one user's workspace.

### def `run_benchmark` (home.py)

- **Arguments**: `df, analyzer`

- **Returns**: `None`

No description available.

### def `app` (home.py)

- **Arguments**: ``

- **Returns**: `None`

No description available.

---

## import_data.py

_Import page: the three-step import flow._

**Imports**:
`ui.components.page_header`, `ui.import_flow.render_import_flow`, `ui.state.get_state`

---

## rag_monitoring.py

_RAG Monitoring Dashboard

Streamlit page for monitoring and visualizing RAG system quality metrics.
Provides dashboards for:
- Quality overview with trends

- Evaluation details and drill-down

- Embedding health visualization

- Manual evaluation triggers_

**Imports**:
`json`, `sys`, `datetime.datetime`, `pathlib.Path`, `streamlit`, `ui.components.text.plain`

### def `render_rag_monitoring` (rag_monitoring.py)

- **Arguments**: ``

- **Returns**: `None`

Main entry point for the RAG Monitoring page.

### def `render_quality_overview` (rag_monitoring.py)

- **Arguments**: `evaluator`

- **Returns**: `None`

Render the quality overview dashboard.

### def `render_evaluation_details` (rag_monitoring.py)

- **Arguments**: `evaluator`

- **Returns**: `None`

Render the evaluation details view.

### def `render_embedding_health` (rag_monitoring.py)

- **Arguments**: `evaluator`

- **Returns**: `None`

Render embedding health visualization.

### def `render_run_evaluation` (rag_monitoring.py)

- **Arguments**: `evaluator`

- **Returns**: `None`

Render the manual evaluation trigger with progress tracking.

### def `run_single_evaluation` (rag_monitoring.py)

- **Arguments**: `evaluator, rag, query: str, eval_function: str`

- **Returns**: `None`

Run a single evaluation with progress tracking.

### def `run_test_suite_with_progress` (rag_monitoring.py)

- **Arguments**: `evaluator, rag, test_cases: list, selected_ids: list`

- **Returns**: `None`

Run test suite with progress tracking.

### def `run_demo_evaluation` (rag_monitoring.py)

- **Arguments**: `evaluator`

- **Returns**: `None`

Run a demo evaluation without the full RAG system.

### def `render_export_options` (rag_monitoring.py)

- **Arguments**: `evaluator`

- **Returns**: `None`

Render export buttons.

---

## rag_sidebar.py

_AI engine configuration (Settings page).

Provider-agnostic controls for the AI providers (local Ollama by default; Google Gemini, OpenAI, Mistral
and Anthropic Claude as cloud providers that require an explicit data-sharing consent; Claude has no
embedding model, so another provider embeds the documents), model and embedding
selection, the document library, retrieval parameters and engine initialisation. Laid out as cards:
provider and models, data sharing and document library on the left; retrieval, the vector database
(LanceDB or Qdrant, see ``manage.vectorstores``) and the engine on the right. Widget keys are stable
because other pages read them from the session._

**Imports**:
`os`, `dataclasses.dataclass`, `datetime.datetime`, `pathlib.Path`, `urllib.parse.urlparse`, `streamlit`, `core.registry.Selection`, `core.registry.as_code`, `core.workspace.remove_pdf`, `core.workspace.safe_filename`, `manage.ai_payload.cloud_model`, `manage.ai_payload.host_of`, `manage.ai_payload.on_this_machine`, `manage.ai_payload.outside_private_network`, `manage.ai_payload.provider_label`, `manage.ai_payload.recipient`, `manage.ai_payload.server_recipient`, `manage.ai_payload.store_recipient`, `manage.iam.authz.is_admin_role`, `manage.rag.ConnectorFactory`, `manage.rag.ProviderConfig`, `manage.rag.ProviderType`, `manage.rag.providers.AI_PROVIDERS`, `manage.rag.providers.declared`, `manage.rag.providers.default_provider`, `manage.rag.providers.embedding_providers`, `manage.rag.providers.file_readers`, `manage.rag.providers.provider_setting`, `manage.rag.providers.server_key`, `manage.server_addresses.ADDRESS_REMOVED`, `manage.server_addresses.NAME_MAX_LENGTH`, `manage.server_addresses.OLLAMA_ADDRESSES`, `manage.server_addresses.QDRANT_ADDRESSES`, `manage.server_addresses.Added`, `manage.server_addresses.AddressList`, `manage.server_addresses.NotAllowed`, `manage.server_addresses.address_name`, `manage.vectorstores.VECTOR_STORES`, `manage.vectorstores.IndexManifest`, `manage.vectorstores.MachineProfile`, `manage.vectorstores.VectorEngine`, `manage.vectorstores.VectorStoreSettings`, `manage.vectorstores.estimate_chunks`, `manage.vectorstores.probe_machine`, `manage.vectorstores.qdrant_embedded_warnings`, `manage.vectorstores.recommend`, `manage.vectorstores.qdrant_index.QdrantIndex`, `manage.vectorstores.qdrant_index.adopt_earlier_index`, `manage.vectorstores.qdrant_index.explicit_address`, `manage.vectorstores.qdrant_index.server_folder`, `manage.vectorstores.qdrant_index.server_indexes`, `manage.vectorstores.qdrant_index.unrecorded_collection`, `ui.preferences`, `ui.components.confirm`, `ui.components.connector_selector.connector_selector`, `ui.components.connector_selector.explain_choice`, `ui.components.text.plain`, `ui.services.ai_engine`, `ui.session_scope.get_current_scope`, `utils.data_paths.get_vectors_dir`, `utils.settings.ADDRESS_MAX_LENGTH`

### class `_Place` (rag_sidebar.py)

What Settings needs to offer one kind of server's addresses, its sentences written in full: its list, the
settings that set the deployment's (the first address's, then the others'), the option that uses no server (Qdrant's
embedded mode), and what is said.

### def `_documents_dir` (rag_sidebar.py)

- **Arguments**: ``

- **Returns**: `str`

The signed-in user's document library (audit E4); the legacy shared folder outside a session.

### def `_discover_ollama_models` (rag_sidebar.py)

- **Arguments**: `base_url: str`

- **Returns**: `list[str]`

Ollama model list, probed at most once a minute per endpoint (audit E11).

### def `_runs_at` (rag_sidebar.py)

- **Arguments**: `base_url: str, model: str`

- **Returns**: `ComplexType`

Where the Ollama server runs a model when not itself (a cloud model, a copy of one), asked at most once a
minute per endpoint and model, as the engine asks it when it starts. A silence raises NoAnswer, which is never
kept (a call that raises is not cached): Settings never states where a model runs from a server that did not say.

### def `_remote_models` (rag_sidebar.py)

- **Arguments**: `base_url: str`

- **Returns**: `dict[ComplexType]`

The models the Ollama server lists that it runs elsewhere, to mark them in the list.

### def `_discover_cloud_models` (rag_sidebar.py)

- **Arguments**: `_manager, provider: str, api_key: str`

- **Returns**: `list[str]`

Cloud model list, fetched at most once an hour per key (audit E11).

### def `_api_key_input` (rag_sidebar.py)

- **Arguments**: `provider: str, label: str, key: str, help_text: str`

- **Returns**: `ComplexType`

Session-scoped API key entry.

The server-side key is never shown or pre-filled and nothing is written to ``os.environ``
(audit E6). Leaving the field empty falls back to the server-configured key, if any.

### def `_ensure_manager` (rag_sidebar.py)

- **Arguments**: ``

- **Returns**: `None`

No description available.

### def `_running` (rag_sidebar.py)

- **Arguments**: ``

- **Returns**: `dict`

The provider and chat model of the engine that runs (empty when it does not run).

### def `_address_badge` (rag_sidebar.py)

- **Arguments**: `url: str, carried: str`

- **Returns**: `None`

Where an address (an Ollama's, a Qdrant server's) sends the data: the kind of place in a badge, the host in a
caption that wraps (in a badge, a Qdrant Cloud host was cut before its region, which says where the data go).

### def `_is_admin` (rag_sidebar.py)

- **Arguments**: ``

- **Returns**: `bool`

No description available.

### def `_account` (rag_sidebar.py)

- **Arguments**: ``

- **Returns**: `str`

The signed-in account's name, as the user signs in (the workspace folder may be a code of it).

### def `_forget` (rag_sidebar.py)

- **Arguments**: `name: str`

- **Returns**: `None`

No description available.

### def `_remember` (rag_sidebar.py)

- **Arguments**: `key: str, place: _Place`

- **Returns**: `None`

Keep the address chosen in ``key`` for the next session: only one the list holds ("" for the embedded Qdrant).
An on_change runs before the page, while the session holds the browser's raw value: a forged one is never kept.

### def `_chosen` (rag_sidebar.py)

- **Arguments**: `key: str, place: _Place`

- **Returns**: `None`

A choice's on_change: the note of a removed choice goes, and the new one is kept for the next session.

### def `_shown` (rag_sidebar.py)

- **Arguments**: `address: str`

- **Returns**: `str`

No description available.

### def `_options` (rag_sidebar.py)

- **Arguments**: `place: _Place`

- **Returns**: `list[str]`

No description available.

### def `_named` (rag_sidebar.py)

- **Arguments**: `value: str, names: ComplexType`

- **Returns**: `str`

A choice in a sentence: its name first, when an administrator gave one, then the address as code; or the
server-free option.

### def `_labelled` (rag_sidebar.py)

- **Arguments**: `value: str, names: dict[ComplexType]`

- **Returns**: `str`

An option of the list: its name, when an administrator gave one, then the address (cut in its middle).

### def `_seed_choice` (rag_sidebar.py)

- **Arguments**: `key: str, place: _Place`

- **Returns**: `None`

A choice's first value in a session: the one kept from the last session while the list holds it. A kept one the
list no longer holds is named, and the first option is kept in its place, as a removal within a session does (for
Qdrant, the embedded one, never the deployment's server). Else the deployment's first address (Qdrant: its server,
else the embedded Qdrant).

### def `_short` (rag_sidebar.py)

- **Arguments**: `address: str, limit: int`

- **Returns**: `str`

An address in a narrow list, cut in its middle past ``limit`` characters: its end (a Qdrant Cloud host's region
and port) stays readable.

### def `_address_choice` (rag_sidebar.py)

- **Arguments**: `key: str, label: str, help_text: str, place: _Place`

- **Returns**: `str`

A server's address chosen from the server's list (manage.server_addresses), never typed: the server sends no
request to an address a user writes, which on a shared server could reach a machine of its network. A value the
list does not hold is never used: an address chosen, then removed by an administrator, is named with the one used
instead; one the browser forged is replaced by the first option. The caller draws the administrators' list
(``_manage_addresses``) below the choice and its badge.

### def `_day` (rag_sidebar.py)

- **Arguments**: `moment: str`

- **Returns**: `str`

A date in a row, never broken across two lines ("9 Nov 2026", its spaces no-break ones).

### def `_manage_addresses` (rag_sidebar.py)

- **Arguments**: `key: str, place: _Place`

- **Returns**: `None`

The administrators' list of a server's addresses every user may choose, below the choice and its badge: add one
(offered at once, and chosen here; Enter adds), rename or remove one added here; the deployment's are changed in its
settings.
The form and its outcome come first, the list after, in a box of its own height, so that the popover keeps one size.

### def `_rename_form` (rag_sidebar.py)

- **Arguments**: `key: str, place: _Place, item: Added`

- **Returns**: `None`

The row of the address being renamed: a box holding its name (left empty, no name), saved with Enter.

### def `_start_rename` (rag_sidebar.py)

- **Arguments**: `key: str, item: Added`

- **Returns**: `None`

Rename's callback: the row turns into a box holding the address's name.

### def `_rename_address` (rag_sidebar.py)

- **Arguments**: `key: str, place: _Place, item: Added`

- **Returns**: `None`

Save's callback: the new name, for every user ("" removes it); the address this page uses is left as it is.

### def `_choose` (rag_sidebar.py)

- **Arguments**: `key: str, place: _Place, address: str`

- **Returns**: `None`

Choose ``address`` on this page, and keep it for the next session, when the list holds it.

### def `_add_address` (rag_sidebar.py)

- **Arguments**: `key: str, place: _Place`

- **Returns**: `None`

Add's callback (Enter in either box too): the address typed is stored for every user (an administrator only), and
chosen here. One already listed is left chosen or not: a name typed for one added here renames it, and one the
deployment sets takes no name.

### def `_add_back` (rag_sidebar.py)

- **Arguments**: `key: str, place: _Place, removed: tuple[ComplexType]`

- **Returns**: `None`

Add it back's callback: the address just removed, added again with its name; chosen again only on the page that
had it chosen, as it was.

### def `_remove_address` (rag_sidebar.py)

- **Arguments**: `key: str, place: _Place, address: str`

- **Returns**: `None`

Remove's callback: every user's AI engine on the address stops at its next use (ui.services); this page moves to
the first option when it had chosen this one, and says so.

### def `_provider_card` (rag_sidebar.py)

- **Arguments**: ``

- **Returns**: `dict`

Provider, connection and model choice; returns the engine configuration.

### def `_silence` (rag_sidebar.py)

- **Arguments**: `cfg: dict`

- **Returns**: `str`

Why a server's answer on where it runs the chosen model is missing, the model and the address named.

### def `_seed_model` (rag_sidebar.py)

- **Arguments**: `key: str, options: list[str], running: ComplexType, saved: ComplexType, kind: str`

- **Returns**: `list[str]`

The model a control starts from, when the session has none: the running engine's, then the one kept from the last
session. A saved model that is no longer offered is not used: the notice returned says so, and the file keeps it
until the user chooses again.

### def `_claude_embedding_block` (rag_sidebar.py)

- **Arguments**: `cfg: dict`

- **Returns**: `tuple[ComplexType]`

Claude has no embedding model: choose the provider that embeds the documents (local Ollama by default).

### def `_roles` (rag_sidebar.py)

- **Arguments**: `cfg: dict, vector: ComplexType`

- **Returns**: `tuple[ComplexType]`

The recipients that need the confirmation, by role: the chat provider when it is a cloud one, an Ollama model run
elsewhere (its tag, or the server's ``remote_host``) or an Ollama outside the private network; the embedding
provider of Claude when it is another such recipient (it receives the documents and every search text); the
Qdrant server outside the private network that keeps the index; the Ollama server outside the private network
when its model runs elsewhere (it receives the prompts too).

### def `_recipients` (rag_sidebar.py)

- **Arguments**: `cfg: dict, vector: ComplexType`

- **Returns**: `list[str]`

The recipients of a configuration that need the confirmation, as one box confirms them together.

### def `_names` (rag_sidebar.py)

- **Arguments**: `recipients: list[str], embedding: ComplexType`

- **Returns**: `str`

No description available.

### def `_consent_key` (rag_sidebar.py)

- **Arguments**: `recipients: list[str]`

- **Returns**: `str`

The session key of the confirmation of exactly these recipients.

### def `_consent_recipients` (rag_sidebar.py)

- **Arguments**: `key: str`

- **Returns**: `list[str]`

No description available.

### def `_consent_changed` (rag_sidebar.py)

- **Arguments**: `key: str, recipients: tuple[ComplexType]`

- **Returns**: `None`

The user ticked or unticked a box: only then does the confirmation change. Acting on how a box is shown made
an untouched box for another configuration withdraw, and stop, the engine that runs.

### def `_notice` (rag_sidebar.py)

- **Arguments**: `recipients: list[str], cfg: dict, vector: ComplexType`

- **Returns**: `str`

What the recipients of one box receive, each for its role (a Qdrant server receives no column summary).

### def `_engine_recipients` (rag_sidebar.py)

- **Arguments**: `manager`

- **Returns**: `tuple[ComplexType]`

The recipients the running engine sends data to, by role as it uses them (never by position: a Qdrant server
is no embedding provider), and its embedding provider among them.

### def `_running_engine` (rag_sidebar.py)

- **Arguments**: `manager`

- **Returns**: `None`

Where the running engine sends data, whatever provider is selected below, and how to stop it at once.

### def `_consent_card` (rag_sidebar.py)

- **Arguments**: `cfg: dict, vector: ComplexType`

- **Returns**: `bool`

Explicit, session-scoped confirmation before any data is sent outside this machine and the private network
(audit E6). One box per set of recipients, naming each (with Claude, a cloud embedding provider too; a Qdrant
server there that keeps the index); ticking or unticking it is what counts, and unticking stops an engine that
sends data to one of them.

### def `_library_card` (rag_sidebar.py)

- **Arguments**: ``

- **Returns**: `ComplexType`

Upload PDFs into the user's document library, choose the ones to index and remove the others (the
index is rebuilt from the remaining documents at the next initialisation: ``manage.vectorstores.index_matches``).

### def `_retrieval_card` (rag_sidebar.py)

- **Arguments**: ``

- **Returns**: `tuple[ComplexType]`

No description available.

### def `_machine` (rag_sidebar.py)

- **Arguments**: `storage: str`

- **Returns**: `MachineProfile`

CPU, memory and free disk, probed at most every five minutes.

### def `_vectors_dir` (rag_sidebar.py)

- **Arguments**: ``

- **Returns**: `Path`

No description available.

### def `_vector_card` (rag_sidebar.py)

- **Arguments**: `selected_files: ComplexType`

- **Returns**: `dict`

Choice of the vector database for this machine and library; returns the engine settings.

### def `_engine_card` (rag_sidebar.py)

- **Arguments**: `cfg: dict, consent: bool, adherence: float, temperature: float, selected_files: ComplexType, vector: dict`

- **Returns**: `None`

No description available.

### def `render_ai_settings` (rag_sidebar.py)

- **Arguments**: ``

- **Returns**: `None`

AI provider, keys, consent, document library, retrieval parameters and engine (Settings page).

---

## reproduce.py

_Reproduce page: traces, artifacts, replay and reports._

**Imports**:
`app_pages.reproduce_analysis.app`, `ui.components.page_header`

---

## reproduce_analysis.py

_No module description._

**Imports**:
`difflib`, `json`, `os`, `pandas`, `streamlit`, `core.lineage.verify_trace`, `core.workspace.safe_filename`, `ingest.loaders.UPLOAD_TYPES`, `manage.paths.PathResolver`, `manage.paths.normalize_path`, `manage.replay.engine.NeededFile`, `manage.replay.engine.files_needed`, `manage.replay.engine.locate`, `manage.replay.engine.run_trace`, `manage.replay.engine.trace_problems`, `manage.trace_documenter.TraceDocumenter`, `ui.components.text.plain`, `ui.progress.current_trace_session`, `ui.progress.mark_step`, `ui.services.services`, `ui.session_scope.get_current_scope`, `ui.state.get_state`, `utils.data_paths.get_traces_dir`, `utils.data_paths.get_uploads_dir`

### def `_uploads_dir` (reproduce_analysis.py)

- **Arguments**: ``

- **Returns**: `str`

No description available.

### def `_traces_dir` (reproduce_analysis.py)

- **Arguments**: ``

- **Returns**: `str`

No description available.

### def `_artifacts_dir` (reproduce_analysis.py)

- **Arguments**: ``

- **Returns**: `str`

No description available.

### def `_resolver` (reproduce_analysis.py)

- **Arguments**: ``

- **Returns**: `PathResolver`

Where the files a trace names are looked for: the user's workspace (the session's services, manage.paths).

### def `_located` (reproduce_analysis.py)

- **Arguments**: `needed: NeededFile`

- **Returns**: `ComplexType`

The file of this workspace that stands for a file the trace needs, found as the replay finds it, with the files
chosen on this page for recorded paths (manage.replay.engine.locate).

### def `_missing_files` (reproduce_analysis.py)

- **Arguments**: `trace_data: dict`

- **Returns**: `dict[ComplexType]`

The files the replay needs and this workspace lacks, by recorded path: the source and each merge's file (a
snapshot is optional: without it, its step is skipped with a warning). A file read twice is asked for once.

### def `_resolve_user_path` (reproduce_analysis.py)

- **Arguments**: `text: ComplexType`

- **Returns**: `ComplexType`

A manually entered path must exist inside the user's workspace (audit E4).

### def `render_trace_integrity` (reproduce_analysis.py)

- **Arguments**: `trace: dict`

- **Returns**: `None`

The hash chain of the trace (schema 2): verified, not verifiable (older traces) or broken.

### def `app` (reproduce_analysis.py)

- **Arguments**: ``

- **Returns**: `None`

No description available.

### def `jump_to_step` (reproduce_analysis.py)

- **Arguments**: `trace_data, target_step_index, rerun_mode`

- **Returns**: `None`

Reproduces the analysis up to the target step index.

---

## settings.py

_Settings page: AI engine and consent, graph engine, import defaults, workspace maintenance._

**Imports**:
`streamlit`, `ui.components.confirm`, `ui.components.page_header`, `ui.preferences.DAYFIRST_DEFAULT`, `ui.preferences.IMPORT_DAYFIRST`, `ui.preferences.save_from`, `ui.shell.clear_workspace_statistics`, `ui.state.get_state`, `utils.optional.extra_installed`, `utils.optional.missing_message`

---

## start.py

_Start page: welcome banner, dataset figures, the three-step workflow, saved versions, sample cohort, AI status._

**Imports**:
`streamlit`, `core.progress.DONE`, `core.progress.STALE`, `ui.components.page_header`, `ui.components.page_link`, `ui.components.status_chip`, `ui.components.switch_to`, `ui.components.workflow_map.workflow_map`, `ui.progress.current_progress`, `ui.state.SAMPLE_DICTIONARY_PATH`, `ui.state.TUTORIAL_COHORT_PATH`, `ui.state.TUTORIAL_CONFIG_PATH`, `ui.state.TUTORIAL_DICTIONARY_PATH`, `ui.state.TUTORIAL_FOLLOWUP_PATH`, `ui.state.get_state`, `ui.workflow.BRANCHES`, `ui.workflow.CORE_STEPS`, `ui.workflow.Link`

### def `progress_chip` (start.py)

- **Arguments**: `step: str`

- **Returns**: `None`

The step's status for the current data, as on the map.

### def `card_links` (start.py)

- **Arguments**: `links: tuple[ComplexType]`

- **Returns**: `None`

A card's links side by side, each opening its page (on one of its tabs when it names one).

---

## study_dossier.py

_Reproduce > Study dossier (follow-up programme M3): the participant flow of the analysis population with the
analyses added to the report, the list of those results (current, or computed on other data or another
population), and the Word study dossier that documents them._

**Imports**:
`__future__.annotations`, `typing.Any`, `pandas`, `streamlit`, `core.attrition.AnalysisCount`, `core.criteria.describe_definition`, `core.lineage.verify_trace`, `core.population.family_words`, `core.state.checksum_text`, `core.steps.DOCUMENT`, `core.steps.LOAD`, `core.steps.POPULATION`, `core.steps.step_title`, `core.steps.ids`, `enrich.transformation_logic.added_columns`, `ui.components.no_dataset`, `ui.components.section`, `ui.components.status_chip`, `ui.components.flow.current_flow`, `ui.components.flow.render_flow`, `ui.components.study.status_text`, `ui.components.study.study_status`, `ui.components.study.study_strip`, `ui.results.CURRENT`, `ui.results.STALE_CODE`, `ui.results.STALE_DATA`, `ui.results.ReportItem`, `ui.results.remove_from_report`, `ui.results.report_items`, `ui.results.report_status`, `ui.state.get_state`, `utils.environment.environment_text`

### def `imputations_of` (study_dossier.py)

- **Arguments**: `value: Any`

- **Returns**: `ComplexType`

m when the result pools multiply imputed datasets (biostats.mi.analyze.MIResult), else None.

### def `analysis_counts` (study_dossier.py)

- **Arguments**: `items: list[ReportItem]`

- **Returns**: `list[AnalysisCount]`

What each result of the report analysed, for the boxes under the population.

### def `_stale_text` (study_dossier.py)

- **Arguments**: `status: str`

- **Returns**: `str`

No description available.

### def `_versions` (study_dossier.py)

- **Arguments**: ``

- **Returns**: `str`

No description available.

### def `_provenance_rows` (study_dossier.py)

- **Arguments**: `state, trace: dict`

- **Returns**: `list[tuple[ComplexType]]`

No description available.

### def `_criteria_table` (study_dossier.py)

- **Arguments**: `population`

- **Returns**: `pd.DataFrame`

No description available.

### def `_data_steps` (study_dossier.py)

- **Arguments**: `trace: dict`

- **Returns**: `list[tuple[ComplexType]]`

No description available.

### def `_missing_table` (study_dossier.py)

- **Arguments**: `frame: pd.DataFrame`

- **Returns**: `pd.DataFrame`

No description available.

### def `_mi_section` (study_dossier.py)

- **Arguments**: `state`

- **Returns**: `ComplexType`

No description available.

### def `session_parts` (study_dossier.py)

- **Arguments**: `state, items: list[ReportItem]`

- **Returns**: `dict[ComplexType]`

What the dossier needs from the session, read during the page run: the download builds the document later,
outside the run, where the session is not available (finish_input and build_dossier are pure).

### def `finish_input` (study_dossier.py)

- **Arguments**: `parts: dict[ComplexType]`

- **Returns**: `None`

The DossierInput from the session parts: result tables, figures and references (no session needed).

### def `dossier_input` (study_dossier.py)

- **Arguments**: `state, items: list[ReportItem]`

- **Returns**: `None`

Everything the Word dossier needs (manage.dossier.DossierInput), read from the session now.

### def `app` (study_dossier.py)

- **Arguments**: ``

- **Returns**: `None`

No description available.

### def `_word_bytes` (study_dossier.py)

- **Arguments**: `parts: dict[ComplexType]`

- **Returns**: `bytes`

No description available.

### def `build_dossier_bytes` (study_dossier.py)

- **Arguments**: `state, items: list[ReportItem]`

- **Returns**: `bytes`

No description available.

---

## survival.py

_Survival & regression page: Kaplan-Meier, log-rank, Cox, logistic and count regression._

**Imports**:
`streamlit`, `app_pages.analysis_survival.run_survival`, `ui.components.no_dataset`, `ui.components.page_header`, `ui.components.mi_mode.mi_banner`, `ui.components.population.population_banner`, `ui.state.get_state`

---

## users.py

_Users page (administrators)._

**Imports**:
`app_pages.users_management.app`, `ui.components.page_header`

---

## users_management.py

_Users page (Admin > Users), for administrators only

This page allows administrators to:
- View all registered users with their status

- Activate/Deactivate user accounts

- Reset user passwords

- Delete user accounts_

**Imports**:
`streamlit`, `manage.iam.IAMProviderType`, `manage.iam.get_iam_manager`, `manage.iam.authz.MIN_PASSWORD_LENGTH`, `manage.iam.authz.is_admin_role`, `manage.iam.providers.IAM_PROVIDERS`

### def `app` (users_management.py)

- **Arguments**: ``

- **Returns**: `None`

Render the Users page.

---

## validate.py

_Validate page: data quality dashboard, implausible values, inclusion criteria, outliers, version comparison._

**Imports**:
`app_pages.data_monitoring.app`, `ui.components.page_header`

---

## validate_dates.py

_Clean > Dates (Validate's Date checks tab until M6's Clean page): the text columns that look like dates, and their
conversion. Each column is read with one
format (ingest.dates): detected from the shape of its values, the order of day and month from the values that show
it, else asked and pre-filled from the import's day-first setting; two-digit years with a pivot; Excel serial numbers
only when accepted; the placeholder dates the user keeps ticked set to missing. The conversion is one step of the
trace (method parse_dates, each column's format recorded), which the replay reads with the same function, never
detecting again. The columns already dates (date cells read on load, or converted here) are counted the same way, and
their placeholder dates set to missing on request: one step recorded as a conversion with the format "dates". A tab
of its own since M6: above Validate's tabs, the conversion read those columns whenever Validate opened on new data,
whichever tab was wanted._

**Imports**:
`__future__.annotations`, `calendar`, `collections.abc.Collection`, `datetime.date`, `typing.Any`, `pandas`, `streamlit`, `core.registry.as_code`, `ingest.dates.AS_DATES`, `ingest.dates.CHOSEN`, `ingest.dates.DATE_FORMATS`, `ingest.dates.METHOD`, `ingest.dates.PLACEHOLDERS`, `ingest.dates.DateReport`, `ingest.dates.Detection`, `ingest.dates.convert`, `ingest.dates.date_report`, `ingest.dates.detect`, `ingest.dates.placeholders_found`, `ingest.dates.read_dates`, `ingest.dates.set_to_missing`, `ui.components.no_dataset`, `ui.preferences.import_dayfirst`, `ui.state.get_state`

### def `_loaded_on` (validate_dates.py)

- **Arguments**: `state`

- **Returns**: `date`

The day the dataset was loaded: the extraction date proposed.

### def `_dayfirst` (validate_dates.py)

- **Arguments**: `state`

- **Returns**: `bool`

The import's day-first setting, else the user's default for imports.

### def `detections` (validate_dates.py)

- **Arguments**: `df: pd.DataFrame`

- **Returns**: `dict[ComplexType]`

The columns that look like dates (detect() gives them a format, chosen or proposed), and how.

### def `_detected` (validate_dates.py)

- **Arguments**: `state`

- **Returns**: `dict[ComplexType]`

The dataset's detections, kept until the data or the day-first setting change.

### def `date_candidates` (validate_dates.py)

- **Arguments**: `state`

- **Returns**: `list[str]`

The text columns that look like dates, by the shape of their values: nothing is read as a date, and the result is
kept until the data change, so that the line above Validate's tabs can count them on every click (it links to
Clean > Dates).

### def `dates_found` (validate_dates.py)

- **Arguments**: `n: int`

- **Returns**: `str`

How many text columns look like dates, in the line above Validate's tabs and at the top of the tab.

### def `_label` (validate_dates.py)

- **Arguments**: `format_id: str`

- **Returns**: `str`

No description available.

### def `_spread` (validate_dates.py)

- **Arguments**: `dates: pd.Series`

- **Returns**: `str`

Where a reading puts the dates: every one in a month it names, or over how many months.

### def `_why` (validate_dates.py)

- **Arguments**: `detection: Detection, values: ComplexType`

- **Returns**: `str`

Why a column's reading is asked: counts only, never a value.

### def `_choose` (validate_dates.py)

- **Arguments**: `column: str, detection: Detection, values: ComplexType`

- **Returns**: `ComplexType`

How a column is read: as detected when the data decide, else as chosen here. None: left as text.

### def `_reports` (validate_dates.py)

- **Arguments**: `df: pd.DataFrame, chosen: dict[ComplexType], extraction: ComplexType, planned: list[str]`

- **Returns**: `tuple[ComplexType]`

Each chosen column read as it would be converted (before the placeholders are set to missing), and what the
reading found; a column that may hold planned dates is not counted after the extraction date.

### def `_day` (validate_dates.py)

- **Arguments**: `value`

- **Returns**: `str`

A date as no reader takes for the other order, 1 Jan 2024: the Earliest and Latest columns sit beside the order
being chosen, and 01/12/2024 reads as 1 December one way and 12 January the other.

### def `_placeholders` (validate_dates.py)

- **Arguments**: `chosen: Collection[str], readings: dict[ComplexType]`

- **Returns**: `dict[ComplexType]`

Per column, the placeholder dates set to missing, chosen from the whole catalogue: a list of only the dates a
column holds told the trace that a participant was born on 1 January 1970, and one choice for every column set a
real birth date to missing with a placeholder visit. The counts are this column's, on screen only.

### def `_study_dates` (validate_dates.py)

- **Arguments**: `state, options: list[str]`

- **Returns**: `tuple[ComplexType]`

The extraction date and the columns that may hold planned dates: asked once, for both kinds of columns.

### def `_heaping` (validate_dates.py)

- **Arguments**: `reports: dict[ComplexType]`

- **Returns**: `None`

No description available.

### def `_extraction_held` (validate_dates.py)

- **Arguments**: `extraction: ComplexType, dates: list[pd.Series]`

- **Returns**: `None`

A warning when 1 to 4 values of the data fall on the extraction date: a participant's date typed in the box is
recorded with the step as typed. A count, on screen only.

### def `already_dates` (validate_dates.py)

- **Arguments**: `df: pd.DataFrame`

- **Returns**: `list[str]`

The columns that already hold dates without a time zone: date cells read on load, or a conversion's.

### def `_dated_reports` (validate_dates.py)

- **Arguments**: `df: pd.DataFrame, dated: list[str], extraction: ComplexType, planned: list[str]`

- **Returns**: `dict[ComplexType]`

What each column already dates holds, counted as for a conversion: nothing to read, the dates themselves.

### def `render_date_checks` (validate_dates.py)

- **Arguments**: ``

- **Returns**: `None`

The Date checks tab: how the text columns that look like dates would be read, what is asked, and their conversion,
one traced step, on request; then what the columns already dates hold. Dates used to be re-parsed in place on
every visit of the page, untraced and invisible to the replay.

### def `_text_columns` (validate_dates.py)

- **Arguments**: `state, df: pd.DataFrame, found: dict[ComplexType], dated: list[str]`

- **Returns**: `ComplexType`

The text columns that look like dates: how each would be read, and their conversion. Returns the extraction date
and the planned-date columns it asked for (None: no column chosen, nothing asked).

### def `_date_columns` (validate_dates.py)

- **Arguments**: `state, df: pd.DataFrame, dated: list[str], study: ComplexType`

- **Returns**: `None`

The columns already dates (date cells read on load, or converted here): what they hold, counted as for a
conversion, and the placeholder dates ticked set to missing on request, one step recorded as a conversion with the
format "dates", which the replay reads the same way. They had no check: an Excel file's 1900-01-01, or a date after
the extraction, stayed uncounted.

---

## variable_map.py

_Variable map page: taxonomy of variables, formulas, clustering and the knowledge graph._

**Imports**:
`streamlit`, `app_pages.data_insight.app`, `ui.components.page_header`, `utils.optional.extra_installed`, `utils.optional.missing_message`

---

## visualization.py

_No module description._

**Imports**:
`functools`, `numpy`, `pandas`, `plotly.express`, `plotly.figure_factory`, `plotly.graph_objects`, `streamlit`, `plotly.subplots.make_subplots`, `scipy.stats`, `utils.export_utils`, `utils.visualization_utils`, `app_pages.data_monitoring`, `ui.components.no_dataset`, `ui.components.text.plain`, `ui.state.get_state`

### def `hover_columns_input` (visualization.py)

- **Arguments**: `df: pd.DataFrame`

- **Returns**: `ComplexType`

Extra columns for the tooltip of each point, passed to Plotly Express (``hover_data``), which matches them to the
points of every trace. Offered where each mark is one row (scatter, line without aggregation): on a bar or a
box, a row's value has no point to show it.

### def `app` (visualization.py)

- **Arguments**: ``

- **Returns**: `None`

No description available.

---

## visualize.py

_Visualize page: distributions, categorical charts, time series, relationships (correlations, clustermap, ROC)._

**Imports**:
`app_pages.visualization.app`, `ui.components.page_header`, `ui.components.population.population_banner`

---

## which_test.py

_Which test? Four questions -> the recommended analysis, why, and where to run it._

**Imports**:
`streamlit`, `ui.components.glossary_popover`, `ui.components.page_header`, `ui.components.page_link`, `ui.components.section`, `ui.components.status_chip`, `ui.guidance.DESIGNS`, `ui.guidance.DISTRIBUTIONS`, `ui.guidance.GROUPS`, `ui.guidance.OUTCOMES`, `ui.guidance.recommend`, `ui.guidance.test_chooser.OTHER_QUESTIONS`

---
