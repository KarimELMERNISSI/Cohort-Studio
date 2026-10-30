"""
The session keys that one part of the application writes and another reads (M5b, runtime).

Once only the open tab of a page runs (``st.tabs`` with ``on_change="rerun"``), a key that one tab writes and another
tab, or a later run, reads is how they share state. Each such key is declared here with its owner (the module that
writes it), its lifetime and the other modules that read it, so that the sharing stays visible. The lifetimes decide
what ``AppState`` clears: ``ui.state.DERIVED_KEYS`` and ``KEPT_ON_DATA_CHANGE`` are derived from them. A key the
code writes into the session without a declaration here is counted by a quality gate, and that count only falls.

Lifetimes:

- ``session``: kept for the whole session;
- ``page``: a widget's own state, kept while the user stays on its page (``reset_on_dataset``: also cleared when
  another dataset is loaded);
- ``dataset``: cleared when another dataset is loaded, kept when the same dataset changes;
- ``data_change``: cleared on any change of the data, and when another dataset is loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Lifetime = Literal["session", "page", "dataset", "data_change"]


@dataclass(frozen=True)
class Key:
    name: str
    owner: str  # the module that writes it
    lifetime: Lifetime
    readers: tuple[str, ...] = ()  # the other modules that read it
    note: str = ""
    reset_on_dataset: bool = (
        False  # a page key that AppState also clears when another dataset is loaded
    )


KEYS: tuple[Key, ...] = (
    # the dataset of the session, replaced when another is loaded (AppState.set_dataset), never cleared
    Key(
        "data",
        "ui.state",
        "session",
        (
            "app_pages.data_enrichment",
            "app_pages.data_monitoring",
            "app_pages.epidemiology",
            "app_pages.home",
            "app_pages.rag_sidebar",
            "app_pages.visualization",
            "app_pages.views.survival",
            "ui.progress",
        ),
        "the dataset: every row of the current data",
    ),
    Key(
        "dataset_ref",
        "ui.state",
        "session",
        ("ui.progress",),
        "where the dataset comes from (DatasetRef)",
    ),
    Key(
        "current_dataset_name",
        "ui.state",
        "session",
        ("app_pages.home", "ui.progress"),
        "the dataset's name, with its version",
    ),
    Key(
        "last_uploaded_file_hash",
        "ui.state",
        "session",
        note="the file read last: read again only when it changes",
    ),
    Key(
        "import_default_dayfirst",
        "ui.preferences",
        "session",
        ("ui.import_flow",),
        "the user's saved choice of day-first dates, seeded once per session",
    ),
    Key(
        "rag_qdrant_url",
        "app_pages.rag_sidebar",
        "page",
        note="Settings: the Qdrant server's address, seeded from the server's setting",
    ),
    Key(
        "config",
        "ui.state",
        "session",
        ("app_pages.data_monitoring", "app_pages.data_enrichment"),
        "the study configuration: the criteria definitions (edited on Validate), the enrichments, the folders",
    ),
    Key(
        "_preferences",
        "ui.preferences",
        "session",
        note="the user's choices kept from the last session (manage.preferences), read once per session",
    ),
    Key(
        "_preferences_seeded",
        "ui.preferences",
        "session",
        note="the user whose preferences seeded this session",
    ),
    Key(
        "validate_tab",
        "app_pages.data_monitoring",
        "page",
        note="the open tab of Validate: only that tab runs",
    ),
    Key(
        "clean_tab",
        "app_pages.data_cleaning",
        "page",
        note="the open tab of Clean (Categories, Dates): only that tab runs",
    ),
    Key(
        "clean_done",
        "app_pages.clean_categories",
        "page",
        note="what the last recode added, said once after the rerun",
    ),
    Key(
        "home_tab",
        "app_pages.home",
        "page",
        note="the open tab of Statistics & correlations: only that tab runs",
    ),
    Key(
        "epi_tab",
        "app_pages.epidemiology",
        "page",
        note="the open tab of Hypothesis tests & epidemiology: only that tab runs",
    ),
    Key(
        "anomaly_check_done",
        "app_pages.data_monitoring",
        "dataset",
        note="Validate: the plausibility check's results stay shown after the click that ran it",
    ),
    Key(
        "inclusion_check_done",
        "app_pages.data_monitoring",
        "dataset",
        note="Validate: the same, for the inclusion criteria",
    ),
    Key(
        "validate_config_uploaded",
        "app_pages.data_monitoring",
        "page",
        note="Validate: what the uploaded configuration gave (loaded, with its notices, or why not), merged once by the "
        "uploader's callback, shown while the file stays in the uploader and dropped once it is empty",
    ),
    Key(
        "validity_outlier_method",
        "app_pages.data_quality_dashboard",
        "dataset",
        note="the outlier detection settings, read by the scorecard above them; kept wherever the data change",
    ),
    Key(
        "validity_outlier_params",
        "app_pages.data_quality_dashboard",
        "dataset",
        note="the parameters of that method",
    ),
    *(
        Key(
            control,
            "app_pages.data_quality_dashboard",
            "page",
            note="a control of those settings, read at the top of the dashboard by the scorecard",
            reset_on_dataset=True,
        )
        for control in (
            "validity_detection_method",
            "validity_iqr_mult",
            "validity_zscore_thresh",
            "validity_q_lower",
            "validity_q_upper",
        )
    ),
    Key(
        "rename_dict",
        "app_pages.home",
        "data_change",
        ("app_pages.data_insight",),
        "the renames pending in the Renaming tab, which the AI suggestions add to",
    ),
    Key(
        "epidemiology_results",
        "app_pages.epidemiology",
        "data_change",
        note="written by the Analysis tab, read by the Power tab to prefill it",
    ),
    Key(
        "epidemiology_group_col",
        "app_pages.epidemiology",
        "data_change",
        note="the grouping variable of those results",
    ),
    # the dataset's own state, which follows its trace: kept when it changes, cleared when another is loaded
    Key(
        "working_df",
        "ui.state",
        "dataset",
        (
            "app_pages.data_enrichment",
            "app_pages.data_insight",
            "app_pages.data_monitoring",
            "app_pages.document_insight",
            "app_pages.home",
            "app_pages.rag_sidebar",
        ),
        "the analysis population: the rows that the ticked criteria keep, which the analyses use",
    ),
    Key(
        "enriched_df",
        "ui.state",
        "dataset",
        ("app_pages.data_enrichment",),
        "the Enrich draft, before it is applied to the dataset",
    ),
    Key(
        "_mi",
        "ui.state",
        "dataset",
        note="the multiple imputation (StoredMI): its status follows the rows",
    ),
    Key(
        "_results",
        "ui.results",
        "dataset",
        note="the results: a stamped result goes stale on a change",
    ),
    Key(
        "_report",
        "ui.results",
        "dataset",
        note="the results added to the study dossier, stale like them",
    ),
    Key(
        "workflow_marks",
        "ui.progress",
        "dataset",
        note="the steps of the Start page done on this dataset",
    ),
    Key(
        "quality_history",
        "ui.progress",
        "dataset",
        note="the quality scores of this dataset as it changed",
    ),
    Key(
        "loaded_path",
        "app_pages.data_insight",
        "dataset",
        note="Data insight: the file of the dataset",
    ),
    Key(
        "last_enrichment_file_hash",
        "app_pages.data_enrichment",
        "dataset",
        note="Enrich: the enrichment file read last, read again only when it changes",
    ),
    # what a page computed on the data: cleared on any change of the data
    *(
        Key(name, owner, "data_change", note=note)
        for name, owner, note in (
            ("zscore_results", "app_pages.epidemiology", "Hypothesis tests: the z-score results"),
            (
                "zscore_config_key",
                "app_pages.epidemiology",
                "the settings those results were computed with",
            ),
            (
                "mwu_p_xy",
                "app_pages.epidemiology",
                "the Mann-Whitney effect size, for the Power tab",
            ),
            ("last_power_selected_groups", "app_pages.epidemiology", "the Power tab's groups"),
            ("last_power_group_col", "app_pages.epidemiology", "the Power tab's grouping variable"),
            (
                "univariate_cache",
                "app_pages.epidemiology",
                "the univariate tests, by their settings",
            ),
            ("fig_clustermap", "app_pages.visualization", "Visualize: the clustered heat map"),
            (
                "fig_clustermap_error",
                "app_pages.visualization",
                "why that heat map could not be drawn",
            ),
            (
                "preview_new_variable",
                "app_pages.data_enrichment",
                "Enrich: a new variable, before it is added",
            ),
            (
                "imputation_suggestions",
                "app_pages.data_enrichment",
                "Enrich: the AI engine's imputations",
            ),
            (
                "ai_variable_suggestions",
                "app_pages.data_enrichment",
                "Enrich: the AI engine's variables",
            ),
            (
                "ai_domain_analysis",
                "app_pages.data_enrichment",
                "Enrich: the AI engine's reading of the domain",
            ),
            ("ai_renaming_suggestions", "app_pages.home", "Statistics: the AI engine's renames"),
            (
                "enrichment_candidates",
                "app_pages.data_insight",
                "Data insight: the files to enrich with",
            ),
            (
                "enrichment_formulas",
                "app_pages.data_insight",
                "Data insight: the formulas suggested with them",
            ),
            (
                "clustering_results",
                "app_pages.data_insight",
                "Data insight: the clusters of the rows",
            ),
            (
                "doc_coverage_results",
                "app_pages.document_insight",
                "Document insight: what the documents cover",
            ),
        )
    ),
    Key(
        "_services",
        "ui.services",
        "session",
        note="the session's services (manage.services.Services), for the user they were built for",
    ),
    Key(
        "db_manager",
        "ui.services",
        "session",
        ("app_pages.data_enrichment", "app_pages.home"),
        "the dataset store of the session's services, under the key the pages read",
    ),
    Key(
        "transformation_manager",
        "ui.services",
        "session",
        ("app_pages.data_enrichment", "ui.progress", "ui.results"),
        "the trace of the session's services, under the key the pages read",
    ),
    Key(
        "rag_manager",
        "ui.services",
        "session",
        (
            "app_pages.data_enrichment",
            "app_pages.data_insight",
            "app_pages.data_monitoring",
            "app_pages.document_insight",
            "app_pages.enrichment_derived",
            "app_pages.home",
            "app_pages.rag_monitoring",
            "app_pages.rag_sidebar",
            "ui.progress",
            "ui.state",
        ),
        "the AI engine, built on first use (ui.services.ai_engine); a page that only uses a running one reads it",
    ),
)

BY_NAME: dict[str, Key] = {key.name: key for key in KEYS}

__all__ = ["BY_NAME", "KEYS", "Key", "Lifetime"]
