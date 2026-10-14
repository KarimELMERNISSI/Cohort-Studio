"""
Session-bound application state: the Streamlit facade over :class:`core.session.DataSession`.

``AppState`` is the only code that changes the session data (audit E8, follow-up programme M1):

- :meth:`AppState.set_dataset` installs a dataset: derived results are cleared, a trace starts in the
  user's workspace, the source file is stored and the parsing options are logged;
- :meth:`AppState.commit_change` replaces the committed table with one traced step;
- :meth:`AppState.stage`, :meth:`AppState.commit_staging` and :meth:`AppState.discard_staging` handle the
  Enrich draft, which reaches the dataset only when applied and never over newer data;
- :meth:`AppState.set_ticked` and :meth:`AppState.activate_family` choose the analysis population:
  inclusion and exclusion criteria are filters and never delete rows
  (:meth:`AppState.freeze_population` does, on request).

Pages read the legacy keys ``data`` and ``working_df`` (the analysis population) and ``enriched_df``
(the Enrich draft, or the dataset). They are views reinstalled at the start of every run
(:meth:`AppState.begin_run`): modifying them in place changes nothing, and in strict mode
(``CMP_STRICT_STATE=1``, the test suite) it raises :class:`UntracedMutationError`.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from core.fingerprint import frame_fingerprint
from core.population import PopulationResult, catalogue
from core.session import (
    EMPTY,
    DataIdentity,
    DataSession,
    PopulationError,
    StagingError,
    StagingStatus,
)
from core.state import DatasetRef, ParseOptions, dataset_base_name
from utils.data_paths import get_traces_dir
from utils.session_scope import get_current_scope

logger = logging.getLogger(__name__)

SAMPLE_COHORT_PATH = Path(__file__).resolve().parents[1] / "assets" / "sample_cohort.csv"
SAMPLE_DICTIONARY_PATH = SAMPLE_COHORT_PATH.with_name("sample_cohort_dictionary.md")
# The tutorial pack: a richer cohort, its study configuration (anomaly rules, inclusion criteria),
# a follow-up file to merge, and its dictionary (dev_scripts/make_tutorial_cohort.py).
TUTORIAL_DIR = SAMPLE_COHORT_PATH.parent / "tutorial"
TUTORIAL_COHORT_PATH = TUTORIAL_DIR / "tutorial_cohort.csv"
TUTORIAL_CONFIG_PATH = TUTORIAL_DIR / "tutorial_config.json"
TUTORIAL_FOLLOWUP_PATH = TUTORIAL_DIR / "tutorial_followup_labs.csv"
TUTORIAL_DICTIONARY_PATH = TUTORIAL_DIR / "tutorial_dictionary.md"

# Results derived from the current dataset: cleared whenever another dataset is loaded.
DERIVED_KEYS: tuple[str, ...] = (
    "rename_dict",
    "enriched_df",
    "working_df",
    "epidemiology_results",
    "zscore_results",
    "zscore_config_key",
    "fig_clustermap",
    "fig_clustermap_error",
    "preview_new_variable",
    "enrichment_candidates",
    "imputation_suggestions",
    "ai_variable_suggestions",
    "ai_renaming_suggestions",
    "ai_domain_analysis",
    "enrichment_formulas",
    "loaded_path",
    "last_main_file_hash",
    "last_enrichment_file_hash",
    "mwu_p_xy",
    "last_power_selected_groups",
    "last_power_group_col",
    "epidemiology_group_col",
    "validity_outlier_params",
    "validity_outlier_method",
    "univariate_cache",
    "diag_result",
    "ba_result",
    "workflow_marks",
    "quality_history",
    "anomaly_check_done",
    "inclusion_check_done",
    "anomaly_data_applied",
    "inclusion_data_applied",
    "outliers_data_applied",
    # Analysis results kept by the analysis pages; they describe the previous dataset after a swap.
    "surv_result",
    "cox_result",
    "logit_result",
    "pois_result",
    "pair_result",
    "one_result",
    "mc_result",
    "mh_result",
    "mi_result",
    "clustering_results",
    "doc_coverage_results",
)
# Kept when the same dataset changes (they follow the trace, or describe the check rather than the data).
KEPT_ON_DATA_CHANGE: frozenset[str] = frozenset(
    {
        "enriched_df",
        "working_df",
        "workflow_marks",
        "quality_history",
        "loaded_path",
        "last_main_file_hash",
        "last_enrichment_file_hash",
        "anomaly_check_done",
        "inclusion_check_done",
        "anomaly_data_applied",
        "inclusion_data_applied",
        "outliers_data_applied",
    }
)
DATA_DERIVED_KEYS: tuple[str, ...] = tuple(k for k in DERIVED_KEYS if k not in KEPT_ON_DATA_CHANGE)

_SESSION_KEY = "_data_session"
_VIEWS_KEY = "_data_views"


class UntracedMutationError(RuntimeError):
    """The session data was changed outside AppState (strict mode only)."""


def strict_state() -> bool:
    return os.environ.get("CMP_STRICT_STATE", "").strip() == "1"


def tutorial_config() -> dict:
    """The tutorial study configuration on top of the default configuration (folders, thresholds, ...)."""
    import json

    from utils.config_loader import create_empty_config

    config = create_empty_config()
    loaded = json.loads(TUTORIAL_CONFIG_PATH.read_text(encoding="utf-8"))
    for key, value in loaded.items():
        if key != "folder_names":  # machine-specific: keep this installation's folders
            config[key] = value
    return config


class AppState:
    """Facade over ``st.session_state`` (construct it with :func:`get_state`)."""

    # ------------------------------------------------------------------ defaults and managers

    def ensure_defaults(self) -> None:
        ss = st.session_state
        if "data" not in ss:
            ss["data"] = None
        if "config" not in ss:
            from utils.config_loader import create_empty_config

            ss["config"] = create_empty_config()
        if "dataset_ref" not in ss:
            ss["dataset_ref"] = None

    @property
    def db_manager(self):
        if "db_manager" not in st.session_state:
            from manage.db_manager import DBManager

            st.session_state["db_manager"] = DBManager()
        return st.session_state["db_manager"]

    @property
    def transformation_manager(self):
        if "transformation_manager" not in st.session_state:
            from manage.transformation_manager import TransformationManager

            st.session_state["transformation_manager"] = TransformationManager()
        return st.session_state["transformation_manager"]

    @property
    def rag_manager(self):
        if "rag_manager" not in st.session_state:
            from manage.rag_manager import RAGManager

            st.session_state["rag_manager"] = RAGManager()
        return st.session_state["rag_manager"]

    @property
    def username(self) -> str | None:
        return st.session_state.get("username")

    @property
    def role(self) -> str | None:
        return st.session_state.get("user_role")

    @property
    def config(self) -> dict:
        config = st.session_state.get("config")
        return config if isinstance(config, dict) else {}

    # ------------------------------------------------------------------ the session data

    @property
    def session(self) -> DataSession:
        ss = st.session_state
        session = ss.get(_SESSION_KEY)
        if session is None or not hasattr(session, "install"):
            session = DataSession()
            ss[_SESSION_KEY] = session
        session.trace = self.transformation_manager
        return session

    @property
    def base(self) -> pd.DataFrame | None:
        """The committed dataset, every row, as a copy-on-write copy: changing it changes nothing in the
        session (the analysis population is :meth:`analysis_frame`)."""
        base = self.session.base
        return None if base is None else base.copy(deep=False)

    @property
    def df(self) -> pd.DataFrame | None:
        return self.base

    @property
    def dataset(self) -> DatasetRef | None:
        ref = st.session_state.get("dataset_ref")
        base = self.base
        if ref is None and base is not None:
            name = st.session_state.get("current_dataset_name") or "dataset"
            ref = DatasetRef.from_saved_name(str(name), n_rows=len(base), n_cols=base.shape[1])
            st.session_state["dataset_ref"] = ref
        return ref

    @property
    def has_data(self) -> bool:
        return self.session.loaded

    def set_dataset(
        self,
        df: pd.DataFrame,
        ref: DatasetRef,
        *,
        source_bytes: bytes | None = None,
        source_name: str | None = None,
        description: str | None = None,
        parent: dict[str, Any] | None = None,
    ) -> DatasetRef:
        """Install ``df`` as the session dataset: reset derived state, start a trace, store the source."""
        tm = self.transformation_manager
        tm.initialize_session(source_name or ref.name, username=self.username, parent=parent)
        session_id = tm.session_id

        source_path = ref.source_path
        if source_bytes is not None and source_name:
            scope = get_current_scope()
            artifact_dir = (
                scope.artifacts_dir(session_id)
                if scope
                else Path(get_traces_dir()) / "artifacts" / session_id
            )
            artifact_dir.mkdir(parents=True, exist_ok=True)
            ext = os.path.splitext(source_name)[1] or ".csv"
            artifact = artifact_dir / f"source{ext}"
            artifact.write_bytes(source_bytes)
            source_path = str(artifact)
            tm.source_dataset = source_path
            tm.save_trace()

        ref = replace(
            ref,
            source_path=source_path,
            n_rows=int(len(df)),
            n_cols=int(df.shape[1]),
            source_name=source_name or ref.source_name,
        )
        ss = st.session_state
        for key in DERIVED_KEYS:
            ss.pop(key, None)
        self.session.install(
            df,
            params={
                "filename": source_name or ref.name,
                "source_path": source_path,
                "dataset_name": ref.name,
                "version": ref.version,
                "origin": ref.origin,
                "file_hash": ref.file_hash,
                "parse_options": ref.parse.to_dict() if ref.parse else None,
            },
            description=description or f"Loaded dataset: {ref.label}",
        )
        ss["current_dataset_name"] = f"{ref.name}_v{ref.version}" if ref.version else ref.name
        ss["last_uploaded_file_hash"] = ref.file_hash
        ss["dataset_ref"] = ref
        self._install_views()
        return ref

    def clear_dataset(self) -> None:
        ss = st.session_state
        for key in DERIVED_KEYS:
            ss.pop(key, None)
        self.session.clear()
        ss["dataset_ref"] = None
        ss.pop("current_dataset_name", None)
        ss.pop("last_uploaded_file_hash", None)
        self._install_views()

    def commit_change(
        self,
        df: pd.DataFrame,
        step: str,
        params: dict[str, Any] | None,
        description: str,
        *,
        expected_fp: str | None = None,
        population: PopulationResult | dict[str, Any] | None = None,
        replayable: bool = True,
        output_dataset_path: str | None = None,
    ) -> bool:
        """Make ``df`` the dataset, as one traced step; ``False`` when it changes nothing."""
        change = self.session.commit(
            df,
            step,
            params,
            description,
            expected_fp=expected_fp,
            population=population,
            replayable=replayable,
            output_dataset_path=output_dataset_path,
        )
        if change is None:
            return False
        self._after_data_change()
        return True

    def install_replay(
        self, df: pd.DataFrame, trace: dict[str, Any], *, upto: int, mode: str
    ) -> DatasetRef:
        """A replayed state becomes the session dataset, linked to the trace it was replayed from."""
        session_id = str(trace.get("session_id") or "trace")
        parent = {
            "trace_session": session_id,
            "trace_head": trace.get("head_hash"),
            "upto_step": upto + 1,
            "mode": mode,
        }
        ref = DatasetRef(name=dataset_base_name(f"{session_id}_step{upto + 1}"), origin="replay")
        return self.set_dataset(
            df,
            ref,
            description=f"Replayed {session_id} up to step {upto + 1} ({mode})",
            parent=parent,
        )

    def _after_data_change(self) -> None:
        ss = st.session_state
        for key in DATA_DERIVED_KEYS:
            ss.pop(key, None)
        ref = ss.get("dataset_ref")
        base = self.base
        if ref is not None and base is not None:
            ss["dataset_ref"] = replace(ref, n_rows=int(len(base)), n_cols=int(base.shape[1]))
        self._install_views()

    # ------------------------------------------------------------------ the Enrich draft

    def working_frame(self) -> pd.DataFrame | None:
        """What Enrich works on: the draft when there is one, otherwise the dataset (every row), as a copy."""
        frame = self.session.working_frame()
        return None if frame is None else frame.copy(deep=False)

    def stage(
        self,
        df: pd.DataFrame,
        step: str,
        params: dict[str, Any] | None,
        description: str,
        *,
        population: PopulationResult | dict[str, Any] | None = None,
        replayable: bool = True,
        output_dataset_path: str | None = None,
    ) -> bool:
        """Add a change to the Enrich draft (traced when applied); ``False`` when it changes nothing."""
        pending = self.session.stage(
            df,
            step,
            params,
            description,
            population=population,
            replayable=replayable,
            output_dataset_path=output_dataset_path,
        )
        self._install_views()
        return pending is not None

    def staging_status(self) -> StagingStatus:
        if not self.session.loaded:
            return StagingStatus(EMPTY)
        return self.session.staging_status(self.config)

    def pending_steps(self) -> list:
        draft = self.session.draft
        return list(draft.steps) if draft is not None else []

    def commit_staging(self) -> tuple[bool, str]:
        status = self.staging_status()
        if status.state == EMPTY:
            return False, "There is no change to apply."
        try:
            self.session.commit_staging(self.config)
        except StagingError as exc:
            return False, str(exc)
        self._after_data_change()
        n = status.n_steps
        return True, f"{n} change{'s' if n != 1 else ''} applied to the dataset."

    def discard_staging(self) -> None:
        self.session.discard_staging()
        self._install_views()

    # ------------------------------------------------------------------ the analysis population

    @property
    def population(self) -> PopulationResult | None:
        return self.session.population(self.config) if self.session.loaded else None

    def analysis_frame(self) -> pd.DataFrame:
        """Rows kept by the ticked criteria; raises PopulationError when a ticked criterion is broken."""
        return self.session.analysis_frame(self.config)

    def identity(self) -> DataIdentity | None:
        return self.session.identity(self.config) if self.session.loaded else None

    @property
    def ticked(self) -> tuple[str, ...]:
        return self.session.ticked

    def set_ticked(self, ids) -> None:
        self.session.set_ticked(ids)
        self._install_views()

    def tick(self, criterion_id: str, on: bool = True) -> None:
        self.session.tick(criterion_id, on)
        self._install_views()

    def activate_family(self, family: str, on: bool = True) -> None:
        """Tick (or untick) every criterion of a family."""
        members = [c.id for c in catalogue(self.config) if c.family == family]
        ticked = [c for c in self.session.ticked if c not in members]
        self.set_ticked(ticked + (members if on else []))

    def freeze_population(self) -> bool:
        """Keep only the analysis population in the dataset (traced; the criteria are then unticked)."""
        change = self.session.freeze_population(self.config)
        if change is None:
            return False
        self._after_data_change()
        return True

    def record_population_use(self) -> None:
        if self.session.loaded:
            with contextlib.suppress(PopulationError):
                self.session.record_population_use(self.config)

    # ------------------------------------------------------------------ views and run guards

    def _install_views(self) -> None:
        ss = st.session_state
        session = self.session
        if not session.loaded:
            ss["data"] = None
            ss.pop("working_df", None)
            ss.pop("enriched_df", None)
            ss[_VIEWS_KEY] = {}
            return
        try:
            analysis = session.analysis_frame(self.config)
        except PopulationError:
            analysis = session.base  # the population banner stops the analysis pages
        data_view = analysis.copy(deep=False)
        enrich_view = session.working_frame().copy(deep=False)
        ss["data"] = data_view
        ss["working_df"] = data_view
        ss["enriched_df"] = enrich_view
        record: dict[str, Any] = {"data": id(data_view), "enriched_df": id(enrich_view)}
        if strict_state():
            record["fp"] = {
                "data": frame_fingerprint(data_view),
                "enriched_df": frame_fingerprint(enrich_view),
            }
        ss[_VIEWS_KEY] = record

    def begin_run(self) -> None:
        """Reinstall the dataset views; call once per run, before the sidebar and the page."""
        ss = st.session_state
        session = self.session
        record = ss.get(_VIEWS_KEY) or {}
        current = ss.get("data")
        draft_view = ss.get("enriched_df")
        if not session.loaded:
            if isinstance(current, pd.DataFrame):  # put there before this session layer existed
                session.install(current, record=False)
        elif isinstance(current, pd.DataFrame) and id(current) != record.get("data"):
            self._external_write("data", current)
        elif isinstance(draft_view, pd.DataFrame) and id(draft_view) != record.get("enriched_df"):
            self._external_write("enriched_df", draft_view)
        self._install_views()

    def end_run(self) -> None:
        """Strict mode: fail when a page modified the dataset views in place during the run."""
        if not strict_state():
            return
        ss = st.session_state
        record = ss.get(_VIEWS_KEY) or {}
        for key, fp in (record.get("fp") or {}).items():
            view = ss.get(key)
            same_view = isinstance(view, pd.DataFrame) and id(view) == record.get(key)
            if same_view and frame_fingerprint(view) != fp:
                raise UntracedMutationError(
                    f"st.session_state['{key}'] was modified in place during the run; "
                    "changes to the data go through AppState (commit_change or stage)."
                )

    def _external_write(self, key: str, frame: pd.DataFrame) -> None:
        message = (
            f"st.session_state['{key}'] was replaced outside AppState; "
            "changes to the data go through AppState (commit_change or stage)."
        )
        if strict_state():
            raise UntracedMutationError(message)
        logger.warning(message)
        if key == "data":
            self.session.commit(
                frame,
                "external_write",
                {"key": key},
                "Data replaced outside AppState",
                replayable=False,
            )
        else:
            self.session.stage(
                frame,
                "external_write",
                {"key": key},
                "Data replaced outside AppState",
                replayable=False,
            )

    # ------------------------------------------------------------------ sources

    def load_parsed(
        self,
        df: pd.DataFrame,
        *,
        filename: str,
        data: bytes,
        options: ParseOptions | None,
        name: str | None = None,
        origin: str = "upload",
    ) -> DatasetRef:
        ref = DatasetRef(
            name=name or dataset_base_name(filename),
            origin=origin,
            file_hash=hashlib.md5(data).hexdigest(),
            source_name=filename,
            parse=options,
        )
        return self.set_dataset(df, ref, source_bytes=data, source_name=filename)

    def load_sample_cohort(self) -> DatasetRef:
        from ingest.loaders import read_table

        data = SAMPLE_COHORT_PATH.read_bytes()
        result = read_table(data, SAMPLE_COHORT_PATH.name, ParseOptions(dayfirst=True))
        return self.load_parsed(
            result.df,
            filename=SAMPLE_COHORT_PATH.name,
            data=data,
            options=result.options,
            name="sample_cohort",
            origin="sample",
        )

    def load_tutorial_cohort(self, *, with_config: bool = True) -> DatasetRef:
        """The tutorial cohort; its study configuration (anomaly rules, inclusion criteria) is loaded too."""
        from ingest.loaders import read_table

        data = TUTORIAL_COHORT_PATH.read_bytes()
        result = read_table(data, TUTORIAL_COHORT_PATH.name, ParseOptions(dayfirst=True))
        ref = self.load_parsed(
            result.df,
            filename=TUTORIAL_COHORT_PATH.name,
            data=data,
            options=result.options,
            name="tutorial_cohort",
            origin="sample",
        )
        if with_config:
            st.session_state["config"] = tutorial_config()
            self._install_views()
        return ref

    def saved_versions(self) -> list[str]:
        try:
            return list(self.db_manager.get_available_datasets(username=self.username))
        except Exception:
            return []

    def load_saved_version(self, saved_name: str) -> tuple[bool, str]:
        df, msg = self.db_manager.load_dataset(saved_name)
        if df is None:
            return False, msg
        path = os.path.join(self.db_manager.dataset_dir, f"{saved_name}.parquet")
        ref = DatasetRef.from_saved_name(
            saved_name, source_path=path, source_name=f"{saved_name}.parquet"
        )
        self.set_dataset(df, ref, description=f"Loaded saved version: {saved_name}")
        return True, msg

    def save_version(self, base_name: str | None = None) -> tuple[bool, str, str | None]:
        base = self.base
        if base is None:
            return False, "No dataset loaded.", None
        ref = self.dataset
        name = dataset_base_name(base_name or (ref.name if ref else "cohort"))
        ok, msg, saved = self.db_manager.save_dataset(base, base_name=name, username=self.username)
        if ok and saved:
            new_ref = DatasetRef.from_saved_name(
                saved,
                file_hash=ref.file_hash if ref else None,
                source_name=ref.source_name if ref else None,
                source_path=os.path.join(self.db_manager.dataset_dir, f"{saved}.parquet"),
                n_rows=int(len(base)),
                n_cols=int(base.shape[1]),
                parse=ref.parse if ref else None,
            )
            st.session_state["dataset_ref"] = new_ref
            st.session_state["current_dataset_name"] = saved
            with contextlib.suppress(Exception):
                self.transformation_manager.add_step(
                    "save_version", {"dataset_name": saved}, f"Saved version: {saved}"
                )
        return ok, msg, saved

    def note(self, step: str, params: dict[str, Any] | None, description: str) -> None:
        """A trace step that documents something without changing the data (parse report, export...)."""
        tm = self.transformation_manager
        if tm.session_id:
            with contextlib.suppress(Exception):
                tm.add_step(step, params or {}, description)

    # ------------------------------------------------------------------ misc

    def ai_status(self) -> dict[str, Any]:
        rm = st.session_state.get("rag_manager")
        initialized = bool(getattr(rm, "initialized", False))
        provider = getattr(rm, "provider", None) if rm is not None else None
        return {
            "initialized": initialized,
            "provider": provider,
            "consent": bool(getattr(rm, "cloud_consent", False)),
        }


def get_state() -> AppState:
    state = AppState()
    state.ensure_defaults()
    return state


__all__ = [
    "DATA_DERIVED_KEYS",
    "DERIVED_KEYS",
    "SAMPLE_COHORT_PATH",
    "SAMPLE_DICTIONARY_PATH",
    "TUTORIAL_CONFIG_PATH",
    "TUTORIAL_COHORT_PATH",
    "TUTORIAL_DICTIONARY_PATH",
    "TUTORIAL_FOLLOWUP_PATH",
    "AppState",
    "UntracedMutationError",
    "get_state",
    "strict_state",
    "tutorial_config",
]
