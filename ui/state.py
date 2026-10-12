"""
Session-bound application state with ONE invalidation point (audit E8).

``AppState.set_dataset`` is the only code that installs a dataset in the session: it resets every
derived result, starts the transformation trace, stores the source artifact in the user's
workspace and logs the parsing options. Pages keep reading the legacy keys (``data``,
``working_df``, ``enriched_df``, ``current_dataset_name``) until they are migrated.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from core.state import DatasetRef, ParseOptions, dataset_base_name
from utils.data_paths import get_traces_dir
from utils.session_scope import get_current_scope

SAMPLE_COHORT_PATH = Path(__file__).resolve().parents[1] / "assets" / "sample_cohort.csv"
SAMPLE_DICTIONARY_PATH = SAMPLE_COHORT_PATH.with_name("sample_cohort_dictionary.md")

# Results derived from the current dataset: cleared whenever the dataset changes.
DERIVED_KEYS: tuple[str, ...] = (
    "rename_dict", "enriched_df", "working_df", "epidemiology_results", "zscore_results", "zscore_config_key",
    "fig_clustermap", "fig_clustermap_error", "preview_new_variable", "enrichment_candidates",
    "imputation_suggestions", "ai_variable_suggestions", "ai_renaming_suggestions", "ai_domain_analysis",
    "enrichment_formulas", "loaded_path", "last_main_file_hash", "last_enrichment_file_hash", "mwu_p_xy",
    "last_power_selected_groups", "last_power_group_col", "epidemiology_group_col", "validity_outlier_params",
    "validity_outlier_method",
)


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

    # ------------------------------------------------------------------ dataset

    @property
    def df(self) -> pd.DataFrame | None:
        return st.session_state.get("data")

    @property
    def dataset(self) -> DatasetRef | None:
        ref = st.session_state.get("dataset_ref")
        if ref is None and self.df is not None:
            # Dataset installed by legacy code (direct session writes): synthesise a reference.
            name = st.session_state.get("current_dataset_name") or "dataset"
            ref = DatasetRef.from_saved_name(str(name), n_rows=len(self.df), n_cols=self.df.shape[1])
            st.session_state["dataset_ref"] = ref
        return ref

    @property
    def has_data(self) -> bool:
        return self.df is not None

    def set_dataset(
        self,
        df: pd.DataFrame,
        ref: DatasetRef,
        *,
        source_bytes: bytes | None = None,
        source_name: str | None = None,
        description: str | None = None,
    ) -> DatasetRef:
        """Install ``df`` as the session dataset: reset derived state, start a trace, store the source."""
        tm = self.transformation_manager
        tm.initialize_session(source_name or ref.name, username=self.username)
        session_id = tm.session_id

        source_path = ref.source_path
        if source_bytes is not None and source_name:
            scope = get_current_scope()
            artifact_dir = scope.artifacts_dir(session_id) if scope else Path(get_traces_dir()) / "artifacts" / session_id
            artifact_dir.mkdir(parents=True, exist_ok=True)
            ext = os.path.splitext(source_name)[1] or ".csv"
            artifact = artifact_dir / f"source{ext}"
            artifact.write_bytes(source_bytes)
            source_path = str(artifact)
            tm.source_dataset = source_path
            tm.save_trace()

        ref = replace(ref, source_path=source_path, n_rows=int(len(df)), n_cols=int(df.shape[1]),
                      source_name=source_name or ref.source_name)
        tm.add_step(
            "initial_load",
            {
                "filename": source_name or ref.name,
                "source_path": source_path,
                "dataset_name": ref.name,
                "version": ref.version,
                "origin": ref.origin,
                "file_hash": ref.file_hash,
                "parse_options": ref.parse.to_dict() if ref.parse else None,
            },
            description or f"Loaded dataset: {ref.label}",
        )

        ss = st.session_state
        for key in DERIVED_KEYS:
            ss.pop(key, None)
        ss["data"] = df
        ss["working_df"] = df.copy()
        ss["enriched_df"] = df.copy()
        ss["current_dataset_name"] = f"{ref.name}_v{ref.version}" if ref.version else ref.name
        ss["last_uploaded_file_hash"] = ref.file_hash
        ss["dataset_ref"] = ref
        return ref

    def clear_dataset(self) -> None:
        ss = st.session_state
        for key in DERIVED_KEYS:
            ss.pop(key, None)
        ss["data"] = None
        ss["dataset_ref"] = None
        ss.pop("current_dataset_name", None)
        ss.pop("last_uploaded_file_hash", None)

    # ------------------------------------------------------------------ sources

    def load_parsed(self, df: pd.DataFrame, *, filename: str, data: bytes, options: ParseOptions | None,
                    name: str | None = None, origin: str = "upload") -> DatasetRef:
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
        return self.load_parsed(result.df, filename=SAMPLE_COHORT_PATH.name, data=data, options=result.options,
                                name="sample_cohort", origin="sample")

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
        ref = DatasetRef.from_saved_name(saved_name, source_path=path, source_name=f"{saved_name}.parquet")
        self.set_dataset(df, ref, description=f"Loaded saved version: {saved_name}")
        return True, msg

    def save_version(self, base_name: str | None = None) -> tuple[bool, str, str | None]:
        if self.df is None:
            return False, "No dataset loaded.", None
        ref = self.dataset
        base = dataset_base_name(base_name or (ref.name if ref else "cohort"))
        ok, msg, saved = self.db_manager.save_dataset(self.df, base_name=base, username=self.username)
        if ok and saved:
            new_ref = DatasetRef.from_saved_name(
                saved,
                file_hash=ref.file_hash if ref else None,
                source_name=ref.source_name if ref else None,
                source_path=os.path.join(self.db_manager.dataset_dir, f"{saved}.parquet"),
                n_rows=int(len(self.df)),
                n_cols=int(self.df.shape[1]),
                parse=ref.parse if ref else None,
            )
            st.session_state["dataset_ref"] = new_ref
            st.session_state["current_dataset_name"] = saved
            try:
                self.transformation_manager.add_step("save_version", {"dataset_name": saved}, f"Saved version: {saved}")
            except Exception:
                pass
        return ok, msg, saved

    # ------------------------------------------------------------------ misc

    def ai_status(self) -> dict[str, Any]:
        rm = st.session_state.get("rag_manager")
        initialized = bool(getattr(rm, "initialized", False))
        provider = getattr(rm, "provider", None) if rm is not None else None
        return {"initialized": initialized, "provider": provider, "consent": bool(getattr(rm, "cloud_consent", False))}


def get_state() -> AppState:
    state = AppState()
    state.ensure_defaults()
    return state


__all__ = ["DERIVED_KEYS", "SAMPLE_COHORT_PATH", "SAMPLE_DICTIONARY_PATH", "AppState", "get_state"]
