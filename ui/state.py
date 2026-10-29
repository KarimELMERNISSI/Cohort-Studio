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
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
import streamlit as st

from core.derived import DerivedVariable, decided_on_data, record, with_definitions
from core.derived import from_config as derived_from_config
from core.fingerprint import frame_fingerprint, ids_to_ranges, ranges_to_ids
from core.lineage import canonical_json
from core.population import PopulationResult, catalogue, criteria_from_trace, evaluate_population
from core.session import (
    EMPTY,
    DataIdentity,
    DataSession,
    PopulationError,
    StagingError,
    StagingStatus,
    merge_cells,
)
from core.state import DatasetRef, ParseOptions, dataset_base_name, file_checksum
from core.steps import FITTED, FORMULA
from core.steps import ids as step_ids
from ingest.analyzer import DataAnalyzer
from manage.snapshots import delete_unreferenced
from manage.snapshots import folders as snapshot_folders
from manage.snapshots import save as save_snapshot_to
from ui.keys import KEYS
from ui.session_scope import get_current_scope
from utils.data_paths import get_data_root, get_traces_dir
from utils.environment import environment
from utils.settings import get_settings

if TYPE_CHECKING:
    from enrich.derived import DerivedCheck, FillPlan
    from explore.data_quality_auditor import DataQualityAuditor
    from manage.services import Services

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

# The keys AppState clears, from their declarations (ui/keys.py): when another dataset is loaded, every key of the
# dataset (its own state and what was computed on it) and the page keys that go with it; on any change of the data,
# what was computed on it.
DERIVED_KEYS: tuple[str, ...] = tuple(
    key.name for key in KEYS if key.lifetime in ("dataset", "data_change") or key.reset_on_dataset
)
# Kept when the same dataset changes (they follow the trace, or describe the check rather than the data).
KEPT_ON_DATA_CHANGE: frozenset[str] = frozenset(
    key.name for key in KEYS if key.lifetime == "dataset" or key.reset_on_dataset
)
DATA_DERIVED_KEYS: tuple[str, ...] = tuple(k for k in DERIVED_KEYS if k not in KEPT_ON_DATA_CHANGE)

_SESSION_KEY = "_data_session"
MI_KEY = "_mi"
_VIEWS_KEY = "_data_views"


class UntracedMutationError(RuntimeError):
    """The session data was changed outside AppState (strict mode only)."""


@dataclass(frozen=True)
class PopulationScope:
    """The rows an imputation of the Enrich draft is computed on and fills (:meth:`AppState.population_scope`)."""

    population: PopulationResult | None  # recorded with the step, also when no criterion is ticked
    blocked: bool  # a ticked criterion cannot be evaluated: no imputation until it is corrected
    scoped: bool  # criteria are ticked: the rows are those of the analysis population
    rows: pd.Index


@dataclass(frozen=True)
class SingleImputation:
    """What :meth:`AppState.impute_single` staged: the imputation, and the derived variables' values computed exactly
    before it and from its imputed values after it."""

    outcome: Any  # enrich.imputation_service.ImputationOutcome
    staged: bool
    exact_first: int = 0
    derived: int = 0
    # the derived variables kept out still missing (a variable of their formula has no value there), by variable
    left: Mapping[str, int] = field(default_factory=dict)
    # rows where a recorded derived variable no longer matches its imputed variables (no inverse recovered them)
    contradicted: Mapping[str, int] = field(default_factory=dict)


def strict_state() -> bool:
    return get_settings().development.strict_state


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
    def services(self) -> Services:
        """The session's services, for the signed-in user (ui.services: built once, again for another user)."""
        from ui.services import services

        return services()

    @property
    def db_manager(self):
        return self.services.datasets

    @property
    def transformation_manager(self):
        return self.services.trace

    @property
    def rag_manager(self):
        from ui.services import ai_engine

        return ai_engine()

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
        # a pending draft is dropped with the dataset it was built on (checked against the trace that ends here)
        dropped = self._draft_snapshots()
        tm = self.transformation_manager
        ended = list(tm.history)
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
            tm.source_dataset = self.recorded_path(source_path)
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
                "source_path": self.recorded_path(source_path),
                "dataset_name": ref.name,
                "version": ref.version,
                "origin": ref.origin,
                "file_hash": ref.file_hash,
                "parse_options": ref.parse.to_dict() if ref.parse else None,
                "environment": environment(),  # the software the results that follow are computed with
            },
            description=description or f"Loaded dataset: {ref.label}",
        )
        ss["current_dataset_name"] = f"{ref.name}_v{ref.version}" if ref.version else ref.name
        ss["last_uploaded_file_hash"] = ref.file_hash
        ss["dataset_ref"] = ref
        self._install_views()
        self._delete_snapshots(dropped, ended)
        return ref

    def clear_dataset(self) -> None:
        ss = st.session_state
        for key in DERIVED_KEYS:
            ss.pop(key, None)
        dropped = self._draft_snapshots()
        self.session.clear()
        self._delete_snapshots(dropped)
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
            output_dataset_path=self.recorded_path(output_dataset_path),
        )
        if change is None:
            return False
        self._after_data_change()
        return True

    def install_replay(
        self,
        df: pd.DataFrame,
        trace: dict[str, Any],
        *,
        upto: int,
        mode: str,
        single_imputed: Mapping[str, Any] | None = None,
        exact_imputed: Mapping[str, Any] | None = None,
    ) -> DatasetRef:
        """A replayed state becomes the session dataset, linked to the trace it was replayed from (with the
        cells its single imputations and its exact formulas filled, when the replay recomputed them)."""
        session_id = str(trace.get("session_id") or "trace")
        parent = {
            "trace_session": session_id,
            "trace_head": trace.get("head_hash"),
            "upto_step": upto + 1,
            "mode": mode,
        }
        ref = DatasetRef(name=dataset_base_name(f"{session_id}_step{upto + 1}"), origin="replay")
        out = self.set_dataset(
            df,
            ref,
            description=f"Replayed {session_id} up to step {upto + 1} ({mode})",
            parent=parent,
        )
        if single_imputed:
            self.session.restore_single_imputed(single_imputed)
        if exact_imputed:
            self.session.restore_exact_imputed(exact_imputed)
        return out

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

    def draft_frame(self) -> pd.DataFrame | None:
        """What Enrich works on: the draft when there is one, otherwise the dataset (every row), as a copy."""
        frame = self.session.draft_frame()
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
        try:
            pending = self.session.stage(
                df,
                step,
                params,
                description,
                population=population,
                replayable=replayable,
                output_dataset_path=self.recorded_path(output_dataset_path),
            )
        except Exception:  # the change did not join the draft: neither does its snapshot
            self._delete_snapshots([output_dataset_path])
            raise
        if pending is None:  # nothing changed: the snapshot written for the change is not needed
            self._delete_snapshots([output_dataset_path])
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
        dropped = self._draft_snapshots()
        self.session.discard_staging()
        self._delete_snapshots(dropped)
        self._install_views()

    def _draft_snapshots(self) -> list[str | None]:
        """The snapshots written for the draft's changes: they reach the trace only when the draft is applied."""
        draft = self.session.draft
        return [step.output_dataset_path for step in draft.steps] if draft is not None else []

    def _delete_snapshots(
        self, paths: list[str | None], steps: list[dict[str, Any]] | None = None
    ) -> None:
        """
        Delete the snapshots of changes that will never reach the trace, except a file that a trace (the session's
        trace, or ``steps``) or a change still in the draft names. Best effort: a failure is logged, and the change
        goes on.
        """
        if not any(paths):
            return
        try:
            history = self.transformation_manager.history if steps is None else steps
            pending = [{"output_dataset_path": path} for path in self._draft_snapshots()]
            delete_unreferenced(paths, get_current_scope(), [*history, *pending])
        except Exception as exc:
            logger.warning("The snapshots of the draft were not deleted (%s)", type(exc).__name__)

    def rebase_staging(self, *, refit_population: bool = False) -> tuple[bool, str]:
        """Redo the draft's changes on the current dataset (the draft was built on older data), or refit its
        population-scoped imputations on the current population. Only replayable changes can be redone; a redone
        imputation is recorded as what it did on the current data (its rows, counts and description)."""
        from manage.reproduction_manager import apply_steps

        session = self.session
        draft = session.draft
        if draft is None or not draft.steps:
            return False, "There is no change to re-apply."
        blocked = [p.description for p in draft.steps if not p.replayable]
        if blocked:
            return (
                False,
                f"'{blocked[0]}' cannot be redone automatically: discard the draft and redo it.",
            )
        if any(decided_on_data(p.params.get("derived")) for p in draft.steps):
            return (
                False,
                "The draft holds values filled from the declared derived variables, decided on the data it was "
                "built on (their rounding, whether they count as observed): discard the draft and fill them again.",
            )
        population = self.population if refit_population else None
        if population is not None and not population.ok:
            return False, "The analysis population cannot be evaluated."
        frame = session.base
        assert frame is not None
        redone = []
        for pending in draft.steps:
            step = {
                "function": pending.function,
                "params": dict(pending.params),
                "description": pending.description,
            }
            block = None
            # redo on the population it was fitted on, as it is on the current data; or refit on the
            # current population (every row when no criterion is ticked). Evaluated on the dataset, as the page
            # does for every change of a draft, not on the table the earlier changes filled
            if pending.population is not None:
                block = (
                    self._on_current_data(pending.population)
                    if population is None
                    else population.to_trace()
                )
                step["population"] = block
            result = apply_steps(
                frame,
                [step],
                frames={session.data_fp: session.base},
                paths=self.services.paths,
            )
            if result.error or result.df is None:
                return False, result.error or f"'{pending.description}' could not be redone."
            # an imputation: recorded with the rows it used and the values it filled there
            record = result.records.get(0)
            if record is not None:
                step["params"], step["description"] = record.params, record.description
            frame = result.df
            redone.append((pending, frame, step, block))
        # the snapshots show the changes on the older data: the replay recomputes the redone changes
        dropped = self._draft_snapshots()
        session.discard_staging()
        left_out = []  # a change that changes nothing on the current data is not kept
        for pending, staged, step, block in redone:
            kept = session.stage(
                staged,
                pending.function,
                step["params"],
                step["description"],
                population=block,
                replayable=pending.replayable,
                output_dataset_path=None,
            )
            if kept is None:
                left_out.append(f"'{pending.description}'")
        self._delete_snapshots(dropped)
        self._install_views()
        n = len(redone) - len(left_out)
        if not n:
            changes = "changes" if len(left_out) == 1 else "change"
            return (
                True,
                f"The draft is empty: {', '.join(left_out)} {changes} nothing on the current data.",
            )
        message = f"{n} change{'s' if n != 1 else ''} redone on the current data."
        if left_out:
            message += f" Left out, as they change nothing there: {', '.join(left_out)}."
        return True, message

    def _on_current_data(self, block: Mapping[str, Any]) -> Mapping[str, Any]:
        """A recorded population evaluated again, with its criteria, on the current dataset: a draft redone on new
        data is then not taken for one on another population. The recorded one when it cannot be evaluated."""
        session = self.session
        base = session.base
        assert base is not None
        evaluated = evaluate_population(base, criteria_from_trace(block), data_fp=session.data_fp)
        return evaluated.to_trace() if evaluated.ok else block

    def population_scope(self, frame: pd.DataFrame) -> PopulationScope:
        """The rows of ``frame`` an imputation is computed on and fills: those of the analysis population, every row
        when no criterion is ticked; none while a ticked criterion cannot be evaluated (``blocked``)."""
        population = self.population
        blocked = population is not None and not population.ok
        scoped = population is not None and bool(population.criteria) and not blocked
        rows = (
            frame.index.intersection(population.included)
            if scoped and population is not None
            else frame.index
        )
        return PopulationScope(population, blocked, scoped, rows)

    def imputations_on_other_rows(self) -> list[str]:
        """
        Imputations applied in this session that were computed on other rows than the current population (every row,
        when no criterion was ticked yet). A statistical imputation, or a formula of other rows (a mean), depends on
        the rows it was computed on; a formula of each row's own values only matters for the rows the population gained
        since and left missing.
        """
        population = self.population
        if population is None or not population.ok:
            return []
        current = population.rows_fingerprint
        fitted, formulas = step_ids(fills=FITTED), step_ids(fills=FORMULA)
        base = self.session.base
        out = []
        for step in self.transformation_manager.history:
            block = step.get("population") if isinstance(step, dict) else None
            function = step.get("function")
            params = step.get("params") or {}
            if not (block and block.get("rows_fp") and block["rows_fp"] != current):
                continue
            if function in formulas and params.get("row_wise"):
                recorded = block.get("included")  # the row ids, written when there are not too many
                if recorded is not None and base is not None:
                    gained = population.included.difference(ranges_to_ids(recorded))
                    target = params.get("target_variable")
                    if target not in base.columns or not base.loc[gained, target].isna().any():
                        continue  # each row's own values, and no row gained since is left missing
            elif function not in fitted and function not in formulas:
                continue
            # a step on a population names its rows in its description, one on every row does not
            every = "" if block.get("criteria") else f" (all {block.get('n_included', '?')} rows)"
            out.append(f"{step.get('description')}{every}")
        return out

    # ------------------------------------------------------------------ derived variables

    @property
    def derived_variables(self) -> tuple[DerivedVariable, ...]:
        """The derived variables declared with the study's configuration (core.derived)."""
        return derived_from_config(self.config)

    def set_derived_variables(self, definitions: Iterable[DerivedVariable]) -> None:
        """Declare these derived variables instead (the configuration is replaced, never changed in place)."""
        st.session_state["config"] = with_definitions(self.config, definitions)

    def derived_check(
        self, definition: DerivedVariable, *, slot: str | None = None
    ) -> DerivedCheck:
        """
        The definition checked against the Enrich data (enrich.derived), kept until those data or what the check reads
        change (its variable, formula, tolerance and kind; not its source or its acceptance). One slot per variable by
        default, so that editing one definition does not check the others again; ``slot`` for another use (a preview).
        """
        from enrich.derived import check_derived

        session = self.session
        frame = session.draft_frame()
        assert frame is not None
        read = (definition.target, definition.formula, definition.tolerance, definition.kind)
        check = session.memo(
            slot or f"derived_check::{definition.target}",
            (session.draft_fp, read),
            lambda: check_derived(definition, frame, imputed=self.imputed_cells()),
        )
        return replace(check, definition=definition)  # its acceptance as it is now

    def derived_inverse_check(self, definition: DerivedVariable, name: str) -> DerivedCheck:
        """The inverse of ``definition`` that recovers ``name``, checked as a definition of its own."""
        from enrich.derived import inverse_definition

        return self.derived_check(
            inverse_definition(definition, name),
            slot=f"derived_inverse::{definition.target}::{name}",
        )

    def imputed_cells(self, *, base: bool = False) -> dict[str, Any]:
        """The cells of the Enrich data an imputation filled, single or exact (row ids per column): values no one
        recorded, which a derived variable's check leaves out. ``base``: those of the dataset, a pending draft's
        aside (Validate audits the dataset)."""
        session = self.session
        source = session.draft if session.draft is not None and not base else session
        return merge_cells(dict(source.single_imputed), dict(source.exact_imputed))

    def exact_fill_plan(self) -> FillPlan:
        """What filling the declared derived variables exactly would do on the Enrich data, within the analysis
        population (enrich.derived.plan_exact_fills), with the checks the page shows; kept until those change."""
        from enrich.derived import plan_exact_fills

        session = self.session
        frame = session.draft_frame()
        assert frame is not None
        scope = self.population_scope(frame)
        definitions = self.derived_variables
        population = scope.population.fingerprint if scope.scoped and scope.population else ""
        read = tuple(canonical_json({"target": d.target, **d.to_config()}) for d in definitions)
        source = session.draft if session.draft is not None else session
        return session.memo(
            "derived_fill_plan",
            (session.draft_fp, population, scope.blocked, read),
            lambda: plan_exact_fills(
                definitions,
                frame,
                scope.rows if scope.scoped else None,
                imputed=self.imputed_cells(),
                single=dict(source.single_imputed),
                check=self.derived_check,
                inverse_check=self.derived_inverse_check,
            ),
        )

    def fill_exactly(self) -> int:
        """
        Stage what filling exactly gives on the current data (:meth:`exact_fill_plan`) in the Enrich draft, each fill a
        targeted imputation recorded with the definition it used (core.derived.record), and with the decisions its
        replay repeats rather than makes again (the formula, whether the values count as observed, their rounding):
        the replay never reads the declared list. One snapshot of the filled table for the run, with its last step; it
        is deleted when the run does not reach the draft. Returns the values filled; none while a ticked criterion
        cannot be evaluated.
        """
        from enrich.derived import fill_steps

        session = self.session
        frame = session.draft_frame()
        assert frame is not None
        scope = self.population_scope(frame)
        plan = self.exact_fill_plan()
        if scope.blocked or not plan.fills:
            return 0
        # the same fills on the same data record the same steps
        run = hashlib.sha256(
            canonical_json(
                [
                    session.draft_fp,
                    scope.population.fingerprint if scope.scoped and scope.population else "",
                    [(f.target, f.definition.semantic_hash) for f in plan.fills],
                ]
            ).encode("utf-8")
        ).hexdigest()[:12]
        steps = fill_steps(plan, frame, scope.rows if scope.scoped else None, run)
        snapshot = save_snapshot_to(
            steps[-1][0], "derived_fill", snapshot_folders(get_current_scope())[2]
        )
        filled = 0
        try:
            for position, (filled_frame, params, description) in enumerate(steps, start=1):
                if self.stage(
                    filled_frame,
                    "targeted_imputation",
                    params,
                    description,
                    population=scope.population,
                    output_dataset_path=snapshot if position == len(steps) else None,
                ):
                    filled += int(params["filled_count"])
        except Exception:  # the run did not reach the draft whole: its snapshot is no one's
            self._delete_snapshots([snapshot])
            raise
        return filled

    @contextlib.contextmanager
    def draft_savepoint(self) -> Iterator[None]:
        """Changes staged inside reach the Enrich draft together or not at all: on an error the draft is put back as it
        was, and the snapshots of the changes taken back are deleted."""
        mark = self.session.draft_mark()
        try:
            yield
        except Exception:
            dropped = self.session.rollback_draft(mark)
            self._delete_snapshots([step.output_dataset_path for step in dropped])
            self._install_views()
            raise

    def impute_single(
        self, settings: Mapping[str, Any], *, recompute: bool, progress_placeholder: Any = None
    ) -> SingleImputation:
        """
        The single imputation of the Enrich page, staged in the draft, fitted on and filling the analysis population
        (every row when no criterion is ticked); ``settings`` are enrich.imputation_service.impute_rows's. With
        ``recompute`` and declared derived variables (enrich.derived.recomputed): what their formulas give exactly
        first, then the imputation with each derived variable still missing kept out, neither imputed nor used to impute
        the others, then each computed by its formula from the imputed values. A complete one informs the others as any
        recorded variable does. The parts reach the draft together or not at all.
        """
        from enrich.derived import contradicted, recomputed
        from enrich.imputation_service import describe_imputation, impute_rows

        frame = self.session.draft_frame()
        assert frame is not None
        scope = self.population_scope(frame)
        if scope.blocked:
            raise PopulationError("a ticked criterion cannot be evaluated")
        pairs = [
            (d, c)
            for d, c in recomputed(self.derived_variables, self.derived_check)
            if d.target in frame.columns
        ]
        rows = scope.rows if scope.scoped else None
        with self.draft_savepoint():
            exact_first = self.fill_exactly() if recompute and pairs else 0
            # kept out: what is still missing after the exact fills, which only the imputed values can give
            missing = self.exact_fill_plan().left if recompute and pairs else {}
            kept_out = [(d, c) for d, c in pairs if missing.get(d.target)]
            aside = [d.target for d, _ in kept_out]
            working = self.session.draft_frame()
            assert working is not None
            outcome = impute_rows(
                working,
                rows,
                **settings,
                progress_placeholder=progress_placeholder,
                set_aside=aside,
            )
            params = dict(outcome.params)
            declined = pairs if not recompute else []
            if (
                kept_out or declined
            ):  # the choice, and the definitions kept out or declined (a record only)
                params["derived"] = {
                    "recompute": bool(kept_out),
                    "definitions": [
                        record(d, (), check.summary()) for d, check in (kept_out or declined)
                    ],
                }
            snapshot = save_snapshot_to(
                outcome.frame, "imputation", snapshot_folders(get_current_scope())[2]
            )
            staged = self.stage(
                outcome.frame,
                "imputation",
                params,
                describe_imputation(
                    str(settings.get("numerical", "Median")),
                    str(settings.get("categorical", "Most_Frequent")),
                    None if rows is None else len(rows),
                    outcome.params.get("set_aside") or (),
                ),
                # also with no criterion ticked: the draft then knows when one is ticked later
                population=scope.population,
                output_dataset_path=snapshot,
            )
            derived = self.fill_exactly() if kept_out else 0
        # what the formulas could not give, and recorded values their imputed variables no longer match
        left = self.exact_fill_plan().left if kept_out else {}
        final = self.session.draft_frame()
        source = self.session.draft if self.session.draft is not None else self.session
        cells = dict(source.single_imputed)
        mismatched = (
            {d.target: contradicted(d, c, final, cells) for d, c in pairs if final is not None}
            if recompute
            else {}
        )
        return SingleImputation(
            outcome,
            staged,
            exact_first,
            derived,
            {name: n for name, n in left.items() if name in aside},
            {name: n for name, n in mismatched.items() if n},
        )

    def derived_fills_edited(self) -> list[str]:
        """The variables the Enrich draft filled, or kept out of an imputation, with a definition that was edited or
        removed since: the draft keeps the values of the definition it used (its record), which the declared list no
        longer says."""
        from core.derived import from_record

        draft = self.session.draft
        if draft is None:
            return []
        declared = {d.target: d for d in self.derived_variables}
        edited: list[str] = []
        for step in draft.steps:
            derived = step.params.get("derived")
            if not decided_on_data(derived):
                continue
            blocks = (
                [derived] if "definition" in derived else list(derived.get("definitions") or [])
            )
            for block in blocks:
                used, uses, _ = from_record(block)
                if used is None:
                    continue
                now = declared.get(used.target)
                same = now is not None and now.semantic_hash == used.semantic_hash
                for name in uses:
                    if name != used.target:
                        same = (
                            same and name in now.inverses and now.inverse_hash(name) == uses[name]
                        )
                if not same and used.target not in edited:
                    edited.append(used.target)
        return edited

    # ------------------------------------------------------------------ the analysis population

    @property
    def population(self) -> PopulationResult | None:
        return self.session.population(self.config) if self.session.loaded else None

    def analysis_frame(self) -> pd.DataFrame:
        """Rows kept by the ticked criteria; raises PopulationError when a ticked criterion is broken."""
        return self.session.analysis_frame(self.config)

    def identity(self) -> DataIdentity | None:
        return self.session.identity(self.config) if self.session.loaded else None

    def single_imputed_counts(self, columns: Iterable[str] | None = None) -> dict[str, int]:
        """Values filled by a single imputation in the analysis population, per column (``columns`` only)."""
        if not self.session.loaded or not self.session.single_imputed:
            return {}
        try:
            rows = self.analysis_frame().index
        except PopulationError:
            return {}
        return self.session.single_imputed_counts(rows, columns)

    def exact_imputed_counts(self, columns: Iterable[str] | None = None) -> dict[str, int]:
        """Values imputed by an exact formula in the analysis population, per column (``columns`` only)."""
        if not self.session.loaded or not self.session.exact_imputed:
            return {}
        try:
            rows = self.analysis_frame().index
        except PopulationError:
            return {}
        return self.session.exact_imputed_counts(rows, columns)

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

    # ------------------------------------------------------------------ derived from the data

    def analyzer(self, rows: Literal["base", "analysis"] = "analysis") -> DataAnalyzer | None:
        """
        A DataAnalyzer of every row (``"base"``, as Validate checks the data) or of the analysis population
        (``"analysis"``, the rows of ``st.session_state["data"]``). A new one on each call, which a page may
        refresh; the column profile behind it is computed once per dataset and population (``DataSession.memo``).
        """
        session = self.session
        if not session.loaded or session.base is None:
            return None
        frame, slot, key = session.base, "analyzer:base", ""
        if rows == "analysis":
            population = session.population(self.config)
            # every row when no criterion is ticked, or a ticked one is broken (as the views, _install_views)
            if population.ok and population.criteria:
                frame = session.analysis_frame(self.config)
                slot, key = "analyzer:analysis", population.fingerprint
        profile = session.memo(slot, key, lambda: DataAnalyzer(frame.copy(deep=False)).profile)
        return DataAnalyzer.from_profile(frame.copy(deep=False), profile)

    def quality_audit(
        self, method: str, params: Mapping[str, Any] | None
    ) -> tuple[DataQualityAuditor, dict[str, Any], list[dict[str, str]]]:
        """
        The quality audit of every row (the Validate dashboard): the auditor, its metrics and its advice. It is kept
        until the data, the criteria definitions, the declared derived variables or the validity method change; the
        page only reads the auditor. One audit is kept per session, with the rows that break an anomaly rule
        (``clinical_anomalies_df``).
        """
        import copy
        import hashlib

        from core.derived import CONFIG_KEY
        from core.lineage import canonical_json
        from explore.data_quality_auditor import DataQualityAuditor

        session = self.session
        base = session.base
        if base is None:
            raise RuntimeError("no dataset is loaded")
        config = (
            self.config
        )  # changed in place by the page: its content is the key, read on every run
        rules = canonical_json(config.get("mask_families") or {})
        derived = canonical_json(config.get(CONFIG_KEY) or {})  # checked by the advice
        settings = dict(params or {})
        key = (
            hashlib.sha256(rules.encode("utf-8")).hexdigest(),
            hashlib.sha256(derived.encode("utf-8")).hexdigest(),
            str(method),
            canonical_json(settings),
        )

        def audit() -> tuple[DataQualityAuditor, dict[str, Any], list[dict[str, str]]]:
            # the cells an imputation filled (the derived variables' check does not compare them) change only with
            # the data, which empty the memo
            auditor = DataQualityAuditor(
                base.copy(deep=False),
                config=copy.deepcopy(config),
                imputed=self.imputed_cells(base=True),
            )
            metrics = auditor.run_audit(validity_method=method, validity_params=settings)
            return auditor, metrics, list(auditor.advice)

        auditor, metrics, advice = session.memo("quality_audit", key, audit)
        return auditor, dict(metrics), list(advice)

    # ------------------------------------------------------------------ multiple imputation

    @property
    def mi(self) -> Any:
        """The multiple imputation of this dataset (biostats.mi.binding.StoredMI), if any."""
        return st.session_state.get(MI_KEY)

    def set_mi(self, run: Any, *, record: bool = True) -> Any:
        """
        Keep an imputation run (biostats.mi.engine.ImputationRun) bound to the current data and population, and
        record it in the trace (``mi_impute``: the specification and the fingerprint of the imputed cells, which
        the replay re-computes and checks), with how the declared derived variables were computed when they were
        (``mi.info["derived"]``, enrich.derived.impute_and_derive), which the replay repeats.
        """
        from biostats.mi.binding import bind

        identity = self.identity()
        population = None
        with contextlib.suppress(PopulationError, RuntimeError):
            population = self.population
        stored = bind(
            run,
            data_fp=identity.data_fp if identity else "",
            population_fp=identity.population_fp if identity else "",
            population=population.to_trace() if population is not None else None,
        )
        st.session_state[MI_KEY] = stored
        tm = self.transformation_manager
        if record and tm.session_id:
            import numpy
            import sklearn

            from biostats.mi.engine import ENGINE
            from biostats.mi.report import incomplete_share
            from biostats.mi.spec import DERIVED

            mi = run.mi
            params = {
                "spec": run.spec.model_dump(),
                "methods": dict(run.methods),
                "m": mi.m,
                "n_rows": int(len(mi.base)),
                "n_imputed": mi.n_imputed(),
                # the derived variables computed from the others are not counted, as on the page
                "incomplete": incomplete_share(
                    mi.base, [c for c in run.spec.columns if run.methods.get(c) != DERIVED]
                ),
                "cells_fp": mi.fingerprint(),
                "engine": ENGINE,
                "versions": {"numpy": numpy.__version__, "scikit-learn": sklearn.__version__},
            }
            if mi.info.get("derived"):
                params["derived"] = dict(mi.info["derived"])
            label = ", ".join(f"{c} ({m})" for c, m in run.methods.items())
            with contextlib.suppress(Exception):
                tm.add_step(
                    "mi_impute",
                    params,
                    f"Multiple imputation, m = {mi.m}: {label}",
                    population=stored.binding.population,
                )
        return stored

    def mi_derived_edited(self) -> list[str]:
        """The variables the kept imputation computed with a definition that was edited or removed since: it keeps
        the values of the formula it used (its record), which the declared list no longer says."""
        from core.derived import from_record

        stored = self.mi
        if stored is None:
            return []
        declared = {d.target: d for d in self.derived_variables}
        edited = []
        for step in (stored.mi.info.get("derived") or {}).get("steps") or []:
            used, _, _ = from_record(step.get("record") or {})
            if used is None:
                continue
            now = declared.get(used.target)
            if now is None or now.semantic_hash != used.semantic_hash:
                edited.append(used.target)
        return edited

    def clear_mi(self) -> None:
        st.session_state.pop(MI_KEY, None)

    def mi_status(self) -> Any:
        """fresh, by_subset, subset or stale against the current analysis population (None without imputation)."""
        from biostats.mi.binding import MIStatus, status

        stored = self.mi
        if stored is None or not self.has_data:
            return None
        try:
            frame = self.analysis_frame()
        except PopulationError as exc:
            return MIStatus("stale", f"the analysis population cannot be evaluated: {exc}", 0)
        return status(stored, frame)

    def analysis_mi(self, *, allow_subset: bool = False) -> Any:
        """The imputed datasets restricted to the analysis population, when the imputation still fits it."""
        from biostats.mi.binding import restrict

        current = self.mi_status()
        if current is None or not (current.usable or (allow_subset and current.kind == "subset")):
            return None
        return restrict(self.mi, self.analysis_frame().index)

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
        enrich_view = session.draft_frame().copy(deep=False)
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

    @staticmethod
    def recorded_path(path: str | None) -> str | None:
        """
        How a file is written in the trace: relative to the user's workspace (or to the data folder outside a
        session), never as the server's absolute path, which names the account. A trace is shared; the replay
        resolves the relative path inside the workspace again (manage.paths.PathResolver). A path outside both
        is kept as given.
        """
        if not path:
            return path
        scope = get_current_scope()
        base = scope.root if scope is not None else Path(get_data_root())
        try:
            return Path(os.path.realpath(path)).relative_to(os.path.realpath(base)).as_posix()
        except ValueError:
            return path

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
            file_hash=file_checksum(data),
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
        """Load a saved version; its lineage sidecar links the new trace to the one that produced it, restores
        the ticked criteria and detects a file modified outside CohortStudio."""
        df, msg = self.db_manager.load_dataset(saved_name)
        if df is None:
            return False, msg
        path = os.path.join(self.db_manager.dataset_dir, f"{saved_name}.parquet")
        ref = DatasetRef.from_saved_name(
            saved_name, source_path=path, source_name=f"{saved_name}.parquet"
        )
        lineage = self.db_manager.load_lineage(saved_name) or {}
        trace = lineage.get("trace") or {}
        parent = (
            {
                "dataset": saved_name,
                "trace_session": trace.get("session_id"),
                "trace_head": trace.get("head_hash"),
                "n_steps": trace.get("n_steps"),
                "fingerprint": lineage.get("fingerprint"),
            }
            if lineage
            else None
        )
        self.set_dataset(df, ref, description=f"Loaded saved version: {saved_name}", parent=parent)
        if lineage:
            if lineage.get("fingerprint") and lineage["fingerprint"] != self.session.data_fp:
                msg += " Warning: the file differs from the version CohortStudio saved (modified outside the app)."
            known = {c.id for c in catalogue(self.config)}
            ticked = [
                i for i in (lineage.get("population") or {}).get("ticked") or [] if i in known
            ]
            if ticked:
                self.set_ticked(ticked)
            cells = lineage.get("single_imputed") or {}
            if cells:
                self.session.restore_single_imputed(
                    {col: ranges_to_ids(ranges) for col, ranges in cells.items()}
                )
            exact = (
                lineage.get("exact_imputed") or {}
            )  # absent from the lineage files of older versions
            if exact:
                self.session.restore_exact_imputed(
                    {col: ranges_to_ids(ranges) for col, ranges in exact.items()}
                )
        from manage.mi_store import has_mi, load_mi

        if has_mi(self.db_manager.dataset_dir, saved_name):
            try:
                stored = load_mi(self.db_manager.dataset_dir, saved_name, df)
            except (ValueError, OSError) as exc:
                msg += f" Its multiple imputation could not be restored: {exc}"
            else:
                st.session_state[MI_KEY] = stored
                msg += f" Multiple imputation restored (m = {stored.mi.m})."
        return True, msg

    def save_version(self, base_name: str | None = None) -> tuple[bool, str, str | None]:
        base = self.base
        if base is None:
            return False, "No dataset loaded.", None
        ref = self.dataset
        name = dataset_base_name(base_name or (ref.name if ref else "cohort"))
        ok, msg, saved = self.db_manager.save_dataset(base, base_name=name, username=self.username)
        if ok and saved:
            self.session.record_population_use(self.config)
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
            tm = self.transformation_manager
            with contextlib.suppress(Exception):
                tm.add_step(
                    "save_version",
                    {"dataset_name": saved, "fingerprint": self.session.data_fp},
                    f"Saved version: {saved}",
                )
            self._save_lineage(saved, ref)
            current = self.mi_status()
            if current is not None:
                if current.usable:
                    from manage.mi_store import save_mi

                    save_mi(self.mi, self.db_manager.dataset_dir, saved)
                    msg += f" Saved with its multiple imputation (m = {self.mi.mi.m})."
                else:
                    msg += f" Its multiple imputation was not saved ({current.kind}: {current.reason})."
        return ok, msg, saved

    def _save_lineage(self, saved: str, ref: DatasetRef | None) -> None:
        from datetime import datetime

        tm = self.transformation_manager
        session = self.session
        population = None
        with contextlib.suppress(PopulationError, RuntimeError):
            population = session.population(self.config)
        lineage = {
            "schema": 1,
            "dataset": saved,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "fingerprint": session.data_fp,
            "n_rows": int(len(session.base)) if session.base is not None else 0,
            "n_cols": int(session.base.shape[1]) if session.base is not None else 0,
            "row_id": "__row_id",
            "single_imputed": {
                col: ids_to_ranges(ids.tolist()) for col, ids in session.single_imputed.items()
            },
            "exact_imputed": {
                col: ids_to_ranges(ids.tolist()) for col, ids in session.exact_imputed.items()
            },
            "parent_dataset": ref.label if ref else None,
            "trace": {
                "session_id": tm.session_id,
                "file": os.path.basename(tm.get_trace_path() or ""),
                "head_hash": getattr(tm, "head_hash", None),
                "n_steps": len(tm.history),
            },
            "population": {
                "ticked": list(session.ticked),
                "block": population.to_trace()
                if population is not None and population.criteria
                else None,
            },
            "source": {
                "name": ref.source_name if ref else None,
                "file_hash": ref.file_hash if ref else None,
                "parse": ref.parse.to_dict() if ref and ref.parse else None,
            },
        }
        try:
            self.db_manager.save_lineage(saved, lineage)
        except OSError as exc:
            logger.warning("Lineage sidecar of %s not written: %s", saved, exc)

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
        uses_cloud = getattr(rm, "uses_cloud", None)
        # data leaves this machine and the private network (a cloud provider, or an Ollama elsewhere)
        sends_out = uses_cloud() if callable(uses_cloud) else provider not in (None, "ollama")
        return {
            "initialized": initialized,
            "provider": provider,
            "consent": bool(getattr(rm, "cloud_consent", False)),
            "local": not sends_out,
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
