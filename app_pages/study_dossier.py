"""
Reproduce > Study dossier (follow-up programme M3): the participant flow of the analysis population with the
analyses added to the report, the list of those results (current, or computed on other data or another
population), and the Word study dossier that documents them.
"""

from __future__ import annotations

import platform
import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from core.attrition import AnalysisCount, describe_definition
from core.lineage import verify_trace
from ui.components import no_dataset, section, status_chip
from ui.components.flow import current_flow, render_flow
from ui.results import (
    CURRENT,
    STALE_DATA,
    ReportItem,
    remove_from_report,
    report_items,
    report_status,
)
from ui.state import get_state

ROOT = Path(__file__).resolve().parents[1]
# steps of the trace that do not change the data (or have their own section)
NOT_DATA_HANDLING = {
    "initial_load",
    "parse_report",
    "population_definition",
    "save_version",
    "mi_impute",
    "freeze_population",
}
LIBRARIES = ("numpy", "pandas", "scipy", "statsmodels", "scikit-learn", "lifelines", "streamlit")


def imputations_of(value: Any) -> int | None:
    """m when the result pools multiply imputed datasets (biostats.mi.analyze.MIResult), else None."""
    if hasattr(value, "m_total") and isinstance(getattr(value, "m", None), int):
        return int(value.m)
    if isinstance(value, dict):  # the pooled group comparison keeps one pooled result per variable
        for pooled in (value.get("results") or {}).values():
            m = imputations_of(getattr(pooled, "result", None))
            if m is not None:
                return m
    return None


def analysis_counts(items: list[ReportItem]) -> list[AnalysisCount]:
    """What each result of the report analysed, for the boxes under the population."""
    return [
        AnalysisCount(
            item.title,
            item.stamp.n_population,
            item.stamp.n_analysed,
            not_used=item.stamp.not_used,
            imputations=imputations_of(item.value),
            note=item.settings,
        )
        for item in items
    ]


def _stale_text(status: str) -> str:
    if status == STALE_DATA:
        return "computed on data that have changed since"
    return "computed on another analysis population"


def _app_version() -> str:
    try:
        return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        return "unknown"


def _versions() -> str:
    parts = [f"CohortStudio {_app_version()}", f"Python {platform.python_version()}"]
    for name in LIBRARIES:
        try:
            parts.append(f"{name} {version(name)}")
        except PackageNotFoundError:
            continue
    return ", ".join(parts)


def _provenance_rows(state, trace: dict) -> list[tuple[str, str]]:
    ref = state.dataset
    identity = state.identity()
    check = verify_trace(trace) if trace.get("steps") else None
    integrity = {
        "verified": "verified (hash chain intact)",
        "unverifiable": "not verifiable (recorded before traces were chained)",
    }
    rows = [
        ("Dataset", ref.label if ref else "dataset"),
        ("Source file", (ref.source_name if ref else None) or "n/a"),
        ("Source file hash (MD5)", (ref.file_hash if ref else None) or "n/a"),
        ("Parse options", str(ref.parse.to_dict()) if ref and ref.parse else "n/a"),
        ("Data fingerprint", identity.data_fp if identity else "n/a"),
        ("Revision of the data", str(identity.revision) if identity else "n/a"),
        ("Analysis population", identity.population_label if identity else "n/a"),
        ("Trace", str(trace.get("session_id") or "n/a")),
        ("Trace head", str(trace.get("head_hash") or "n/a")),
        (
            "Trace integrity",
            "n/a" if check is None else integrity.get(check.status, f"FAILED: {check.message}"),
        ),
        ("Software", _versions()),
    ]
    return rows


def _criteria_table(population) -> pd.DataFrame:
    counts = {c.criterion.id: c for c in population.counts}
    return pd.DataFrame(
        [
            {
                "Family": c.family.replace("_", " "),
                "Criterion": c.name,
                "Kind": c.kind,
                "Definition": describe_definition(c.name, c.definition),
                "Excluded first": f"{counts[c.id].sequential:,}" if c.id in counts else "n/a",
            }
            for c in population.criteria
        ]
    )


def _data_steps(trace: dict) -> list[tuple[str, str]]:
    from manage.trace_documenter import TraceDocumenter

    names = TraceDocumenter({})
    out = []
    for step in trace.get("steps") or []:
        function = str(step.get("function") or "")
        data = step.get("data") or {}
        if function in NOT_DATA_HANDLING or not data.get("fp_out"):
            continue
        text = str(step.get("description") or function)
        params = step.get("params") or {}
        formula = params.get("formula") or params.get("expression")
        if formula:
            text += f" (formula: {formula})"
        if data.get("columns_added"):
            text += f"; added: {', '.join(data['columns_added'])}"
        out.append((names._get_readable_function_name(function), text))
    return out


def _missing_table(frame: pd.DataFrame) -> pd.DataFrame:
    missing = frame.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    return pd.DataFrame(
        {
            "Variable": missing.index,
            "Missing": [f"{n:,}" for n in missing],
            "Share": [f"{n / max(len(frame), 1):.1%}" for n in missing],
        }
    )


def _mi_section(state) -> dict[str, Any] | None:
    stored = state.mi
    if stored is None:
        return None
    from biostats.mi.report import incomplete_share, methods_paragraph

    binding, mi = stored.binding, stored.mi
    spec = binding.spec or {}
    current = state.mi_status()
    rows = [
        ("Imputed datasets (m)", str(spec.get("m", mi.m))),
        ("Iterations", str(spec.get("maxit", "n/a"))),
        ("Seed", str(spec.get("seed", "n/a"))),
        ("Imputation within the levels of", str(spec.get("by") or "none")),
        ("Planned Cox model", " and ".join(spec["survival"]) if spec.get("survival") else "none"),
        ("Rows of the analysis population", f"{binding.n_rows:,}"),
        ("Engine", str((mi.info or {}).get("engine", "CohortStudio"))),
        ("Created", str(binding.created_at)),
    ]
    rhat = (stored.diagnostics or {}).get("rhat") or {}
    if rhat:
        name, value = max(rhat.items(), key=lambda kv: float(kv[1]) if kv[1] is not None else 0.0)
        rows.append(("Largest indicative R-hat", f"{float(value):.3f} ({name})"))
    methods = pd.DataFrame(
        [
            {"Variable": col, "Method": method or "not imputed", "Imputed values": f"{len(mi.imputations.get(col, [])):,}"}
            for col, method in binding.methods.items()
        ]
    )
    return {
        "methods_paragraph": methods_paragraph(
            binding, n_rows=binding.n_rows, incomplete=incomplete_share(mi.base, list(binding.columns)), info=mi.info
        ),
        "settings": rows,
        "methods": methods,
        "status": None if current is None else f"Status against the current population: {current.kind} ({current.reason}).",
    }


def session_parts(state, items: list[ReportItem], *, include_stale: bool) -> dict[str, Any]:
    """
    What the dossier needs from the session, read during the page run: the download builds the document later,
    outside the run, where the session is not available (finish_input and build_dossier are pure).
    """
    population = state.population
    try:
        trace = state.transformation_manager.get_trace() or {}
    except Exception:
        trace = {}
    kept, omitted = [], []
    for item in items:
        status = report_status(item)
        if status == CURRENT or include_stale:
            kept.append((item, status))
        else:
            omitted.append(f"{item.title} ({_stale_text(status)})")
    return {
        "flow": current_flow(state, analysis_counts([item for item, status in kept if status == CURRENT])),
        "kept": kept,
        "omitted": omitted,
        "population_label": population.label if population else "n/a",
        "provenance": _provenance_rows(state, trace),
        "criteria": _criteria_table(population) if population is not None and population.criteria else None,
        "data_steps": _data_steps(trace),
        "missing": _missing_table(state.analysis_frame()),
        "single_imputed": state.single_imputed_counts(),
        "mi": _mi_section(state),
    }


def finish_input(parts: dict[str, Any], *, title: str, author: str):
    """The DossierInput from the session parts: result tables, figures and references (no session needed)."""
    from manage.dossier import METHOD_REFERENCES, DossierInput, DossierResult
    from manage.flow_figure import flow_png
    from manage.report_tables import result_blocks

    results = []
    references: list[str] = [METHOD_REFERENCES["strobe"]]
    pooled = parts["mi"] is not None
    for item, status in parts["kept"]:
        blocks = result_blocks(item.value)
        references.extend(blocks.references)
        pooled = pooled or imputations_of(item.value) is not None
        results.append(
            DossierResult(
                title=item.title,
                settings=item.settings,
                provenance=item.stamp.caption,
                blocks=blocks,
                stale=None if status == CURRENT else _stale_text(status),
                single_imputed=item.stamp.single_imputed,
            )
        )
    if pooled:
        references += [
            METHOD_REFERENCES[k] for k in ("rubin1987", "barnard1999", "vanbuuren2018", "white2011", "sterne2009")
        ]
    flow = parts["flow"]
    return DossierInput(
        title=title,
        author=author,
        flow=flow,
        population_label=parts["population_label"],
        provenance=parts["provenance"],
        flow_png=flow_png(flow) if flow is not None else None,
        criteria=parts["criteria"],
        data_steps=parts["data_steps"],
        missing=parts["missing"],
        single_imputed=parts["single_imputed"],
        mi=parts["mi"],
        results=results,
        references=references,
        omitted=parts["omitted"],
    )


def dossier_input(state, items: list[ReportItem], *, include_stale: bool, title: str, author: str):
    """Everything the Word dossier needs (manage.dossier.DossierInput), read from the session now."""
    return finish_input(session_parts(state, items, include_stale=include_stale), title=title, author=author)


def app() -> None:
    state = get_state()
    if not state.has_data:
        no_dataset()
        return
    items = report_items()
    current = [item for item in items if report_status(item) == CURRENT]

    section(
        "Participant flow",
        "From the source file to the analysis population and each analysis of the dossier (STROBE item 13).",
    )
    render_flow(current_flow(state, analysis_counts(current)), key="dossier_flow")

    section(
        "Results in the dossier",
        "Add a result with the button under it on the analysis pages; it keeps what it was computed on.",
    )
    if not items:
        st.info(
            "No result yet. Run an analysis (Analyze) and click **Add to the study dossier** under the result.",
            icon=":material/note_add:",
        )
    for item in items:
        status = report_status(item)
        with st.container(border=True):
            head, chip, remove = st.columns([6, 1.4, 1], vertical_alignment="center")
            head.markdown(f"**{item.title}**" + (f" · {item.settings}" if item.settings else ""))
            with chip:
                status_chip("current" if status == CURRENT else "stale", "ok" if status == CURRENT else "caution")
            remove.button(
                "Remove",
                key=f"dossier_remove::{item.id}",
                icon=":material/delete:",
                on_click=remove_from_report,
                args=(item.id,),
            )
            st.caption(f":material/fingerprint: {item.stamp.caption}")
            if status != CURRENT:
                st.caption(
                    f":material/warning: {_stale_text(status).capitalize()}: the dossier leaves it out unless you "
                    "include stale results, marked as such. Run the analysis again to update it."
                )

    section(
        "Word dossier",
        "Provenance, analysis population and participant flow, data handling, missing data, the results above "
        "and the references.",
    )
    ref = state.dataset
    c1, c2 = st.columns([2, 1])
    title = c1.text_input(
        "Title", value=f"Study dossier: {ref.label if ref else 'dataset'}", key="dossier_title"
    )
    author = c2.text_input("Author", value=state.username or "", key="dossier_author")
    include_stale = st.checkbox(
        "Include the results computed on other data or another population, marked as not current",
        key="dossier_stale",
    )
    # read from the session now: the download builds the document after the run, without the session
    parts = session_parts(state, items, include_stale=include_stale)
    st.download_button(
        "Download the Word dossier",
        data=lambda: _word_bytes(parts, title=title, author=author),
        file_name=f"{(ref.name if ref else 'study')}_dossier.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        icon=":material/download:",
        on_click="ignore",
        key="dossier_download",
        type="primary",
    )
    st.caption(
        "The dossier supports the reporting of the participant flow (STROBE item 13) and of multiple imputation "
        "(Sterne et al. 2009); it does not certify compliance and is not an electronically signed record."
    )


def _word_bytes(parts: dict[str, Any], *, title: str, author: str) -> bytes:
    from manage.dossier import build_dossier

    return build_dossier(finish_input(parts, title=title, author=author)).getvalue()


def build_dossier_bytes(state, items: list[ReportItem], *, include_stale: bool, title: str, author: str) -> bytes:
    return _word_bytes(session_parts(state, items, include_stale=include_stale), title=title, author=author)


__all__ = ["analysis_counts", "app", "build_dossier_bytes", "dossier_input", "finish_input", "imputations_of", "session_parts"]
