"""
Clean > Categories: the values of a categorical variable as written, each given a clean value or set to missing, and
written into a new column beside the original, which is kept (enrich.recode). Spaces and capitals are pre-filled; a
one-letter code, a rare near spelling and a code that usually means missing are offered, never applied. The page shows
each value's own rows, on all rows, never an outcome beside them: merging values after seeing results is a choice the
intervals ignore. Apply is one traced step, ``recode_levels``, which the replay redoes from the recorded mapping.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.registry import as_code
from enrich.recode import Offer, level_counts, recode_column, suggest
from ui.components.levels import show_level
from ui.components.text import plain
from ui.quality_charts import chart_config
from ui.results import RESULTS_KEY
from ui.state import get_state
from ui.theme import OKABE_ITO

MAX_VALUES = 200  # more distinct values: free text or an identifier, not listed
CHOSEN = "chosen"  # a clean value typed, or given with the buttons
AS_BEFORE = "as cleaned before"
_BAR = OKABE_ITO[0]
_MISSING_BAR = "rgba(128, 140, 145, 0.55)"
WRITTEN, ROWS, CLEAN, MISSING, WHY = "Value as written", "Rows", "Clean value", "Missing", "Why"


def candidates(df: pd.DataFrame) -> list[tuple[str, int, int]]:
    """
    The text and categorical columns with 2 to MAX_VALUES values as written: (column, values, values the suggestions
    name), those with suggestions first, then in the dataset's order.
    """
    found = []
    for position, column in enumerate(df.columns):
        values = df[column]
        if (
            pd.api.types.is_bool_dtype(values)
            or pd.api.types.is_numeric_dtype(values)
            or pd.api.types.is_datetime64_any_dtype(values)
        ):
            continue
        counts = level_counts(values)
        if not 2 <= len(counts) <= MAX_VALUES:
            continue
        hints = suggest(counts)
        found.append((position, str(column), len(counts), len(hints.spellings) + len(hints.offers)))
    return [(column, n, k) for _, column, n, k in sorted(found, key=lambda f: (f[3] == 0, f[0]))]


def _recoded_before(state, source: str, target: str) -> dict[str, Any] | None:
    """The mapping of the last recode of ``source`` into ``target`` in the trace: cleaning again starts from it."""
    mapping = None
    for step in state.transformation_manager.history:
        params = step.get("params") or {}
        if step.get("function") == "recode_levels" and (
            params.get("source"),
            params.get("target"),
        ) == (
            source,
            target,
        ):
            mapping = dict(params.get("mapping") or {})
    return mapping


def _results_reading(column: str) -> int:
    """The stored results whose settings name ``column``: the step records how many (Methods: after analysis)."""
    store = st.session_state.get(RESULTS_KEY) or {}
    needle = json.dumps(column)
    return sum(
        1
        for kept in store.values()
        if needle in json.dumps(getattr(kept, "config", None), default=repr)
    )


def _keys(column: str) -> dict[str, str]:
    return {
        "manual": f"clean_manual::{column}",  # value as written -> clean value or None, given with the buttons
        "revision": f"clean_revision::{column}",  # the table starts again when it changes
        "editor": f"clean_editor_key::{column}",  # the key of the table drawn last
        "pick": f"clean_pick::{column}",
        "into": f"clean_into::{column}",
        "target": f"clean_target::{column}",
        "view": f"clean_view::{column}",
    }


def _fold(column: str, values: list[str]) -> None:
    """The table's edits into the values given with the buttons, before the table starts again (it would lose them)."""
    keys = _keys(column)
    manual = dict(st.session_state.get(keys["manual"]) or {})
    edits = (st.session_state.get(st.session_state.get(keys["editor"], "")) or {}).get(
        "edited_rows"
    ) or {}
    table = st.session_state.get(f"{keys['editor']}::start") or {}
    for row, changed in edits.items():
        value = values[int(row)]
        clean, missing = table.get(value, (value, False))
        clean = str(changed.get(CLEAN, clean) or "").strip()
        missing = bool(changed.get(MISSING, missing))
        manual[value] = None if missing else clean
    st.session_state[keys["manual"]] = manual
    st.session_state[keys["revision"]] = int(st.session_state.get(keys["revision"], 0)) + 1


def _give(column: str, values: list[str], how: str) -> None:
    """Callback of Map, Set to missing and Keep as written: the picked values given a clean value."""
    _fold(column, values)
    keys = _keys(column)
    manual = st.session_state[keys["manual"]]
    into = str(st.session_state.get(keys["into"]) or "").strip()
    for value in st.session_state.get(keys["pick"]) or []:
        manual[value] = None if how == "missing" else value if how == "keep" else into
    st.session_state[keys["pick"]] = []


def _start(
    counts: dict[str, int], column: str, offers: list[Offer], loaded: dict[str, Any] | None
) -> dict[str, tuple[str | None, str]]:
    """Each value as written: its clean value (None: missing) and why, before the table's own edits."""
    hints = suggest(counts)
    manual = st.session_state.get(_keys(column)["manual"]) or {}
    start: dict[str, tuple[str | None, str]] = {}
    for value in counts:
        if loaded is not None:
            clean, why = loaded.get(value, value), AS_BEFORE if value in loaded else ""
        else:
            clean, why = hints.spellings.get(value, value), hints.reasons.get(value, "")
        for offer in offers:
            if clean == offer.value:
                clean, why = offer.clean, offer.why
        if value in manual:
            clean, why = manual[value], CHOSEN
        start[value] = (clean, why)
    return start


def _figure(counts: dict[str, int], mapping: dict[str, Any], *, after: bool) -> go.Figure:
    """One bar per clean value (after) or per value as written (before); the values set to missing apart, in grey."""
    bars: dict[str, int] = {}
    merged: dict[str, list[str]] = {}
    missing = 0
    for value, n in counts.items():
        clean = mapping.get(value, value) if after else value
        if clean is None:
            missing += n
            continue
        bars[clean] = bars.get(clean, 0) + n
        merged.setdefault(clean, []).append(show_level(value))
    names = sorted(bars, key=lambda name: (-bars[name], name))
    labels = [f"{show_level(name)} ({bars[name]:,})" for name in names]
    sizes = [bars[name] for name in names]
    colors = [_BAR] * len(names)
    hovers = [
        ", ".join(merged[name][:12]) + (" ..." if len(merged[name]) > 12 else "") for name in names
    ]
    if missing:
        labels.append(f"missing ({missing:,})")
        sizes.append(missing)
        colors.append(_MISSING_BAR)
        hovers.append("set to missing")
    fig = go.Figure(
        go.Bar(
            x=sizes,
            y=labels,
            orientation="h",
            marker={"color": colors, "cornerradius": 4},
            hovertext=hovers,
            hovertemplate="%{y}<br>%{hovertext}<extra></extra>",
        )
    )
    fig.update_layout(
        height=min(60 + 26 * len(labels), 520),
        margin={"l": 10, "r": 10, "t": 10, "b": 30},
        bargap=0.35,
        showlegend=False,
        xaxis={"title": "Rows", "showgrid": True},
        yaxis={"autorange": "reversed", "automargin": True},
    )
    return fig


def category_editor(state, df: pd.DataFrame, column: str) -> None:
    """The editor of one variable's values: what each becomes, a preview, and Apply."""
    keys = _keys(column)
    counts = level_counts(df[column])
    values = list(counts)
    target = st.text_input(
        "New column",
        value=f"{column}_clean",
        key=keys["target"],
        persist_state="page",
        help="The clean values go into this column, right after the original, which is kept.",
    ).strip()
    loaded = _recoded_before(state, column, target) if target in df.columns else None
    hints = suggest(counts)
    if hints.offers:
        st.markdown("**Possible changes, not applied**")
        st.caption(
            "Tick one to apply it. A one-letter code, a near spelling or a code for unknown can name a category of "
            "its own: check what it means in the source first."
        )
        held: dict[str, int] = {}
        for value, n in counts.items():
            spelling = hints.spellings.get(value, value)
            held[spelling] = held.get(spelling, 0) + n
        for offer in hints.offers:
            into = as_code(show_level(offer.clean)) if offer.clean is not None else "missing"
            st.checkbox(
                f"{as_code(show_level(offer.value))} ({held.get(offer.value, 0):,} rows) to {into}: {offer.why}",
                key=f"clean_offer::{column}::{offer.value}",
                persist_state="page",
                on_change=_fold,
                args=(column, values),
            )
    accepted = [
        o for o in hints.offers if st.session_state.get(f"clean_offer::{column}::{o.value}")
    ]
    start = _start(counts, column, accepted, loaded)

    with st.container(horizontal=True, vertical_alignment="bottom"):
        st.multiselect(
            "Values",
            values,
            format_func=show_level,
            key=keys["pick"],
            persist_state="page",
            placeholder="Choose values to map together",
        )
        st.text_input(
            "Clean value", key=keys["into"], persist_state="page", placeholder="What they become"
        )
        st.button("Map", key=f"clean_map::{column}", on_click=_give, args=(column, values, "map"))
        st.button(
            "Set to missing",
            key=f"clean_missing::{column}",
            on_click=_give,
            args=(column, values, "missing"),
        )
        st.button(
            "Keep as written",
            key=f"clean_keep::{column}",
            on_click=_give,
            args=(column, values, "keep"),
        )

    editor = f"clean_editor::{column}::{st.session_state.get(keys['revision'], 0)}::{len(accepted)}"
    st.session_state[keys["editor"]] = editor
    st.session_state[f"{keys['editor']}::start"] = {
        value: ("" if clean is None else clean, clean is None)
        for value, (clean, _) in start.items()
    }
    edited = st.data_editor(
        pd.DataFrame(
            {
                WRITTEN: [show_level(v) for v in values],
                ROWS: list(counts.values()),
                CLEAN: ["" if clean is None else clean for clean, _ in start.values()],
                MISSING: [clean is None for clean, _ in start.values()],
                WHY: [why for _, why in start.values()],
            }
        ),
        key=editor,
        hide_index=True,
        width="stretch",
        disabled=[WRITTEN, ROWS, WHY],
        column_config={
            WRITTEN: st.column_config.TextColumn(help="A hidden space shows as ␣."),
            ROWS: st.column_config.NumberColumn(format="%d"),
            CLEAN: st.column_config.TextColumn(
                help="The same clean value in several rows merges them."
            ),
            MISSING: st.column_config.CheckboxColumn(help="Set this value to missing."),
        },
    )
    mapping: dict[str, Any] = {}
    reasons: dict[str, str] = {}
    empty: list[str] = []
    for row, value in enumerate(values):
        missing = bool(edited.at[row, MISSING])
        clean = None if missing else str(edited.at[row, CLEAN] or "").strip()
        if clean == "":
            empty.append(value)
            continue
        if clean == value:
            continue
        mapping[value] = clean
        first, why = start[value]
        reasons[value] = why if clean == first and why else CHOSEN

    gone = sum(counts[v] for v, c in mapping.items() if c is None)
    if gone:
        st.caption(
            f"{gone:,} row{'s' if gone != 1 else ''} set to missing leave the complete-case analyses of "
            f"{as_code(column)}, or are imputed by a multiple imputation, which assumes them missing at random. Set "
            "to missing only a value that means unknown: 'not applicable' or 'declined' may need to stay a value."
        )
    after = st.segmented_control(
        "Show", ["After", "Before"], default="After", key=keys["view"], persist_state="page"
    )
    st.plotly_chart(
        _figure(counts, mapping, after=after != "Before"),
        config=chart_config(),
        width="stretch",
        key=f"clean_chart::{column}",
    )

    replaces = loaded is not None
    blocked = ""
    if not target or not target.isprintable():
        blocked = "Name the new column."
    elif target == column:
        blocked = "The clean values go into a new column: the original is kept."
    elif target in df.columns and not replaces:
        blocked = f"{as_code(target)} is already a column: choose another name."
    elif empty:
        blocked = f"Give {as_code(show_level(empty[0]))} a clean value, or tick Missing."
    elif not mapping:
        blocked = "Nothing to change: every value keeps how it is written."
    if blocked:
        st.caption(blocked)
    label = (
        f"Write {plain(target)} again ({plain(column)} kept)"
        if replaces
        else f"Add {plain(target)} ({plain(column)} kept)"
    )
    if st.button(label, key=f"clean_apply::{column}", type="primary", disabled=bool(blocked)):
        new, _ = recode_column(df, column, target, mapping, replaces=replaces)
        clean_values = {mapping.get(v, v) for v in values} - {None}
        changed = sum(counts[v] for v in mapping)
        params = {
            "source": column,
            "target": target,
            "mapping": mapping,
            "reasons": reasons,
            "replaces": replaces,
            "values_before": len(values),
            "values_after": len(clean_values),
            "rows_changed": changed,
            "rows_set_to_missing": gone,
            "results_before": _results_reading(column),
        }
        summary = f"{len(values)} values to {len(clean_values)}" + (
            f", {gone} row{'s' if gone != 1 else ''} set to missing" if gone else ""
        )
        state.commit_change(
            new, "recode_levels", params, f"Recoded {column} into {target}: {summary}"
        )
        st.session_state["clean_done"] = f"Added {as_code(target)}: {summary}."
        st.rerun()


def render_categories() -> None:
    """The Categories tab: a variable, then its editor."""
    state = get_state()
    df = state.base
    if df is None:
        return
    done = st.session_state.pop("clean_done", None)
    if done:
        st.success(done, icon=":material/check:")
    found = state.session.memo("clean_candidates", None, lambda: candidates(df))
    if not found:
        st.caption(f"No text column holds 2 to {MAX_VALUES} values: nothing to clean here.")
        return
    about = {column: (n, k) for column, n, k in found}
    column = st.selectbox(
        "Variable",
        list(about),
        format_func=lambda c: (
            f"{c} · {about[c][1]} to look at" if about[c][1] else f"{c} · {about[c][0]} values"
        ),
        key="clean_variable",
        persist_state="page",
    )
    if column is not None:
        category_editor(state, df, column)


__all__ = ["MAX_VALUES", "candidates", "category_editor", "render_categories"]
