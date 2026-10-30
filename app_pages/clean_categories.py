"""
Clean > Categories: the values of a categorical variable as written, each given a clean value or set to missing, and
written into a new column beside the original, which is kept (enrich.recode). Values that differ by spaces, and words
that differ by capitals, take their group's spelling (the fold rule, recorded as a rule); a one-letter code, a rare
near spelling, a case variant and a code that may mean missing are offered, never applied. A value set to missing says
what missing means there; two values that each hold 5 % of the rows, merged, say why. The page shows each value's own
rows, on all rows, never an outcome beside them: merging values after seeing results is a choice the intervals ignore.
Apply is one traced step, ``recode_levels``, which the replay redoes.

The choices live in one session key, ``clean_manual`` (ui.keys: cleared when another dataset is loaded), per column;
the widgets that edit them carry a short id of the dataset's trace and fold what they change into it as they change,
so that no choice is lost on a page change and none follows the user to another dataset.
"""

from __future__ import annotations

import hashlib
import html
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.fingerprint import frame_fingerprint
from core.registry import as_code
from enrich.recode import (
    CHOSEN,
    FOLD,
    GROUPING,
    MAX_VALUES,
    MEANINGS,
    Candidate,
    Offer,
    candidates,
    changes,
    groupings,
    last_record,
    level_counts,
    plan,
    recode_column,
    recoded_columns,
    suggest,
)
from ui.components.levels import show_level
from ui.components.text import plain
from ui.quality_charts import chart_config
from ui.state import get_state
from ui.theme import OKABE_ITO

STORE = "clean_manual"  # {column: choices}, declared in ui.keys (lifetime "dataset")
DONE = "clean_done"  # what the last Apply did, said once after the rerun
COLUMN = "clean_column"  # the variable open in the editor, declared in ui.keys
KEPT = "kept as written"  # the Why of a fold put back (shown, never stored)
_BAR = OKABE_ITO[0]
_MISSING_BAR = "rgba(128, 140, 145, 0.55)"
WRITTEN, ROWS, CLEAN, MISSING, MEANS, WHY = (
    "Value as written",
    "Rows",
    "Clean value",
    "Missing",
    "Means",
    "Why",
)


def _rows(n: int) -> str:
    return f"{n:,} row{'s' if n != 1 else ''}"


def _scope(state) -> str:
    """A short id of the dataset's trace: the widget keys carry it, so a widget's state never meets another dataset."""
    session_id = str(state.transformation_manager.session_id or "")
    return hashlib.sha256(session_id.encode()).hexdigest()[:8]


def _empty() -> dict[str, Any]:
    # "shown": what the table shows, so that its edits are read against it
    return {
        "mapping": {},
        "kinds": {},
        "kept": [],
        "meanings": {},
        "reasons": {},
        "revision": 0,
        "shown": {},
    }


def _seeded(steps: list, column: str, target: str) -> dict[str, Any]:
    """The choices of the last recode of ``column`` into ``target``: cleaning again starts from them, kinds kept."""
    record = last_record(steps, column, target)
    if record is None:
        return _empty()
    mapping = dict(record.get("mapping") or {})
    kinds = dict(
        record.get("kinds") or record.get("reasons") or {}
    )  # "reasons": before the fold rule
    return {
        "mapping": mapping,
        "kinds": {k: str(kinds.get(k, CHOSEN)) for k in mapping},
        "kept": list(record.get("kept") or []),
        "meanings": dict(record.get("missing_meaning") or {}),
        "reasons": dict(record.get("groupings") or {}),
        "revision": 0,
        "shown": {},
    }


def _choices(column: str) -> dict[str, Any]:
    return st.session_state.setdefault(STORE, {}).setdefault(column, _empty())


def _keys(scope: str, column: str, revision: int) -> dict[str, str]:
    base = f"{scope}::{column}"
    return {
        "target": f"clean_target::{base}",
        "pick": f"clean_pick::{base}",
        "into": f"clean_into::{base}",
        "means": f"clean_means::{base}",
        "view": f"clean_view::{base}",
        "offers": f"clean_offers::{base}::{revision}",  # starts again from the store when it changes
        "editor": f"clean_editor::{base}::{revision}",
    }


def _give(
    choices: dict[str, Any],
    value: str,
    clean: str | None,
    spelling: str,
    *,
    meaning: str | None = None,
) -> None:
    """One value given a clean value (None: missing): a spelling's choice is its group's, a folded value's its own."""
    mapping, kinds, kept = choices["mapping"], choices["kinds"], choices["kept"]
    if value in kept:
        kept.remove(value)
    if clean is not None and value != spelling:
        follows = mapping.get(spelling, spelling)
        if clean == value:  # its fold put back
            mapping.pop(value, None)
            kinds.pop(value, None)
            kept.append(value)
            return
        if clean == follows:  # its group's clean value again
            mapping.pop(value, None)
            kinds.pop(value, None)
            return
    if clean is not None and value == spelling and clean == value:  # as written: no choice
        mapping.pop(value, None)
        kinds.pop(value, None)
        return
    if mapping.get(value, "\0") != clean:
        kinds[value] = CHOSEN
    mapping[value] = clean
    if clean is None:
        if meaning:
            choices["meanings"][value] = meaning
    else:
        choices["meanings"].pop(value, None)


def _table_changed(column: str, key: str, values: list[str], spellings: dict[str, str]) -> None:
    """The table's edits into the store, as they are made; the table then starts again from it."""
    choices = _choices(column)
    shown = choices.get("shown") or {}
    for row, changed in ((st.session_state.get(key) or {}).get("edited_rows") or {}).items():
        value = values[int(row)]
        spelling = spellings.get(value, value)
        clean, missing, means = shown.get(value, (value, False, None))
        typed = changed.get(CLEAN, clean)
        clean = (
            "" if typed is None or pd.isna(typed) else str(typed).strip()
        )  # a cleared cell comes back NaN
        missing = bool(changed.get(MISSING, missing))
        means = changed.get(MEANS, means)
        means = str(means) if means in MEANINGS else None
        if missing:
            _give(choices, value, None, spelling, meaning=means)
        elif MISSING in changed and not clean:  # missing unticked: back to what the fold gives
            choices["mapping"].pop(value, None)
            choices["kinds"].pop(value, None)
        else:
            _give(choices, value, clean, spelling)
    choices["revision"] += 1


def _offers_changed(column: str, key: str, offers: dict[str, Offer]) -> None:
    """A ticked offer fills the table (its group's choice); unticked, its values take their spelling again."""
    choices = _choices(column)
    picked = set(st.session_state.get(key) or [])
    for value, offer in offers.items():
        applied = value in choices["mapping"] and choices["mapping"][value] == offer.clean
        if value in picked and not applied:
            choices["mapping"][value] = offer.clean
            choices["kinds"][value] = offer.why
            if offer.clean is None and offer.meaning:
                choices["meanings"][value] = offer.meaning
            if value in choices["kept"]:
                choices["kept"].remove(value)
        elif applied and value not in picked:
            for part in ("mapping", "kinds", "meanings"):
                choices[part].pop(value, None)
    choices["revision"] += 1


def _buttons(column: str, keys: dict[str, str], spellings: dict[str, str], how: str) -> None:
    """Map, Set to missing and Keep as written: the picked values given a clean value."""
    choices = _choices(column)
    into = st.session_state.get(keys["into"])
    means = st.session_state.get(keys["means"])
    for value in st.session_state.get(keys["pick"]) or []:
        spelling = spellings.get(value, value)
        if how == "missing":
            _give(choices, value, None, spelling, meaning=means)
        elif how == "keep":
            _give(choices, value, value, spelling)
        elif into is not None:
            _give(choices, value, str(into).strip(), spelling)
    st.session_state[keys["pick"]] = []
    choices["revision"] += 1


def _rows_shown(
    counts: dict[str, int],
    spellings: dict[str, str],
    reasons: dict[str, str],
    choices: dict[str, Any],
) -> dict[str, tuple[str | None, str, str | None]]:
    """Each value as written: its clean value (None: missing), why, and what its missing value means."""
    mapping, kinds, kept, meanings = (
        choices["mapping"],
        choices["kinds"],
        choices["kept"],
        choices["meanings"],
    )
    shown: dict[str, tuple[str | None, str, str | None]] = {}
    for value in counts:
        spelling = spellings.get(value, value)
        key = value if value != spelling and value in mapping else spelling
        if value in kept:
            shown[value] = (value, KEPT, None)
        elif key in mapping:
            clean = mapping[key]
            shown[value] = (
                clean,
                kinds.get(key, CHOSEN),
                meanings.get(key) if clean is None else None,
            )
        else:
            shown[value] = (spelling, reasons.get(value, ""), None)
    return shown


def _figure(counts: dict[str, int], full: dict[str, Any], *, after: bool) -> go.Figure:
    """One bar per clean value (after) or per value as written (before); the values set to missing apart, in grey.
    The values are drawn as text: a level holding markup is shown, never rendered."""
    bars: dict[str, int] = {}
    merged: dict[str, list[str]] = {}
    missing = 0
    for value, n in counts.items():
        clean = full.get(value, value) if after else value
        if clean is None:
            missing += n
            continue
        bars[clean] = bars.get(clean, 0) + n
        merged.setdefault(clean, []).append(html.escape(show_level(value)))
    names = sorted(bars, key=lambda name: (-bars[name], name))
    labels = [f"{html.escape(show_level(name))} ({bars[name]:,})" for name in names]
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
        height=min(60 + 26 * len(labels), 480),
        margin={"l": 10, "r": 10, "t": 10, "b": 30},
        bargap=0.35,
        showlegend=False,
        xaxis={"title": "Rows", "showgrid": True},
        yaxis={"autorange": "reversed", "automargin": True},
    )
    return fig


def _replaceable(df: pd.DataFrame, steps: list, column: str, target: str) -> str:
    """Why ``target`` cannot be written again ("" when it can): it is the copy of ``column`` that the last recode
    wrote, and no later step changed it (its fingerprint, or the recode of the source as it is now, still gives it)."""
    record = last_record(steps, column, target)
    if record is None:
        return f"{as_code(target)} is already a column: choose another name."
    fp = record.get("target_fp")
    if fp is None or frame_fingerprint(df[[target]]) == fp:
        return ""
    try:  # a later step removed rows from both (a freeze): the copy is still what the recode writes
        again, _ = recode_column(
            df,
            column,
            target,
            record.get("mapping") or {},
            fold_rule=record.get("fold"),
            kept=record.get("kept") or (),
            replaces=True,
        )
        if again[target].equals(df[target]):
            return ""
    except ValueError:
        pass
    return f"{as_code(target)} was changed by a later step: write the clean values into another column."


def _recorded(counts: dict[str, int], choices: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """The choices that change something over the fold rule (a choice its group or the fold already makes is left
    out, never recorded), and the folds put back."""
    mapping = {k: v for k, v in choices["mapping"].items() if v != ""}
    kept = [v for v in choices["kept"] if v in counts]
    full = plan(counts, mapping, kept=kept)
    needed = {}
    for key, clean in mapping.items():
        without = {k: v for k, v in mapping.items() if k != key}
        if plan(counts, without, kept=kept) != full:
            needed[key] = clean
    return needed, [
        v for v in kept if plan(counts, needed, kept=[k for k in kept if k != v]) != full
    ]


def category_editor(
    state, df: pd.DataFrame, column: str, *, header: Any = None, done: str | None = None
) -> None:
    """The editor of one variable's values: the offers, what each value becomes, a preview, and Apply. ``header``:
    the container beside the variable's choice that holds the new column's name."""
    steps = state.transformation_manager.history
    scope = _scope(state)
    counts = level_counts(df[column])
    values = list(counts)
    hints = suggest(counts)
    spellings = hints.spellings
    cleaned_into = recoded_columns(steps).get(column, "")
    store = st.session_state.setdefault(STORE, {})
    if column not in store:
        target = cleaned_into if cleaned_into in df.columns else f"{column}_clean"
        store[column] = _seeded(steps, column, target)
    choices = store[column]
    keys = _keys(scope, column, choices["revision"])

    with header if header is not None else st.container():
        target = st.text_input(
            "New column",
            value=cleaned_into if cleaned_into in df.columns else f"{column}_clean",
            key=keys["target"],
            persist_state="page",
            help="The clean values go into this column, right after the original, which is kept.",
        ).strip()

    held: dict[str, int] = {}
    for value, n in counts.items():
        spelling = spellings.get(value, value)
        held[spelling] = held.get(spelling, 0) + n
    offers = {offer.value: offer for offer in hints.offers}
    if offers:
        st.markdown("**Check before merging**")

        def label(value: str) -> str:
            offer = offers[value]
            into = as_code(show_level(offer.clean)) if offer.clean is not None else "missing"
            return (
                f"{as_code(show_level(value))} → {into} · {offer.why} · {_rows(held.get(value, 0))}"
            )

        applied = [
            v
            for v, o in offers.items()
            if v in choices["mapping"] and choices["mapping"][v] == o.clean
        ]
        st.pills(
            "Possible changes",
            list(offers),
            selection_mode="multi",
            default=applied,
            format_func=label,
            # drawn from the store (default=), not kept: its key changes with each revision of the choices
            key=f"clean_offers::{scope}::{column}::{choices['revision']}",
            on_change=_offers_changed,
            args=(column, keys["offers"], offers),
            label_visibility="collapsed",
        )
        st.caption(
            f"Ticking fills the table; the data change when you add {as_code(target or column + '_clean')}. A "
            "one-letter code, a near spelling, a case variant or a code for unknown can name a category of its own: "
            "check what it means in the source first."
        )

    shown = _rows_shown(counts, spellings, hints.reasons, choices)
    full_choices = {k: v for k, v in choices["mapping"].items() if v != ""}
    full = plan(counts, full_choices, kept=[v for v in choices["kept"] if v in counts])
    table, preview = st.columns([3, 2], gap="medium")

    with table:
        clean_values = sorted(
            {c for c, _, _ in shown.values() if c},
            key=lambda c: -sum(counts[v] for v, (cc, _, _) in shown.items() if cc == c),
        )
        with st.container(horizontal=True, vertical_alignment="bottom"):
            picked = st.multiselect(
                "Values",
                values,
                format_func=show_level,
                key=keys["pick"],
                persist_state="page",
                placeholder="Choose values",
            )
            into = st.selectbox(
                "Clean value",
                clean_values,
                index=None,
                key=keys["into"],
                persist_state="page",
                accept_new_options=True,
                placeholder="What they become",
            )
        n = len(picked)
        with st.container(horizontal=True, vertical_alignment="bottom"):
            st.button(
                f"Map {n} value{'s' if n != 1 else ''} to {plain(str(into))}"
                if n and into
                else "Map",
                key=f"clean_map::{scope}::{column}",
                on_click=_buttons,
                args=(column, keys, spellings, "map"),
                disabled=not (n and into),
            )
            means = st.selectbox(
                "What missing means",
                MEANINGS,
                index=None,
                key=keys["means"],
                persist_state="page",
                placeholder="What missing means",
                label_visibility="collapsed",
            )
            st.button(
                "Set to missing",
                key=f"clean_missing::{scope}::{column}",
                on_click=_buttons,
                args=(column, keys, spellings, "missing"),
                disabled=not (n and means),
                help="Say first what missing means for these values.",
            )
            st.button(
                "Keep as written",
                key=f"clean_keep::{scope}::{column}",
                on_click=_buttons,
                args=(column, keys, spellings, "keep"),
                disabled=not n,
            )
        choices["shown"] = {
            v: ("" if c is None else c, c is None, m) for v, (c, _, m) in shown.items()
        }
        st.data_editor(
            pd.DataFrame(
                {
                    WRITTEN: [show_level(v) for v in values],
                    ROWS: list(counts.values()),
                    CLEAN: ["" if c is None else c for c, _, _ in shown.values()],
                    MISSING: [c is None for c, _, _ in shown.values()],
                    MEANS: [
                        m or "" for _, _, m in shown.values()
                    ],  # empty, not "None", where nothing is missing
                    WHY: [why for _, why, _ in shown.values()],
                }
            ),
            key=keys["editor"],
            on_change=_table_changed,
            args=(column, keys["editor"], values, spellings),
            hide_index=True,
            width="stretch",
            # what missing means, once a value is set to missing: an empty choice draws "None" in every row
            column_order=[
                WRITTEN,
                ROWS,
                CLEAN,
                MISSING,
                *([MEANS] if any(c is None for c, _, _ in shown.values()) else []),
                WHY,
            ],
            disabled=[WRITTEN, ROWS, WHY],
            column_config={
                WRITTEN: st.column_config.TextColumn(help="A hidden space shows as ␣."),
                ROWS: st.column_config.NumberColumn(format="%d"),
                CLEAN: st.column_config.TextColumn(
                    help="The same clean value in several rows merges them."
                ),
                MISSING: st.column_config.CheckboxColumn(help="Set this value to missing."),
                MEANS: st.column_config.SelectboxColumn(
                    options=list(MEANINGS), help="What missing means for a value set to missing."
                ),
            },
        )
        if any(show_level(v) != v for v in values):
            st.caption("␣ marks a hidden space.")

    with preview:
        view = st.segmented_control(
            "Show",
            ["After", "Before"],
            default="After",
            key=keys["view"],
            persist_state="page",
            label_visibility="collapsed",
        )
        st.plotly_chart(
            _figure(counts, full, after=view != "Before"),
            config=chart_config(),
            width="stretch",
            key=f"clean_chart::{scope}::{column}",
        )
        gone = sum(counts[v] for v, c in full.items() if c is None)
        if gone:
            st.caption(
                f"{_rows(gone)} set to missing: complete-case analyses using {as_code(target or column)} leave them "
                "out, and a multiple imputation fills them as missing at random given its variables. A value that "
                "means not applicable or declined is often better kept as a value."
            )
        grouped = groupings(counts, full)
        for clean, parts in grouped.items():
            reason = st.text_input(
                f"Why are {', '.join(show_level(p) for p in parts)} grouped as {show_level(clean)}?",
                value=choices["reasons"].get(clean, ""),
                key=f"clean_reason::{scope}::{column}::{clean}",
                max_chars=200,
                help="Merging values that each hold 5 % of the rows or more is a grouping of categories, not a "
                "correction: the methods say why.",
            ).strip()
            if reason:
                choices["reasons"][clean] = reason
            else:
                choices["reasons"].pop(clean, None)

        blocked = ""
        replaces = bool(target) and target in df.columns and target == cleaned_into
        unmeant = [
            k for k, c in choices["mapping"].items() if c is None and not choices["meanings"].get(k)
        ]
        empty = [v for v, (c, _, _) in shown.items() if c == ""]
        if not target or not target.isprintable():
            blocked = "Name the new column."
        elif target == column:
            blocked = "The clean values go into a new column: the original is kept."
        elif target in df.columns and not replaces:
            blocked = f"{as_code(target)} is already a column: choose another name."
        elif replaces and (why := _replaceable(df, steps, column, target)):
            blocked = why
        elif empty:
            blocked = f"Give {as_code(show_level(empty[0]))} a clean value, or tick Missing."
        elif unmeant:
            blocked = (
                f"Say what missing means for {as_code(show_level(unmeant[0]))}: its Means cell in the table, or "
                "pick it with Values, choose what missing means, then Set to missing."
            )
        elif missing_reason := [c for c in grouped if not choices["reasons"].get(c)]:
            blocked = f"Say why the values grouped as {as_code(show_level(missing_reason[0]))} are grouped."
        elif not full:
            blocked = "Nothing to change: every value keeps how it is written."
        if blocked:
            st.caption(blocked)
        if state.pending_steps():
            st.caption(
                "An Enrich draft is open: this changes the data under it, and the draft will ask to be redone."
            )
        label = (
            f"Write {plain(target)} again ({plain(column)} kept)"
            if replaces
            else f"Add {plain(target)} ({plain(column)} kept)"
        )
        if st.button(label, key=f"clean_apply::{column}", type="primary", disabled=bool(blocked)):
            _apply(state, df, column, target, counts, choices, grouped, replaces=replaces)
        if done:
            st.success(done, icon=":material/check:")


def _apply(
    state,
    df: pd.DataFrame,
    column: str,
    target: str,
    counts: dict[str, int],
    choices: dict[str, Any],
    grouped: dict[str, list[str]],
    *,
    replaces: bool,
) -> None:
    mapping, kept = _recorded(counts, choices)
    kinds = {k: choices["kinds"].get(k, CHOSEN) for k in mapping}
    for key, clean in mapping.items():
        if clean in grouped:
            kinds[key] = GROUPING
    meanings = {k: choices["meanings"][k] for k, c in mapping.items() if c is None}
    new, kept_as_written = recode_column(df, column, target, mapping, kept=kept, replaces=replaces)
    full = plan(counts, mapping, kept=kept)
    after = {full.get(v, v) for v in counts} - {None}
    gone = sum(counts[v] for v, c in full.items() if c is None)
    params = {
        "source": column,
        "target": target,
        "fold": FOLD,
        "mapping": mapping,
        "kinds": kinds,
        "kept": kept,
        "missing_meaning": meanings,
        "groupings": {c: choices["reasons"][c] for c in grouped},
        "replaces": replaces,
        "values_before": len(counts),
        "values_after": len(after),
        "kept_as_written": kept_as_written,
        "rows_changed": sum(counts[v] for v in full),
        "rows_set_to_missing": gone,
        "changes": changes(counts, mapping, kinds, kept=kept, meanings=meanings),
        "target_fp": frame_fingerprint(new[[target]]),
    }
    summary = f"{len(after)} values instead of {len(counts)}" + (
        f", {_rows(gone)} set to missing" if gone else ""
    )
    wrote = state.commit_change(
        new, "recode_levels", params, f"Recoded {column} into {target}: {summary}"
    )
    st.session_state.setdefault(STORE, {}).pop(column, None)  # the trace holds the choices now
    if wrote:
        verb = "Wrote" if replaces else "Added"
        st.session_state[DONE] = f"{verb} {as_code(target)} beside {as_code(column)}: {summary}."
    else:
        st.session_state[DONE] = f"Nothing written: {as_code(target)} already holds these values."
    st.rerun()


def _choice_label(c: Candidate) -> str:
    if c.cleaned_into:
        return f"{c.column} · cleaned into {c.cleaned_into}"
    if c.to_check:
        return f"{c.column} · {c.values} values, {c.to_check} to check"
    return f"{c.column} · {c.values} values"


def render_categories() -> None:
    """The Categories tab: a variable and the new column's name on one row, then its editor."""
    state = get_state()
    df = state.base
    if df is None:
        return
    done = st.session_state.pop(DONE, None)
    steps = state.transformation_manager.history
    found = state.session.memo(
        "clean_candidates", None, lambda: candidates(df, recoded_columns(steps))
    )
    if not found:
        st.caption(f"No text column holds 2 to {MAX_VALUES} values: nothing to clean here.")
        return
    about = {c.column: c for c in found}
    options = list(about)
    chosen = st.session_state.get(COLUMN)
    pick, header = st.columns([2, 1], vertical_alignment="bottom")
    with pick:
        column = st.selectbox(
            "Variable",
            options,
            index=options.index(chosen) if chosen in options else 0,
            format_func=lambda c: _choice_label(about[c]),
            # a new widget after each change of the data, so that its label says what Apply did (the browser keeps
            # the text of a widget that did not change); the choice itself is kept in COLUMN
            key=f"clean_variable::{state.session.data_fp}",
        )
    st.session_state[COLUMN] = column
    if column is not None:
        category_editor(state, df, column, header=header, done=done)


__all__ = ["category_editor", "render_categories"]
