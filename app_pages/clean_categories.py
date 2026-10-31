"""
Clean > Categories: the values of a categorical variable as written, each given a clean value or set to missing, and
written into a new column beside the original, which is kept (enrich.recode). A value written with extra spaces takes
its spelling with the spaces collapsed (the fold rule, recorded as a rule); a case variant, a one-letter code, a variant
(accents, British and American spellings), a spelling one letter apart, a code that may mean missing and what the AI
engine proposes are offered, never applied. A value set to missing says what missing means there; a grouping of
categories says why. The page shows each value's own rows, on all rows, never an outcome beside them: merging values
after seeing results is a choice the intervals ignore. Apply is one traced step, ``recode_levels``, which the replay
redoes; what it records always gives the preview (enrich.recode.recorded).

The choices live in one session key, ``clean_manual`` (ui.keys: cleared when another dataset is loaded), per column,
changed only through enrich.recode (give, tick, untick, unset_missing); the widgets that edit them carry a short id of
the dataset's trace and fold what they change into it as they change, so that no choice is lost on a page change and
none follows the user to another dataset. The table is drawn from the store: it reads each edit once, and starts
again from the store whenever the store and what the browser shows of its edits differ.
"""

from __future__ import annotations

import hashlib
import html
from collections import defaultdict
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.fingerprint import frame_fingerprint
from core.registry import as_code
from enrich.recode import (
    AI,
    CAPITALS,
    CHOSEN,
    CLOSE,
    FIRST_LETTER,
    FOLD,
    GROUPING,
    MAX_VALUES,
    MEANINGS,
    MISSING_CODE,
    OFFER_KINDS,
    RESULT,
    SPACES,
    SPACES_AND_CAPITALS,
    SPELLING,
    VARIANT,
    Candidate,
    Offer,
    ai_offers,
    candidates,
    changes,
    choices_of,
    free_text,
    give,
    groupings,
    last_record,
    level_counts,
    new_choices,
    numeric_codes,
    outcome,
    plan,
    recode_column,
    recoded_columns,
    recorded,
    replaceable,
    resolve,
    rule_of,
    sort_key,
    suggest,
    swaps,
    tick,
    ticked,
    unset_missing,
    untick,
    values_fp,
)
from manage.ai_payload import category_values, host_of, on_this_machine, provider_label, recipient
from ui.components.levels import show_level
from ui.components.text import plain
from ui.quality_charts import chart_config
from ui.services import engine_model, running_engine
from ui.state import get_state
from ui.theme import OKABE_ITO

STORE = "clean_manual"  # {column: choices}, declared in ui.keys (lifetime "dataset")
AI_STORE = "clean_ai"  # {column: what the AI engine was sent and proposed}, declared in ui.keys (lifetime "dataset")
DONE = "clean_done"  # what the last Apply did, said once after the rerun
COLUMN = "clean_column"  # the variable open in the editor, declared in ui.keys
_BAR = OKABE_ITO[0]
_MISSING_BAR = "rgba(128, 140, 145, 0.55)"
WRITTEN, ROWS, CLEAN, MISSING, WHY = "Value as written", "Rows", "Clean value", "Missing", "Why"
# the Missing column: not missing, missing and what it means, missing until the user says what it means, a result
NOT_MISSING = "—"
ASK = "missing: say what it means"
A_TO_Z, MOST_ROWS = "A-Z", "Most rows"
# what each kind of choice reads on the page (the stored ids never change)
WORDS = {
    CAPITALS: "capitals",
    SPACES: "spaces",
    SPACES_AND_CAPITALS: "spaces and capitals",
    FIRST_LETTER: "first-letter code",
    VARIANT: "spelling variant",
    SPELLING: "one letter apart",
    CLOSE: "one letter apart",
    MISSING_CODE: "code for missing",
    CHOSEN: "chosen",
    GROUPING: "grouping",
    AI: "AI proposal",
}
# the offers, in groups; "Tick all" only where each offer is the same value written another way, never where two
# values may be two categories (one letter apart, the AI engine's), each of which the user checks one by one
GROUPS = (
    ("written", "Written another way", (CAPITALS, VARIANT, FIRST_LETTER), True),
    ("letter", "One letter apart: check that both mean the same", (SPELLING, CLOSE), False),
    ("missing", "Codes that may mean missing", (MISSING_CODE,), True),
    ("ai", "Proposed by the AI engine: check that both name the same category", (AI,), False),
)
# a group's title already says why: its pills do not repeat it
SAID = frozenset({SPELLING, CLOSE, MISSING_CODE})
_UNSEEN = object()  # a cell of the table no earlier edit set


def _rows(n: int) -> str:
    return f"{n:,} row{'s' if n != 1 else ''}"


def _scope(state) -> str:
    """A short id of the dataset's trace: the widget keys carry it, so a widget's state never meets another dataset."""
    session_id = str(state.transformation_manager.session_id or "")
    return hashlib.sha256(session_id.encode()).hexdigest()[:8]


def _choices(column: str) -> dict[str, Any]:
    return st.session_state.setdefault(STORE, {}).setdefault(
        column, {**new_choices(), "revision": 0, "shown": {}}
    )


def _keys(scope: str, column: str, revision: int, order: str) -> dict[str, str]:
    base = f"{scope}::{column}"
    return {
        "target": f"clean_target::{base}",
        "pick": f"clean_pick::{base}",
        "into": f"clean_into::{base}",
        "means": f"clean_means::{base}",
        "view": f"clean_view::{base}",
        "order": f"clean_order::{base}",
        # starts again from the store when a pill, a button or the order changes it (not on the table's own edits,
        # which would lose its scroll position and focus); the rows are matched to values by position: the order
        # enters the key
        "editor": f"clean_editor::{base}::{revision}::{order}",
    }


def _again(column: str) -> None:
    _choices(column)["revision"] += 1


def _cell(value: Any) -> Any:
    """A cell as the browser sends it: a cleared one comes back None or NaN."""
    return None if value is None or (isinstance(value, float) and pd.isna(value)) else value


def _edits(key: str) -> dict[int, dict[str, Any]]:
    return {
        int(row): {name: _cell(v) for name, v in cells.items()}
        for row, cells in ((st.session_state.get(key) or {}).get("edited_rows") or {}).items()
    }


def _table_changed(column: str, key: str, values: list[str], counts: dict[str, int]) -> None:
    """
    The table's last edit into the store. The browser sends every edit made since the table was drawn, by row: only
    the cells that changed since the last call are read, so that an earlier edit is never read again over a later one
    (which, read in row order against a store the later one already changed, could undo it).
    """
    choices = _choices(column)
    shown = choices.get("shown") or {}
    edits = _edits(key)
    seen = choices.get("seen") or {}
    before = seen.get("rows", {}) if seen.get("key") == key else {}
    choices["seen"] = {"key": key, "rows": edits}
    for row, cells in edits.items():
        changed = {
            name: v for name, v in cells.items() if before.get(row, {}).get(name, _UNSEEN) != v
        }
        if not changed or row >= len(values):
            continue
        value = values[row]
        clean, cell = shown.get(value, (value, NOT_MISSING))
        typed = changed.get(CLEAN, clean)
        clean = "" if typed is None else str(typed).strip()
        cell = changed.get(MISSING, cell)
        if MISSING in changed and cell in MEANINGS:
            give(choices, counts, value, None, meaning=cell)  # set to missing, and what it means
        elif MISSING in changed and cell == ASK:
            give(choices, counts, value, None)  # missing: Apply waits for what it means
        elif MISSING in changed and cell == RESULT:
            give(
                choices, counts, value, value
            )  # a result (ND: not detected): kept as a value, as written
        elif CLEAN in changed and clean:
            give(choices, counts, value, clean)  # a clean value typed: a value
        elif MISSING in changed and cell == NOT_MISSING:
            unset_missing(choices, counts, value)
        elif CLEAN in changed:
            give(choices, counts, value, clean)  # cleared: Apply waits for a clean value


def _ticked(column: str, key: str, offers: dict[str, Offer], counts: dict[str, int]) -> None:
    """A group's pills: a ticked offer fills the table (its group's choice); unticked, its values take their spelling
    again."""
    choices = _choices(column)
    picked = set(st.session_state.get(key) or [])
    for value, offer in offers.items():
        if value in picked and not ticked(choices, counts, offer):
            tick(choices, counts, offer, by=_proposed_by(column) if offer.why == AI else None)
        elif value not in picked and ticked(choices, counts, offer):
            untick(choices, counts, offer)
    _again(column)


def _tick_all(column: str, offers: dict[str, Offer], counts: dict[str, int]) -> None:
    choices = _choices(column)
    for offer in offers.values():
        if not ticked(choices, counts, offer):
            tick(choices, counts, offer)
    _again(column)


def _buttons(column: str, keys: dict[str, str], counts: dict[str, int], how: str) -> None:
    """Map, Set to missing and Keep as written: the picked values given a clean value; the picks and the meaning are
    cleared after use."""
    choices = _choices(column)
    into = st.session_state.get(keys["into"])
    means = st.session_state.get(keys["means"])
    for value in st.session_state.get(keys["pick"]) or []:
        if how == "missing":
            give(choices, counts, value, None, meaning=means)
        elif how == "keep":
            give(choices, counts, value, value)
        elif into is not None:
            give(choices, counts, value, str(into).strip())
    st.session_state[keys["pick"]] = []
    st.session_state[keys["means"]] = None
    _again(column)


def _shown(
    counts: dict[str, int],
    spellings: dict[str, str],
    reasons: dict[str, str],
    choices: dict[str, Any],
):
    """Each value as written: its clean value (None: missing), what missing means there, and why."""
    mapping, kinds, kept, meanings = (
        choices["mapping"],
        choices["kinds"],
        choices["kept"],
        choices["meanings"],
    )
    shown: dict[str, tuple[str | None, str | None, str]] = {}
    for value in counts:
        spelling = spellings.get(value, value)
        clean, first, last = resolve(value, spellings, mapping, kept)
        meaning = meanings.get(last) if clean is None and last is not None else None
        if value in kept:
            why = "kept as written"
        elif first is not None and first == value:
            why = WORDS.get(kinds.get(first, CHOSEN), kinds.get(first, CHOSEN))
        elif value != spelling:  # folded: why, or the spelling whose choice it follows
            folded = WORDS.get(reasons.get(value, SPACES), SPACES)
            why = folded if clean == spelling else f"follows {show_level(spelling)}"
        elif first is not None:
            why = WORDS.get(kinds.get(first, CHOSEN), kinds.get(first, CHOSEN))
        else:
            why = ""
        shown[value] = (clean, meaning, why)
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
        height=min(60 + 22 * len(labels), 320),  # capped: Apply stays in view above it
        margin={"l": 10, "r": 10, "t": 10, "b": 30},
        bargap=0.35,
        showlegend=False,
        xaxis={"title": "Rows", "showgrid": True},
        yaxis={"autorange": "reversed", "automargin": True},
    )
    return fig


def _offer_groups(
    column: str,
    scope: str,
    choices: dict[str, Any],
    counts: dict[str, int],
    offers: tuple[Offer, ...],
    into: str,
) -> None:
    """The offers in their groups, each with "Tick all"; ticking fills the table, never the data."""
    held: dict[str, int] = defaultdict(int)
    spellings = suggest(counts, near=False).spellings
    for value, n in counts.items():
        held[spellings.get(value, value)] += n
    many = len(offers) > 8
    # many offers: a box that scrolls, so that the table, which each offer fills, stays near
    area = st.container(height=260, border=False) if many else st.container()
    for name, title, kinds, all_at_once in GROUPS:
        group = {o.value: o for o in offers if o.why in kinds}
        if not group:
            continue

        def label(value: str, group: dict[str, Offer] = group) -> str:
            offer = group[value]
            clean = as_code(show_level(offer.clean)) if offer.clean is not None else "missing"
            if offer.why == AI:
                why = f" · {plain(offer.reason, limit=80)}" if offer.reason else ""
            else:
                why = "" if offer.why in SAID else f" · {WORDS[offer.why]}"
            return f"{as_code(show_level(value))} → {clean}{why} · {_rows(held.get(value, 0))}"

        on = [v for v, o in group.items() if ticked(choices, counts, o)]
        # drawn from the store (default=), never kept: its key follows the store's revision and what it shows ticked,
        # so that an edit of the table (which leaves the revision) can never leave a pill stale, to be undone by the
        # next click on another
        drawn = hashlib.sha256("\x1f".join(on).encode()).hexdigest()[:8]
        key = f"clean_offers::{scope}::{column}::{name}::{choices['revision']}::{drawn}"
        with area:
            st.pills(  # the whole width: the pills wrap
                title,
                list(group),
                selection_mode="multi",
                default=on,
                format_func=label,
                key=f"clean_offers::{scope}::{column}::{name}::{choices['revision']}::{drawn}",
                on_change=_ticked,
                args=(column, key, group, counts),
            )
            if all_at_once and len(group) > 1:
                left = len(group) - len(on)
                st.button(
                    f"Tick all {len(group)}"
                    if not on
                    else f"Tick the other {left}"
                    if left
                    else "All ticked",
                    key=f"clean_tick_all::{scope}::{column}::{name}",
                    type="tertiary",
                    icon=":material/done_all:",
                    on_click=_tick_all,
                    args=(column, group, counts),
                    disabled=not left,
                )
    st.caption(
        f"Ticking fills the table; the data change when you {into}. A one-letter code, a variant, a spelling one "
        "letter apart or a code for unknown can name a category of its own: check what it means in the source first."
    )


def category_editor(
    state, df: pd.DataFrame, column: str, *, header: Any = None, done: str | None = None
) -> None:
    """The editor of one variable's values: the offers, what each value becomes, a preview, and Apply. ``header``:
    the container beside the variable's choice that holds the new column's name."""
    steps = state.transformation_manager.history
    scope = _scope(state)
    counts = level_counts(df[column])
    hints = suggest(counts)
    spellings, reasons = hints.spellings, hints.reasons
    cleaned_into = recoded_columns(steps).get(column, "")
    store = st.session_state.setdefault(STORE, {})
    if column not in store:
        target = cleaned_into if cleaned_into in df.columns else f"{column}_clean"
        store[column] = {
            **choices_of(last_record(steps, column, target)),
            "revision": 0,
            "shown": {},
        }
    choices = store[column]
    order = str(st.session_state.get(f"clean_order::{scope}::{column}") or A_TO_Z)
    keys = _keys(scope, column, choices["revision"], order)

    with header if header is not None else st.container():
        target = st.text_input(
            "New column",
            value=cleaned_into if cleaned_into in df.columns else f"{column}_clean",
            key=keys["target"],
            persist_state="page",
            help="The clean values go into this column, right after the original, which is kept.",
        ).strip()
    replaces = bool(target) and target in df.columns and target == cleaned_into
    name = target or f"{column}_clean"  # drawn through as_code, which escapes it
    proposed = _ai_proposals(column, counts, hints.offers)
    with st.container(horizontal=True, vertical_alignment="center"):
        if hints.offers or proposed:
            st.markdown("**Check before changing**")
        _ask_ai(column, scope, counts)
    if hints.offers or proposed:
        into = f"write {as_code(name)} again" if replaces else f"add {as_code(name)}"
        _offer_groups(column, scope, choices, counts, (*hints.offers, *proposed), into)

    shown = _shown(counts, spellings, reasons, choices)
    full = plan(
        counts, {k: v for k, v in choices["mapping"].items() if v != ""}, kept=choices["kept"]
    )
    table, preview = st.columns([3, 2], gap="medium")

    with table:
        st.segmented_control(
            "Order",
            [A_TO_Z, MOST_ROWS],
            default=A_TO_Z,
            key=keys["order"],
            persist_state="page",
            on_change=_again,
            args=(column,),
            help="A-Z sets case, spaces and accents aside, so that a value's spellings sit together.",
        )
        values = sorted(counts, key=sort_key) if order == A_TO_Z else list(counts)
        clean_values = sorted({c for c, _, _ in shown.values() if c}, key=sort_key)
        with st.container(horizontal=True, vertical_alignment="bottom"):
            picked = st.multiselect(
                "Values",
                sorted(counts, key=sort_key),
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
                placeholder="Choose or type",
            )
            n = len(picked)
            st.button(
                f"Map {n} value{'s' if n != 1 else ''} to {plain(str(into))}"
                if n and into
                else "Map",
                key=f"clean_map::{scope}::{column}",
                on_click=_buttons,
                args=(column, keys, counts, "map"),
                disabled=not (n and into),
            )
        with st.container(horizontal=True, vertical_alignment="bottom"):
            means = st.selectbox(
                "Missing means",
                MEANINGS,
                index=None,
                key=keys["means"],
                persist_state="page",
                placeholder="What missing means",
            )
            st.button(
                "Set to missing",
                key=f"clean_missing::{scope}::{column}",
                on_click=_buttons,
                args=(column, keys, counts, "missing"),
                disabled=not (n and means),
                help="Choose the values and what missing means for them first.",
            )
            st.button(
                "Keep as written",
                key=f"clean_keep::{scope}::{column}",
                on_click=_buttons,
                args=(column, keys, counts, "keep"),
                disabled=not n,
                help="Undo the fold or a choice: the values stay as they are written.",
            )
        choices["shown"] = {
            v: ("" if c is None else c, (m or ASK) if c is None else NOT_MISSING)
            for v, (c, m, _) in shown.items()
        }
        frame = pd.DataFrame(
            {
                WRITTEN: [show_level(v) for v in values],
                ROWS: [counts[v] for v in values],
                CLEAN: [choices["shown"][v][0] for v in values],
                MISSING: [choices["shown"][v][1] for v in values],
                WHY: [shown[v][2] for v in values],
            }
        )
        # the browser paints the edits it made since the table was drawn over the values sent: where one no longer
        # matches the store (a choice that changed other rows, a result kept as written), the table starts again
        if any(
            row >= len(values) or str(frame.at[row, name] or "") != str(v or "")
            for row, cells in _edits(keys["editor"]).items()
            for name, v in cells.items()
            if name in (CLEAN, MISSING)
        ):
            _again(column)
            keys = _keys(scope, column, choices["revision"], order)
        st.data_editor(
            frame,
            key=keys["editor"],
            on_change=_table_changed,
            args=(column, keys["editor"], values, counts),
            hide_index=True,
            width="stretch",
            disabled=[WRITTEN, ROWS, WHY],
            column_config={  # widths so that Why, the last, keeps the room it needs
                WRITTEN: st.column_config.TextColumn(width=115, help="A hidden space shows as ␣."),
                ROWS: st.column_config.NumberColumn(width=45, format="%d"),
                CLEAN: st.column_config.TextColumn(
                    width=100, help="The same clean value in several rows merges them."
                ),
                WHY: st.column_config.TextColumn(width=120),
                MISSING: st.column_config.SelectboxColumn(
                    options=[NOT_MISSING, *MEANINGS, ASK, RESULT],
                    width=135,
                    help="Set a value to missing by saying what missing means there; a code such as ND (not "
                    "detected) can be a result, kept as a value.",
                    required=True,
                ),
            },
        )
        if any(show_level(v) != v for v in values):
            st.caption("␣ marks a hidden space.")

    with preview:
        try:
            record = recorded(counts, choices)
        except ValueError:  # never: the record would not give the preview
            record = None
        grouped = (
            groupings(counts, record[0], record[1], kept=record[2]) if record is not None else {}
        )
        for clean in grouped:  # the reasons as typed in this run: the inputs come after Apply
            typed = st.session_state.get(f"clean_reason::{scope}::{column}::{clean}")
            if typed is not None:
                choices["reasons"][clean] = str(typed).strip()[:200]
        blocked = _blocked(
            df, steps, column, target, cleaned_into, counts, choices, shown, full, grouped, record
        )
        with st.container(horizontal=True, vertical_alignment="center"):
            view = st.segmented_control(
                "Show",
                ["After", "Before"],
                default="After",
                key=keys["view"],
                persist_state="page",
                label_visibility="collapsed",
            )
            label = (
                f"Write {plain(target)} again ({plain(column)} kept)"
                if replaces
                else f"Add {plain(target)} ({plain(column)} kept)"
            )
            applied = st.button(
                label, key=f"clean_apply::{column}", type="primary", disabled=bool(blocked)
            )
        if blocked:
            st.caption(blocked)
        if done:
            st.success(done, icon=":material/check:")
        if state.pending_steps():
            st.caption(
                "An Enrich draft is open: this changes the data under it, and the draft will ask to be redone."
            )
        gone = sum(counts[v] for v, c in full.items() if c is None)
        if gone:
            them = "it" if gone == 1 else "them"
            st.caption(
                f"{_rows(gone)} set to missing: complete-case analyses using {as_code(name)} leave {them} out, and a "
                f"multiple imputation fills {them} as missing at random given its variables. A value that means not "
                "applicable or declined is often better kept as a value."
            )
        for clean, parts in grouped.items():
            listed = [as_code(show_level(p)) for p in parts]
            reason = st.text_input(
                f"Why group {', '.join(listed[:-1])} and {listed[-1]} as {as_code(show_level(clean))}?",
                value=choices["reasons"].get(clean, ""),
                key=f"clean_reason::{scope}::{column}::{clean}",
                max_chars=200,
                help="Merging categories is a grouping, not a correction: the methods say why.",
            ).strip()[:200]  # the browser's limit, enforced here too
            if reason:
                choices["reasons"][clean] = reason
            else:
                choices["reasons"].pop(clean, None)
        st.plotly_chart(
            _figure(counts, full, after=view != "Before"),
            config=chart_config(),
            width="stretch",
            key=f"clean_chart::{scope}::{column}",
        )
        if applied and not blocked and record is not None:
            _apply(state, df, column, target, counts, choices, record, grouped, replaces=replaces)


def _blocked(
    df, steps, column, target, cleaned_into, counts, choices, shown, full, grouped, record
) -> str:
    """Why Apply waits ("" when it does not)."""
    replaces = bool(target) and target in df.columns and target == cleaned_into
    unmeant = [v for v, (c, m, _) in shown.items() if c is None and not m]
    empty = [v for v, (c, _, _) in shown.items() if c == ""]
    if not target or not target.isprintable():
        return "Name the new column."
    if target == column:
        return "The clean values go into a new column: the original is kept."
    if target in df.columns and not replaces:
        return f"{as_code(target)} is already a column: choose another name."
    if replaces and not replaceable(df, steps, column, target):
        return f"{as_code(target)} was changed by a later step: write the clean values into another column."
    if empty:
        return f"Give {as_code(show_level(empty[0]))} a clean value, or set it to missing in the Missing column."
    if unmeant:
        return f"Say what missing means for {as_code(show_level(unmeant[0]))} in the table's Missing column."
    if swapped := swaps(counts, choices):
        named = " and ".join(as_code(show_level(v)) for v in swapped[:2])
        return (
            f"{named} take each other's name: the copy would swap them, not merge them. Give both one clean value, or "
            "keep one as written."
        )
    if record is None:
        return "These choices cannot be recorded as shown: undo the last one."
    if waiting := [c for c in grouped if not choices["reasons"].get(c)]:
        return f"Say below why {as_code(show_level(waiting[0]))} groups several categories."
    if not full:
        return "Nothing to change: every value keeps how it is written."
    last = last_record(steps, column, target) if replaces else None
    if last is not None and last.get("target_fp") == frame_fingerprint(df[[target]]):
        # the copy is what that recode wrote, with its rule: the same outcome would write it again
        then = choices_of(last)
        before = outcome(
            counts, then["mapping"], then["meanings"], kept=then["kept"], fold_rule=rule_of(last)
        )
        if before == outcome(counts, record[0], record[3], kept=record[2]):
            return f"Nothing changed since {as_code(target)} was written."
    return ""


def _apply(
    state,
    df: pd.DataFrame,
    column: str,
    target: str,
    counts: dict[str, int],
    choices: dict[str, Any],
    record: tuple,
    grouped: dict[str, list[str]],
    *,
    replaces: bool,
) -> None:
    mapping, kinds, kept, meanings = record
    for (
        key,
        clean,
    ) in (
        mapping.items()
    ):  # a hand-made merge into a grouping is one; an AI proposal keeps who proposed
        if clean in grouped and kinds[key] not in OFFER_KINDS and kinds[key] != AI:
            kinds[key] = GROUPING
    spellings = suggest(counts, near=False).spellings
    held: dict[str, int] = defaultdict(int)
    for value, n in counts.items():
        held[spellings.get(value, value)] += n
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
        "groupings": {
            clean: {
                "reason": choices["reasons"][clean][:200],
                "values": parts,
                "each_5_rows": all(
                    held.get(p, 0) >= 5 for p in parts
                ),  # the report names them only then
            }
            for clean, parts in grouped.items()
        },
        "replaces": replaces,
        "values_before": len(counts),
        "values_after": len(after),
        "kept_as_written": kept_as_written,
        "rows_changed": sum(counts[v] for v in full),
        "rows_missing_before": int(
            df[column].isna().sum()
        ),  # recorded as missing, apart from those set to missing
        "rows_set_to_missing": gone,
        "changes": changes(counts, mapping, kinds, kept=kept, meanings=meanings),
        "values_fp": values_fp(counts),
        "target_fp": frame_fingerprint(new[[target]]),
    }
    # who proposed each AI choice (provider, model, where it runs), kept when the copy is written again: never the
    # prompt, never the answer
    proposed = {k: choices.get("ai_by", {}).get(k, {}) for k, kind in kinds.items() if kind == AI}
    if proposed:
        params["ai_by"] = proposed
    summary = f"{len(after)} values instead of {len(counts)}" + (
        f", {_rows(gone)} set to missing" if gone else ""
    )
    wrote = state.commit_change(
        new, "recode_levels", params, f"Recoded {column} into {target}: {summary}"
    )
    # the trace holds the choices now: the store starts again from them, its revision moving on (the table, drawn
    # again under a key it had before, would bring back the browser's edits)
    st.session_state.setdefault(STORE, {})[column] = {
        **choices_of(params),
        "revision": choices["revision"] + 1,
        "shown": {},
    }
    if wrote:
        verb = "Wrote" if replaces else "Added"
        st.session_state[DONE] = f"{verb} {as_code(target)} beside {as_code(column)}: {summary}."
    else:
        st.session_state[DONE] = f"Nothing written: {as_code(target)} already holds these values."
    st.rerun()


def _ai_proposals(
    column: str, counts: dict[str, int], offers: tuple[Offer, ...]
) -> tuple[Offer, ...]:
    """What the AI engine proposed for this column's values as they are now (a change of the values asks again)."""
    asked = (st.session_state.get(AI_STORE) or {}).get(column) or {}
    if asked.get("fp") != values_fp(counts):
        return ()
    taken = {o.value for o in offers}
    return ai_offers(
        asked.get("proposals") or (), counts, sent=asked.get("sent") or (), taken=taken
    )


def _ask_ai(column: str, scope: str, counts: dict[str, int]) -> None:
    """
    "Ask the AI engine": a preview of exactly what is sent (the variable's name and its values, no count, no row;
    to any recipient but an Ollama that runs its model on this machine, only the values held by 5 rows or more), then
    Send, which closes it; what came back is said under the heading. Never for free text.
    """
    if free_text(counts):
        return
    engine = running_engine()
    popover = f"clean_ai_open::{scope}::{column}"
    with st.popover(
        "Ask the AI engine",
        type="tertiary",
        icon=":material/auto_awesome:",
        disabled=engine is None,
        help="Start the AI engine in Settings > AI engine first." if engine is None else None,
        key=popover,
        on_change="rerun",  # its preview is made only while it is open; Send closes it
    ) as box:
        if engine is not None and box.open:
            _preview(engine, column, scope, counts, popover)
    asked = (st.session_state.get(AI_STORE) or {}).get(column) or {}
    if asked.get("fp") == values_fp(counts):  # what came back, once
        if asked.get("error"):
            st.caption(f"The AI engine could not answer: {plain(asked['error'], limit=200)}")
        else:
            n = len(_ai_proposals(column, counts, ()))
            st.caption(
                f"The AI engine proposed {n} correspondence{'s' if n != 1 else ''}: under Proposed by the AI engine."
                if n
                else "The AI engine proposed no correspondence the rules accept for these values."
            )


def _preview(engine, column: str, scope: str, counts: dict[str, int], popover: str) -> None:
    """Exactly what Send gives the engine, then Send."""
    model = engine_model(engine)
    sent, held = category_values(counts, engine.provider, engine.base_url, model)
    to, _ = _receiver(engine, model)
    rare = (
        f" {held:,} value{'s' if held != 1 else ''} held by fewer than 5 rows {'are' if held != 1 else 'is'} not sent."
        if held
        else ""
    )
    if not sent:
        st.caption(f"No value is held by 5 rows or more: nothing can be sent to {plain(to)}.")
        return
    st.caption(
        f"Send to {plain(to)}: the name {as_code(column)} and these {len(sent):,} values, as written, A to Z; no "
        f"count, no row.{rare}"
    )
    st.dataframe(  # every value sent, as it is sent: drawn as text, never as markup
        pd.DataFrame({"Values sent": [" ".join(v.split())[:80] for v in sent]}),
        hide_index=True,
        height=min(38 + 35 * len(sent), 260),
    )
    st.button(
        "Send",
        key=f"clean_ai_send::{scope}::{column}",
        type="primary",
        on_click=_send,
        args=(column, counts, sent, held, popover),
    )


def _send(column: str, counts: dict[str, int], sent: list[str], held: int, popover: str) -> None:
    """
    Send, as the click's callback: the previewed values, to the engine that runs now, once the rule run again on it
    gives the same values (an engine changed since the preview sends nothing); its answer into the store, said under
    the heading, and the popover closed.
    """
    engine = running_engine()
    st.session_state[popover] = False
    if engine is None:
        return
    model = engine_model(engine)
    if category_values(counts, engine.provider, engine.base_url, model) != (sent, held):
        error, proposals = (
            "the AI engine changed since the preview: open it again to see what is sent",
            [],
        )
    else:
        with st.spinner("The AI engine reads the values..."):
            proposals, error = engine.suggest_category_groups(column, sent)
    st.session_state.setdefault(AI_STORE, {})[column] = {
        "fp": values_fp(counts),
        "sent": sent,
        "proposals": [
            {k: str(p.get(k, ""))[:120] for k in ("value", "clean", "reason")} for p in proposals
        ],
        "error": error,
        "provider": str(engine.provider or ""),
        "model": model,
        "where": _receiver(engine, model)[1],
        "values_sent": len(sent),
        "values_held": held,
    }


def _receiver(engine, model: str) -> tuple[str, str]:
    """Who receives a request, and where it runs: the chat recipient only (the embedding provider receives none)."""
    found = recipient(engine.provider, engine.base_url, model)
    if found:
        return provider_label(found), "cloud"
    if on_this_machine(engine.base_url):
        return "the AI engine on this machine", "this machine"
    return f"the AI engine at {host_of(engine.base_url)}, on your network", "your network"


def _proposed_by(column: str) -> dict[str, Any]:
    """Who proposed (provider, model, where it runs) and how many values it was sent and not sent: kept with each AI
    choice, recorded with the step."""
    asked = (st.session_state.get(AI_STORE) or {}).get(column) or {}
    return {
        **{k: str(asked.get(k, "")) for k in ("provider", "model", "where")},
        **{k: int(asked.get(k) or 0) for k in ("values_sent", "values_held")},
    }


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
    codes = numeric_codes(df)
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
    if codes:
        shown = ", ".join(as_code(c) for c in codes[:5]) + (" ..." if len(codes) > 5 else "")
        st.caption(f"Columns of numbers are not listed, codes such as 1 and 2 included ({shown}).")
    if column is not None:
        category_editor(state, df, column, header=header, done=done)


__all__ = ["category_editor", "render_categories"]
