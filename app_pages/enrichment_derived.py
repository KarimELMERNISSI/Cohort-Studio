"""
Data > Enrich > Handle Missing Data > Derived variables: the variables computed from others in the same row (a BMI from
the weight and the height, a total from its items), declared and checked against the data (core.derived,
enrich.derived). Optional: with no definition nothing changes, and a definition is used only when the user fills with
it or ticks it in an imputation tab.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import streamlit as st

from core.derived import KINDS, DerivedVariable, cycles
from core.registry import as_code
from enrich.derived import AGREES, CANNOT, DISAGREES, MOSTLY, TOO_FEW, DerivedCheck
from enrich.formulas import evaluate_formula_safely, read_formula
from manage.ai_payload import column_kinds, typical_ranges
from ui.components import help_popover
from ui.components.text import plain
from ui.state import get_state
from utils.data_analyzer import DataAnalyzer

_BADGES = {
    AGREES: (":material/check_circle:", "green"),
    MOSTLY: (":material/rule:", "orange"),
    DISAGREES: (":material/cancel:", "red"),
    TOO_FEW: (":material/help:", "gray"),
    CANNOT: (":material/block:", "red"),
}
_ORIGINS = {"user": "you", "ai": "the AI engine", "import": "an imported file"}
_KIND_LABELS = {
    "": "Not said",
    "conversion": "Unit conversion",
    "ratio": "Ratio or product (a BMI)",
    "total": "Total of items",
    "index": "Index or score",
    "equation": "Clinical equation (an eGFR)",
    "class": "Class of a variable",
}
# the one form that adds and edits a definition
_TARGET, _FORMULA, _KIND, _NOTE = (
    "derived_form_target",
    "derived_form_formula",
    "derived_form_kind",
    "derived_form_note",
)
_EDITING = "derived_editing"  # the variable whose definition the form edits
_NOTICE, _CHANGED = "_derived_notice", "_derived_changed"
_PROPOSALS, _RANGES = "derived_proposals", "derived_send_ranges"
_ORDER = {AGREES: 0, MOSTLY: 1, TOO_FEW: 2, DISAGREES: 3, CANNOT: 4}  # the best evidence first


def _badge(check: DerivedCheck) -> None:
    icon, colour = _BADGES[check.verdict]
    st.badge(check.label.capitalize(), icon=icon, color=colour)


def _check_lines(check: DerivedCheck) -> None:
    """The check in words: its evidence, why, and what it could not test."""
    with st.container(horizontal=True, vertical_alignment="center"):
        _badge(check)
        st.caption(plain(check.evidence))
    if check.reason and check.verdict != CANNOT and check.reason != check.evidence:
        st.caption(plain(check.reason))
    for warning in check.warnings:
        st.caption(f":material/info: {plain(warning)}")


def _inputs(definitions: list[DerivedVariable]) -> dict[str, tuple[str, ...]]:
    return {d.target: read_formula(d.formula).names for d in definitions}


def _circle(definition: DerivedVariable, definitions: tuple[DerivedVariable, ...]) -> str | None:
    """Why ``definition`` cannot join the others: it and another would compute each other."""
    others = [d for d in definitions if d.target != definition.target]
    found = cycles([*others, definition], _inputs([*others, definition]))
    if not found:
        return None
    other = next((n for n in found[0] if n != definition.target), found[0][0])
    return (
        f"{definition.target} and {other} would compute each other: add {definition.target} as an inverse of "
        f"{other} instead."
    )


# --- changes, as callbacks: they read the current list, and the whole page reruns with the new one -----------------


def _changed(notice: str) -> None:
    st.session_state[_NOTICE] = notice
    st.session_state[_CHANGED] = True


def _current(target: str) -> DerivedVariable | None:
    return next((d for d in get_state().derived_variables if d.target == target), None)


def _put(definition: DerivedVariable) -> None:
    """``definition`` in the list, where its variable's was (at the end for a new one)."""
    state = get_state()
    definitions = list(state.derived_variables)
    places = [i for i, d in enumerate(definitions) if d.target == definition.target]
    if places:
        definitions[places[0]] = definition
    else:
        definitions.append(definition)
    state.set_derived_variables(definitions)


def _clear_form() -> None:
    # the widgets' values are assigned, not removed: the browser then shows the empty form
    st.session_state.update({_TARGET: None, _FORMULA: "", _KIND: "", _NOTE: ""})
    st.session_state.pop(_EDITING, None)


def _save(definition: DerivedVariable) -> None:
    editing = st.session_state.get(_EDITING) == definition.target
    _put(definition)
    _clear_form()
    _changed(f"{definition.target} is {'changed' if editing else 'declared'}.")


def _edit(target: str) -> None:
    definition = _current(target)
    if definition is None:
        return
    st.session_state[_EDITING] = target
    st.session_state[_FORMULA] = definition.formula
    st.session_state[_KIND] = definition.kind
    st.session_state[_NOTE] = definition.note


def _remove(target: str) -> None:
    state = get_state()
    state.set_derived_variables([d for d in state.derived_variables if d.target != target])
    if st.session_state.get(_EDITING) == target:
        _clear_form()
    _changed(f"{target} is no longer declared.")


def _accept(target: str, key: str) -> None:
    definition = _current(target)
    if definition is not None:
        _put(replace(definition, accepted=bool(st.session_state.get(key))))
        _changed(f"{target} is {'used' if st.session_state.get(key) else 'no longer used'}.")


def _add_inverse(target: str, name_key: str, formula_key: str) -> None:
    definition, name = _current(target), st.session_state.get(name_key)
    formula = str(st.session_state.get(formula_key) or "").strip()
    if definition is None or not name or not formula:
        return
    _put(replace(definition, inverses={**definition.inverses, str(name): formula}))
    st.session_state[formula_key] = ""
    _changed(f"{name} is recovered from {target}.")


def _remove_inverse(target: str, name: str) -> None:
    definition = _current(target)
    if definition is not None:
        _put(
            replace(
                definition, inverses={k: v for k, v in definition.inverses.items() if k != name}
            )
        )
        _changed(f"{name} is no longer recovered from {target}.")


# --- the page --------------------------------------------------------------------------------------------------------


def _summary(definition: DerivedVariable, check: DerivedCheck, to_fill: int) -> str:
    """The line of a definition: its variable, its check, what it would fill, whether it is used."""
    parts = [plain(definition.target), check.label]
    if to_fill:
        parts.append(f"{to_fill:,} to fill")
    if not check.usable:
        parts.append(
            "not used" if check.verdict in (DISAGREES, CANNOT) else "not used until you accept it"
        )
    if definition.origin != "user":
        parts.append(f"from {_ORIGINS.get(definition.origin, definition.origin)}")
    return " · ".join(parts)


def _disagreeing(definition: DerivedVariable, check: DerivedCheck, data: pd.DataFrame) -> None:
    rows = check.disagreeing.intersection(data.index)
    with st.expander(f"The {len(rows):,} rows that disagree", icon=":material/table_rows:"):
        shown = data.loc[rows, list(dict.fromkeys([definition.target, *check.inputs]))].copy()
        ok, computed, _ = evaluate_formula_safely(
            definition.formula, "__derived__", shown, return_series=True
        )
        if ok:
            shown.insert(1, "formula gives", pd.Series(computed, index=shown.index))
        st.dataframe(shown, alt=f"Rows where {definition.target} disagrees with its formula")
        st.caption(
            "Shown here only, never saved. They may be entry errors: a criterion on Data > Validate can exclude "
            "them from the analysis population."
        )


def _inverses(definition: DerivedVariable) -> None:
    """Its inverses: each fills a missing variable of the formula from the others (a weight from the BMI)."""
    state = get_state()
    st.markdown("**Inverses**")
    if not definition.inverses:
        st.caption(
            "None. An inverse fills a missing variable of the formula from the others: a weight from the BMI and "
            "the height."
        )
    for name, formula in definition.inverses.items():
        with st.container(horizontal=True, vertical_alignment="center"):
            st.markdown(f"{as_code(name)} = {as_code(formula, limit=300)}")
            st.button(
                "Remove",
                icon=":material/close:",
                key=f"derived_inverse_remove::{definition.target}::{name}",
                on_click=_remove_inverse,
                args=(definition.target, name),
                type="tertiary",
            )
        _check_lines(state.derived_inverse_check(definition, name))


def _new_inverse(definition: DerivedVariable, check: DerivedCheck) -> None:
    candidates = [n for n in check.inputs if n not in definition.inverses]
    if not candidates:
        return
    with st.popover("Add an inverse", icon=":material/undo:"):
        name_key = f"derived_inverse_name::{definition.target}"
        formula_key = f"derived_inverse_formula::{definition.target}"
        st.selectbox("Variable", candidates, key=name_key)
        st.text_input(
            "Formula",
            key=formula_key,
            placeholder="bmi * (height_cm / 100) ** 2",
            help=f"Computes the variable from {definition.target} and the formula's other variables.",
        )
        st.button(
            "Add",
            icon=":material/add:",
            key=f"derived_inverse_add::{definition.target}",
            on_click=_add_inverse,
            args=(definition.target, name_key, formula_key),
        )


def _details(definition: DerivedVariable, check: DerivedCheck, data: pd.DataFrame) -> None:
    target = definition.target
    st.markdown(f"{as_code(target)} = {as_code(definition.formula, limit=300)}")
    _check_lines(check)
    if definition.note:
        st.caption(f"Source: {plain(definition.note, limit=300)}")
    if check.verdict in (MOSTLY, TOO_FEW):
        key = f"derived_accept::{target}"
        st.session_state[key] = (
            definition.accepted
        )  # mirrors the definition, which keeps the acceptance
        st.toggle(
            "Use it all the same",
            key=key,
            on_change=_accept,
            args=(target, key),
            help=(
                "The data mostly agree with it, or have too few rows to check it. Accepted, it fills and imputes "
                "like a definition the data agree with; the acceptance is kept with the definition."
            ),
        )
    if len(check.disagreeing):
        _disagreeing(definition, check, data)
    _inverses(definition)
    with st.container(horizontal=True):
        st.button(
            "Edit",
            icon=":material/edit:",
            key=f"derived_edit::{target}",
            on_click=_edit,
            args=(target,),
        )
        _new_inverse(definition, check)
        st.button(
            "Remove the definition",
            icon=":material/delete:",
            key=f"derived_remove::{target}",
            on_click=_remove,
            args=(target,),
        )


def _form(data: pd.DataFrame, definitions: tuple[DerivedVariable, ...]) -> None:
    """The one form that adds a definition, or edits one (its inverses kept), checked before it is saved."""
    state = get_state()
    editing = st.session_state.get(_EDITING)
    current = next((d for d in definitions if d.target == editing), None)
    label = f"Edit {editing}" if current else "Add a definition"
    with st.expander(
        label, icon=":material/edit:" if current else ":material/add:", expanded=current is not None
    ):
        if current is not None:
            st.caption(f"The definition of {as_code(current.target)}: its inverses are kept.")
            target: str | None = current.target
        else:
            declared = {d.target for d in definitions}
            target = st.selectbox(
                "Variable",
                [c for c in data.columns if c not in declared],
                index=None,
                placeholder="The variable the formula computes",
                key=_TARGET,
            )
        formula = st.text_input(
            "Formula",
            key=_FORMULA,
            placeholder="weight_kg / (height_cm / 100) ** 2",
            help="The language of Create New Variables: column names, + - * / **, np.where, np.sqrt...",
        )
        with st.container(horizontal=True):
            # "not said" first: the default, with no index (the form sets its key)
            kinds = ["", *(k for k in KINDS if k)]
            kind = st.selectbox(
                "Kind", kinds, format_func=lambda k: _KIND_LABELS.get(k, k), key=_KIND
            )
            note = st.text_input("Source", key=_NOTE, placeholder="WHO; weight in kg, height in cm")
        formula = str(formula or "").strip()
        proposed, check, circle = None, None, None
        if not target or not formula:
            st.caption("The formula is checked against your data before it is saved.")
        else:
            same = current is not None and formula == current.formula
            proposed = DerivedVariable(
                str(target),
                formula,
                inverses=dict(current.inverses) if current else {},
                kind=str(kind or ""),
                origin=current.origin if same and current else "user",
                note=str(note or "").strip(),
                tolerance=current.tolerance if current else None,
                accepted=bool(same and current and current.accepted),
            )
            check = state.derived_check(proposed, slot="derived_preview")
            _check_lines(check)
            circle = _circle(proposed, definitions)
            if circle:
                st.warning(plain(circle), icon=":material/sync_problem:")
        with st.container(horizontal=True):
            if proposed is not None and check is not None:
                st.button(
                    "Save" if current else "Add",
                    type="primary",
                    icon=":material/check:" if current else ":material/add:",
                    key="derived_save",
                    disabled=check.verdict == CANNOT or bool(circle),
                    on_click=_save,
                    args=(proposed,),
                )
            if current is not None:
                st.button("Cancel", key="derived_cancel", on_click=_clear_form)


def _engine():
    """The AI engine when it runs (Settings > AI engine), else None."""
    engine = st.session_state.get("rag_manager")
    return engine if engine is not None and getattr(engine, "initialized", False) else None


def _find(data: pd.DataFrame, engine) -> None:
    """One request for the whole dataset; its proposals are kept for this version of the data."""
    columns = [str(c) for c in data.columns]
    taxonomy = getattr(engine, "variable_taxonomy", None) or {}
    units = {
        name: str(entry["unit"])
        for name, entry in taxonomy.items()
        if isinstance(entry, dict) and entry.get("unit") and name in columns
    }
    dates = DataAnalyzer(data).date_cols
    send = st.session_state.get(_RANGES, True)
    ranges = typical_ranges(data, columns, date_columns=dates) if send else {}
    kinds = column_kinds(data, columns, date_columns=dates)
    found, error = engine.suggest_derived_variables(
        columns, column_kinds=kinds, units=units, ranges=ranges
    )
    st.session_state[_PROPOSALS] = {
        "fp": get_state().session.working_fp,
        "found": list(found),
        "error": error,
    }


def _adopt(indexes: list[int]) -> None:
    """The chosen proposals join the list: never one the data disagree with, one per variable, and none that would
    compute another definition's variable from its own."""
    state = get_state()
    stored = st.session_state.get(_PROPOSALS) or {}
    found: list[DerivedVariable] = list(stored.get("found") or [])
    added, replaced, refused, taken = [], [], [], set()
    declared = {d.target for d in state.derived_variables}
    for i in indexes:
        if not 0 <= i < len(found):
            continue
        proposal = found[i]
        check = state.derived_check(proposal, slot=f"derived_proposal::{i}")
        if check.verdict in (DISAGREES, CANNOT):
            refused.append(f"{proposal.target} ({check.label})")
        elif proposal.target in taken:
            refused.append(f"{proposal.target} (one definition per variable)")
        elif _circle(proposal, state.derived_variables):
            refused.append(f"{proposal.target} (it would compute another definition's variable)")
        else:
            # its inverses join it only when the data agree with them too
            kept = {
                name: formula
                for name, formula in proposal.inverses.items()
                if state.derived_check(
                    DerivedVariable(name, formula, origin="ai"),
                    slot=f"derived_proposal::{i}::{name}",
                ).verdict
                == AGREES
            }
            refused.extend(
                f"{name} from {proposal.target}" for name in proposal.inverses if name not in kept
            )
            _put(replace(proposal, inverses=kept))
            (replaced if proposal.target in declared else added).append(i)
            taken.add(proposal.target)
    stored["found"] = [p for i, p in enumerate(found) if i not in added and i not in replaced]
    st.session_state[_PROPOSALS] = stored
    parts = []
    if added:
        parts.append(f"Added: {', '.join(found[i].target for i in added)}.")
    if replaced:
        parts.append(f"Replaced your definition of {', '.join(found[i].target for i in replaced)}.")
    if refused:
        parts.append(
            f"Not added, as the data do not agree or it cannot be used: {', '.join(refused)}."
        )
    _changed(" ".join(parts) or "Nothing was added.")


def _dismiss() -> None:
    st.session_state.pop(_PROPOSALS, None)


def _proposals(definitions: tuple[DerivedVariable, ...]) -> None:
    """The proposals of the AI engine, each checked against the data, the best evidence first, for the user to
    choose: nothing is added by itself."""
    state = get_state()
    stored = st.session_state.get(_PROPOSALS)
    if not stored or stored.get("fp") != state.session.working_fp:
        return
    if stored.get("error"):
        st.error(f"The AI engine could not answer: {plain(stored['error'], limit=300)}")
        return
    found: list[DerivedVariable] = stored.get("found") or []
    if not found:
        st.caption("The AI engine found no derived variable among your columns.")
        return
    checks = [state.derived_check(p, slot=f"derived_proposal::{i}") for i, p in enumerate(found)]
    declared = {d.target for d in definitions}
    order = sorted(range(len(found)), key=lambda i: (_ORDER[checks[i].verdict], found[i].target))
    table = pd.DataFrame(
        [
            {
                "add": False,
                "variable": found[i].target,
                "formula": found[i].formula,
                "check": checks[i].label
                + (" · replaces yours" if found[i].target in declared else ""),
                "evidence": checks[i].evidence
                + (
                    f"; {checks[i].unit_hint}"
                    if checks[i].unit_hint and checks[i].unit_hint != checks[i].evidence
                    else ""
                ),
                "inverses": "; ".join(f"{k} = {v}" for k, v in found[i].inverses.items()),
                "source": found[i].note,
            }
            for i in order
        ]
    )
    edited = st.data_editor(
        table,
        hide_index=True,
        disabled=["variable", "formula", "check", "evidence", "inverses", "source"],
        column_config={
            "add": st.column_config.CheckboxColumn("Add", help="Tick the proposals to add."),
            "variable": "Variable",
            "formula": st.column_config.TextColumn("Formula", width="medium"),
            "check": "Check",
            "evidence": st.column_config.TextColumn("On your data", width="large"),
            "inverses": st.column_config.TextColumn(
                "Inverses", help="Added with it only when the data agree with them too."
            ),
            "source": "Source",
        },
        key=f"derived_review::{stored['fp']}::{len(found)}",
        alt="The AI engine's proposals of derived variables, checked against your data",
    )
    ticked = [order[j] for j, add in enumerate(edited["add"].tolist()) if add]
    agreeing = [i for i in order if checks[i].verdict == AGREES and found[i].target not in declared]
    with st.container(horizontal=True):
        st.button(
            f"Add the {len(agreeing)} that agree" if agreeing else "None agrees with your data",
            type="primary",
            icon=":material/done_all:",
            key="derived_adopt_agreeing",
            disabled=not agreeing,
            on_click=_adopt,
            args=(agreeing,),
        )
        st.button(
            "Add the ticked ones",
            icon=":material/add:",
            key="derived_adopt_ticked",
            disabled=not ticked,
            on_click=_adopt,
            args=(ticked,),
        )
        st.button("Dismiss", icon=":material/close:", key="derived_dismiss", on_click=_dismiss)


def _ai_search(data: pd.DataFrame, definitions: tuple[DerivedVariable, ...]) -> None:
    engine = _engine()
    with st.container(border=True):
        st.markdown("**Find them with the AI engine**")
        st.caption(
            "The AI engine is sent the names of your columns, how each is stored, their units in the variable map "
            "and, if you tick it, the typical values of each number column (its 5th, 50th and 95th percentiles, "
            "never the minimum or the maximum, which an outlier would move): no row. Each proposal is then checked "
            "here, on your data, and added only when you choose it."
        )
        with st.container(horizontal=True, vertical_alignment="center"):
            st.checkbox(
                "Send the typical values of number columns, to help find their units",
                value=True,
                key=_RANGES,
                persist_state="session",  # unticked, it stays so for the session, whatever the page
            )
            if st.button(
                "Find derived variables",
                icon=":material/auto_awesome:",
                key="derived_find",
                disabled=engine is None,
                help=None
                if engine is not None
                else "Start the AI engine in Settings > AI engine first.",
            ):
                with st.spinner("The AI engine reads your columns..."):
                    _find(data, engine)
        _proposals(definitions)


@st.fragment
def render_derived_variables() -> None:
    state = get_state()
    if not state.has_data:
        st.info("Load a dataset first.")
        return
    data = state.working_frame()
    assert data is not None
    st.markdown("### Derived variables")
    st.caption(
        "Variables computed from others in the same row: a BMI from the weight and the height, a total from its "
        "items, a unit conversion. Each one declared here is checked against your data; you then choose whether to "
        "fill its missing values from its formula, and to keep it out of the single or multiple imputation, "
        "recomputed from the imputed values. With no definition, nothing changes."
    )
    notice = st.session_state.pop(_NOTICE, None)
    if notice:
        st.toast(plain(notice), icon=":material/check:")

    definitions = state.derived_variables
    scope = state.population_scope(data)
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.markdown("**Definitions**")
            help_popover(
                "Each definition is computed on the rows where its variable and every variable of its formula are "
                "recorded, and compared with the recorded value: within the precision the variable is recorded with "
                "(±0.05 for a BMI to 0.1), or what rounding its variables moves it. It **agrees** when 99% of these "
                "rows give the recorded value (beyond chance), **mostly agrees** at 90%, which you may accept; it "
                "**disagrees** otherwise, also when it disagrees for one group of a text or code variable (a factor "
                "wrong for women only). Under 60 rows, it cannot be checked, which you may accept too. Nothing "
                "leaves this machine, and agreeing with your data says nothing of the published reference.",
                label="How the check works",
            )
        _form(data, definitions)
        if not definitions:
            st.caption("No derived variable is declared.")
        for definition in definitions:
            check = state.derived_check(definition)
            to_fill = len(check.fillable_rows.intersection(scope.rows)) if check.fills else 0
            with st.expander(_summary(definition, check, to_fill), icon=_BADGES[check.verdict][0]):
                _details(definition, check, data)
    _ai_search(data, definitions)
    if st.session_state.pop(_CHANGED, False):
        st.rerun(scope="app")  # the other tabs read the definitions too


__all__ = ["render_derived_variables"]
