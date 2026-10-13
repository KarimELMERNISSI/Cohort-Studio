"""
Three-step import flow (audit U3): choose a file, check how it is read (auto-detected options the
user can override, a 20-row preview and a parse report), then name and load it.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
from typing import Any

import pandas as pd
import streamlit as st

from core.state import DEFAULT_NA_TOKENS, ParseOptions, dataset_base_name
from ingest.loaders import (
    DELIMITER_CANDIDATES,
    DELIMITER_LABELS,
    SUPPORTED_EXTENSIONS,
    LoadResult,
    excel_sheet_names,
    file_kind,
    read_table,
    sniff,
)
from ui.components import error_card, page_link, section
from ui.state import AppState

ENCODINGS = ("utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-15", "utf-16")
THOUSANDS = {
    "none": None,
    "comma ( , )": ",",
    "dot ( . )": ".",
    "space": " ",
    "apostrophe ( ' )": "'",
}
DRAFT_KEY = "import_draft"


def _new_draft(uploaded: Any) -> dict[str, Any]:
    data = uploaded.getvalue()
    kind = file_kind(uploaded.name)
    options = sniff(data, uploaded.name)
    if st.session_state.get("import_default_dayfirst"):
        options = ParseOptions(**{**options.to_dict(), "dayfirst": True})
    sheet_names = excel_sheet_names(data) if kind == "excel" else []
    if sheet_names:
        options = ParseOptions(**{**options.to_dict(), "sheets": (sheet_names[0],)})
    return {
        "file_id": uploaded.file_id,
        "name": uploaded.name,
        "data": data,
        "kind": kind,
        "options": options,
        "sheet_names": sheet_names,
        "file_hash": hashlib.md5(data).hexdigest(),
        "parsed_for": None,
        "parsed": None,
        "loaded": False,
    }


def _common_columns(data: bytes, sheets: list[str]) -> list[str]:
    try:
        with pd.ExcelFile(io.BytesIO(data)) as xls:
            common: set[str] | None = None
            for sheet in sheets:
                cols = {str(c) for c in pd.read_excel(xls, sheet_name=sheet, nrows=0).columns}
                common = cols if common is None else common & cols
        return sorted(common or [])
    except Exception:
        return []


def _render_options(draft: dict[str, Any]) -> ParseOptions:
    current: ParseOptions = draft["options"]
    kind = draft["kind"]
    if kind == "parquet":
        st.caption("Parquet files carry their own types; nothing to configure.")
        return current

    fields = current.to_dict()
    if kind == "csv":
        c1, c2, c3, c4 = st.columns(4)
        enc_options = list(dict.fromkeys([current.encoding, *ENCODINGS]))
        fields["encoding"] = c1.selectbox(
            "Encoding",
            enc_options,
            index=0,
            key=f"imp_enc_{draft['file_id']}",
            help="Detected automatically; change it if accents look wrong in the preview.",
        )
        sep_index = (
            DELIMITER_CANDIDATES.index(current.sep) if current.sep in DELIMITER_CANDIDATES else 0
        )
        fields["sep"] = c2.radio(
            "Delimiter",
            DELIMITER_CANDIDATES,
            index=sep_index,
            format_func=lambda s: DELIMITER_LABELS[s],
            key=f"imp_sep_{draft['file_id']}",
            help="The character between columns.",
        )
        fields["decimal"] = c3.radio(
            "Decimal separator",
            (".", ","),
            index=0 if current.decimal == "." else 1,
            key=f"imp_dec_{draft['file_id']}",
            help="Use ',' for files exported from French/German locales.",
        )
        thousands_label = next((k for k, v in THOUSANDS.items() if v == current.thousands), "none")
        fields["thousands"] = THOUSANDS[
            c4.selectbox(
                "Thousands separator",
                list(THOUSANDS),
                index=list(THOUSANDS).index(thousands_label),
                key=f"imp_thou_{draft['file_id']}",
                help="Character grouping thousands in numbers (e.g. 1 234,5 or 1,234.5).",
            )
        ]
    else:
        names = draft["sheet_names"]
        chosen = st.multiselect(
            "Sheet(s)",
            names,
            default=list(current.sheets) or names[:1],
            key=f"imp_sheets_{draft['file_id']}",
            help="Select one sheet, or several to merge on a shared identifier column.",
        )
        fields["sheets"] = tuple(chosen)
        if len(chosen) > 1:
            common = _common_columns(draft["data"], chosen)
            if common:
                default_idx = common.index(current.merge_key) if current.merge_key in common else 0
                fields["merge_key"] = st.selectbox(
                    "Merge key (shared identifier column)",
                    common,
                    index=default_idx,
                    key=f"imp_merge_{draft['file_id']}",
                    help="Rows of the selected sheets are joined on this column (outer join).",
                )
            else:
                st.warning(
                    "The selected sheets share no column name; they will be stacked vertically."
                )
                fields["merge_key"] = None
        else:
            fields["merge_key"] = None

    c5, c6, c7, c8 = st.columns(4)
    fields["header"] = (
        int(
            c5.number_input(
                "Header row (1 = first line)",
                min_value=1,
                max_value=100,
                value=current.header + 1,
                key=f"imp_header_{draft['file_id']}",
                help="The line holding the column names, counted after any skipped lines.",
            )
        )
        - 1
    )
    fields["skiprows"] = int(
        c6.number_input(
            "Lines to skip before the header",
            min_value=0,
            max_value=1000,
            value=current.skiprows,
            key=f"imp_skip_{draft['file_id']}",
            help="Use this for files that start with titles or notes before the table.",
        )
    )
    fields["dayfirst"] = c7.checkbox(
        "Dates are day-first (31/12/2024)",
        value=current.dayfirst,
        key=f"imp_dayfirst_{draft['file_id']}",
        help="European date order. Used when dates are parsed later.",
    )
    na_text = c8.text_input(
        "Missing-value tokens",
        value=", ".join(t for t in current.na_values if t),
        key=f"imp_na_{draft['file_id']}",
        help="Comma-separated. Empty cells are always treated as missing.",
    )
    tokens = tuple(dict.fromkeys(["", *[t.strip() for t in na_text.split(",") if t.strip()]]))
    fields["na_values"] = tokens or DEFAULT_NA_TOKENS
    return ParseOptions(**fields)


def _parse(draft: dict[str, Any], options: ParseOptions) -> LoadResult | None:
    if draft.get("parsed_for") == options and draft.get("parsed") is not None:
        return draft["parsed"]
    try:
        result = read_table(draft["data"], draft["name"], options)
    except Exception as exc:
        error_card("The file could not be parsed with these options.", exc)
        draft["parsed_for"], draft["parsed"] = None, None
        return None
    draft["parsed_for"], draft["parsed"] = options, result
    return result


def _render_preview(result: LoadResult) -> None:
    report = result.report
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows", f"{report.n_rows:,}")
    m2.metric("Columns", report.n_cols)
    m3.metric(
        "Malformed lines",
        report.bad_lines,
        help="Lines with a different number of fields; they are skipped, never silently merged.",
    )
    m4.metric("Empty columns", len(report.empty_columns))
    for warning in report.warnings:
        st.warning(warning)
    if report.bad_line_examples:
        with st.expander("Examples of malformed lines"):
            for example in report.bad_line_examples:
                st.code(example)
    df = result.df
    types = pd.DataFrame(
        {
            "column": [str(c) for c in df.columns],
            "type": [str(t) for t in df.dtypes],
            "missing": [int(df[c].isna().sum()) for c in df.columns],
        }
    )
    st.dataframe(df.head(20), width="stretch", height=320)
    with st.expander("Column types and missing counts"):
        st.dataframe(types, width="stretch", hide_index=True)
    numeric = sum(pd.api.types.is_numeric_dtype(t) for t in df.dtypes)
    st.caption(
        f"{numeric} numeric column(s), {len(df.columns) - numeric} text/other column(s). "
        "If numbers show as text, check the decimal and thousands separators."
    )


def render_import_flow(state: AppState) -> None:
    section("1. Choose a file")
    uploaded = st.file_uploader(
        "Dataset file",
        type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
        key="import_uploader",
        help="CSV / TSV / TXT (any delimiter and encoding), Excel (one or several sheets) or Parquet. Max 500 MB.",
    )
    draft = st.session_state.get(DRAFT_KEY)
    if uploaded is None:
        if draft is not None:
            st.session_state.pop(DRAFT_KEY, None)
        st.caption(
            "Or pick a saved version from the sidebar, or load the sample cohort from the Start page."
        )
        return
    if draft is None or draft.get("file_id") != uploaded.file_id:
        try:
            draft = _new_draft(uploaded)
        except Exception as exc:
            error_card("This file type is not supported or the file is unreadable.", exc)
            return
        st.session_state[DRAFT_KEY] = draft

    section("2. Check how the file is read")
    options = _render_options(draft)
    draft["options"] = options
    result = _parse(draft, options)
    if result is None:
        return
    _render_preview(result)

    section("3. Name and load")
    c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
    name = c1.text_input(
        "Dataset name",
        value=dataset_base_name(draft["name"]),
        key=f"imp_name_{draft['file_id']}",
        help="Used for saved versions (name_v1, name_v2, ...) and in traces.",
    )
    save_now = c2.checkbox(
        "Save as version v1 in my workspace",
        value=True,
        key=f"imp_save_{draft['file_id']}",
        help="Stores the loaded table as a Parquet version you can reopen from the sidebar.",
    )
    if st.button("Load dataset", type="primary", key=f"imp_load_{draft['file_id']}"):
        ref = state.load_parsed(
            result.df,
            filename=draft["name"],
            data=draft["data"],
            options=options,
            name=dataset_base_name(name),
        )
        with contextlib.suppress(Exception):
            state.transformation_manager.add_step(
                "parse_report", result.report.to_dict(), "Parse report of the imported file"
            )
        if save_now:
            ok, msg, saved = state.save_version(ref.name)
            if not ok:
                st.warning(f"Loaded, but the version could not be saved: {msg}")
        draft["loaded"] = True
        st.rerun()
    if draft.get("loaded"):
        st.success(
            f"Loaded {state.dataset.label if state.dataset else name}. Next: validate the data, then explore it."
        )
        col_a, col_b = st.columns(2)
        with col_a:
            page_link("validate", "Validate the data", icon=":material/fact_check:")
        with col_b:
            page_link("explore", "Explore statistics", icon=":material/analytics:")


__all__ = ["render_import_flow"]
