"""Import engine: detection of encoding/delimiter/decimal, parse reports, Excel sheets (audit U3)."""

import io
from pathlib import Path

import pandas as pd
import pytest

from core.state import ParseOptions
from ingest.loaders import (
    detect_decimal,
    detect_delimiter,
    detect_encoding,
    excel_sheet_names,
    file_kind,
    read_path,
    read_table,
    sniff,
)

FRENCH_CSV = "id;poids;taille;ville\n1;72,5;1,80;Esch-sur-Alzette\n2;65,0;1,65;Ettelbrück\n3;80,25;1,75;Luxembourg\n"


def test_sniff_semicolon_decimal_comma_latin1() -> None:
    data = FRENCH_CSV.encode("latin-1")
    options = sniff(data, "cohorte.csv")
    assert options.sep == ";"
    assert options.decimal == ","
    result = read_table(data, "cohorte.csv", options)
    assert list(result.df.columns) == ["id", "poids", "taille", "ville"]
    assert result.df["poids"].tolist() == [72.5, 65.0, 80.25]
    assert pd.api.types.is_float_dtype(result.df["taille"])
    assert options.encoding == "cp1252"
    assert "Ettelbrück" in result.df["ville"].tolist()
    assert result.report.bad_lines == 0


def test_detect_encoding_prefers_utf8_and_handles_bom() -> None:
    assert detect_encoding("a,b\n1,é\n".encode()) == "utf-8"
    assert detect_encoding(b"\xef\xbb\xbfa,b\n1,2\n") == "utf-8-sig"
    assert detect_encoding("a,b\n1,é\n".encode("latin-1")) != "utf-8"


def test_delimiter_detection_prefers_consistency_and_decimal_comma_tiebreak() -> None:
    assert detect_delimiter("a\tb\tc\n1\t2\t3\n") == "\t"
    assert detect_delimiter("a|b\n1|2\n") == "|"
    assert detect_delimiter("a,b,c\n1,2,3\n4,5,6\n") == ","
    # one comma and one semicolon per line: the decimal comma breaks the tie towards ';'
    assert detect_delimiter("x;y\n1,5;2\n3,5;4\n") == ";"
    assert detect_decimal("x;y\n1,5;2\n", ";") == ","
    assert detect_decimal("x;y\n1.5;2\n", ";") == "."
    assert detect_decimal("x,y\n1.5,2\n", ",") == "."


def test_malformed_lines_are_counted_not_dropped_silently() -> None:
    data = b"a,b\n1,2\n3,4,5\n6,7\n"
    result = read_table(data, "bad.csv")
    assert len(result.df) == 2
    assert result.report.bad_lines == 1
    assert result.report.bad_line_examples == ["3,4,5"]
    assert any("malformed" in w for w in result.report.warnings)


def test_column_report_flags_unnamed_and_empty_columns() -> None:
    data = b"a,,c\n1,,\n2,,\n"
    result = read_table(data, "cols.csv")
    assert result.report.unnamed_columns == ["Unnamed: 1"]
    assert set(result.report.empty_columns) >= {"Unnamed: 1", "c"}


def test_na_tokens_and_header_options() -> None:
    data = b"skip me\nid,value\n1,NA\n2,-\n3,7\n"
    options = ParseOptions(skiprows=1, na_values=("NA", "-"))
    result = read_table(data, "na.csv", options)
    assert result.df["value"].isna().sum() == 2
    assert result.df["value"].iloc[2] == 7


def _excel_bytes() -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        pd.DataFrame({"id": [1, 2, 3], "age": [50, 60, 70]}).to_excel(
            writer, sheet_name="Baseline", index=False
        )
        pd.DataFrame({"id": [1, 2, 4], "event": ["No", "Yes", "No"]}).to_excel(
            writer, sheet_name="Outcome", index=False
        )
    return buf.getvalue()


def test_excel_single_sheet_merge_and_stack() -> None:
    data = _excel_bytes()
    assert excel_sheet_names(data) == ["Baseline", "Outcome"]
    first = read_table(data, "study.xlsx")
    assert list(first.df.columns) == ["id", "age"]
    merged = read_table(
        data, "study.xlsx", ParseOptions(sheets=("Baseline", "Outcome"), merge_key="id")
    )
    assert len(merged.df) == 4 and {"age", "event"} <= set(merged.df.columns)
    stacked = read_table(data, "study.xlsx", ParseOptions(sheets=("Baseline", "Outcome")))
    assert len(stacked.df) == 6
    with pytest.raises(ValueError):
        read_table(data, "study.xlsx", ParseOptions(sheets=("Nope",)))


def test_parquet_and_path_loading(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    path = tmp_path / "t.parquet"
    df.to_parquet(path)
    assert read_path(str(path)).df.equals(df)
    assert file_kind("x.tsv") == "csv" and file_kind("x.XLSX") == "excel"
    with pytest.raises(ValueError):
        file_kind("report.docx")


def test_a_legacy_xls_workbook_is_read_like_xlsx() -> None:
    """Legacy Excel (.xls, BIFF8) through xlrd: sheets, numbers, decimals and accented text."""
    data = (Path(__file__).with_name("fixtures") / "legacy_cohort.xls").read_bytes()
    assert file_kind("legacy_cohort.xls") == "excel"
    assert excel_sheet_names(data) == ["baseline", "follow_up"]
    baseline = read_table(data, "legacy_cohort.xls").df
    assert list(baseline.columns) == ["id", "age", "weight_kg", "site"]
    assert baseline["weight_kg"].tolist() == [72.5, 80.25, 65.0]
    assert baseline["site"].iloc[2] == "Ettelbrück"
    merged = read_table(
        data, "legacy_cohort.xls", ParseOptions(sheets=("baseline", "follow_up"), merge_key="id")
    ).df
    assert len(merged) == 3 and "sbp_6m" in merged.columns


def test_sample_cohort_loads_with_documented_properties() -> None:
    path = Path(__file__).resolve().parents[2] / "assets" / "sample_cohort.csv"
    result = read_path(str(path), ParseOptions(dayfirst=True))
    df = result.df
    assert df.shape == (500, 19)
    assert df["age"].max() == 212  # the deliberate anomaly
    risk = df.groupby("arm")["event"].apply(lambda s: (s == "Yes").mean())
    assert risk["Placebo"] / risk["Treatment"] > 1.4
    assert (
        df.groupby("arm")["biomarker"].apply(lambda s: s.isna().mean())["Treatment"] > 0.15
    )  # MAR
    assert 0.03 < df["lab_mcar"].isna().mean() < 0.15  # MCAR
