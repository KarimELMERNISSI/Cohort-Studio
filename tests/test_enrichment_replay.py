"""
Files read for an enrichment: every entry point reads them with the import engine, and the replay reads them
again as the page did (M5). Before, a file given by its path went through a legacy loader: a semicolon file with
decimal commas came out as one column, "-" and "?" stayed text, a chosen sheet replayed as the first sheet, date
keys were parsed on the page only, and the merged columns changed order with Python's hash seed.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import enrich.external_data as eed
from manage.reproduction_manager import _read_any
from utils.multipage import _load_excel_with_selection, load_dataframe

ROOT = Path(__file__).resolve().parents[1]

SEMICOLON = "id;sbp;ldl;crp;hba1c;weight;height\n" + "\n".join(
    f"{i};{120 + i},5;{2 + i / 10:.1f}".replace(".", ",") + f";{i},2;6,{i};{70 + i},0;1,{70 + i}"
    for i in range(1, 6)
)
TOKENS = "id,score\n1,80\n2,-\n3,?\n4,95\n"


class _Upload(io.BytesIO):
    """What st.file_uploader returns: the bytes, with a name."""

    def __init__(self, data: bytes, name: str) -> None:
        super().__init__(data)
        self.name = name


@pytest.fixture()
def files(tmp_path: Path) -> dict[str, Path]:
    paths = {"semicolon": tmp_path / "labs_fr.csv", "tokens": tmp_path / "scores.csv", "workbook": tmp_path / "visits.xlsx"}
    paths["semicolon"].write_text(SEMICOLON, encoding="utf-8")
    paths["tokens"].write_text(TOKENS, encoding="utf-8")
    with pd.ExcelWriter(paths["workbook"]) as writer:
        pd.DataFrame({"note": ["first sheet"]}).to_excel(writer, sheet_name="cover", index=False)
        pd.DataFrame({"id": [1, 2, 3, 4], "potassium": [4.1, 3.9, 5.2, 4.4]}).to_excel(writer, sheet_name="labs", index=False)
    return paths


def test_a_file_read_by_path_is_the_table_the_replay_reads(files: dict[str, Path]) -> None:
    by_path = load_dataframe(str(files["semicolon"]))
    assert by_path.shape == (5, 7) and by_path["sbp"].dtype == float  # decimal commas read as numbers
    pd.testing.assert_frame_equal(by_path, _read_any(str(files["semicolon"])))
    tokens = load_dataframe(str(files["tokens"]))
    assert tokens["score"].isna().tolist() == [False, True, True, False]  # "-" and "?" are missing values
    pd.testing.assert_frame_equal(tokens, _read_any(str(files["tokens"])))


def test_an_upload_is_the_table_the_replay_reads(files: dict[str, Path]) -> None:
    upload = load_dataframe(_Upload(files["semicolon"].read_bytes(), "labs_fr.csv"))
    pd.testing.assert_frame_equal(upload, _read_any(str(files["semicolon"])))
    assert upload.attrs["parse_options"]["sep"] == ";" and upload.attrs["parse_options"]["decimal"] == ","


def _record(tmp_path: Path, main: pd.DataFrame, enrichment: pd.DataFrame, path: Path, identifier: list[str]):
    """A session that merges ``enrichment`` (read from ``path``) as the Enrich page does: align, merge, stage."""
    from core.session import DataSession
    from core.state import ParseOptions
    from ingest.loaders import read_path
    from manage.transformation_manager import TransformationManager

    src = tmp_path / "cohort.csv"
    main.to_csv(src, index=False)
    tm = TransformationManager(trace_dir=str(tmp_path / "traces"))
    tm.initialize_session("cohort.csv")
    tm.source_dataset = str(src)
    session = DataSession(trace=tm)
    options = ParseOptions()
    session.install(read_path(str(src), options).df, params={"source_path": str(src), "parse_options": options.to_dict()})
    working = session.working_frame()
    if "visit_date" in working.columns:  # the main dataset holds dates (a recorded conversion)
        converted = working.assign(visit_date=pd.to_datetime(working["visit_date"]))
        params = {"columns": ["visit_date"], "new_type": "datetime"}
        session.stage(converted, "data_type_conversion", params, "Dates")
        session.commit_staging()
        working = session.working_frame()
    enrichment = eed.align_date_keys(working, enrichment)
    params = {
        "enrichment_file_path": str(path),
        "parse_options": enrichment.attrs.get("parse_options"),
        "strategy": "left",
        "conflict_resolution": "keep",
        "identifier": identifier,
        "retained_main_columns": [str(c) for c in working.columns],
        "retained_enrichment_columns": [str(c) for c in enrichment.columns],
    }
    merged = eed.add_data(
        df=working,
        additional_df=enrichment,
        left_id_names=identifier,
        right_id_names=identifier,
        additional_cols=[c for c in enrichment.columns if c not in identifier],
        strategy="left",
        conflict_resolution="keep",
    )
    session.stage(merged, "enrichment", params, "Enriched with external data")
    session.commit_staging()
    return tm, session


def _replays(tm, session) -> None:
    from core.fingerprint import frame_fingerprint
    from manage.reproduction_manager import run_trace

    result = run_trace(json.loads(json.dumps(tm.get_trace(), default=str)))
    assert result.error is None and result.reproducible, result.warnings
    assert frame_fingerprint(result.df) == session.data_fp


def test_an_enrichment_read_by_path_replays(tmp_path: Path, files: dict[str, Path]) -> None:
    main = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6], "age": [50, 61, 72, 45, 58, 66]})
    tm, session = _record(tmp_path, main, load_dataframe(str(files["semicolon"])), files["semicolon"], ["id"])
    assert session.base["sbp"].notna().sum() == 5
    _replays(tm, session)


@pytest.mark.parametrize("sep, chosen", [("^", False), ("*", True)])
def test_an_enrichment_with_an_unusual_delimiter_replays(tmp_path: Path, sep: str, chosen: bool) -> None:
    """'^' is detected; '*' is chosen by the user ('How the file is read'): the replay reads the file the same way."""
    from core.state import ParseOptions

    path = tmp_path / "lab_export.txt"
    path.write_text(f"id{sep}potassium\n1{sep}4,1\n2{sep}3,9\n3{sep}5,2\n", encoding="utf-8")
    options = ParseOptions(sep=sep, decimal=",") if chosen else None
    labs = load_dataframe(str(path), options)
    assert labs.attrs["parse_options"]["sep"] == sep and labs["potassium"].tolist() == [4.1, 3.9, 5.2]
    main = pd.DataFrame({"id": [1, 2, 3, 4], "age": [50, 61, 72, 45]})
    tm, session = _record(tmp_path, main, labs, path, ["id"])
    assert session.base["potassium"].notna().sum() == 3
    _replays(tm, session)


def test_an_enrichment_from_a_chosen_sheet_replays(tmp_path: Path, files: dict[str, Path]) -> None:
    data = files["workbook"].read_bytes()
    labs = _load_excel_with_selection(data, ["cover", "labs"], ["labs"])
    main = pd.DataFrame({"id": [1, 2, 3, 4], "age": [50, 61, 72, 45]})
    tm, session = _record(tmp_path, main, labs, files["workbook"], ["id"])
    assert session.base["potassium"].tolist() == [4.1, 3.9, 5.2, 4.4]
    _replays(tm, session)  # read as the page did: the sheet "labs", not the first sheet


def test_an_enrichment_on_a_date_key_replays(tmp_path: Path) -> None:
    visits = tmp_path / "visits.csv"
    visits.write_text("visit_date,ldl\n2024-01-03,2.9\n2024-02-07,3.4\n", encoding="utf-8")
    main = pd.DataFrame({"visit_date": ["2024-01-03", "2024-02-07", "2024-03-01"], "age": [50, 61, 72]})
    tm, session = _record(tmp_path, main, load_dataframe(str(visits)), visits, ["visit_date"])
    assert session.base["ldl"].notna().sum() == 2  # the text dates of the file matched the dates of the main table
    _replays(tm, session)


MERGE = """
import pandas as pd
import enrich.external_data as eed
main = pd.DataFrame({"id": [1, 2], "age": [50, 60]})
extra = pd.DataFrame({"id": [1, 2], "zeta": [1, 2], "alpha": [3, 4], "mid": [5, 6], "beta": [7, 8]})
print(list(eed.add_data(main, extra, ["id"], ["id"], additional_cols=["zeta", "alpha", "mid", "beta"]).columns))
"""


def test_the_merged_columns_keep_their_order_under_any_hash_seed() -> None:
    orders = set()
    for seed in ("1", "2", "3"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(ROOT)}
        out = subprocess.run([sys.executable, "-c", MERGE], capture_output=True, text=True, env=env, check=True)
        orders.add(out.stdout.strip())
    assert orders == {"['id', 'age', 'zeta', 'alpha', 'mid', 'beta']"}


def test_missing_values_stay_missing_through_the_merge(files: dict[str, Path]) -> None:
    main = pd.DataFrame({"id": [1, 2, 3, 4]})
    merged = eed.add_data(main, load_dataframe(str(files["tokens"])), ["id"], ["id"], additional_cols=["score"])
    assert merged["score"].tolist()[0] == 80 and np.isnan(merged["score"].tolist()[1])
