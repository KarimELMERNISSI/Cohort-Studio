"""Data > Enrich > Handle Missing Data > Derived variables (AppTest): declared, checked before they are saved, edited,
accepted when the data mostly agree, and kept with the study's configuration."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import utils.data_paths as dp

BMI = "weight_kg / (height_cm / 100) ** 2"
WEIGHT = "bmi * (height_cm / 100) ** 2"


@pytest.fixture
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path))
    return tmp_path


def _tab_script(mixed_units: bool = False) -> None:
    import numpy as np
    import pandas as pd
    import streamlit as st

    from app_pages.enrichment_derived import render_derived_variables
    from core.state import DatasetRef
    from ui.state import get_state

    state = get_state()
    if "seeded" not in st.session_state:
        st.session_state["seeded"] = True
        rng = np.random.default_rng(1)
        n = 200
        weight = np.round(rng.normal(75, 12, n), 1)
        height = np.round(rng.normal(170, 9, n), 0)
        frame = pd.DataFrame(
            {
                "weight_kg": weight,
                "height_cm": height,
                "bmi": np.round(weight / (height / 100) ** 2, 1),
            }
        )
        frame.loc[:9, "bmi"] = np.nan
        if mixed_units:  # a site recording another unit
            frame.loc[10:19, "bmi"] *= 10_000
        state.set_dataset(frame, DatasetRef(name="d", origin="upload", file_hash="h"))
    render_derived_variables()
    st.session_state["probe"] = {"config": state.config}


def _ok(at: AppTest) -> None:
    assert not at.exception, [str(e.value)[:300] for e in at.exception]


def _declared(at: AppTest) -> dict:
    return at.session_state["probe"]["config"].get("derived_variables") or {}


def _declare(at: AppTest, target: str, formula: str) -> None:
    at.selectbox(key="derived_form_target").select(target).run()
    at.text_input(key="derived_form_formula").input(formula).run()
    _ok(at)


def test_a_definition_is_checked_before_it_is_added(data_root: Path) -> None:
    at = AppTest.from_function(_tab_script, default_timeout=120).run()
    _ok(at)
    assert any("No derived variable is declared" in c.value for c in at.caption)
    _declare(at, "bmi", BMI)
    evidence = "gives the recorded value in 190 of 190 rows where bmi, weight\\_kg and height\\_cm are recorded"
    assert any(evidence in c.value for c in at.caption), [c.value for c in at.caption]
    at.button(key="derived_save").click().run()
    _ok(at)
    assert _declared(at) == {"bmi": {"formula": BMI}}
    assert [e.label for e in at.expander if e.label.startswith("bmi")] == [
        "bmi · agrees · 10 to fill"
    ]


def test_what_the_check_cannot_use_or_a_circle_cannot_be_saved(data_root: Path) -> None:
    at = AppTest.from_function(_tab_script, default_timeout=120).run()
    _declare(at, "bmi", "bmi * 1")  # its own variable
    assert at.button(key="derived_save").disabled
    _declare(at, "bmi", BMI)
    at.button(key="derived_save").click().run()
    # the weight from the BMI, while the BMI comes from the weight: an inverse of the BMI instead
    _declare(at, "weight_kg", WEIGHT)
    assert at.button(key="derived_save").disabled
    assert any("would compute each other" in w.value for w in at.warning)


def test_an_inverse_is_added_and_kept_when_the_definition_is_edited(data_root: Path) -> None:
    at = AppTest.from_function(_tab_script, default_timeout=120).run()
    _declare(at, "bmi", BMI)
    at.button(key="derived_save").click().run()
    at.selectbox(key="derived_inverse_name::bmi").select("weight_kg").run()
    at.text_input(key="derived_inverse_formula::bmi").input(WEIGHT).run()
    at.button(key="derived_inverse_add::bmi").click().run()
    _ok(at)
    assert _declared(at)["bmi"]["inverses"] == {"weight_kg": WEIGHT}
    captions = " ".join(c.value for c in at.caption)
    assert "where weight\\_kg, bmi and height\\_cm are recorded" in captions
    # edited: a new source, the same formula, the inverse kept
    at.button(key="derived_edit::bmi").click().run()
    assert at.text_input(key="derived_form_formula").value == BMI
    at.text_input(key="derived_form_note").input("WHO").run()
    at.button(key="derived_save").click().run()
    _ok(at)
    assert _declared(at)["bmi"] == {
        "formula": BMI,
        "inverses": {"weight_kg": WEIGHT},
        "note": "WHO",
    }
    at.button(key="derived_remove::bmi").click().run()
    _ok(at)
    assert _declared(at) == {}


def test_a_definition_the_data_mostly_agree_with_is_used_once_accepted(data_root: Path) -> None:
    at = AppTest.from_function(_tab_script, kwargs={"mixed_units": True}, default_timeout=120).run()
    _declare(at, "bmi", BMI)
    at.button(key="derived_save").click().run()
    _ok(at)
    labels = [e.label for e in at.expander if e.label.startswith("bmi")]
    assert labels == ["bmi · mostly agrees · 10 to fill · not used until you accept it"]
    assert any("10,000 times the formula" in c.value for c in at.caption)
    at.toggle(key="derived_accept::bmi").set_value(True).run()
    _ok(at)
    assert _declared(at)["bmi"]["accepted"] is True
    assert [e.label for e in at.expander if e.label.startswith("bmi")] == [
        "bmi · mostly agrees · 10 to fill"
    ]


def test_the_enrich_page_opens_its_missing_data_tabs_with_the_derived_variables(
    data_root: Path,
) -> None:
    from ui import nav

    os.environ.setdefault("CMP_IAM_PROVIDER", "local")
    main = Path(__file__).resolve().parents[2] / "main.py"
    at = AppTest.from_file(str(main), default_timeout=300)
    at.run()
    at.session_state["user_authenticated"] = True
    at.session_state["username"] = "derived_user"
    at.session_state["user_role"] = "researcher"
    at.run()
    at.switch_page(nav.page_path("start")).run()
    at.button(key="start_tutorial").click().run()
    at.switch_page(nav.page_path("enrich")).run()
    _ok(at)
    labels = [t.label for t in at.tabs]
    derived = labels.index("Derived variables")
    assert derived + 1 == labels.index("Targeted Imputation (Custom Formula)")
    assert any("No derived variable is declared" in c.value for c in at.caption)
