"""
Multiple imputation with declared derived variables, "impute, then derive" (enrich.derived.derivation_steps, derive,
impute_and_derive): each derived variable still missing is left out of the chains, so neither imputed nor, while
incomplete, a predictor, then computed by its formula in every completed dataset from the imputed values; the replay
computes the same cells.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from biostats.mi import ImputationSpec
from core.derived import DerivedVariable
from core.fingerprint import ensure_row_ids
from enrich.derived import (
    DERIVED_METHOD,
    IMPUTE_THEN_DERIVE,
    check_derived,
    derivation_steps,
    derive,
    impute_and_derive,
)

BMI = DerivedVariable("bmi", "weight_kg / (height_cm / 100) ** 2")


def _cohort(n: int = 300, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    weight = np.round(rng.normal(78, 13, n), 1)
    height = np.round(rng.normal(171, 9, n), 1)
    frame = pd.DataFrame(
        {
            "weight_kg": weight,
            "height_cm": height,
            "bmi": np.round(weight / (height / 100) ** 2, 1),
            "age": np.round(rng.normal(65, 10, n), 0),
        }
    )
    frame.loc[0:19, ["height_cm", "bmi"]] = (
        np.nan
    )  # neither: the height is imputed, then the BMI computed
    frame.loc[20:29, ["weight_kg", "bmi"]] = np.nan
    return ensure_row_ids(frame)


def _steps(frame: pd.DataFrame, definitions=(BMI,)) -> list[dict]:
    return derivation_steps(list(definitions), lambda d: check_derived(d, frame), frame)


SPEC = ImputationSpec(columns=["weight_kg", "height_cm", "bmi", "age"], m=3, maxit=3, seed=1)


def test_what_is_computed_instead_of_imputed_is_recorded() -> None:
    frame = _cohort()
    (step,) = _steps(frame)
    assert {k: step[k] for k in ("target", "formula", "round_decimals")} == {
        "target": "bmi",
        "formula": BMI.formula,
        "round_decimals": 1,
    }
    assert step["record"]["definition"] == {"target": "bmi", "formula": BMI.formula}
    assert "only_levels" not in step
    assert (
        _steps(frame.assign(bmi=frame["bmi"].fillna(25.0))) == []
    )  # nothing missing: nothing to compute


def test_a_derived_variable_is_left_out_of_the_chains_and_follows_its_formula() -> None:
    frame = _cohort()
    run = impute_and_derive(frame, SPEC, _steps(frame))
    assert run.spec.methods == {"bmi": ""} and run.methods["bmi"] == DERIVED_METHOD
    assert "bmi" not in run.predictors["height_cm"] and "bmi" not in run.predictors["weight_kg"]
    assert run.mi.n_imputed()["bmi"] == 30
    for _, completed in run.mi.completed():
        formula = completed["weight_kg"] / (completed["height_cm"] / 100) ** 2
        assert completed["bmi"].notna().all()
        assert np.allclose(completed["bmi"], formula, atol=0.05 + 1e-9)  # rounded as recorded
    # recorded values kept
    observed = frame.index[frame["bmi"].notna()]
    assert run.mi.complete(1).loc[observed, "bmi"].equals(frame.loc[observed, "bmi"])
    block = run.mi.info[
        "derived"
    ]  # kept with the imputation: a saved version and the methods paragraph read it
    assert block["approach"] == IMPUTE_THEN_DERIVE and [s["target"] for s in block["steps"]] == [
        "bmi"
    ]
    assert not any("is incomplete and not imputed" in w for w in run.warnings)
    again = impute_and_derive(frame, SPEC, _steps(frame))
    assert again.mi.fingerprint() == run.mi.fingerprint()


def test_a_value_its_formula_cannot_give_stays_missing() -> None:
    frame = _cohort().assign(note=lambda f: np.where(f.index % 7 == 0, None, "x"))
    rule = DerivedVariable("bmi", "weight_kg / (height_cm / 100) ** 2 + 0 * (note == 'x')")
    spec = ImputationSpec(columns=["weight_kg", "height_cm", "bmi", "age"], m=2, maxit=2, seed=1)
    steps = derivation_steps([rule], lambda d: check_derived(d, frame), frame)
    run = impute_and_derive(frame, spec, steps)
    # the note is outside the model and never imputed: where it is missing, the BMI stays missing
    unseen = frame.index[frame["bmi"].isna() & frame["note"].isna()]
    assert len(unseen) and run.mi.complete(1).loc[unseen, "bmi"].isna().all()
    assert any(f"bmi: {len(unseen):,} values stay missing" in w for w in run.warnings)


def test_a_derived_variable_is_there_for_a_later_formula() -> None:
    frame = _cohort()
    frame["obese"] = np.where(frame["bmi"].isna(), None, np.where(frame["bmi"] >= 30, "yes", "no"))
    obese = DerivedVariable("obese", "np.where(bmi >= 30, 'yes', 'no')")
    spec = ImputationSpec(
        columns=["weight_kg", "height_cm", "bmi", "age", "obese"], m=2, maxit=2, seed=1
    )
    steps = _steps(frame, (obese, BMI))
    assert [s["target"] for s in steps] == ["bmi", "obese"]
    run = impute_and_derive(frame, spec, steps)
    for _, completed in run.mi.completed():
        expected = np.where(completed["bmi"] >= 30, "yes", "no")
        assert (completed["obese"] == expected).all()
    assert run.methods["obese"] == DERIVED_METHOD


def test_derive_alone_needs_no_imputation() -> None:
    frame = _cohort()
    run = impute_and_derive(frame, SPEC, [])
    assert (
        "bmi" in run.mi.imputations and run.methods["bmi"] == "pmm"
    )  # imputed as a variable of its own
    mi, left = derive(run.mi, [])
    assert mi is run.mi and left == {}


def test_the_replay_computes_the_same_cells() -> None:
    from manage.reproduction_manager import apply_steps

    frame = _cohort()
    steps = _steps(frame)
    run = impute_and_derive(frame, SPEC, steps)
    params = {
        "spec": run.spec.model_dump(),
        "methods": dict(run.methods),
        "cells_fp": run.mi.fingerprint(),
        "derived": {"approach": IMPUTE_THEN_DERIVE, "recompute": True, "steps": steps},
    }
    replay = apply_steps(frame, [{"function": "mi_impute", "params": params}])
    assert replay.error is None and replay.mi_divergence == [], replay.warnings
    assert replay.imputations[0].fingerprint() == run.mi.fingerprint()
    # without the derivation, the replay would impute the BMI as recorded: it is told apart
    plain = {**params, "derived": None}
    assert apply_steps(frame, [{"function": "mi_impute", "params": plain}]).mi_divergence == [0]


@pytest.mark.parametrize("approach", [IMPUTE_THEN_DERIVE])
def test_the_approach_id_never_changes(approach: str) -> None:
    assert approach == "impute_then_derive" and DERIVED_METHOD == "derived"


def test_the_methods_paragraph_says_how_a_derived_variable_was_computed() -> None:
    from biostats.mi.binding import bind
    from biostats.mi.report import methods_paragraph

    frame = _cohort()
    run = impute_and_derive(frame, SPEC, _steps(frame))
    stored = bind(run)
    text = methods_paragraph(stored.binding, n_rows=300, incomplete=0.1, info=dict(run.mi.info))
    assert "The imputation model included 3 variables (weight_kg, height_cm, age)" in text
    assert "weight_kg, height_cm by predictive mean matching (5 donors)." in text
    assert (
        "bmi was not in the imputation model: its missing values were computed in each imputed dataset by its "
        "formula from the imputed values (bmi = weight_kg / (height_cm / 100) ** 2" in text
    )
    assert "von Hippel 2009" in text and "remained" not in text


def test_a_recorded_value_beside_an_imputed_variable_is_counted() -> None:
    # heights missing where the BMI is recorded, and no inverse: the imputed heights no longer match those BMIs
    frame = _cohort()
    frame.loc[40:59, "height_cm"] = np.nan
    run = impute_and_derive(frame, SPEC, _steps(frame))
    mismatched = run.mi.info["derived"]["mismatched"]
    assert 15 <= mismatched["bmi"] <= 20
    assert any("no longer matches its imputed variables" in w for w in run.warnings)


def test_a_variable_of_a_formula_outside_the_model_is_watched() -> None:
    from biostats.mi.binding import bind, status

    frame = _cohort()
    spec = ImputationSpec(
        columns=["height_cm", "bmi", "age"], m=2, maxit=2, seed=1
    )  # the weight is not in it
    run = impute_and_derive(frame, spec, _steps(frame))
    stored = bind(run)
    assert stored.binding.watched == ["weight_kg"]
    assert status(stored, frame).kind == "fresh"
    edited = frame.copy()
    edited.loc[100, "weight_kg"] += 5
    assert status(stored, edited).kind == "stale"


def test_an_unknown_approach_stops_the_replay_and_a_declined_one_imputes() -> None:
    from manage.reproduction_manager import apply_steps

    frame = _cohort()
    steps = _steps(frame)
    run = impute_and_derive(frame, SPEC, steps)
    params = {"spec": run.spec.model_dump(), "cells_fp": run.mi.fingerprint()}
    unknown = {
        **params,
        "derived": {"approach": "impute_in_the_chains", "recompute": True, "steps": steps},
    }
    result = apply_steps(frame, [{"function": "mi_impute", "params": unknown}])
    assert "unknown derivation approach 'impute_in_the_chains'" in (result.error or "")
    # declined: the definitions on record, the BMI imputed as a variable of its own
    declined = impute_and_derive(frame, SPEC, [], declined=[s["record"] for s in steps])
    block = declined.mi.info["derived"]
    assert block["recompute"] is False and block["steps"] == [] and len(block["declined"]) == 1
    assert declined.methods["bmi"] == "pmm"
