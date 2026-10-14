"""Tamper-evident trace chain and change descriptions."""

import copy
import json

import numpy as np
import pandas as pd

from core.lineage import (
    DISCONTINUOUS,
    TAMPERED,
    UNVERIFIABLE,
    VERIFIED,
    canonical_json,
    describe_change,
    genesis_hash,
    normalise,
    seal_step,
    verify_trace,
)


def _trace(n: int = 3) -> dict:
    steps, prev = [], genesis_hash("s1")
    fp = "f0"
    for i in range(n):
        step = seal_step(
            {
                "function": f"step{i}",
                "params": {"k": i},
                "data": {"fp_in": fp, "fp_out": f"f{i + 1}"},
            },
            prev,
        )
        steps.append(step)
        prev, fp = step["hash"], f"f{i + 1}"
    return {"schema_version": 2, "session_id": "s1", "steps": steps, "head_hash": prev}


def test_an_intact_chain_verifies_after_a_json_round_trip() -> None:
    trace = json.loads(json.dumps(_trace()))
    result = verify_trace(trace)
    assert result.status == VERIFIED and result.ok and result.n_steps == 3
    assert result.head_hash == trace["head_hash"]


def test_an_edited_step_is_reported() -> None:
    trace = _trace()
    trace["steps"][1]["params"]["k"] = 99
    result = verify_trace(trace)
    assert result.status == TAMPERED and result.first_bad_step == 2


def test_removed_and_reordered_steps_are_reported() -> None:
    removed = _trace()
    del removed["steps"][1]
    assert verify_trace(removed).status == TAMPERED
    truncated = _trace()
    truncated["steps"].pop()
    assert verify_trace(truncated).status == TAMPERED  # the head hash no longer matches
    swapped = _trace()
    swapped["steps"][0], swapped["steps"][1] = swapped["steps"][1], swapped["steps"][0]
    assert verify_trace(swapped).first_bad_step == 1


def test_a_gap_in_the_data_fingerprints_is_reported() -> None:
    steps, prev = [], genesis_hash("s1")
    for fp_in, fp_out in (("a", "b"), ("c", "d")):
        step = seal_step({"function": "x", "data": {"fp_in": fp_in, "fp_out": fp_out}}, prev)
        steps.append(step)
        prev = step["hash"]
    result = verify_trace({"schema_version": 2, "session_id": "s1", "steps": steps})
    assert result.status == DISCONTINUOUS and result.first_bad_step == 2


def test_older_traces_are_unverifiable() -> None:
    assert verify_trace({"session_id": "old", "steps": [{"function": "x"}]}).status == UNVERIFIABLE


def test_normalise_makes_steps_json_safe() -> None:
    value = normalise(
        {"a": np.int64(3), "b": (1, 2), "c": float("nan"), "d": np.float32(1.5), 4: {"x"}}
    )
    assert value == {"a": 3, "b": [1, 2], "c": None, "d": 1.5, "4": ["x"]}
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    original = {"k": [np.int64(1)]}
    copy.deepcopy(original)
    seal_step(original, "x")
    assert isinstance(original["k"][0], np.int64)  # sealing does not touch the caller's step


def test_describe_change_counts_rows_and_columns() -> None:
    before = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    after = before.drop(index=[1]).assign(c=1.0, a=lambda d: d["a"].astype(float))
    change = describe_change(before, after)
    assert (change.rows_in, change.rows_out, change.cols_in, change.cols_out) == (3, 2, 2, 3)
    assert change.columns_added == ("c",) and change.columns_retyped == ("a",)
    assert change.rows_removed == 1 and change.rows_added == 0 and change.changed
    assert json.dumps(change.to_dict())
