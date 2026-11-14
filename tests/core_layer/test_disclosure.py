"""
The small-cell rule of a shared package (M8): a count of participants is exact or "fewer than k", k is never below
5, and the participant flow is shown so that no withheld count comes back by subtraction (the reasons that are too
small on one line, an exit that is too small merged with the next transition, and only the population when every
exclusion together is too small).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.attrition import (
    DATA,
    LOADED,
    POPULATION,
    STAGE,
    AnalysisCount,
    FlowBox,
    FlowExit,
    ParticipantFlow,
    participant_flow,
    source_from_trace,
)
from core.disclosure import (
    DEFAULT_K,
    FLOOR_K,
    DisclosedBox,
    DisclosedFlow,
    DisclosurePolicy,
    disclose_flow,
    part_of,
    recoverable,
    shown,
)
from core.population import catalogue
from core.session import DataSession

TUTORIAL = Path(__file__).resolve().parents[2] / "assets" / "tutorial"
K10 = DisclosurePolicy()


class Sink:
    def __init__(self) -> None:
        self.steps: list[dict] = []

    def add_step(self, function_name, params, description, output_dataset_path=None, **extra):
        self.steps.append(
            {"function": function_name, "params": params, "description": description, **extra}
        )


# ----------------------------------------------------------------------------- the policy


def test_k_is_ten_unless_the_study_says_otherwise_and_never_below_five() -> None:
    assert (DEFAULT_K, FLOOR_K) == (10, 5)
    assert DisclosurePolicy().k == 10 and DisclosurePolicy(5).k == 5
    # a study's k on an installation: the larger of what the study asks for and of the installation's floor
    assert DisclosurePolicy.for_study(None, floor=5).k == 10
    assert DisclosurePolicy.for_study(5, floor=5).k == 5
    assert DisclosurePolicy.for_study(5, floor=10).k == 10
    assert DisclosurePolicy.for_study(20, floor=10).k == 20
    for k in (4, 0, -1):
        with pytest.raises(ValueError, match="never below 5"):
            DisclosurePolicy(k)
    with pytest.raises(ValueError, match="never below 5"):
        DisclosurePolicy.for_study(4, floor=5)
    with pytest.raises(ValueError, match="floor is never below 5"):
        DisclosurePolicy.for_study(10, floor=3)
    for floor in (float("nan"), 7.5, True, None):
        with pytest.raises(ValueError, match="floor is a whole number"):
            DisclosurePolicy.for_study(5, floor=floor)  # type: ignore[arg-type]
    with pytest.raises(TypeError):  # no floor: the package would follow no installation
        DisclosurePolicy.for_study(5)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="whole number"):
        DisclosurePolicy(7.5)  # type: ignore[arg-type]
    assert DisclosurePolicy(5).to_dict() == {"k": 5, "version": 2}
    # a package says why its k is not the study's
    raised = DisclosurePolicy.for_study(5, floor=10)
    assert raised.to_dict() == {"k": 10, "asked": 5, "floor": 10, "version": 2}
    assert raised == DisclosurePolicy(10)  # the same rule, whatever it was made for


def test_a_count_is_exact_or_fewer_than_k_and_zero_is_shown() -> None:
    assert [shown(n, K10).text for n in (0, 1, 9, 10, 1206)] == [
        "0",
        "fewer than 10",
        "fewer than 10",
        "10",
        "1,206",
    ]
    assert shown(3, K10).to_json() == {"fewer_than": 10} and shown(12, K10).to_json() == 12
    assert shown(7, DisclosurePolicy(5)).text == "7"


def test_a_part_of_a_total_is_given_only_when_the_rest_could_be() -> None:
    assert part_of(200, 415, K10) == "200 of the 415"
    assert (
        part_of(412, 415, K10) == "all but fewer than 10 of the 415"
    )  # the rest, 3, would come back
    assert part_of(3, 415, K10) == "fewer than 10 of the 415"
    assert part_of(0, 415, K10) == "0 of the 415" and part_of(415, 415, K10) == "415 of the 415"
    with pytest.raises(ValueError):
        part_of(5, 3, K10)


# ----------------------------------------------------------------------------- the flow


def _flow(loaded: int, *transitions: tuple[dict[str, int] | int, int]) -> ParticipantFlow:
    """A flow from ``loaded`` rows through transitions, each (reasons or a count leaving, rows added)."""
    main = [FlowBox("loaded", "Loaded: cohort", loaded, LOADED)]
    exits = []
    for i, (leaving, added) in enumerate(transitions):
        reasons = tuple(leaving.items()) if isinstance(leaving, dict) else ()
        n = sum(leaving.values()) if isinstance(leaving, dict) else leaving
        if n:
            exits.append(
                FlowExit(f"exit{i}", main[-1].id, f"Excluded: family {i}", n, reasons=reasons)
            )
        last = i == len(transitions) - 1
        main.append(
            FlowBox(
                "population" if last else f"after{i}",
                "Analysis population" if last else f"After: family {i}",
                main[-1].n - n + added,
                POPULATION if last else (DATA if added else STAGE),
                added=added,
            )
        )
    flow = ParticipantFlow(tuple(main), tuple(exits))
    assert flow.inconsistencies() == []
    return flow


def _numbers(disclosed: DisclosedFlow) -> list[int]:
    return [
        *(b.n for b in disclosed.boxes),
        *(n for t in disclosed.transitions for n in (t.excluded, t.added)),
        *(r.n for t in disclosed.transitions for r in t.reasons),
        *(r.lacking for t in disclosed.transitions for r in t.reasons if r.lacking is not None),
    ]


def test_reasons_that_are_too_small_are_shown_on_one_line_with_the_next_smallest() -> None:
    flow = _flow(1000, ({"age": 3, "kidney": 12, "consent": 30}, 0))
    disclosed = disclose_flow(flow, K10)
    (transition,) = disclosed.transitions
    assert [b.n for b in disclosed.boxes] == [1000, 955] and transition.excluded == 45
    # the line of the others names none of them: it would say that each holds at least one row
    assert [(r.names, r.n) for r in transition.reasons] == [(("consent",), 30), ((), 15)]
    assert 3 not in _numbers(disclosed) and 12 not in _numbers(disclosed)
    assert recoverable(flow, disclosed) == []
    written = json.dumps(disclosed.to_dict())
    assert "age" not in written and "kidney" not in written and "parts" not in written


def test_when_one_line_would_take_every_reason_the_total_says_it_all() -> None:
    flow = _flow(1206, ({f"implausible {i}": n for i, n in enumerate((1, 1, 2, 2, 2, 2, 3, 3))}, 0))
    (transition,) = disclose_flow(flow, K10).transitions
    assert transition.excluded == 16 and transition.reasons == ()
    two = _flow(1000, ({"age": 3, "kidney": 12}, 0))
    assert disclose_flow(two, K10).transitions[0].reasons == ()  # 3 and 12 together are the total


def test_an_exit_that_is_too_small_is_merged_with_the_next_transition() -> None:
    flow = _flow(1000, ({"age": 3}, 0), ({"kidney": 30, "consent": 12}, 0))
    disclosed = disclose_flow(flow, K10)
    assert [b.id for b in disclosed.boxes] == [
        "loaded",
        "population",
    ]  # the box between them is not shown
    assert [b.n for b in disclosed.boxes] == [1000, 955]
    (transition,) = disclosed.transitions
    assert transition.exits == ("exit0", "exit1") and transition.excluded == 45
    assert transition.labels == ("Excluded: family 0", "Excluded: family 1")
    assert [(r.names, r.n) for r in transition.reasons] == [(("kidney",), 30), ((), 15)]
    assert 997 not in _numbers(disclosed) and 3 not in _numbers(disclosed)
    assert recoverable(flow, disclosed) == []


def test_a_last_exit_that_is_too_small_joins_the_one_before() -> None:
    flow = _flow(1000, (40, 0), ({"age": 2, "sex": 1}, 0))
    disclosed = disclose_flow(flow, K10)
    assert [b.n for b in disclosed.boxes] == [1000, 957]
    (transition,) = disclosed.transitions
    assert transition.excluded == 43 and transition.reasons == ()  # 40 alone would give the 3 back
    assert recoverable(flow, disclosed) == []


def test_when_every_exclusion_together_is_too_small_only_the_population_is_shown() -> None:
    flow = _flow(1000, ({"age": 2}, 0), ({"sex": 1}, 0))
    disclosed = disclose_flow(flow, K10)
    assert [(b.id, b.n) for b in disclosed.boxes] == [("population", 997)]
    assert disclosed.transitions == () and disclosed.withheld == (
        "fewer than 10 rows excluded in all",
    )
    assert "1000" not in json.dumps(disclosed.to_dict())  # the rows loaded appear nowhere
    assert recoverable(flow, disclosed) == []


def test_rows_added_follow_the_same_rule() -> None:
    flow = _flow(1000, (0, 4), ({"age": 50}, 0))
    disclosed = disclose_flow(flow, K10)
    assert disclosed.withheld == ("fewer than 10 rows added in all",) and len(disclosed.boxes) == 1
    large = _flow(1000, (0, 40), ({"age": 50}, 0))
    assert [(t.excluded, t.added) for t in disclose_flow(large, K10).transitions] == [
        (0, 40),
        (50, 0),
    ]
    both = _flow(1000, (3, 40), ({"age": 50}, 0))
    (merged,) = disclose_flow(both, K10).transitions
    assert (merged.excluded, merged.added) == (53, 40)


def test_nothing_is_merged_when_every_count_can_be_shown() -> None:
    flow = _flow(1000, ({"age": 30, "sex": 0, "kidney": 12}, 0), (0, 0), ({"consent": 20}, 0))
    disclosed = disclose_flow(flow, K10)
    assert [b.n for b in disclosed.boxes] == [1000, 958, 958, 938]
    assert [(r.names, r.n) for r in disclosed.transitions[0].reasons] == [
        (("age",), 30),
        (("kidney",), 12),
    ]
    assert disclose_flow(flow, DisclosurePolicy(5)).to_dict()["k"] == 5


def test_two_criteria_of_one_name_are_not_confused() -> None:
    main = (
        FlowBox("loaded", "Loaded: cohort", 500, LOADED),
        FlowBox("population", "Analysis population", 455, POPULATION),
    )
    leaving = FlowExit(
        "stage:a", "loaded", "Excluded: a", 45, reasons=(("old", 3), ("old", 12), ("frail", 30))
    )
    flow = ParticipantFlow(main, (leaving,))
    disclosed = disclose_flow(flow, K10)
    lines = disclosed.transitions[0].reasons
    assert [(r.names, r.n, r.parts) for r in lines] == [
        (("frail",), 30, ("stage:a#2",)),
        ((), 15, ("stage:a#0", "stage:a#1")),
    ]
    assert recoverable(flow, disclosed) == []


def test_a_line_of_ten_for_ten_criteria_names_none_of_them() -> None:
    """Ten criteria of one row each beside one of 25: the line of the others read "10" with ten names, which gave
    each of them, since a criterion is listed only when it holds a row."""
    reasons = {f"rare {i}": 1 for i in range(10)} | {"frequent": 25}
    flow = _flow(1000, (reasons, 0))
    disclosed = disclose_flow(flow, K10)
    lines = disclosed.transitions[0].reasons
    assert [(r.names, r.n) for r in lines] == [(("frequent",), 25), ((), 10)]
    assert "rare" not in json.dumps(disclosed.to_dict())
    assert recoverable(flow, disclosed) == []


def test_the_rows_lacking_a_value_are_given_only_when_the_rest_could_be() -> None:
    main = (
        FlowBox("loaded", "Loaded: cohort", 500, LOADED),
        FlowBox("population", "Analysis population", 380, POPULATION),
    )
    leaving = FlowExit(
        "stage:inclusion",
        "loaded",
        "Not meeting: inclusion",
        120,
        reasons=(("adult", 60), ("not pregnant", 60)),
        missing=(("adult", 3), ("not pregnant", 40)),
    )
    lines = disclose_flow(ParticipantFlow(main, (leaving,)), K10).transitions[0].reasons
    assert [(r.names, r.n, r.lacking) for r in lines] == [
        (("adult",), 60, None),
        (("not pregnant",), 60, 40),
    ]


def test_a_population_below_k_is_refused() -> None:
    flow = _flow(40, ({"age": 33}, 0))
    with pytest.raises(ValueError, match="fewer than 10 rows: nothing of it can be shared"):
        disclose_flow(flow, K10)
    assert disclose_flow(flow, DisclosurePolicy(5)).boxes[-1].n == 7


def test_a_reader_s_arithmetic_finds_what_a_careless_flow_gives_back() -> None:
    """What the check is for: the two exits are shown as one (45), but the box between them is left in."""
    flow = _flow(1000, ({"age": 3}, 0), ({"kidney": 30, "consent": 12}, 0))
    sound = disclose_flow(flow, K10)
    assert recoverable(flow, sound) == []
    careless = DisclosedFlow(
        10,
        (sound.boxes[0], DisclosedBox("after0", "After: family 0", 997, STAGE), sound.boxes[1]),
        sound.transitions,
    )
    # 1,000 then 997 with no row added (the rows added are 0 in all, so 0 at each step): the exit of 3 comes back,
    # and with it its only reason
    assert recoverable(flow, careless) == ["exit:exit0", "reason:exit0#0"]


def test_no_small_count_comes_back_from_thousands_of_random_flows() -> None:
    rng = np.random.default_rng(20261110)
    shown_whole, merged, population_only = 0, 0, 0
    for _ in range(3000):
        loaded = int(rng.integers(200, 5000))
        transitions = []
        left = loaded
        for _ in range(int(rng.integers(1, 5))):
            reasons = {
                f"c{j}": int(n)
                for j, n in enumerate(
                    rng.choice([0, 1, 2, 3, 5, 9, 10, 11, 14, 25, 60], int(rng.integers(0, 5)))
                )
                if n
            }
            if sum(reasons.values()) > left - 20:
                reasons = {}
            added = int(rng.choice([0, 0, 0, 2, 7, 12, 40]))
            leaving: dict[str, int] | int = reasons if rng.random() < 0.8 else sum(reasons.values())
            transitions.append((leaving, added))
            left = left - sum(reasons.values()) + added
        flow = _flow(loaded, *transitions)
        population = flow.main[-1].n
        analyses = []
        for j in range(int(rng.integers(0, 3))):
            left = int(rng.choice([0, 0, 1, 3, 9, 10, 40]))
            if left > population - 5:
                left = 0
            analysed = int(rng.integers(max(1, (population - left) // 2), population - left + 1))
            analyses.append(
                AnalysisCount(
                    f"analysis {j}",
                    population,
                    analysed,
                    n_left_out=left,
                    left_out="not in the compared groups",
                )
            )
        for k in (5, 10):
            policy = DisclosurePolicy(k)
            disclosed = disclose_flow(flow, policy, analyses)  # raises if a count comes back
            assert recoverable(flow, disclosed, analyses) == []
            assert all(policy.shows(a.left_out) for a in disclosed.analyses)
            assert len(disclosed.analyses) == sum(policy.shows(a.n_left_out) for a in analyses)
            assert all(policy.shows(n) for n in _numbers(disclosed)), (transitions, k)
            assert disclosed.boxes[-1].n == flow.main[-1].n  # the population is exact
            # what is shown still adds up
            for transition, before, after in zip(
                disclosed.transitions, disclosed.boxes, disclosed.boxes[1:], strict=False
            ):
                assert after.n == before.n - transition.excluded + transition.added
                if transition.reasons:
                    assert sum(r.n for r in transition.reasons) == transition.excluded
            if any("in all" in said for said in disclosed.withheld):
                population_only += 1
            elif len(disclosed.boxes) < len(flow.main):
                merged += 1
            else:
                shown_whole += 1
    assert min(shown_whole, merged, population_only) > 100  # each way of showing a flow was met


# ----------------------------------------------------------------------------- the tutorial cohort


def test_the_tutorial_flow_shows_no_count_below_ten() -> None:
    from core.state import ParseOptions
    from ingest.loaders import read_table

    config = json.loads((TUTORIAL / "tutorial_config.json").read_text(encoding="utf-8"))
    session = DataSession(trace=Sink())
    session.install(
        read_table(
            (TUTORIAL / "tutorial_cohort.csv").read_bytes(),
            "tutorial_cohort.csv",
            ParseOptions(dayfirst=True),
        ).df,
        params={"filename": "tutorial_cohort.csv"},
    )
    session.set_ticked([c.id for c in catalogue(config)])
    population = session.population(config)
    source, n_loaded, steps = source_from_trace({"steps": session.trace.steps})
    flow = participant_flow(
        population,
        dataset="tutorial_cohort",
        source=source,
        n_loaded=n_loaded,
        steps=steps,
    )
    analyses = [AnalysisCount("Cox model", population.n_included, 718)]
    small = [n for e in flow.exits for _, n in e.reasons if 0 < n < 10]
    assert len(small) == 9  # the eight implausible-value reasons and one inclusion criterion

    disclosed = disclose_flow(flow, K10, analyses)
    # never the file's or the dataset's name
    assert disclosed.boxes[0].label == "Records read in the source file"
    assert "tutorial_cohort" not in json.dumps(disclosed.to_dict())
    assert [b.n for b in disclosed.boxes] == [1206, 1190, 928]
    implausible, inclusion = disclosed.transitions
    assert (
        implausible.excluded == 16 and implausible.reasons == ()
    )  # eight reasons of 1 to 3: the total only
    assert inclusion.excluded == 262
    together = inclusion.reasons[-1]
    # the reason of 5 with the next smallest, named neither
    assert together.names == () and len(together.parts) == 2 and together.n >= 10
    assert sum(r.n for r in inclusion.reasons) == 262
    assert all(K10.shows(n) for n in _numbers(disclosed))
    # the number analysed and the rows with a missing value are exact
    assert [(a.label, a.n, a.missing, a.left_out) for a in disclosed.analyses] == [
        ("Cox model", 718, 210, 0)
    ]
    assert recoverable(flow, disclosed, analyses) == []

    five = disclose_flow(flow, DisclosurePolicy(5))
    assert five.transitions[0].reasons == () and len(five.transitions[1].reasons) > len(
        inclusion.reasons
    )


# ----------------------------------------------------------------------------- the analyses


def test_an_analysis_that_left_out_a_few_rows_for_another_reason_is_withheld() -> None:
    """A comparison of two groups among three: the population (960) and the rows it analysed (957) are exact, so the
    third group's 3 rows came back by subtraction. They are no missing values."""
    flow = _flow(1000, ({"age": 40}, 0))
    compared = AnalysisCount(
        "Group comparison", 960, 957, n_left_out=3, left_out="not in the compared groups"
    )
    missing = AnalysisCount(
        "Cox model", 960, 957
    )  # 3 rows with a missing value: exact, as reporting asks
    large = AnalysisCount(
        "Group comparison", 960, 900, n_left_out=40, left_out="not in the compared groups"
    )
    disclosed = disclose_flow(flow, K10, [compared, missing, large])
    assert [
        (a.label, a.n, a.missing, a.left_out, a.left_out_words) for a in disclosed.analyses
    ] == [
        ("Cox model", 957, 3, 0, ""),
        ("Group comparison", 900, 20, 40, "not in the compared groups"),
    ]
    assert disclosed.withheld == (
        "1 analysis withheld: fewer than 10 rows were left out for another reason than a missing value",
    )
    assert recoverable(flow, disclosed, [compared, missing, large]) == []
    # shown as it is, the analysis gives its 3 rows back: the check finds them
    from core.disclosure import DisclosedAnalysis, DisclosedFlow

    careless = DisclosedFlow(
        10,
        disclosed.boxes,
        disclosed.transitions,
        (DisclosedAnalysis("Group comparison", 957, None, 0, 3, "not in the compared groups"),),
    )
    assert recoverable(flow, careless, [compared]) == ["left_out:0"]


def test_a_pooled_analysis_and_one_of_several_variables_keep_their_numbers() -> None:
    flow = _flow(1000, ({"age": 72}, 0))
    pooled = AnalysisCount("Cox model, pooled", 928, imputations=20)
    several = AnalysisCount("Group comparison", 928, 778, n_most=928)
    disclosed = disclose_flow(flow, K10, [pooled, several])
    assert [(a.n, a.n_most, a.missing, a.imputations) for a in disclosed.analyses] == [
        (928, None, 0, 20),
        (778, 928, 150, None),
    ]


def test_an_analysis_computed_on_another_population_is_withheld() -> None:
    """A Cox model computed when only the first criterion was ticked (960 rows), shown under the flow of both (957):
    its number beside the population's gave the second criterion's 3 rows, which the flow had merged away."""
    flow = _flow(1000, ({"a": 40}, 0), ({"b": 3}, 0))
    assert flow.main[-1].n == 957
    before = AnalysisCount("Cox model", 960, 960)
    now = AnalysisCount("Cox model", 957, 950)
    disclosed = disclose_flow(flow, K10, [before, now])
    assert [b.n for b in disclosed.boxes] == [1000, 957]  # the two exclusions as one
    assert [(a.label, a.n, a.missing) for a in disclosed.analyses] == [("Cox model", 950, 7)]
    assert disclosed.withheld == (
        "1 analysis withheld: computed on another population than this one (run again to include)",
    )
    assert recoverable(flow, disclosed, [before, now]) == []


def test_a_pooled_analysis_counts_the_rows_it_left_out() -> None:
    """Pooled curves of the groups large enough to draw: the analysis used the population less the 15 rows of the
    others, and read "960 analysed, -15 missing"."""
    flow = _flow(1000, ({"age": 40}, 0))
    pooled = AnalysisCount(
        "Kaplan-Meier, pooled",
        960,
        imputations=5,
        n_left_out=15,
        left_out="in a group too small to draw",
    )
    (shown_as,) = disclose_flow(flow, K10, [pooled]).analyses
    assert (shown_as.n, shown_as.missing, shown_as.left_out, shown_as.imputations) == (
        945,
        0,
        15,
        5,
    )
    # the rows its datasets read, when they were counted: 20 lack the event, which no imputation fills
    read = AnalysisCount(
        "Kaplan-Meier, pooled",
        960,
        925,
        imputations=5,
        n_left_out=15,
        left_out="in a group too small to draw",
    )
    (shown_as,) = disclose_flow(flow, K10, [read]).analyses
    assert (shown_as.n, shown_as.missing, shown_as.left_out) == (925, 20, 15)
    # never more than the rows it did not leave out: no missing value is counted below zero
    over = AnalysisCount("Cox model, pooled", 960, 960, imputations=5, n_left_out=15, left_out="x")
    (shown_as,) = disclose_flow(flow, K10, [over]).analyses
    assert (shown_as.n, shown_as.missing, shown_as.left_out) == (945, 0, 15)


def test_exits_listed_for_as_many_rows_are_not_listed() -> None:
    """At k = 5, five steps that each removed one row merge into "5 excluded" with five exits listed: an exit is
    listed only when it holds a row, so each holds one."""
    flow = _flow(1000, (1, 0), (1, 0), (1, 0), (1, 0), (1, 0))
    disclosed = disclose_flow(flow, DisclosurePolicy(5))
    assert [b.n for b in disclosed.boxes] == [995] and disclosed.transitions == ()
    assert disclosed.withheld == (
        "the steps of the flow are withheld: as listed, they would give back a count below 5",
    )
    # six rows over the five exits: one of them holds two, and nothing says which
    six = disclose_flow(_flow(1000, (1, 0), (1, 0), (1, 0), (1, 0), (2, 0)), DisclosurePolicy(5))
    assert [b.n for b in six.boxes] == [1000, 994] and six.transitions[0].excluded == 6
    assert six.withheld == ()


def test_the_closure_takes_what_the_results_add_to_the_flow() -> None:
    """Three comparisons of two groups among four, each leaving out 10 rows or more: the flow alone gives nothing
    back, and a reader who knows which groups each compared (its settings) gets the group of 3. The writer of a
    package adds what each result shows to the flow's closure before it writes anything."""
    from core.disclosure import Closure, flow_closure

    flow = _flow(1040, ({"age": 40}, 0))
    sizes = {"alpha": 600, "beta": 385, "gamma": 12, "delta": 3}
    compared = [("alpha", "beta"), ("gamma", "beta"), ("alpha", "gamma")]
    analyses = [
        AnalysisCount(
            f"Group comparison {place}",
            1000,
            sizes[a] + sizes[b],
            n_left_out=1000 - sizes[a] - sizes[b],
            left_out="not in the compared groups",
        )
        for place, (a, b) in enumerate(compared)
    ]
    disclosed = disclose_flow(flow, K10, analyses)
    assert [a.n for a in disclosed.analyses] == [985, 397, 612]
    assert recoverable(flow, disclosed, analyses) == []
    closure = flow_closure(flow, disclosed, analyses)
    for name, rows in sizes.items():
        closure.unknown(f"group:{name}", rows)
    closure.equation({f"group:{name}": 1 for name in sizes}, 1000)  # the population
    for a, b in compared:
        closure.equation({f"group:{a}": 1, f"group:{b}": 1}, sizes[a] + sizes[b])
    assert closure.comes_back(10) == ["group:delta"]
    assert {
        "group:gamma",
        "group:delta",
    } <= closure.found()  # 12 is worked out too: it is not below k
    # a reader is told nothing false, and a count has one value
    with pytest.raises(ValueError, match="do not satisfy"):
        closure.equation({"group:alpha": 1}, 601)
    with pytest.raises(ValueError, match="two values"):
        closure.unknown("group:alpha", 599)
    # what the reader knows a count holds at least narrows the others: two exits listed for two rows
    listed, unlisted = Closure(), Closure()
    for name in ("a", "b"):
        listed.unknown(name, 1, at_least=1)
        unlisted.unknown(name, 1)
    listed.equation({"a": 1, "b": 1}, 2)
    unlisted.equation({"a": 1, "b": 1}, 2)
    assert listed.comes_back(5) == ["a", "b"] and unlisted.comes_back(5) == []


def test_two_groupings_give_back_a_cell_that_neither_shows() -> None:
    """Sites Alpha (600 rows) and Beta (385) of one country, Gamma (30) of another, and Delta (3) of the first:
    the curves by site show Alpha and Beta, the curves by country show both countries, and the first country's
    988 rows, less the two sites shown in it, are Delta's 3. A closure with one unknown per group of each variable
    does not see it; over the cells of the two variables crossed, where each group shown is a sum of cells, it
    does."""
    from core.disclosure import Closure

    cells = {
        ("Alpha", "first"): 600,
        ("Beta", "first"): 385,
        ("Gamma", "second"): 30,
        ("Delta", "first"): 3,
    }
    sites = {site: rows for (site, _), rows in cells.items()}
    countries = {"first": 988, "second": 30}
    apart = Closure()
    for site, rows in sites.items():
        apart.unknown(f"site:{site}", rows)
    for country, rows in countries.items():
        apart.unknown(f"country:{country}", rows)
    # the sites together are the countries together; Alpha, Beta and the two countries are shown
    apart.equation(
        {**{f"site:{s}": 1 for s in sites}, **{f"country:{c}": -1 for c in countries}}, 0
    )
    for name in ("site:Alpha", "site:Beta", "country:first", "country:second"):
        apart.shown(name)
    assert apart.comes_back(10) == []
    crossed = Closure()
    for (site, country), rows in cells.items():
        crossed.unknown(f"cell:{site}|{country}", rows)
    for site in ("Alpha", "Beta"):
        crossed.equation({f"cell:{s}|{c}": 1 for s, c in cells if s == site}, sites[site])
    for country, rows in countries.items():
        crossed.equation({f"cell:{s}|{c}": 1 for s, c in cells if c == country}, rows)
    assert crossed.comes_back(10) == ["cell:Delta|first"]


# ----------------------------------------------------------------------------- flows that cannot be shared


def test_a_flow_whose_numbers_do_not_add_up_is_refused_first() -> None:
    """A trace that missed a change of the rows gives a flow with a note and a box that does not follow: the check
    of what a reader can work out blamed itself for it."""
    main = (
        FlowBox("loaded", "Loaded: cohort", 1000, LOADED),
        FlowBox("dataset", "Dataset: cohort", 990, DATA),
        FlowBox("population", "Analysis population", 950, POPULATION),
    )
    leaving = FlowExit("stage:a", "dataset", "Excluded: a", 40)
    broken = ParticipantFlow(main, (leaving,), notes=("The dataset has 990 rows, not 1,000",))
    with pytest.raises(ValueError, match="needs a trace that records every change of the rows"):
        disclose_flow(broken, K10)


def test_a_box_is_called_by_its_place_never_by_a_file_s_name() -> None:
    from core.attrition import SOURCE

    main = (
        FlowBox("source", "Records read in patients_dupont.csv", 1010, SOURCE),
        FlowBox("loaded", "Loaded: dupont_cohort", 1000, LOADED),
        FlowBox("population", "Analysis population", 960, POPULATION),
    )
    exits = (
        FlowExit("skipped", "source", "Malformed lines skipped", 10),
        FlowExit("stage:a", "loaded", "Excluded: a", 40),
    )
    disclosed = disclose_flow(ParticipantFlow(main, exits), K10)
    assert [b.label for b in disclosed.boxes] == [
        "Records read in the source file",
        "Records loaded",
        "Analysis population",
    ]
    assert "dupont" not in json.dumps(disclosed.to_dict())


# ----------------------------------------------------------------------------- results

SIDES = "rows, rows with the event or rows without it"
CELL = f"fewer than 10 {SIDES} in a cell compared, or an estimate without a maximum"
REFERENCE = f"its reference cell holds fewer than 10 {SIDES}"
COVER = f"the model's counts, less the cells shown, would give back fewer than 10 {SIDES}"
FIT = "the numbers of the fit as a whole are withheld with them: each would give back what the terms withheld hold"
ADVICE = (
    "join a small level or group to another, or take its rows out of the analysis population by a criterion, and "
    "run the analysis again"
)
MODEL_WITHHELD = f"the model is withheld: fewer than 10 {SIDES}, or residual degrees of freedom"
NO_TERM = f"the model is withheld: none of its terms can be shown ({ADVICE})"
NO_COMPARISON = (
    "the model is withheld: no term of its comparison can be shown, and what is left is its adjustment "
    f"({ADVICE})"
)
DROPS_OUT = (
    "the model is withheld: one of its terms compares a cell of fewer than 10 rows, or one where nobody or "
    "everybody had the event, or has no maximum: the terms left would be those of the model fitted without those "
    f"rows ({ADVICE})"
)
TWO_SIDES = "rows or rows with the event"
CELL_OF_TWO = f"fewer than 10 {TWO_SIDES} in a cell compared, or an estimate without a maximum"
COVER_OF_TWO = (
    f"the model's counts, less the cells shown, would give back fewer than 10 {TWO_SIDES}"
)
# in a Cox or a count model a cell where everybody had the event has a maximum, and stays in the fit
DROPS_OUT_NOBODY = DROPS_OUT.replace("nobody or everybody", "nobody")
A_TABLE = (
    "the model is withheld: its terms all compare categories, so that their numbers are the cells of a table, and "
    f"one of them cannot be shown ({ADVICE})"
)
# the slope of a numeric covariate of a few values has no level to join: its own sentences
A_VALUE = f"one of its values holds fewer than 10 {SIDES} (cap or group its values)"
A_VALUE_OF_TWO = f"one of its values holds fewer than 10 {TWO_SIDES} (cap or group its values)"
CAP = "cap or group the values of the covariate, and run the analysis again"
OFF_ITS_VALUE = (
    "the model is withheld: the term of a numeric covariate is withheld, and the rows off its commonest value are "
    "fewer than 10, or none or all of them had the event, or the term has no maximum: the terms left would be "
    "those of the model fitted without those rows (leave the covariate out of the model, or take those rows out "
    "of the analysis population by a criterion, and run the analysis again)"
)
OFF_ITS_VALUE_NONE = OFF_ITS_VALUE.replace("none or all of them", "none of them")
GROUP_WITHHELD = "1 group withheld: fewer than 10 rows, events or rows without the event"
COVER_WITHHELD = (
    "1 more group withheld: the number analysed, less the groups shown, would give back fewer than 10 rows, "
    "events or rows without the event"
)
TOGETHER = f"to share the other groups together: {ADVICE}"
LOG_RANK = "the log-rank test is withheld: it compares a group that is"
NOT_IN_THE_FLOW = "the result is withheld: the participant flow does not show its analysis"
FEW_LEFT_OUT = "the result is withheld: fewer than 10 rows were left out for another reason than a missing value"
NOT_ITS_ROWS = (
    "the result is withheld: its rows are not those the participant flow shows for its analysis"
)
# what a fit says of all its rows together: written only when no term is withheld
FIT_NAMES = {
    "converged", "llf", "aic", "pseudo_r2", "auc", "auc_ci_low", "auc_ci_high", "concordance", "dispersion",
    "converged_in",
}  # fmt: skip
CURVE_NAMES = {
    "rows", "events", "confidence", "points_every", "time_step", "followed_until", "median", "median_lower",
    "median_upper", "later_than_followed_until", "times", "survival", "lower", "upper", "time_unit_from",
    "decimals",
}  # fmt: skip


def _cohort(n: int = 400, seed: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "age": rng.normal(62, 9, n).round(1),
            "arm": rng.choice(["control", "drug"], n),
            "site": rng.choice(["Beta", "Gamma"], n),
        }
    )
    df["died"] = (rng.random(n) < 0.4).astype(int)
    df["days"] = rng.exponential(np.exp(6.0 + 0.3 * (df["arm"] == "drug"))).round() + 1
    return df


def _sites_of(sizes: dict[str, int], seed: int = 3) -> pd.DataFrame:
    """Sites of these sizes; everybody of Delta died."""
    rng = np.random.default_rng(seed)
    n = sum(sizes.values())
    df = pd.DataFrame(
        {
            "age": rng.normal(62, 9, n).round(1),
            "site": [name for name, rows in sizes.items() for _ in range(rows)],
        }
    )
    df["died"] = (rng.random(n) < 0.4).astype(int)
    df.loc[df["site"] == "Delta", "died"] = 1
    df["days"] = rng.exponential(400, n).round() + 1
    df["visits"] = rng.poisson(2, n)
    return df


def _three_sites(delta: int = 3) -> pd.DataFrame:
    """Two large sites and one of ``delta`` participants, who all died."""
    return _sites_of({"Alpha": 600, "Beta": 400 - delta, "Delta": delta})


def _four_sites() -> pd.DataFrame:
    return _sites_of({"Alpha": 450, "Beta": 397, "Gamma": 150, "Delta": 3})


def _written(disclosed) -> str:
    return "" if disclosed.record is None else disclosed.record.to_json()


def _spec_of(record, compare: str | None = None):
    """A spec of the record's kind: a model's comparison is read in it."""
    from biostats.plan import specify

    roles = {
        "survival": {"time": "days", "event": "died"},
        "cox": {"time": "days", "event": "died", "compare": compare},
        "logistic": {"outcome": "died", "compare": compare},
        "count": {"count": "visits", "compare": compare},
    }.get(record.kind, {})
    return specify(record.kind, pooled=record.pooled, **roles)


def _in_the_flow(record):
    """The record's analysis as a flow shows it: the rows it read and those it left out."""
    from core.disclosure import DisclosedAnalysis

    left_out = sum(
        int(n.value or 0)
        for p in record.parts
        for n in p.numbers
        if n.name in ("rows_skipped", "rows_left_out")
    )
    read = [
        n.value for p in record.parts if p.id == "data" for n in p.numbers if n.name == "rows_min"
    ]
    if record.kind == "survival" and record.pooled and read:
        rows = read[0]  # the fewest rows that one of its imputed datasets read
    elif record.kind == "survival":
        rows = sum(
            p.support.get("rows", p.support.get("rows_min", 0))
            for p in record.parts
            if p.id.startswith("curve#")
        )
    else:
        model = next((p for p in record.parts if p.id == "model"), None)
        rows = 0 if model is None else model.support.get("rows", model.support.get("rows_min", 0))
    return DisclosedAnalysis("analysis", rows, left_out=left_out)


def _disclose(record, policy: DisclosurePolicy = K10, *, compare: str | None = None):
    from core.disclosure import disclose_record

    return disclose_record(
        record, policy, spec=_spec_of(record, compare), analysis=_in_the_flow(record)
    )


def _ids(disclosed) -> list[str]:
    return [part.id for part in disclosed.record.parts]


def test_a_term_of_one_participant_coded_1_withholds_the_model() -> None:
    """One participant is coded 1 and died: the coefficient (+22.9 in the review's probe) gives their outcome, and
    the term carried the whole model's counts. With that term alone withheld, the terms written were those of the
    model fitted without the participant: beside the same model without the covariate, their difference gave the
    participant's outcome, arm and age."""
    from biostats import plan
    from biostats.regression import logistic_regression

    df = _cohort()
    df["transplanted"] = 0
    df.loc[df.index[7], ["transplanted", "died"]] = [1, 1]
    fitted = logistic_regression(df, "died", ["transplanted", "arm", "age"])
    record = plan.record("logistic", fitted)
    assert record is not None and record.part("term:transplanted").support["rows"] == 1
    # what the rule rests on: a participant alone in a cell is fitted by the cell's own parameter
    with_them = fitted.table.set_index("term")["coef"]
    without = logistic_regression(df.drop(index=df.index[7]), "died", ["arm", "age"])
    others = without.table.set_index("term")["coef"]
    assert with_them[others.index].to_numpy() == pytest.approx(others.to_numpy(), abs=1e-6)
    disclosed = _disclose(record, compare="arm")
    assert disclosed.record is None and disclosed.withheld == (DROPS_OUT,)
    # at k = 5 too: one row is below any k
    assert _disclose(record, DisclosurePolicy(5), compare="arm").record is None
    # as the comparison of the model: what is left is the adjustment, whose coefficients are no effects
    compared = _disclose(record, compare="transplanted")
    assert compared.record is None and compared.withheld == (NO_COMPARISON,)


def test_a_term_of_a_cell_with_a_few_events_is_withheld_alone() -> None:
    """Twelve participants coded 1, three of whom died: the cell holds k rows and both outcomes, so its rows stay in
    the fit, and its term alone is withheld, with the numbers of the fit as a whole."""
    from biostats import plan
    from biostats.regression import logistic_regression

    df = _cohort()
    df["transplanted"] = 0
    df.loc[df.index[:12], "transplanted"] = 1
    df.loc[df.index[:12], "died"] = [1, 1, 1] + [0] * 9
    record = plan.record(
        "logistic", logistic_regression(df, "died", ["transplanted", "arm", "age"])
    )
    assert record is not None
    assert record.part("term:transplanted").support == {"rows": 12, "events": 3, "non_events": 9}
    disclosed = _disclose(record, compare="arm")
    assert disclosed.record is not None
    ids = _ids(disclosed)
    assert "term:transplanted" not in ids and "reference:transplanted" not in ids
    assert {"model", "reference:arm", "term:arm#1", "term:age"} <= set(ids)
    assert disclosed.withheld == (f"1 term of transplanted withheld: {CELL}", FIT)
    # at k = 5 the cell still lacks its events
    assert "term:transplanted" not in _ids(_disclose(record, DisclosurePolicy(5), compare="arm"))
    # as the comparison of the model: what is left is the adjustment, whose coefficients are no effects
    compared = _disclose(record, compare="transplanted")
    assert compared.record is None and compared.withheld == (NO_COMPARISON,)


def test_a_rare_reference_level_withholds_the_model() -> None:
    """The first level in sorted order is the reference: held by 2 rows who both died, it made every other level's
    coefficient (-22.8, -23.0) give their outcome, while those terms passed on their own counts."""
    from biostats import plan
    from biostats.regression import logistic_regression

    df = _cohort()
    df.loc[df.index[:2], ["site", "died"]] = ["Aaa-CANARY", 1]
    record = plan.record("logistic", logistic_regression(df, "died", ["site", "age"]))
    assert record is not None and record.part("reference:site").support["rows"] == 2
    disclosed = _disclose(record)
    # the two rows drop out of the fit: the age term left would be that of the model fitted without them
    assert disclosed.record is None and disclosed.withheld == (DROPS_OUT,)
    assert "CANARY" not in " ".join(disclosed.withheld)
    assert _disclose(record, compare="site").withheld == (NO_COMPARISON,)


def test_a_reference_level_with_a_few_events_withholds_every_term_of_its_factor() -> None:
    """A reference of 12 rows, two of whom died, stays in the fit: no term of its factor can be read without it,
    and the other terms of the model are written."""
    from biostats import plan
    from biostats.regression import logistic_regression

    df = _cohort()
    df.loc[df.index[:12], "site"] = "Aaa-CANARY"
    df.loc[df.index[:12], "died"] = [1, 1] + [0] * 10
    record = plan.record("logistic", logistic_regression(df, "died", ["site", "age"]))
    assert record is not None
    assert record.part("reference:site").support == {"rows": 12, "events": 2, "non_events": 10}
    disclosed = _disclose(record)
    assert [i for i in _ids(disclosed) if "site" in i] == [] and "term:age" in _ids(disclosed)
    # said for what it is: the exporter can join the level, or name the levels so that a larger one comes first
    assert disclosed.withheld == (f"2 terms of site withheld: {REFERENCE}", FIT)
    assert "CANARY" not in _written(disclosed) and "Beta" not in _written(disclosed)
    assert "CANARY" not in " ".join(disclosed.withheld)
    assert _disclose(record, compare="site").withheld == (NO_COMPARISON,)


def test_a_model_shows_only_the_numbers_its_rule_names() -> None:
    from biostats import plan
    from biostats.regression import logistic_regression
    from biostats.survival import cox_regression

    df = _cohort()
    record = plan.record("cox", cox_regression(df, "days", "died", ["arm", "site", "age"]))
    disclosed = _disclose(record, compare="arm")
    assert disclosed.withheld == () and disclosed.record is not None
    shown = disclosed.record
    model = shown.part("model")
    assert [n.name for n in model.numbers] == [
        "rows",
        "events",
        "ties",
        "df_resid",
        "converged",
        "concordance",
        "parameters",
        "events_per_parameter",
        "sparse",
    ]  # fmt: skip  (not the rows left out for a negative time: a small count of participants)
    # what the application's warning says, as numbers: no warning's wording travels
    events = record.part("model").support["events"]
    assert model.number("parameters").value == 3
    assert model.number("events_per_parameter").value == events / 3
    assert model.number("sparse").value is False
    assert shown.part("term:arm#1").labels == {
        "variable": "arm",
        "level": "drug",
    }  # a shown level keeps its name
    # the model's own counts travel; the counts behind a term do not
    assert model.support == record.part("model").support
    assert all(part.support == {} for part in shown.parts if part.id != "model")
    assert shown.warnings == ()  # a warning's wording never travels
    assert not any(n.fine for part in shown.parts for n in part.numbers)
    # 12 events for 8 parameters, in a logistic model: written, and said to be sparse
    noise = {f"x{i}": np.random.default_rng(i).normal(size=len(df)) for i in range(8)}
    wide = df.assign(died=0, **noise)
    wide.loc[wide.index[::33][:12], "died"] = 1
    sparse = _disclose(
        plan.record("logistic", logistic_regression(wide, "died", list(noise)))
    ).record.part("model")
    assert sparse.number("events_per_parameter").value == 1.5 and sparse.number("sparse").value
    # a model on a handful of events is withheld whole
    few = df.assign(died=0)
    few.loc[few.index[:6], "died"] = 1
    small = _disclose(plan.record("cox", cox_regression(few, "days", "died", ["age"])))
    assert small.record is None and small.withheld == (MODEL_WITHHELD,)


def test_a_withheld_level_does_not_come_back_from_the_counts_shown() -> None:
    """A site of 15 participants, 3 of whom died: its term was withheld, and the model's rows, events and rows
    without the event, less the reference's and the shown term's, gave 15, 3 and 12. With a site of 3 participants
    who all died, the same subtraction gave the outcome of each of the three."""
    from biostats import plan
    from biostats.regression import logistic_regression
    from biostats.survival import cox_regression

    df = _three_sites(delta=15)
    df.loc[df["site"] == "Delta", "died"] = [1, 1, 1] + [0] * 12
    for kind, result, without in (
        ("logistic", logistic_regression(df, "died", ["site", "age"]), "non_events"),
        ("cox", cox_regression(df, "days", "died", ["site", "age"]), "censored"),
    ):
        record = plan.record(kind, result)
        assert record is not None
        # as computed, the record gives them: this is what must not travel
        left = {
            name: record.part("model").support[name]
            - record.part("reference:site").support[name]
            - record.part("term:site#1").support[name]
            for name in ("rows", "events", without)
        }
        assert left == {"rows": 15, "events": 3, without: 12}
        disclosed = _disclose(record)  # the site is an adjustment here
        # no count travels beside a term, and the other level is withheld with the small one: nothing says what
        # the 15 rows hold
        # a Cox term's cell is judged on its rows and its rows with the event, and says so
        cell, cover = (CELL, COVER) if kind == "logistic" else (CELL_OF_TWO, COVER_OF_TWO)
        assert disclosed.withheld == (
            f"1 term of site withheld: {cell}",
            f"1 more term of site withheld: {cover}",
            FIT,
        )
        shown = disclosed.record
        assert [p.id for p in shown.parts if p.id.startswith(("reference:", "term:"))] == [
            "term:age"
        ]
        assert all(part.support == {} for part in shown.parts if part.id != "model")
        assert not FIT_NAMES & {n.name for n in shown.part("model").numbers}
        assert "Delta" not in _written(disclosed) and "Beta" not in _written(disclosed)
        assert _disclose(record, compare="site").withheld == (NO_COMPARISON,)
    # three participants who all died drop out of the fit: the age term left would be that of the model fitted
    # without them, and the model is withheld whole
    three = _three_sites()
    for kind, result in (
        ("logistic", logistic_regression(three, "died", ["site", "age"])),
        ("cox", cox_regression(three, "days", "died", ["site", "age"])),
    ):
        gone = _disclose(plan.record(kind, result))
        dropped = DROPS_OUT if kind == "logistic" else DROPS_OUT_NOBODY
        assert gone.record is None and gone.withheld == (dropped,), kind
        assert _disclose(plan.record(kind, result), compare="site").withheld == (NO_COMPARISON,)


def test_a_model_of_categories_alone_gives_no_cell_back() -> None:
    """In a model of one factor, the coefficient of a level is the log odds ratio of its cell and the reference's,
    and the square of its standard error the sum of their four reciprocals: with one term shown, a single set of
    whole numbers gave both cells, and the model's totals then left the 3 participants of Delta, who all died. In a
    model of two factors a cell withheld of one came back the same way through the terms of the other."""
    import math

    from biostats import plan
    from biostats.regression import count_regression, logistic_regression
    from biostats.survival import cox_regression

    crude = plan.record("logistic", logistic_regression(_three_sites(), "died", ["site"]))
    assert crude is not None
    beta = crude.part("term:site#1")
    cells = (crude.part("reference:site").support, beta.support)
    odds = [cell["events"] / cell["non_events"] for cell in cells]
    assert beta.number("coef").value == pytest.approx(math.log(odds[1] / odds[0]))
    assert beta.number("se").value ** 2 == pytest.approx(
        sum(1 / cell[name] for cell in cells for name in ("events", "non_events"))
    )
    # Beta is withheld with Delta, and a model none of whose terms is written holds nothing a reader can use
    disclosed = _disclose(crude, compare="site")
    assert disclosed.record is None and disclosed.withheld == (NO_TERM,)
    # with a fourth site, the smaller of the two others covers Delta and Beta's term could be written: but the
    # three rows of Delta drop out of the fit, in each kind of model
    four = _four_sites()
    for key, result in (
        ("logistic", logistic_regression(four, "died", ["site"])),
        ("cox", cox_regression(four, "days", "died", ["site"])),
        ("count", count_regression(four, "visits", ["site"])),
    ):
        record = plan.record(key, result)
        assert record is not None
        disclosed = _disclose(record, compare="site")
        dropped = DROPS_OUT if key == "logistic" else DROPS_OUT_NOBODY
        assert disclosed.record is None and disclosed.withheld == (dropped,), key
    # a fourth site of 15 rows, 3 of whom died, stays in the fit. Gamma covers it, and what the totals would leave
    # beside Beta's term is Gamma and Delta together: but every term written would compare categories, so the model
    # is a table, and is withheld whole
    df = _sites_of({"Alpha": 450, "Beta": 385, "Gamma": 150, "Delta": 15})
    df.loc[df["site"] == "Delta", "died"] = [1, 1, 1] + [0] * 12
    table = _disclose(
        plan.record("logistic", logistic_regression(df, "died", ["site"])), compare="site"
    )
    assert table.record is None and table.withheld == (A_TABLE,)
    # two variables of categories, the comparison being the other one: the same
    df["sex"] = np.where(np.arange(len(df)) % 2 == 0, "F", "M")
    two = plan.record("logistic", logistic_regression(df, "died", ["site", "sex"]))
    assert _disclose(two, compare="sex").withheld == (A_TABLE,)
    # and a numeric covariate of a few values is one of them
    df["score"] = np.arange(len(df)) % 3
    scored = plan.record("logistic", logistic_regression(df, "died", ["site", "score"]))
    assert _disclose(scored, compare="score").withheld == (A_TABLE,)
    # beside a covariate of many values, the terms are no cells of a table: Beta's is written
    for key, result, counted in (
        (
            "logistic",
            logistic_regression(df, "died", ["site", "age"]),
            ("rows", "events", "non_events"),
        ),
        (
            "cox",
            cox_regression(df, "days", "died", ["site", "age"], diagnostics=False),
            ("rows", "events", "censored"),
        ),
    ):
        record = plan.record(key, result)
        assert record is not None
        shown = _disclose(record, compare="site")
        assert [i for i in _ids(shown) if i.startswith(("reference:", "term:"))] == [
            "reference:site",
            "term:age",
            "term:site#1",
        ], key
        cell, cover = (CELL, COVER) if key == "logistic" else (CELL_OF_TWO, COVER_OF_TWO)
        assert shown.withheld == (
            f"1 term of site withheld: {cell}",
            f"1 more term of site withheld: {cover}",
            FIT,
        )
        left = {
            name: record.part("model").support[name]
            - record.part("reference:site").support[name]
            - record.part("term:site#1").support[name]
            for name in counted
        }
        assert left["rows"] == 165 and min(left.values()) >= 10, key
        assert not FIT_NAMES & {n.name for n in shown.record.part("model").numbers}, key
        assert "Delta" not in _written(shown) and "Gamma" not in _written(shown)


def test_the_numbers_of_a_fit_travel_only_when_no_term_is_withheld() -> None:
    """One participant coded 1, who died: the term was withheld and the model written with its log-likelihood, its
    pseudo-R2 and its AUC, each of which says whether that participant had the event. The AUC, less one half, times
    twice the events and the rows without, is the count of the other outcome."""
    from biostats import plan
    from biostats.regression import logistic_regression

    df = _cohort()
    df["transplanted"] = 0
    df.loc[df.index[7], ["transplanted", "died"]] = [1, 1]
    alone = plan.record("logistic", logistic_regression(df, "died", ["transplanted"]))
    assert alone is not None
    model = alone.part("model")
    events, without = model.support["events"], model.support["non_events"]
    assert (model.number("auc").value - 0.5) * 2 * events * without == pytest.approx(without)
    assert _disclose(alone).record is None and _disclose(alone).withheld == (NO_TERM,)
    # adjusted for age, nothing of it is written either: the age term would be that of the other 399 rows
    one = plan.record("logistic", logistic_regression(df, "died", ["transplanted", "age"]))
    assert _disclose(one).withheld == (DROPS_OUT,)
    # twelve participants coded 1, three of whom died: the model is written with its totals and its age term, and
    # none of the fit's numbers
    df = _cohort()
    df["transplanted"] = 0
    df.loc[df.index[:12], "transplanted"] = 1
    df.loc[df.index[:12], "died"] = [1, 1, 1] + [0] * 9
    adjusted = plan.record("logistic", logistic_regression(df, "died", ["transplanted", "age"]))
    shown = _disclose(adjusted).record.part("model")
    assert [n.name for n in shown.numbers] == [
        "family", "rows", "events", "df_resid", "parameters", "events_per_parameter", "sparse",
    ]  # fmt: skip
    # a model none of whose terms is withheld keeps them
    whole = _disclose(plan.record("logistic", logistic_regression(df, "died", ["arm", "age"])))
    assert whole.withheld == ()
    assert {"converged", "llf", "aic", "pseudo_r2", "auc", "auc_ci_low", "auc_ci_high"} <= {
        n.name for n in whole.record.part("model").numbers
    }


def test_a_numeric_covariate_that_a_few_rows_carry_is_read_as_cells() -> None:
    """A dose that is 0 for 398 rows, 1 and 3 for two participants who both died: its term carried the model's
    counts and was written, +22.8: everybody with a dose died. A score of 0, 1 or 2 that two participants hold at 2
    was written the same way, and the slope, its standard error and the likelihood of the crude model left one
    table: those two, and their outcome."""
    from biostats import plan
    from biostats.regression import count_regression, logistic_regression
    from biostats.survival import cox_regression

    df = _cohort()
    df["dose"] = 0.0
    df.loc[df.index[0], ["dose", "died"]] = [1.0, 1]
    df.loc[df.index[1], ["dose", "died"]] = [3.0, 1]
    record = plan.record("logistic", logistic_regression(df, "died", ["dose", "age"]))
    assert record is not None
    dose = record.part("term:dose")
    assert dose.labels == {"variable": "dose", "per": "unit"}  # a slope, not a level
    # each of its values is a cell: the smallest of each count over the values above the first; and, for the
    # model as a whole, the rows off its commonest value
    assert dose.support == {
        "rows": 1, "events": 1, "non_events": 0, "rows_off": 2, "events_off": 2, "non_events_off": 0,
    }  # fmt: skip
    reference = record.part("reference:dose")
    assert reference.labels == {"variable": "dose", "level": "0", "of": "values"}
    assert reference.support["rows"] == 398
    # two rows off the value that all the others hold: they are what the model with the dose and the model
    # without it differ by
    assert _disclose(record).withheld == (OFF_ITS_VALUE,)
    # six participants with an earlier event, five of whom died: the same in the three kinds of model
    df = _cohort()
    df["prior"] = 0
    df.loc[df.index[:6], "prior"] = [1, 1, 2, 1, 3, 1]
    df.loc[df.index[:6], "died"] = [1, 1, 1, 1, 1, 0]
    df["visits"] = np.random.default_rng(2).poisson(2, len(df))
    for key, result in (
        ("logistic", logistic_regression(df, "died", ["prior", "age"])),
        ("cox", cox_regression(df, "days", "died", ["prior", "age"], diagnostics=False)),
        ("count", count_regression(df, "visits", ["prior", "age"])),
    ):
        record = plan.record(key, result)
        assert record is not None and record.part("term:prior").support["rows"] == 1, key
        assert record.part("term:prior").support["rows_off"] == 6, key
        dropped = OFF_ITS_VALUE if key == "logistic" else OFF_ITS_VALUE_NONE
        assert _disclose(record).withheld == (dropped,), key
    # a score of 0, 1 or 2 that two participants hold at 2: crude, and adjusted
    scored = _cohort()
    scored["score"] = np.arange(len(scored)) % 2
    scored.loc[scored.index[:2], "score"] = 2
    crude = plan.record("logistic", logistic_regression(scored, "died", ["score"]))
    assert crude is not None and crude.part("term:score").support["rows"] == 2
    assert _disclose(crude).withheld == (NO_TERM.replace(ADVICE, CAP),)
    # adjusted for age, the slope is withheld and the model is written: a slope has one parameter for the
    # three values, and the two rows at 2 stay in the fit (201 rows are off the commonest value)
    adjusted = plan.record("logistic", logistic_regression(scored, "died", ["score", "age"]))
    assert adjusted is not None and adjusted.part("term:score").support["rows_off"] == 201
    shown = _disclose(adjusted)
    assert _ids(shown) == ["model", "term:age"]
    assert shown.withheld == (f"1 term of score withheld: {A_VALUE}", FIT)
    # a score whose every value many rows hold: its slope is written, per unit, and no cell is called its reference
    scored["score"] = np.arange(len(scored)) % 4
    wide = _disclose(plan.record("logistic", logistic_regression(scored, "died", ["score", "age"])))
    assert wide.withheld == () and _ids(wide) == ["model", "term:score", "term:age"]
    assert wide.record.part("term:score").labels == {"variable": "score", "per": "unit"}
    # a slope rests on every value: it is not marked as resting on a small cell, and its counts do not travel
    assert "sparse" not in [n.name for n in wide.record.part("term:score").numbers]
    assert wide.record.part("term:score").support == {}
    # a covariate of many values that half of the rows hold at 0: its cells are the rows on that value and off it
    df["packs"] = np.where(np.arange(len(df)) % 2 == 0, 0.0, 1.0 + np.arange(len(df)))
    spread = plan.record("logistic", logistic_regression(df, "died", ["packs", "age"]))
    assert spread is not None and spread.part("term:packs").support["rows"] == 200
    assert spread.part("reference:packs").labels == {
        "variable": "packs",
        "level": "0",
        "of": "commonest",
    }
    assert _ids(_disclose(spread)) == ["model", "term:packs", "term:age"]
    # its term is carried by the rows off that value, and says whether they are few
    assert _disclose(spread).record.part("term:packs").number("sparse").value is False
    assert not [name for name in spread.part("term:packs").support if name.endswith("_off")]
    # and age, which no value holds for half of the rows, has the model's counts behind it
    whole = {k: n for k, n in spread.part("model").support.items() if k != "df_resid"}
    assert spread.part("term:age").support == whole
    assert "reference:age" not in [p.id for p in spread.parts]


def _scored(
    sizes: list[int], seed: int = 31, *, slope: float = 0.45, base: float = -1.0
) -> pd.DataFrame:
    """A cohort with a score from 0 that these numbers of rows hold, and death by the score."""
    n = sum(sizes)
    df = _cohort(n, seed)
    rng = np.random.default_rng(seed + 100)
    df["ecog"] = rng.permutation(np.repeat(np.arange(len(sizes)), sizes))
    df["died"] = (rng.random(n) < 1 / (1 + np.exp(-(base + slope * df["ecog"])))).astype(int)
    df["visits"] = rng.poisson(1.5, n)
    return df


def test_a_rare_value_of_a_score_withholds_its_slope_and_not_the_model() -> None:
    """A performance score from 0 to 4 held by 236, 236, 93, 32 and 3 rows, as an adjustment: the three kinds of
    model were withheld whole, at k = 10 and at k = 5, as if the 3 rows dropped out of the fit. They do not: a
    slope has one parameter for every value (without them the arm's term moved by 3e-3 to 1.5e-2; for an
    indicator of the same 3 rows, by 1e-12). The tutorial's deaths by arm, age and number of hospital visits (18
    values, 6 of them held by fewer than 10 rows) wrote nothing."""
    from biostats import plan
    from biostats.regression import count_regression, logistic_regression
    from biostats.survival import cox_regression

    df = _scored([236, 236, 93, 32, 3])
    covariates = ["arm", "age", "ecog"]
    for key, result in (
        ("logistic", logistic_regression(df, "died", covariates)),
        ("cox", cox_regression(df, "days", "died", covariates, diagnostics=False)),
        ("count", count_regression(df, "visits", covariates)),
    ):
        record = plan.record(key, result)
        assert record is not None
        term = record.part("term:ecog")
        # the smallest of each count over its values, and the rows off its commonest value
        assert (term.support["rows"], term.support["rows_off"]) == (3, 364), key
        for policy in (K10, DisclosurePolicy(5)):
            shown = _disclose(record, policy, compare="arm")
            assert _ids(shown)[1:] == ["reference:arm", "term:age", "term:arm#1"], key
            a_value = A_VALUE if key == "logistic" else A_VALUE_OF_TWO
            assert shown.withheld == (
                f"1 term of ecog withheld: {a_value.replace('10', str(policy.k))}",
                FIT,
            ), key
            assert not FIT_NAMES & {n.name for n in shown.record.part("model").numbers}, key
            assert "_off" not in _written(shown), key
    # without age every term that would be written compares categories: the model is a table, and says what
    # answers a slope
    by_arm = plan.record("logistic", logistic_regression(df, "died", ["arm", "ecog"]))
    assert _disclose(by_arm, compare="arm").withheld == (A_TABLE.replace(ADVICE, CAP),)
    # a rare outcome: a value that 23 rows hold, none of whom had the event. The slope has a maximum
    rare = _scored([362, 292, 123, 23], seed=32, slope=0.0, base=-2.7)
    rare.loc[rare["ecog"] == 3, "died"] = 0
    record = plan.record("logistic", logistic_regression(rare, "died", covariates))
    assert record is not None
    term = record.part("term:ecog")
    assert term.support["events"] == 0 and term.number("extreme").value is False
    shown = _disclose(record, compare="arm")
    assert [i for i in _ids(shown) if i.startswith("term:")] == ["term:age", "term:arm#1"]
    # nearly every row at one value: the rows off it are what two models isolate. Ten of them stay in the fit,
    # nine do not
    for sizes, whole in (([590, 5, 3, 2], False), ([591, 4, 3, 2], True)):
        few = _scored(sizes, seed=33)
        record = plan.record("logistic", logistic_regression(few, "died", covariates))
        assert record is not None
        assert record.part("term:ecog").support["rows_off"] == sum(sizes[1:])
        shown = _disclose(record, compare="arm")
        assert (shown.record is None) is whole, sizes
        assert not whole or shown.withheld == (OFF_ITS_VALUE,)
    # twelve rows off the commonest value, none of whom, or all of whom, had the event: a logistic model fitted
    # on them alone has no maximum; a Cox model where all of them had the event has one
    lone = _scored([588, 7, 5], seed=34)
    for died in (0, 1):
        lone.loc[lone["ecog"] > 0, "died"] = died
        record = plan.record("logistic", logistic_regression(lone, "died", covariates))
        assert _disclose(record, compare="arm").withheld == (OFF_ITS_VALUE,), died
    cox = plan.record("cox", cox_regression(lone, "days", "died", covariates, diagnostics=False))
    assert _ids(_disclose(cox, compare="arm"))[1:] == ["reference:arm", "term:age", "term:arm#1"]
    lone.loc[lone["ecog"] > 0, "died"] = 0
    cox = plan.record("cox", cox_regression(lone, "days", "died", covariates, diagnostics=False))
    assert _disclose(cox, compare="arm").withheld == (OFF_ITS_VALUE_NONE,)
    # pooled over imputed ages, the same: the counts are the fewest that a dataset holds
    from biostats.plan import specify
    from core.disclosure import disclose_record

    roles = {"time": "days", "event": "died", "compare": "arm", "adjust": ["age", "ecog"]}
    for sizes, seed, whole in (([236, 236, 93, 32, 3], 31, False), ([591, 4, 3, 2], 33, True)):
        done = plan.run(specify("cox", pooled=True, **roles), mi=_imputed_age(_scored(sizes, seed)))
        record = plan.record("cox_mi", done.result)
        assert record is not None
        assert record.part("term:ecog").support["rows_off"] == sum(sizes) - max(sizes)
        shown = disclose_record(record, K10, spec=done.spec, analysis=_in_the_flow(record))
        assert (shown.record is None) is whole, sizes
        if whole:
            assert shown.withheld == (OFF_ITS_VALUE_NONE,)
        else:
            assert [i for i in _ids(shown) if i.startswith("term:")] == ["term:age", "term:arm#1"]
            assert shown.withheld[0] == f"1 term of ecog withheld: {A_VALUE_OF_TWO}"


def test_a_dose_of_two_values_is_written_per_unit_without_a_reference() -> None:
    """A dose coded 0 or 50 was written as the level "50" against a reference "0", beside an odds ratio for one unit
    of dose."""
    from biostats import plan
    from biostats.regression import logistic_regression

    df = _cohort()
    df["dose"] = np.where(np.arange(len(df)) % 4 == 0, 50, 0)
    df["exposed"] = (df["dose"] > 0).astype(int)
    by_dose = _disclose(plan.record("logistic", logistic_regression(df, "died", ["dose", "age"])))
    assert by_dose.withheld == () and _ids(by_dose) == ["model", "term:dose", "term:age"]
    assert by_dose.record.part("term:dose").labels == {"variable": "dose", "per": "unit"}
    assert "50" not in json.dumps([p.labels for p in by_dose.record.parts])
    # a covariate coded 0 or 1 is the higher value against the lower, and says so
    by_flag = _disclose(
        plan.record("logistic", logistic_regression(df, "died", ["exposed", "age"]))
    )
    assert _ids(by_flag) == ["model", "reference:exposed", "term:exposed", "term:age"]
    assert by_flag.record.part("reference:exposed").labels == {"variable": "exposed", "level": "0"}
    assert by_flag.record.part("term:exposed").labels == {"variable": "exposed", "level": "1"}


def test_a_term_without_a_maximum_withholds_the_model() -> None:
    """The application itself says that such a ratio cannot be read, and the terms left are those of the model
    fitted without the rows that carry it. Beside a term withheld for its counts, "not converged" made a reader
    doubt the sound terms and told them that a withheld level had no maximum."""
    from biostats.plan import EXACT, ITERATIVE, Number, Part, ResultRecord

    counts = {"rows": 400, "events": 160, "non_events": 240}

    def term(name: str, extreme: bool, cell: dict | None = None, **labels: str) -> Part:
        numbers = (Number("coef", 0.2, ITERATIVE), Number("extreme", extreme, EXACT))
        return Part(f"term:{name}", numbers, cell or counts, {"variable": name, **labels})

    model = Part(
        "model",
        (
            Number("rows", 400, EXACT),
            Number("converged", False, EXACT),
            Number("llf", -250.0, ITERATIVE),
        ),
        {**counts, "df_resid": 397},
    )
    record = ResultRecord("logistic", parts=(model, term("age", False), term("marker", True)))
    disclosed = _disclose(record)
    assert disclosed.record is None and disclosed.withheld == (DROPS_OUT,)
    # a term withheld for its counts (12 rows, 3 of them with the event): the model is written, without what its
    # fit says of all its rows together, "not converged" included
    reference = Part(
        "reference:marker",
        (),
        {"rows": 388, "events": 157, "non_events": 231},
        {"variable": "marker", "level": "0"},
    )
    few = term("marker", False, {"rows": 12, "events": 3, "non_events": 9}, level="1")
    counted = ResultRecord("logistic", parts=(model, reference, term("age", False), few))
    disclosed = _disclose(counted)
    assert _ids(disclosed) == ["model", "term:age"]
    assert disclosed.withheld == (f"1 term of marker withheld: {CELL}", FIT)
    assert [n.name for n in disclosed.record.part("model").numbers] == [
        "rows",
        "parameters",
        "events_per_parameter",
        "sparse",
    ]
    assert disclosed.record.part("model").number("events_per_parameter").value == 80.0


def test_a_level_where_everybody_had_the_event_keeps_its_cox_term_and_loses_its_curve() -> None:
    """A site of 15 participants who all died. Its curve is a table ("15 rows, 15 events" says of each of the
    15 that they died): withheld, as a group of curves counts its rows without the event. Its Cox term is
    a ratio, with no count beside it: written, since the rows without the event are not counted for a term
    (counted, they withheld the ratios of the very levels with the most events). A logistic model counts
    both sides: a level where everybody had the event has no maximum, and withholds the model."""
    from biostats import plan
    from biostats.plan import specify
    from biostats.regression import logistic_regression
    from biostats.survival import compare_survival, cox_regression
    from core.disclosure import FOR_THE_OTHER_SIDE

    df = _three_sites(delta=15)
    record = plan.record("cox", cox_regression(df, "days", "died", ["site", "age"]))
    assert record is not None
    assert record.part("term:site#2").support == {"rows": 15, "events": 15, "censored": 0}
    disclosed = _disclose(record)
    assert {"reference:site", "term:site#1", "term:site#2"} <= set(_ids(disclosed))
    assert disclosed.withheld == ()
    # no count stands beside a term: only the model as a whole has its counts
    assert all(part.support == {} for part in disclosed.record.parts if part.id != "model")
    assert disclosed.record.part("model").support["rows"] == 1000
    curves = _disclose(plan.record("survival", compare_survival(df, "days", "died", "site")))
    assert [p.labels["group"] for p in curves.record.parts if p.id.startswith("curve#")] == [
        "Alpha"
    ]
    assert curves.withheld == (GROUP_WITHHELD, COVER_WITHHELD, TOGETHER, LOG_RANK)
    assert "Delta" not in _written(curves) and "Beta" not in _written(curves)
    # in one package: the curves withhold Delta and its cover for their rows without the event alone, which a
    # Cox term does not show, so that the model keeps its terms beside them
    both = [
        plan.run(specify("survival", time="days", event="died", group="site"), df),
        plan.run(specify("cox", time="days", event="died", adjust=["site", "age"]), df),
    ]
    for order in (both, both[::-1]):
        shown, named = _package(order)
        assert named == {"site": {"Delta": FOR_THE_OTHER_SIDE, "Beta": FOR_THE_OTHER_SIDE}}
        by_kind = {one.kind: one for one in shown}
        assert "term:site#2" in _ids(by_kind["cox"]) and "term:site#1" in _ids(by_kind["cox"])
        assert [
            p.labels["group"] for p in by_kind["survival"].record.parts if p.id.startswith("curve#")
        ] == ["Alpha"]
    logistic = _disclose(plan.record("logistic", logistic_regression(df, "died", ["site", "age"])))
    assert logistic.record is None and logistic.withheld == (DROPS_OUT,)


def test_a_count_model_of_a_few_people_with_an_event_is_withheld() -> None:
    """400 rows and 60 visits, all of 3 participants: the model's "events" are a sum of counts, and its term
    describes those 3."""
    from biostats import plan
    from biostats.regression import count_regression

    rng = np.random.default_rng(1)
    df = pd.DataFrame({"age": rng.normal(60, 8, 400).round(1)})
    df["visits"] = 0
    df.loc[df.index[:3], "visits"] = 20
    record = plan.record("count", count_regression(df, "visits", ["age"]))
    assert record is not None
    counts = record.part("model").support
    assert (counts["events"], counts["rows_with_events"], counts["rows_without_events"]) == (
        60,
        3,
        397,
    )
    disclosed = _disclose(record)
    assert disclosed.record is None and disclosed.withheld == (MODEL_WITHHELD,)


def test_a_kind_without_a_rule_is_withheld() -> None:
    from biostats.plan import Number, Part, ResultRecord

    record = ResultRecord("ancova", parts=(Part("model", (Number("rows", 500, "exact"),)),))
    disclosed = _disclose(record)
    assert disclosed.record is None
    assert disclosed.withheld == ("this kind of result has no rule for sharing yet",)


def test_a_part_without_its_counts_is_not_shown() -> None:
    """A contrast whose cells held no counts passed (every count of none reaches k), and a curve needed its events
    only."""
    from biostats.plan import EXACT, ITERATIVE, Number, Part, ResultRecord

    counts = {"rows": 400, "events": 160, "non_events": 240, "df_resid": 396}
    cell = {"rows": 120, "events": 50, "non_events": 70}
    estimate = (Number("coef", 0.2, ITERATIVE), Number("p", 0.3, ITERATIVE))
    model = Part("model", (Number("rows", 400, EXACT),), counts)
    age = Part("term:age", estimate, {}, {"variable": "age"})
    reference = Part("reference:site", (), cell, {"variable": "site", "level": "Alpha"})
    record = ResultRecord(
        "logistic",
        parts=(
            model,
            reference,
            age,
            Part("term:site#1", estimate, cell, {"variable": "site", "level": "Beta"}),
            Part("term:site#2", estimate, {}, {"variable": "site", "level": "Gamma"}),
            Part(
                "term:site#3",
                estimate,
                {"rows": 120, "events": 50},
                {"variable": "site", "level": "Delta"},
            ),
        ),
    )
    disclosed = _disclose(record)
    # two terms without their counts: nothing says that their rows stay in the fit, and nothing is written
    assert disclosed.record is None and disclosed.withheld == (DROPS_OUT,)
    # with both cells counted, Beta's term is written
    counted = ResultRecord("logistic", parts=record.parts[:4])
    assert _ids(_disclose(counted)) == ["model", "reference:site", "term:age", "term:site#1"]
    # a level without its reference's counts cannot be judged
    orphan = ResultRecord("logistic", parts=(model, age, record.parts[3]))
    assert _disclose(orphan).withheld == (DROPS_OUT,)
    # a curve without its rows without the event; a model without its residual degrees of freedom
    curve = Part(
        "curve#0", (Number("rows", 100, EXACT),), {"rows": 100, "events": 40}, {"group": "all"}
    )
    assert _disclose(ResultRecord("survival", parts=(curve,))).record is None
    bare = Part(
        "model",
        (Number("rows", 400, EXACT),),
        {name: n for name, n in counts.items() if name != "df_resid"},
    )
    assert _disclose(ResultRecord("logistic", parts=(bare, age))).record is None


def test_no_number_of_one_participant_passes_inside_the_complete_case_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rule that let one through is a defect, and nothing is written: the complete-case record beside a pooled
    one was not looked at."""
    from biostats.plan import EXACT, Number, Part, ResultRecord
    from core import disclosure
    from core.disclosure import DisclosedRecord

    inner = ResultRecord(
        "survival", parts=(Part("curve#0", (Number("median", 412.0, EXACT, fine=True),)),)
    )
    careless = DisclosedRecord("survival", True, ResultRecord("survival", True, (), (), inner))
    monkeypatch.setitem(
        disclosure.RULES, "survival", lambda record, policy, spec, cover=None: careless
    )
    with pytest.raises(ValueError, match="would share 1 number"):
        _disclose(ResultRecord("survival", True))


def test_a_result_is_withheld_with_the_analysis_the_flow_withholds() -> None:
    """A site of one participant, who died: the flow withheld the Kaplan-Meier analysis (one row in a group too
    small to draw) and its curves, shown all the same, held 999 rows beside a population of 1,000; a Cox model's
    events, less the curves', then gave that participant's death."""
    from biostats import plan
    from biostats.plan import specify
    from biostats.survival import compare_survival, cox_regression
    from core.disclosure import DisclosedAnalysis, disclose_record

    df = _three_sites(delta=1)
    result = compare_survival(df, "days", "died", "site")
    record = plan.record("survival", result)
    assert record is not None and result.skipped_rows == 1
    assert sum(p.support["rows"] for p in record.parts if p.id.startswith("curve#")) == 999
    # whatever line it is given with: its own record counts the row
    line = _in_the_flow(record)
    assert line.left_out == 1
    hidden = disclose_record(record, K10, spec=_spec_of(record), analysis=line)
    assert hidden.record is None and hidden.withheld == (FEW_LEFT_OUT,)
    # a result the flow does not show is not shown either, and the argument has no default
    whole = plan.record(
        "survival", compare_survival(df[df["site"] != "Delta"], "days", "died", "site")
    )
    assert whole is not None and _disclose(whole).record is not None
    unlisted = disclose_record(whole, K10, spec=_spec_of(whole), analysis=None)
    assert unlisted.record is None and unlisted.withheld == (NOT_IN_THE_FLOW,)
    with pytest.raises(TypeError):
        disclose_record(whole, K10, spec=_spec_of(whole))  # type: ignore[call-arg]
    # nor one given with the line of another analysis, or with another kind's spec
    other = DisclosedAnalysis("Cox model", 1000)
    assert disclose_record(whole, K10, spec=_spec_of(whole), analysis=other).withheld == (
        NOT_ITS_ROWS,
    )
    with pytest.raises(ValueError, match="its own spec"):
        disclose_record(whole, K10, spec=specify("cox", time="days", event="died"), analysis=other)
    # two rows with a follow-up time below zero: the Cox model that leaves them out says so in its record
    negative = _cohort()
    negative.loc[negative.index[:2], "days"] = -5.0
    cox = plan.record("cox", cox_regression(negative, "days", "died", ["arm", "age"]))
    assert cox is not None and _in_the_flow(cox).left_out == 2
    assert _disclose(cox, compare="arm").withheld == (FEW_LEFT_OUT,)
    # twelve of them are a count that can be shown: the model is, with the line that counts them
    negative.loc[negative.index[:12], "days"] = -5.0
    twelve = plan.record("cox", cox_regression(negative, "days", "died", ["arm", "age"]))
    assert twelve is not None and _disclose(twelve, compare="arm").record is not None
    assert disclose_record(
        twelve, K10, spec=_spec_of(twelve), analysis=DisclosedAnalysis("Cox model", 388)
    ).withheld == (NOT_ITS_ROWS,)


def _is_multiple(value: float, step: float) -> bool:
    return float(f"{value / step:.9g}").is_integer()


def test_a_curve_is_given_at_multiples_of_one_step_every_k_events() -> None:
    from biostats import plan
    from biostats.survival import compare_survival

    df = _cohort()
    df.loc[df.index[3], ["days", "died"]] = [
        98765.0,
        1,
    ]  # one participant's day of death, the last of all
    record = plan.record("survival", compare_survival(df, "days", "died", "arm"))
    disclosed = _disclose(record)
    assert disclosed.withheld == () and disclosed.record is not None
    step = disclosed.record.parts[0].number("time_step").value
    for part, original in zip(disclosed.record.parts[:2], record.parts[:2], strict=True):
        assert part.id == original.id and part.labels == original.labels
        assert part.number("time_step").value == step  # one step for every time of the result
        assert part.number("points_every").value == 10
        times, survival = part.number("times").value, part.number("survival").value
        assert len(times) >= 5 and times == sorted(set(times))
        assert all(_is_multiple(t, step) for t in times)
        assert len(survival) == len(times) == len(part.number("lower").value)
        steps, events = original.number("times").value, original.number("events_at").value
        full = dict(zip(steps, original.number("survival").value, strict=True))
        # the value at a time given is the full curve's at that time, to the decimals of that time: two, in an
        # arm of about 200 rows
        digits = part.number("decimals").value
        assert set(digits) == {2} and len(digits) == len(times)
        for t, value, places in zip(times, survival, digits, strict=True):
            assert value == round(full[max(s for s in steps if s <= t)], places)
        # at least k events between two times given, and none or at least k after the last
        until = part.number("followed_until").value
        flat = times[-1] == until and survival[-1] == survival[-2]
        points = times[:-1] if flat else times
        last = 0.0
        for t in points:
            assert sum(e for s, e in zip(steps, events, strict=True) if last < s <= t) >= 10
            last = t
        after = sum(e for s, e in zip(steps, events, strict=True) if s > points[-1])
        assert after == 0 or after >= 10
        # k rows are still followed, without the event, at the last time given and at followed_until
        followed = dict(
            zip(
                original.number("follow_times").value,
                original.number("followed").value,
                strict=True,
            )
        )
        assert min(n for t, n in followed.items() if t <= until) >= 10 and _is_multiple(until, step)
        # the median and its limits are multiples of the step, or have no value
        for name in ("median", "median_lower", "median_upper"):
            value = part.number(name).value
            assert value is None or _is_multiple(value, step)
        if part.number("median").value is not None:
            assert part.number("median").value >= original.number("median").value
        assert {n.name for n in part.numbers} == CURVE_NAMES
        # the table under a curve: its rows, its events and its rows without the event
        assert part.support == original.support and set(part.support) == {
            "rows",
            "events",
            "censored",
        }
    assert "98765" not in _written(disclosed)  # no event time as it is, no last follow-up time
    assert not any(n.fine for part in disclosed.record.parts for n in part.numbers)
    assert (
        disclosed.record.part("test:log_rank").number("p").value
        == record.part("test:log_rank").number("p").value
    )
    five = _disclose(record, DisclosurePolicy(5))
    assert len(five.record.parts[0].number("times").value) > len(
        disclosed.record.parts[0].number("times").value
    )


def _curve_shown(frame: pd.DataFrame, group: str | None = None):
    from biostats import plan
    from biostats.survival import compare_survival

    record = plan.record("survival", compare_survival(frame, "days", "died", group))
    assert record is not None
    return [p for p in _disclose(record).record.parts if p.id.startswith("curve#")]


def test_the_end_of_a_curve_gives_no_event_back_and_says_until_when_it_was_followed() -> None:
    """The events after the last time given came back from the group's events and the curve's value (6 of them, the
    events of days 48 to 50), and a curve that stays flat for 950 days stopped at day 47."""
    # 60 events in the first 50 days, then 340 rows followed to day 1000
    rng = np.random.default_rng(5)
    days, died = np.full(400, 1000.0), np.zeros(400, dtype=int)
    who = rng.choice(400, 60, replace=False)
    days[who], died[who] = rng.integers(1, 51, 60), 1
    (plateau,) = _curve_shown(pd.DataFrame({"days": days, "died": died}))
    assert plateau.number("time_step").value == 10
    assert plateau.number("times").value == [10, 30, 40, 50, 1000]
    # fewer than 400 rows are followed from the first time on: two decimals (0.968, 0.925, 0.877 and 0.85)
    assert plateau.number("survival").value == [0.97, 0.92, 0.88, 0.85, 0.85]
    assert plateau.number("decimals").value == [2, 2, 2, 2, 2]
    assert plateau.number("followed_until").value == 1000
    assert plateau.number("median").value is None  # not reached, and not named as later
    assert plateau.number("later_than_followed_until").value == []
    # a trial followed to day 450: both arms are given to the same day
    rng = np.random.default_rng(8)
    both = np.concatenate([rng.exponential(2000, 200), rng.exponential(1500, 200)])
    trial = pd.DataFrame(
        {
            "arm": ["control"] * 200 + ["treated"] * 200,
            "days": np.minimum(both, 450).round() + 1,
            "died": (both <= 450).astype(int),
        }
    )
    control, treated = _curve_shown(trial, "arm")
    assert control.number("time_step").value == treated.number("time_step").value == 5
    assert control.number("times").value == [100, 190, 285, 445, 450]
    assert treated.number("times").value == [80, 165, 415, 450]
    assert control.number("followed_until").value == treated.number("followed_until").value == 450


def test_a_median_is_rounded_outwards_or_said_to_be_later_than_the_follow_up() -> None:
    """A median is an event day: 546 (281 to 717) for 60 rows, where 717 is an event day with 9 rows followed
    after it, beyond the last time given."""
    rng = np.random.default_rng(11)
    follow = 1000 - rng.uniform(0, 900, 60)
    event_time = rng.exponential(700, 60)
    late = pd.DataFrame(
        {
            "days": np.minimum(event_time, follow).round() + 1,
            "died": (event_time <= follow).astype(int),
        }
    )
    (curve,) = _curve_shown(late)
    assert curve.number("time_step").value == 10 and curve.number("times").value == [80, 490]
    assert curve.number("followed_until").value == 710
    # up for the median (the curve is at most one half from there on), down and up for its limits
    assert curve.number("median").value == 550 and curve.number("median_lower").value == 280
    assert curve.number("median_upper").value is None
    assert curve.number("later_than_followed_until").value == ["median_upper"]
    # 30 rows, 12 events, 18 rows censored early: the group passes, and no point of its curve can be given
    days = [5, 8, 12, 15, 20, 22, 25, 30, 33, 35, 40, 44, 50, 55, 60, 70, 80, 90]
    days += [100, 130, 160, 200, 240, 300, 380, 450, 520, 640, 700, 810]
    early = pd.DataFrame({"days": [float(d) for d in days], "died": [0] * 18 + [1] * 12})
    (curve,) = _curve_shown(early)
    assert curve.number("times").value == [] and curve.number("survival").value == []
    assert curve.number("time_step").value == 10 and curve.number("followed_until").value == 150
    assert curve.number("median").value is None and curve.number("median_lower").value == 130
    assert curve.number("later_than_followed_until").value == ["median", "median_upper"]


def test_a_time_one_last_bit_above_a_step_goes_to_the_next_one() -> None:
    """20 events at 2.0 and 20 at the next number a machine can hold were given at 2.0 and 2.0."""
    from core.disclosure import round_down, round_up, step_at_least

    above = float(np.nextafter(2.0, 3.0))
    assert round_up(2.0, 0.1) == 2.0 and round_up(above, 0.1) == 2.1
    assert round_up(0.9, 0.1) == 0.9 and round_up(1272, 20) == 1280 and round_up(40, 20) == 40
    assert round_down(1196, 20) == 1180 and round_down(0.3, 0.1) == 0.3
    assert round_down(float(np.nextafter(0.3, 0.0)), 0.1) == 0.2
    assert [step_at_least(v) for v in (14.07, 5, 0.0348, 250, 1, 0.9)] == [20, 5, 0.05, 500, 1, 1]


def test_a_small_group_is_withheld_with_another_that_covers_it() -> None:
    """A group of 6 rows withheld alone comes back at once: the number analysed (400) less the two curves shown.
    The smallest group shown is withheld with it, until those withheld hold k rows, k events and k rows without
    the event together; the log-rank test, which compares them, is withheld too."""
    from biostats import plan
    from biostats.survival import compare_survival

    df = _cohort()
    df["group"] = df["arm"]
    df.loc[df.index[:6], "group"] = "CANARY-GROUP"
    record = plan.record("survival", compare_survival(df, "days", "died", "group"))
    disclosed = _disclose(record)
    curves = [p for p in record.parts if p.id.startswith("curve#")]
    shown = [p for p in disclosed.record.parts if p.id.startswith("curve#")]
    assert len(curves) == 3 and len(shown) == 1 and shown[0].labels["group"] in ("control", "drug")
    # said with what to do to keep the comparison
    assert disclosed.withheld == (GROUP_WITHHELD, COVER_WITHHELD, TOGETHER, LOG_RANK)
    # what the number analysed gives, less the group shown: the two groups withheld, at least k of each together
    hidden = [p for p in curves if p.id != shown[0].id]
    assert min(p.support["rows"] for p in hidden) == 6
    assert shown[0].support["rows"] == max(
        p.support["rows"] for p in curves
    )  # the larger one stays
    for name in ("rows", "events", "censored"):
        assert sum(p.support[name] for p in hidden) >= 10
    assert "CANARY" not in _written(disclosed) and "CANARY" not in " ".join(disclosed.withheld)
    # two groups only: nothing is left to show once the other covers the small one
    two = df[df["group"] != "drug"]
    alone = _disclose(plan.record("survival", compare_survival(two, "days", "died", "group")))
    assert alone.record is None and alone.withheld == (GROUP_WITHHELD, COVER_WITHHELD, TOGETHER)


def test_a_pooled_curve_is_thinned_on_each_dataset_s_own_events() -> None:
    """Two imputed datasets whose events up to each time are 5, 10, 20, 30, 40 and 9, 12, 20, 30, 40: the fewest so
    far are 5, 10, 20..., whose differences released the second time, where the second dataset holds 3 events since
    the first."""
    from biostats.plan import DETERMINISTIC, EXACT, Number, Part, ResultRecord

    times = [50.0, 100.0, 200.0, 300.0, 400.0]
    each = [[5, 10, 20, 30, 40], [9, 12, 20, 30, 40]]
    days = range(1, 501)  # a follow-up ends every day, and 101 rows are followed to day 500
    curve = Part(
        "curve#0",
        (
            Number("rows_min", 200, EXACT),
            Number("events_min", 40, EXACT),
            Number("median", None, EXACT, fine=True),
            Number("times", times, EXACT, fine=True),
            Number("survival", [0.97, 0.94, 0.90, 0.85, 0.80], DETERMINISTIC, scale=1.0),
            Number("events_so_far_each", each, EXACT, fine=True),
            Number("follow_times", [float(d) for d in days], EXACT, fine=True),
            Number("followed_min", [200 - (d - 1) // 5 for d in days], EXACT, fine=True),
        ),
        {"rows_min": 200, "events_min": 40, "censored_min": 160},
        {"group": "all"},
    )
    five = DisclosurePolicy(5)
    shown = _disclose(ResultRecord("survival", True, (curve,)), five).record.part("curve#0")
    # not at day 100; and to day 500 with the last value, the curve being flat to there
    assert shown.number("times").value == [50.0, 200.0, 300.0, 400.0, 500.0]
    assert shown.number("survival").value == [0.97, 0.90, 0.85, 0.80, 0.80]
    assert shown.number("time_step").value == 5 and shown.number("followed_until").value == 500
    assert not any(n.fine for n in shown.numbers)
    # its counts are the smallest over the datasets, and named so
    assert shown.support == {"rows_min": 200, "events_min": 40, "censored_min": 160}
    # a pooled curve recorded without each dataset's events, or without its follow-up, cannot be given
    for missing in ("events_so_far_each", "follow_times"):
        without = Part(
            curve.id,
            tuple(n for n in curve.numbers if n.name != missing),
            curve.support,
            curve.labels,
        )
        assert _disclose(ResultRecord("survival", True, (without,)), five).record is None, missing


def test_the_complete_case_analysis_is_shown_when_it_differs_by_none_or_by_k() -> None:
    """A pooled model and its complete cases differ by the rows that lack a value, and by their outcomes: 12 rows
    of which 2 had the event passed, the pair being compared on its rows only."""
    from biostats.plan import EXACT, ITERATIVE, Number, Part, ResultRecord

    age = Part("term:age", (Number("se", 0.1, ITERATIVE),), {}, {"variable": "age"})

    def model(rows: int, events: int, *, pooled: bool = False) -> Part:
        counts = {
            "rows": rows,
            "events": events,
            "non_events": rows - events,
            "df_resid": rows - 2,
        }
        if pooled:  # the smallest over the imputed datasets, named so
            counts = {f"{name}_min": n for name, n in counts.items()}
        return Part("model", (Number("rows_min" if pooled else "rows", rows, EXACT),), counts)

    def pair(rows: int, events: int) -> ResultRecord:
        complete = ResultRecord("logistic", parts=(model(rows, events), age))
        return ResultRecord("logistic", True, (model(400, 160, pooled=True), age), (), complete)

    # 43 rows lack a value, 20 of them with the event and 23 without: nothing small comes from the pair
    assert _disclose(pair(357, 140)).record.complete_case is not None
    assert _disclose(pair(400, 160)).record.complete_case is not None  # no row lacks one
    # 12 rows lack a value, 2 of them with the event: the two models' events give the 2
    few_events = _disclose(pair(388, 158))
    assert few_events.record is not None and few_events.record.complete_case is None
    assert few_events.withheld == ("the complete-case analysis is withheld",)
    assert _disclose(pair(395, 160)).record.complete_case is None  # 5 rows lack a value


def _imputed_age(truth: pd.DataFrame | None = None) -> object:
    from biostats.mi import MIData

    truth = _cohort() if truth is None else truth
    rng = np.random.default_rng(6)
    base = truth.copy()
    gap = rng.random(len(base)) < 0.15
    base.loc[gap, "age"] = np.nan
    completed = []
    for _ in range(5):
        one = base.copy()
        one.loc[gap, "age"] = truth.loc[gap, "age"] + rng.normal(0, 5, gap.sum())
        completed.append(one)
    return MIData.from_completed(base, completed, methods={"age": "pmm"})


def test_a_pooled_model_keeps_rubin_s_quantities_and_thins_its_curves() -> None:
    from biostats import plan
    from biostats.plan import specify
    from core.disclosure import disclose_record

    mi = _imputed_age()
    cox = plan.run(
        specify("cox", pooled=True, time="days", event="died", compare="arm", adjust=["age"]), mi=mi
    )
    record = plan.record("cox_mi", cox.result)
    disclosed = disclose_record(record, K10, spec=cox.spec, analysis=_in_the_flow(record))
    assert disclosed.record is not None and disclosed.record.pooled
    age = disclosed.record.part("term:age")
    assert "estimates" not in [n.name for n in age.numbers] and "fmi" in [
        n.name for n in age.numbers
    ]
    assert all(part.support == {} for part in disclosed.record.parts if part.id != "model")
    # the model's counts are the smallest over the datasets, and named so; not the rows a dataset left out
    model = disclosed.record.part("model")
    assert set(model.support) == {"rows_min", "events_min", "censored_min", "df_resid_min"}
    assert [n.name for n in model.numbers] == [
        "rows_min", "rows_max", "events_min", "events_max", "converged_in", "parameters",
        "events_per_parameter_min", "sparse",
    ]  # fmt: skip
    # 62 rows lack a value, 19 of them with the event and 43 without: at least k of each, so the pair is no
    # subtraction
    assert disclosed.record.complete_case is not None

    curves = plan.run(
        specify("survival", pooled=True, time="days", event="died", group="arm"), mi=mi
    )
    record = plan.record("survival_mi", curves.result)
    shown = disclose_record(record, K10, spec=curves.spec, analysis=_in_the_flow(record))
    drug = next(p for p in shown.record.parts if p.labels.get("group") == "drug")
    assert 3 <= len(drug.number("times").value) < len(record.part(drug.id).number("times").value)
    assert set(drug.support) == {"rows_min", "events_min", "censored_min"}
    assert not any(n.fine for part in shown.record.parts for n in part.numbers)
    # its complete cases, here the same rows, are given at the same times
    beside = next(p for p in shown.record.complete_case.parts if p.labels.get("group") == "drug")
    assert beside.number("times").value == drug.number("times").value
    assert beside.number("time_step").value == drug.number("time_step").value


def test_a_pooled_result_and_its_complete_cases_withhold_the_same_groups() -> None:
    """The group lacks a value for 120 rows of one arm and 10 of the other: the pooled groups hold 300, 194 and 6
    rows, the complete cases 180, 184 and 6. Each analysis covered the 6 rows with its own smallest group, and a
    reader held one arm of the comparison from the pooled result and the other from the complete cases."""
    from biostats import plan
    from biostats.mi import MIData
    from biostats.plan import specify
    from core.disclosure import disclose_record

    rng = np.random.default_rng(12)
    truth = pd.DataFrame({"group": ["a_treat"] * 300 + ["b_control"] * 194 + ["c_other"] * 6})
    truth["died"] = (rng.random(500) < 0.4).astype(int)
    truth["days"] = rng.exponential(400, 500).round() + 1
    base = truth.copy()
    base["group"] = base["group"].astype(object)
    base.loc[[*range(120), *range(300, 310)], "group"] = np.nan
    mi = MIData.from_completed(base, [truth.copy() for _ in range(5)], methods={"group": "polyreg"})
    done = plan.run(
        specify("survival", pooled=True, time="days", event="died", group="group"), mi=mi
    )
    record = plan.record("survival_mi", done.result)
    assert record is not None and record.complete_case is not None

    def rows(of) -> list[int]:
        return [
            p.support.get("rows_min", p.support.get("rows"))
            for p in of.parts
            if p.id.startswith("curve#")
        ]

    assert rows(record) == [300, 194, 6] and rows(record.complete_case) == [180, 184, 6]
    disclosed = disclose_record(record, K10, spec=done.spec, analysis=_in_the_flow(record))

    def groups(of) -> list[str]:
        return [p.labels["group"] for p in of.parts if p.id.startswith("curve#")]

    assert groups(disclosed.record) == ["a_treat"]
    assert groups(disclosed.record.complete_case) == ["a_treat"]


def test_a_pooled_result_keeps_the_groups_its_complete_cases_cannot_show() -> None:
    """Groups of 300, 200 and 40 rows in the pooled result; the third one's value is recorded for 6 rows. The
    complete cases covered their 6 rows with the second group, and what they withheld went back to the pooled
    result, which showed one group and no test: for complete cases that, with 4 rows lacking in the first group,
    were not even written."""
    from biostats import plan
    from biostats.mi import MIData
    from biostats.plan import specify
    from core.disclosure import disclose_record

    rng = np.random.default_rng(12)
    truth = pd.DataFrame({"group": ["a"] * 300 + ["b"] * 200 + ["c"] * 40})
    truth["died"] = (rng.random(540) < 0.4).astype(int)
    truth["days"] = rng.exponential(400, 540).round() + 1

    def disclosed_with(lacking_in_a: int):
        base = truth.copy()
        base["group"] = base["group"].astype(object)
        base.loc[[*range(lacking_in_a), *range(300, 330), *range(500, 534)], "group"] = np.nan
        mi = MIData.from_completed(
            base, [truth.copy() for _ in range(5)], methods={"group": "polyreg"}
        )
        done = plan.run(
            specify("survival", pooled=True, time="days", event="died", group="group"), mi=mi
        )
        record = plan.record("survival_mi", done.result)
        assert record is not None and record.complete_case is not None
        return disclose_record(record, K10, spec=done.spec, analysis=_in_the_flow(record))

    def groups(of) -> list[str]:
        return [p.labels["group"] for p in of.parts if p.id.startswith("curve#")]

    # complete cases of 270, 170 and 6 rows: they take the result's cover (none) and their own; the result keeps
    # its three groups and its test
    shown = disclosed_with(30)
    assert groups(shown.record) == ["a", "b", "c"]
    assert "test:log_rank" in [p.id for p in shown.record.parts]
    assert groups(shown.record.complete_case) == ["a"]
    assert all(said.startswith("complete-case analysis: ") for said in shown.withheld)
    assert len(shown.withheld) == 4 and "or the pooled result withholds them" in shown.withheld[1]
    # 4 rows lack the value in the first group: the pair would give them, and the complete cases are not written
    alone = disclosed_with(4)
    assert groups(alone.record) == ["a", "b", "c"] and alone.record.complete_case is None
    assert "test:log_rank" in [p.id for p in alone.record.parts]
    assert alone.withheld == ("the complete-case analysis is withheld",)


def _sites_with_a_few_events(sizes: dict[str, int]) -> pd.DataFrame:
    """Sites of these sizes; 3 of the 15 participants of Delta died: a cell that stays in a fit."""
    df = _sites_of(sizes)
    df.loc[df["site"] == "Delta", "died"] = [1, 1, 1] + [0] * 12
    return df


def test_the_curves_and_a_model_of_one_grouping_withhold_the_same_cells() -> None:
    """Sites Alpha (the reference: first in sorted order), Beta, Gamma and a small Delta. The curves withheld the
    smallest site shown with Delta, whichever it was; a model withheld the smallest term, never its reference.
    With Alpha the smallest, the curves showed Beta and Gamma and the model wrote Gamma's term: its coefficient
    and standard error gave Alpha's cell, and the totals, less Alpha, Beta and Gamma, gave Delta."""
    from biostats import plan
    from biostats.plan import specify
    from core.disclosure import (
        ASKED,
        FOR_ITS_ROWS,
        disclose_record,
        withheld_labels,
    )

    def both(df: pd.DataFrame, **more):
        curves = plan.run(specify("survival", time="days", event="died", group="site"), df)
        model = plan.run(specify("logistic", outcome="died", adjust=["site", "age"]), df)
        found = []
        for done in (curves, model):
            record = plan.record(done.spec.kind, done.result)
            disclosed = disclose_record(
                record, K10, spec=done.spec, analysis=_in_the_flow(record), **more
            )
            found.append((disclosed, withheld_labels(record, disclosed, done.spec, K10, **more)))
        return found

    # the reference is the smallest site shown: the curves withhold it with Delta, and the model every term of
    # the factor (a level cannot be read without its reference)
    smallest = _sites_with_a_few_events({"Alpha": 150, "Beta": 397, "Gamma": 438, "Delta": 15})
    (curves, of_curves), (model, of_model) = both(smallest)
    # Delta for its 3 events, Alpha to cover them
    assert of_curves == {"site": {"Delta": FOR_ITS_ROWS, "Alpha": FOR_ITS_ROWS}}
    assert [p.labels["group"] for p in curves.record.parts if p.id.startswith("curve#")] == [
        "Beta",
        "Gamma",
    ]
    # the model writes no cell of site: it gives nothing to put beside the curves, and names nothing
    assert of_model == {}
    assert _ids(model) == ["model", "term:age"]
    assert model.withheld == (
        f"1 term of site withheld: {CELL}",
        f"2 more terms of site withheld: {COVER}",
        FIT,
    )
    # the reference is the largest: both withhold Gamma with Delta
    largest = _sites_with_a_few_events({"Alpha": 450, "Beta": 385, "Gamma": 150, "Delta": 15})
    (curves, of_curves), (model, of_model) = both(largest)
    assert of_curves == of_model == {"site": {"Delta": FOR_ITS_ROWS, "Gamma": FOR_ITS_ROWS}}
    assert _ids(model) == ["model", "reference:site", "term:age", "term:site#1"]
    # a cover chosen elsewhere (another result of the package, or the person who shares) is withheld first, in
    # both: Beta with Delta, which need no other
    (curves, of_curves), (model, of_model) = both(largest, cover={"site": ["Beta"]})
    assert of_curves == of_model == {"site": {"Beta": ASKED, "Delta": FOR_ITS_ROWS}}
    assert model.withheld == (
        f"1 term of site withheld: {CELL}",
        f"1 more term of site withheld: {COVER}",
        FIT,
    )
    assert "Beta" not in _written(model) and "Beta" not in _written(curves)
    # the reference asked for: no term of the factor is written
    (_, _), (model, of_model) = both(largest, cover={"site": ["Alpha"]})
    assert _ids(model) == ["model", "term:age"] and of_model == {}
    # a grouping that the result does not hold changes nothing
    (curves, of_curves), (model, of_model) = both(largest, cover={"arm": ["drug"]})
    assert of_curves == of_model == {"site": {"Delta": FOR_ITS_ROWS, "Gamma": FOR_ITS_ROWS}}
    # a result that is withheld whole writes no number to put beside another's: it names no cell
    three = _three_sites()
    done = plan.run(specify("logistic", outcome="died", adjust=["site", "age"]), three)
    record = plan.record("logistic", done.result)
    gone = disclose_record(record, K10, spec=done.spec, analysis=_in_the_flow(record))
    assert gone.record is None
    assert withheld_labels(record, gone, done.spec, K10) == {}


def _package(results: list, policy: DisclosurePolicy = K10) -> tuple[list, dict]:
    """The results of one package disclosed as its writer must: in turn, each with the cells the results before
    it named, then each again with all of them. Gives what is shown of each, and the cells named."""
    from biostats import plan
    from core.disclosure import cover_for, disclose_record, named_together, withheld_labels

    records = [(done, plan.record(done.spec.kind, done.result)) for done in results]
    named: dict = {}
    for done, record in records:
        cover = cover_for(record, named)
        shown = disclose_record(
            record, policy, spec=done.spec, analysis=_in_the_flow(record), cover=cover
        )
        own = withheld_labels(record, shown, done.spec, policy, cover=cover)
        named = named_together(named, own)
    final = []
    for done, record in records:
        cover = cover_for(record, named)
        shown = disclose_record(
            record, policy, spec=done.spec, analysis=_in_the_flow(record), cover=cover
        )
        # one turn names every cell: given them all, no result withholds another
        again = withheld_labels(record, shown, done.spec, policy, cover=cover)
        assert named_together(named, again) == named
        final.append(shown)
    return final, named


def test_a_result_withheld_whole_takes_no_other_result_of_its_package_with_it() -> None:
    """Curves by arm, a Cox model by arm, and a logistic model adjusted for sex, which is spelt "f " in two rows:
    the logistic model is withheld whole. It named both arms and every spelling of sex as withheld, and the
    writer's loop then withheld the curves by arm and the Cox model: every result of the grouping, for a level
    of another variable. Nothing is protected by it: a result that is not written gives no number to combine."""
    from biostats import plan
    from biostats.plan import specify

    df = _cohort()
    df["sex"] = np.where(np.arange(len(df)) % 2 == 0, "F", "M")
    df.loc[df.index[:2], "sex"] = "f "
    curves = plan.run(specify("survival", time="days", event="died", group="arm"), df)
    cox = plan.run(specify("cox", time="days", event="died", compare="arm", adjust=["age"]), df)
    logistic = plan.run(
        specify("logistic", outcome="died", compare="arm", adjust=["sex", "age"]), df
    )
    for order in ([curves, cox, logistic], [logistic, cox, curves]):
        shown, named = _package(order)
        by_kind = {one.kind: one for one in shown}
        assert by_kind["logistic"].record is None  # a cell of two rows: nothing of the model
        assert named == {}
        groups = [
            p.labels["group"] for p in by_kind["survival"].record.parts if p.id.startswith("curve#")
        ]
        assert groups == ["control", "drug"]
        assert "term:arm#1" in _ids(by_kind["cox"])


def test_a_model_that_writes_no_cell_of_a_grouping_leaves_the_curves_of_that_grouping() -> None:
    """Sites Alpha (450 rows), Beta, Gamma and Delta (3 rows, who all died). The model adjusted for site is
    withheld whole (a cell of 3 rows drops out of the fit), and a model that writes age alone writes no cell of
    site: each named the four sites, and the curves of Alpha and Beta went with them."""
    from biostats import plan
    from biostats.plan import specify
    from core.disclosure import FOR_ITS_ROWS

    df = _four_sites()
    curves = plan.run(specify("survival", time="days", event="died", group="site"), df)
    crude = plan.run(specify("logistic", outcome="died", adjust=["site"]), df)
    adjusted = plan.run(specify("logistic", outcome="died", adjust=["site", "age"]), df)
    alone = _package([curves])[0][0]
    kept = [p.labels["group"] for p in alone.record.parts if p.id.startswith("curve#")]
    assert kept == ["Alpha", "Beta"]  # Delta, and Gamma to cover it
    for order in ([curves, crude, adjusted], [adjusted, crude, curves]):
        shown, named = _package(order)
        by_spec = dict(zip([id(done) for done in order], shown, strict=True))
        of_curves = by_spec[id(curves)]
        assert [
            p.labels["group"] for p in of_curves.record.parts if p.id.startswith("curve#")
        ] == kept
        assert by_spec[id(crude)].record is None and by_spec[id(adjusted)].record is None
        # the curves alone name cells: the two they withhold
        assert named == {"site": {"Delta": FOR_ITS_ROWS, "Gamma": FOR_ITS_ROWS}}
    # a model that is written, with age and no term of site (its reference is the smallest site shown)
    df = _sites_with_a_few_events({"Alpha": 150, "Beta": 397, "Gamma": 438, "Delta": 15})
    curves = plan.run(specify("survival", time="days", event="died", group="site"), df)
    model = plan.run(specify("logistic", outcome="died", adjust=["site", "age"]), df)
    for order in ([curves, model], [model, curves]):
        shown, named = _package(order)
        by_kind = {one.kind: one for one in shown}
        assert _ids(by_kind["logistic"]) == ["model", "term:age"]
        assert [
            p.labels["group"] for p in by_kind["survival"].record.parts if p.id.startswith("curve#")
        ] == ["Beta", "Gamma"]
        assert named == {"site": {"Delta": FOR_ITS_ROWS, "Alpha": FOR_ITS_ROWS}}


def test_two_results_of_one_outcome_on_different_rows_are_a_subtraction() -> None:
    """A crude result and an adjusted one stand in nearly every package. They differ by the rows that lack a
    covariate, whose number is exempt, and by those rows' events, which are not: 3 rows lack a BMI and all three
    died; curves by arm and a Cox model with BMI were both written, and each of the three had died. Nothing
    compared two results."""
    from biostats import plan
    from biostats.plan import specify
    from core.disclosure import outcome_of, pair_gives_back

    df = _cohort()
    rng = np.random.default_rng(8)
    df["bmi"] = rng.normal(27, 4, len(df)).round(1)
    lacking = df.index[df["died"] == 1][:3]
    df.loc[lacking, "bmi"] = np.nan

    def record(done):
        return plan.record(done.spec.kind, done.result)

    curves = plan.run(specify("survival", time="days", event="died", group="arm"), df)
    crude = plan.run(specify("cox", time="days", event="died", compare="arm"), df)
    with_bmi = plan.run(
        specify("cox", time="days", event="died", compare="arm", adjust=["bmi"]), df
    )
    assert {outcome_of(done.spec) for done in (curves, crude, with_bmi)} == {"died"}
    # each is written on its own: 400 rows and 397
    for done in (curves, crude, with_bmi):
        assert _disclose(record(done), compare="arm").record is not None
    assert pair_gives_back(record(curves), record(with_bmi), K10)  # 3 deaths between them
    assert pair_gives_back(record(crude), record(with_bmi), K10)
    assert pair_gives_back(record(with_bmi), record(crude), K10)  # either way round
    assert not pair_gives_back(record(curves), record(crude), K10)  # the same rows
    # 30 rows lack a value, 12 of them with the event and 18 without: at least k of each, so no subtraction
    df = _cohort()
    df["bmi"] = rng.normal(27, 4, len(df)).round(1)
    gone = [*df.index[df["died"] == 1][:12], *df.index[df["died"] == 0][:18]]
    df.loc[gone, "bmi"] = np.nan
    crude = plan.run(specify("logistic", outcome="died", compare="arm"), df)
    with_bmi = plan.run(specify("logistic", outcome="died", compare="arm", adjust=["bmi"]), df)
    assert not pair_gives_back(record(crude), record(with_bmi), K10)
    # 30 rows again, 7 of them with the event (the tutorial's deaths among the participants without a phenotype)
    df = _cohort()
    df["bmi"] = rng.normal(27, 4, len(df)).round(1)
    gone = [*df.index[df["died"] == 1][:7], *df.index[df["died"] == 0][:23]]
    df.loc[gone, "bmi"] = np.nan
    crude = plan.run(specify("logistic", outcome="died", compare="arm"), df)
    with_bmi = plan.run(specify("logistic", outcome="died", compare="arm", adjust=["bmi"]), df)
    assert pair_gives_back(record(crude), record(with_bmi), K10)
    assert not pair_gives_back(record(crude), record(with_bmi), DisclosurePolicy(5))
    # a record that does not say its counts cannot be compared: not written beside another
    bare = plan.ResultRecord("logistic", parts=(plan.Part(id="model"),))
    assert pair_gives_back(record(crude), bare, K10)
    assert outcome_of(specify("count", count="visits", compare="arm")) == "visits"


def test_two_models_that_differ_by_a_variable_without_a_written_term_are_a_pair() -> None:
    """A flag held by 12 participants of the control arm, 5 of whom died: its term is withheld (5 events) and the
    model is written, since a cell of 12 rows stays in the fit. Beside the same model without the flag, the two
    give the cell's deaths to a reader who knows its 12 rows and its arm."""
    from biostats import plan
    from biostats.plan import specify
    from core.disclosure import disclose_record, nested_by_unwritten

    df = _cohort()
    control = df.index[df["arm"] == "control"][:12]
    df["flag"] = 0
    df.loc[control, "flag"] = 1
    df.loc[control, "died"] = [1] * 5 + [0] * 7

    def shown(**roles):
        done = plan.run(specify("logistic", **roles), df)
        record = plan.record("logistic", done.result)
        return done.spec, disclose_record(
            record, K10, spec=done.spec, analysis=_in_the_flow(record)
        )

    without = shown(outcome="died", compare="arm", adjust=["age"])
    with_flag = shown(outcome="died", compare="arm", adjust=["age", "flag"])
    assert with_flag[1].record is not None and "term:flag" not in _ids(with_flag[1])
    assert "term:arm#1" in _ids(with_flag[1])
    assert nested_by_unwritten(with_flag, without) and nested_by_unwritten(without, with_flag)
    # a variable whose term is written: two models as any two
    with_site = shown(outcome="died", compare="arm", adjust=["age", "site"])
    assert "term:site#1" in _ids(with_site[1])
    assert not nested_by_unwritten(with_site, without)
    # another outcome, the same model, or neither inside the other: no such pair
    df["other"] = (np.random.default_rng(9).random(len(df)) < 0.4).astype(int)
    assert not nested_by_unwritten(with_flag, shown(outcome="other", compare="arm", adjust=["age"]))
    assert not nested_by_unwritten(without, without)
    assert not nested_by_unwritten(with_flag, shown(outcome="died", compare="arm", adjust=["site"]))
    # a model that is not written gives nothing
    gone = (with_flag[0], type(with_flag[1])("logistic", False, None, ("withheld",)))
    assert not nested_by_unwritten(gone, without)


def _classes(without: dict[str, tuple[int, int]], seed: int = 5) -> pd.DataFrame:
    """Classes of these sizes, each with this many participants without the event (``died`` is 0 for them, and
    they have no visit); the others died, and have 1 to 6 visits."""
    rng = np.random.default_rng(seed)
    parts = []
    for name, (rows, spared) in without.items():
        died = np.array([0] * spared + [1] * (rows - spared))
        rng.shuffle(died)
        parts.append(pd.DataFrame({"nyha": name, "died": died}))
    df = pd.concat(parts, ignore_index=True)
    df["age"] = rng.normal(62, 9, len(df)).round(1)
    df["days"] = rng.exponential(400, len(df)).round() + 1
    df["visits"] = np.where(df["died"] == 1, rng.integers(1, 7, len(df)), 0)
    df["years"] = rng.uniform(1, 4, len(df)).round(2)
    return df


NEARLY_ALL = {"I": (300, 120), "II": (300, 120), "III": (100, 9), "IV": (40, 4)}


def test_a_cox_term_does_not_count_the_rows_without_the_event() -> None:
    """Classes III (100 rows, 9 without the event) and IV (40 rows, 4 without): counted, the rows without the
    event withheld the ratios of the two classes with the most events, crude and adjusted. No count stands beside
    a term, so nothing written shows them."""
    from biostats import plan
    from biostats.plan import specify
    from core.disclosure import FOR_THE_OTHER_SIDE

    df = _classes(NEARLY_ALL)
    for adjust in ([], ["age"]):
        done = plan.run(
            specify("cox", time="days", event="died", compare="nyha", adjust=adjust), df
        )
        record = plan.record("cox", done.result)
        assert record.part("term:nyha#3").support == {"rows": 40, "events": 36, "censored": 4}
        shown = _disclose(record, compare="nyha")
        assert {"term:nyha#1", "term:nyha#2", "term:nyha#3"} <= set(_ids(shown)), adjust
        assert shown.withheld == ()
        assert all(part.support == {} for part in shown.record.parts if part.id != "model")
        # its caution reads the counts the cell is judged on: 36 events, not 4 rows without
        assert shown.record.part("term:nyha#3").number("sparse").value is False
    # beside curves by class, which count the three and withhold III and IV: the terms stay
    curves = plan.run(specify("survival", time="days", event="died", group="nyha"), df)
    for order in ([curves, done], [done, curves]):
        both, named = _package(order)
        by_kind = {one.kind: one for one in both}
        assert named == {"nyha": {"III": FOR_THE_OTHER_SIDE, "IV": FOR_THE_OTHER_SIDE}}
        assert [
            p.labels["group"] for p in by_kind["survival"].record.parts if p.id.startswith("curve#")
        ] == ["I", "II"]
        assert {"term:nyha#2", "term:nyha#3"} <= set(_ids(by_kind["cox"]))
    # a table of the classes written elsewhere in the package gives each class's rows: the three counts again
    from core.disclosure import disclose_record

    tabled = disclose_record(
        record, K10, spec=done.spec, analysis=_in_the_flow(record), tabulated=["nyha"]
    )
    assert "term:nyha#1" in _ids(tabled) and "term:nyha#3" not in _ids(tabled)
    assert tabled.withheld[0] == f"2 terms of nyha withheld: {CELL}"


def test_a_cox_cell_of_few_rows_or_few_events_keeps_its_rule() -> None:
    """Fewer than k rows, or nobody with the event, withholds the model; fewer than k events withholds the term,
    with the smallest class as its cover: together they hold k rows and k events."""
    from biostats import plan
    from biostats.plan import specify

    def shown(classes: dict):
        done = plan.run(
            specify("cox", time="days", event="died", compare="nyha", adjust=["age"]),
            _classes(classes),
        )
        return _disclose(plan.record("cox", done.result), compare="nyha")

    few_rows = shown({**NEARLY_ALL, "IV": (9, 4)})
    assert few_rows.record is None and few_rows.withheld == (DROPS_OUT_NOBODY,)
    no_event = shown({**NEARLY_ALL, "IV": (40, 40)})
    assert no_event.record is None and no_event.withheld == (DROPS_OUT_NOBODY,)
    few_events = shown({**NEARLY_ALL, "IV": (40, 34)})  # 6 events
    written = set(_ids(few_events))
    assert {"model", "reference:nyha", "term:nyha#1", "term:age"} <= written
    assert not {"term:nyha#2", "term:nyha#3"} & written
    assert few_events.withheld == (
        f"1 term of nyha withheld: {CELL_OF_TWO}",
        f"1 more term of nyha withheld: {COVER_OF_TWO}",
        FIT,
    )
    nine = shown({**NEARLY_ALL, "IV": (20, 11)})  # 20 rows, 9 events
    assert "term:nyha#3" not in _ids(nine) and "term:nyha#1" in _ids(nine)


def test_a_count_term_does_not_count_the_rows_without_an_event_and_the_model_writes_neither_side() -> (
    None
):
    """A class of 40 rows, 4 of them without a visit: its term is written, with and without an offset. The model
    wrote its rows with a visit and without one: beside a term, which gives its class's events, they gave the
    class's rows without a visit."""
    from biostats import plan
    from biostats.plan import specify
    from core.disclosure import DisclosurePolicy as Policy
    from core.disclosure import FollowedToTheEvent, disclose_record

    df = _classes(NEARLY_ALL)
    for more in ({}, {"person_time": "years"}):
        done = plan.run(
            specify("count", count="visits", compare="nyha", adjust=["age"], **more), df
        )
        record = plan.record("count", done.result)
        held = record.part("model").support
        assert (held["rows_with_events"], held["rows_without_events"]) == (487, 253)
        shown = disclose_record(record, K10, spec=done.spec, analysis=_in_the_flow(record))
        assert {"term:nyha#1", "term:nyha#2", "term:nyha#3"} <= set(_ids(shown)), more
        assert set(shown.record.part("model").support) == {"rows", "events", "df_resid"}
        assert "rows_with_events" not in _written(shown)
    # nearly everybody has a visit: 3 rows without one in all. The model as a whole counts them, unless the study
    # declares that its participants are followed to the outcome
    nearly = _classes({"I": (200, 1), "II": (170, 1), "III": (30, 1)})
    done = plan.run(specify("count", count="visits", compare="nyha", adjust=["age"]), nearly)
    record = plan.record("count", done.result)
    plain = disclose_record(record, K10, spec=done.spec, analysis=_in_the_flow(record))
    assert plain.record is None and plain.withheld == (MODEL_WITHHELD,)
    declared = Policy(declared=(FollowedToTheEvent("visits", reason="every participant is seen"),))
    lifted = disclose_record(record, declared, spec=done.spec, analysis=_in_the_flow(record))
    assert lifted.record is not None and "term:nyha#2" in _ids(lifted)
    said = lifted.record.part("declaration")
    assert said.number("followed_to_the_event").value is True
    assert said.number("share_without_the_event").value == 0.0075  # 3 of 400
    assert said.number("beyond_the_cap").value is False and lifted.withheld == ()


def test_a_yes_no_outcome_fitted_as_a_count_keeps_the_logistic_rule() -> None:
    """A class of 40 rows, 36 with the event: the logistic model withholds its term. Fitted as a Poisson model of
    the same 0 and 1, the term was written, and its standard error gave the 36: every count is 0 or 1, the events
    are the rows with an event, and the cells are judged on three counts."""
    from biostats import plan
    from biostats.plan import specify
    from core.disclosure import DisclosurePolicy as Policy
    from core.disclosure import FollowedToTheEvent, disclose_record

    df = _classes(NEARLY_ALL)
    logistic = plan.run(specify("logistic", outcome="died", compare="nyha", adjust=["age"]), df)
    of_logistic = _disclose(plan.record("logistic", logistic.result), compare="nyha")
    assert "term:nyha#1" in _ids(of_logistic)
    assert not {"term:nyha#2", "term:nyha#3"} & set(_ids(of_logistic))
    as_count = plan.run(specify("count", count="died", compare="nyha", adjust=["age"]), df)
    record = plan.record("count", as_count.result)
    model = record.part("model").support
    assert model["events"] == model["rows_with_events"] == 487
    shown = disclose_record(record, K10, spec=as_count.spec, analysis=_in_the_flow(record))
    assert "term:nyha#1" in _ids(shown)
    assert not {"term:nyha#2", "term:nyha#3"} & set(_ids(shown))
    assert shown.withheld[0] == f"2 terms of nyha withheld: {CELL}"
    # and a declaration does not apply to it: both sides of a yes/no outcome are counted
    declared = Policy(declared=(FollowedToTheEvent("died"),))
    for done, kind in ((as_count, "count"), (logistic, "logistic")):
        record = plan.record(kind, done.result)
        said = disclose_record(record, declared, spec=done.spec, analysis=_in_the_flow(record))
        assert any("does not apply to a model of a yes/no outcome" in w for w in said.withheld), (
            kind
        )
        assert "declaration" not in _ids(said) and "term:nyha#3" not in _ids(said)


def test_a_pooled_cox_term_reads_its_smallest_rows_and_events() -> None:
    from biostats import plan
    from biostats.plan import specify
    from core.disclosure import disclose_record

    mi = _imputed_age(_classes(NEARLY_ALL))
    done = plan.run(
        specify("cox", pooled=True, time="days", event="died", compare="nyha", adjust=["age"]),
        mi=mi,
    )
    record = plan.record("cox_mi", done.result)
    # a cell's counts are the smallest over the imputed datasets (the class is recorded for every row)
    assert record.part("term:nyha#3").support == {"rows": 40, "events": 36, "censored": 4}
    assert record.part("model").support["censored_min"] == 253
    shown = disclose_record(record, K10, spec=done.spec, analysis=_in_the_flow(record))
    assert {"term:nyha#1", "term:nyha#2", "term:nyha#3"} <= set(_ids(shown))


def test_an_outcome_declared_followed_to_the_event_writes_its_small_counts_without_the_event() -> (
    None
):
    """A cohort followed to the event: 500 rows, 13 without it (5, 5 and 3 in its three sites). No group of its
    curves holds 10 rows without the event: nothing was written. The study can declare that its participants are
    followed to that outcome: the few without the event are then no outcome to protect, their count is written,
    and each result says under which declaration."""
    from biostats import plan
    from biostats.plan import specify
    from core.disclosure import DisclosurePolicy as Policy
    from core.disclosure import FollowedToTheEvent, disclose_record, reading_notes

    df = _classes({"A": (200, 5), "B": (150, 5), "C": (150, 3)})
    df["other"] = (np.random.default_rng(2).random(len(df)) < 0.4).astype(int)
    curves = plan.run(specify("survival", time="days", event="died", group="nyha"), df)
    record = plan.record("survival", curves.result)

    def shown(policy, done=curves, of=record):
        return disclose_record(of, policy, spec=done.spec, analysis=_in_the_flow(of))

    assert shown(K10).record is None
    declaration = FollowedToTheEvent(
        "died",
        reason="deaths are read in the national register",
        role="investigator",
        date="2026-11-14",
    )
    declared = Policy(declared=(declaration,))
    written = shown(declared)
    groups = {p.labels["group"]: p for p in written.record.parts if p.id.startswith("curve#")}
    assert set(groups) == {"A", "B", "C"} and groups["C"].support["censored"] == 3
    said = written.record.part("declaration")
    assert [n.name for n in said.numbers] == [
        "followed_to_the_event",
        "share_without_the_event",
        "cap",
        "beyond_the_cap",
    ]
    assert said.number("share_without_the_event").value == 0.026  # 13 of 500
    assert said.number("cap").value == 0.05 and said.number("beyond_the_cap").value is False
    assert written.withheld == ()
    # the policy block of the package and its reading notes say it
    block = declared.to_dict()["declared"]
    assert block == [
        {
            "outcome": "died",
            "declared": "followed to the event",
            "reason": "deaths are read in the national register",
            "role": "investigator",
            "date": "2026-11-14",
            "cap": 0.05,
            "beyond_the_cap": False,
        }
    ]
    assert "followed to died" in reading_notes(declared)[-1] and len(reading_notes(declared)) == 12
    assert "to died" not in " ".join(reading_notes(K10))
    # its Cox model as a whole likewise (13 rows without the event are enough; with 7, the model needed it)
    seven = _classes({"A": (200, 3), "B": (150, 2), "C": (150, 2)})
    cox = plan.run(specify("cox", time="days", event="died", compare="nyha", adjust=["age"]), seven)
    of_cox = plan.record("cox", cox.result)
    assert shown(K10, cox, of_cox).record is None
    assert {"term:nyha#1", "term:nyha#2", "declaration"} <= set(_ids(shown(declared, cox, of_cox)))
    # another outcome of the same package keeps the rule
    others = plan.run(specify("survival", time="days", event="other", group="nyha"), df)
    of_others = plan.record("survival", others.result)
    assert "declaration" not in _ids(shown(declared, others, of_others))
    # a group where nobody had the event stays withheld, and rows and events still need k each
    none = df.copy()
    none.loc[none["nyha"] == "C", "died"] = [1] * 6 + [0] * 144
    few = plan.run(specify("survival", time="days", event="died", group="nyha"), none)
    of_few = plan.record("survival", few.result)
    kept = shown(Policy(declared=(FollowedToTheEvent("died", beyond_the_cap=True),)), few, of_few)
    assert "C" not in [p.labels.get("group") for p in kept.record.parts]
    # declared once, by its outcome, and by a declaration
    with pytest.raises(ValueError, match="declared once"):
        Policy(declared=(declaration, FollowedToTheEvent("died")))
    with pytest.raises(ValueError, match="names its outcome"):
        FollowedToTheEvent(" ")
    with pytest.raises(ValueError):
        Policy(declared=("died",))  # type: ignore[arg-type]


def test_a_declaration_holds_beyond_its_cap_only_by_an_explicit_override_that_each_result_says() -> (
    None
):
    """More than a twentieth of the rows without the event: the outcome is not one its participants are followed
    to, and the declaration does not hold. The person who declares can ask that it hold all the same: off unless
    asked for, with a warning, and written with the actual share where a reader of the result finds it."""
    from biostats import plan
    from biostats.plan import specify
    from core.disclosure import MOST_WITHOUT_THE_EVENT, FollowedToTheEvent, disclose_record
    from core.disclosure import DisclosurePolicy as Policy

    assert MOST_WITHOUT_THE_EVENT == 0.05 and FollowedToTheEvent("died").beyond_the_cap is False
    df = _classes(
        {"A": (200, 9), "B": (150, 9), "C": (150, 9)}
    )  # 27 of 500 without the event: 5.4%
    curves = plan.run(specify("survival", time="days", event="died", group="nyha"), df)
    record = plan.record("survival", curves.result)

    def shown(policy):
        return disclose_record(record, policy, spec=curves.spec, analysis=_in_the_flow(record))

    plain = shown(Policy(declared=(FollowedToTheEvent("died"),)))
    assert (
        plain.record is None
    )  # as without a declaration: no group holds 10 rows without the event
    refused = [w for w in plain.withheld if "does not hold for this result" in w]
    assert (
        len(refused) == 1 and "5.4% of its rows are without the event, more than 5%" in refused[0]
    )
    assert "explicit override" in refused[0]
    asked = Policy(declared=(FollowedToTheEvent("died", beyond_the_cap=True),))
    over = shown(asked)
    assert over.record is not None
    groups = {p.labels["group"]: p for p in over.record.parts if p.id.startswith("curve#")}
    assert set(groups) == {"A", "B", "C"} and groups["A"].support["censored"] == 9
    # the warning to the person who shares, and the record a reader of the result finds
    (warning,) = [w for w in over.withheld if w.startswith("Warning: ")]
    assert "holds for this result by an explicit override" in warning
    assert "5.4% of its rows are without the event, more than 5%" in warning
    said = over.record.part("declaration")
    assert said.number("beyond_the_cap").value is True
    assert (
        said.number("share_without_the_event").value == 0.054 and said.number("cap").value == 0.05
    )
    assert asked.to_dict()["declared"][0]["beyond_the_cap"] is True
    # under the cap, the override changes nothing and nothing warns
    under = _classes({"A": (200, 5), "B": (150, 5), "C": (150, 3)})
    curves = plan.run(specify("survival", time="days", event="died", group="nyha"), under)
    record = plan.record("survival", curves.result)
    quiet = shown(asked)
    assert (
        quiet.withheld == ()
        and quiet.record.part("declaration").number("beyond_the_cap").value is False
    )
    with pytest.raises(ValueError, match="True or False"):
        FollowedToTheEvent("died", beyond_the_cap="yes")  # type: ignore[arg-type]


def test_the_cells_of_a_grouping_are_named_once_with_the_strictest_reason() -> None:
    from core.disclosure import (
        ASKED,
        FOR_ITS_ROWS,
        FOR_THE_OTHER_SIDE,
        cover_for,
        named_together,
    )

    first = {"site": {"Delta": FOR_THE_OTHER_SIDE, "Gamma": ASKED}}
    second = {"site": {"Delta": FOR_ITS_ROWS, "Gamma": FOR_THE_OTHER_SIDE}, "arm": {"drug": ASKED}}
    together = named_together(first, second)
    assert together == {
        "site": {"Delta": FOR_ITS_ROWS, "Gamma": FOR_THE_OTHER_SIDE},
        "arm": {"drug": ASKED},
    }
    assert named_together(second, first) == together and named_together() == {}
    assert first == {"site": {"Delta": FOR_THE_OTHER_SIDE, "Gamma": ASKED}}  # left as given
    # curves and a logistic model count the rows without the event: they withhold every cell named
    from biostats.plan import ResultRecord

    for kind in ("survival", "logistic"):
        assert cover_for(ResultRecord(kind), together) == {
            "site": ("Delta", "Gamma"),
            "arm": ("drug",),
        }
    # a Cox or a count term does not count the rows without the event: a cell withheld for that side alone
    # is not asked of it, unless the package writes a count per level of the variable
    for kind in ("cox", "count"):
        assert cover_for(ResultRecord(kind), together) == {"site": ("Delta",), "arm": ("drug",)}
        assert cover_for(ResultRecord(kind), together, tabulated=["site"]) == {
            "site": ("Delta", "Gamma"),
            "arm": ("drug",),
        }
    assert cover_for(ResultRecord("survival"), {}) == {}


def test_the_time_step_is_read_on_the_usual_gap_between_two_recorded_times() -> None:
    """The data's unit was the smallest gap between two recorded times, and a step five of them at least: a
    follow-up in whole years (1 to 4) got a step of 5 and no time at all, and one row recorded at 45.1 days in a
    study recorded in days brought the step down to 1, where the times given are event days again."""
    rng = np.random.default_rng(3)
    years = pd.DataFrame(
        {"days": rng.integers(1, 5, 400).astype(float), "died": (rng.random(400) < 0.5).astype(int)}
    )
    (curve,) = _curve_shown(years)
    assert curve.number("time_step").value == 1
    assert curve.number("times").value == [1, 2, 3, 4]  # four times can hold a point
    assert curve.number("followed_until").value == 4
    rng = np.random.default_rng(5)
    ninety = pd.DataFrame(
        {
            "days": rng.integers(1, 91, 928).astype(float),
            "died": (rng.random(928) < 0.5).astype(int),
        }
    )
    (curve,) = _curve_shown(ninety)
    times = curve.number("times").value
    assert curve.number("time_step").value == 5 and times == [float(t) for t in range(5, 90, 5)]
    for odd in (45.25, 45.1):
        finer = ninety.copy()
        finer.loc[0, "days"] = odd
        (curve,) = _curve_shown(finer)
        assert curve.number("time_step").value == 5 and curve.number("times").value == times, odd
    # whole months over four years, and one time written 12.1
    months = pd.DataFrame(
        {
            "days": rng.integers(1, 49, 600).astype(float),
            "died": (rng.random(600) < 0.5).astype(int),
        }
    )
    months.loc[0, "days"] = 12.1
    (curve,) = _curve_shown(months)
    assert curve.number("time_step").value == 5
    assert curve.number("times").value == [float(t) for t in range(5, 50, 5)]


def _curve_with_unit(frame: pd.DataFrame, unit=None):
    """The one curve of ``frame`` as a package shows it, with the unit declared for its time variable, and what
    the person who shares is told."""
    from biostats import plan
    from biostats.plan import specify
    from core.disclosure import disclose_record

    done = plan.run(specify("survival", time="days", event="died"), frame)
    record = plan.record("survival", done.result)
    shown = disclose_record(
        record,
        K10,
        spec=done.spec,
        analysis=_in_the_flow(record),
        units=None if unit is None else {"days": unit},
    )
    (curve,) = [p for p in shown.record.parts if p.id.startswith("curve#")]
    return curve, shown.withheld


def _names(part) -> list[str]:
    return [n.name for n in part.numbers]


def test_the_unit_a_study_declares_for_its_times_sets_the_step() -> None:
    """Read on the data alone, the unit is the gap that a twentieth of the gaps stay below: 7% of the rows
    recorded a tenth of a month finer than the others brought the step from 5 months to 0.5, and 10 of the 12
    times given were event months. No reading of the data resists every such record: the study declares the
    unit, and the step is read on the larger of the two."""
    from core.disclosure import TIME_UNITS, TimeUnit

    day, month, year = (TimeUnit(name, 1) for name in ("day", "month", "year"))
    rng = np.random.default_rng(5)
    # a 90-day study with a few rows a tenth of a day later: by the data's own unit the step falls to 1
    ninety = pd.DataFrame(
        {
            "days": rng.integers(1, 91, 928).astype(float),
            "died": (rng.random(928) < 0.5).astype(int),
        }
    )
    finer = ninety.copy()
    finer.loc[finer.index[:60], "days"] += 0.1
    curve, told = _curve_with_unit(finer)
    assert curve.number("time_step").value < 5 and curve.number("time_unit_from").value == "data"
    assert "time_unit" not in _names(curve) and told == ()
    curve, told = _curve_with_unit(finer, day)
    assert curve.number("time_step").value == 5
    assert all(float(t) == int(t) and t % 5 == 0 for t in curve.number("times").value)
    # the package writes the unit, and which unit set the step
    assert curve.number("time_unit").value == "day" and curve.number("time_unit_size").value == 1.0
    assert curve.number("time_unit_from").value == "declared"
    assert (
        len(told) == 1 and "more than 5 times coarser than the unit read on its values" in told[0]
    )
    # whole months over four years, 7% of the rows written x.1
    months = pd.DataFrame(
        {
            "days": rng.integers(1, 49, 600).astype(float),
            "died": (rng.random(600) < 0.5).astype(int),
        }
    )
    months.loc[months.index[:42], "days"] += 0.1
    assert _curve_with_unit(months)[0].number("time_step").value == 0.5
    curve, _ = _curve_with_unit(months, month)
    assert curve.number("time_step").value == 5
    assert all(t % 5 == 0 for t in curve.number("times").value)
    # a day declared in a variable of months, with its size: the step of the data again
    curve, told = _curve_with_unit(months, TimeUnit("day", 1 / 30.4375))
    assert curve.number("time_step").value == 0.5 and curve.number("time_unit_from").value == "data"
    assert told == ()  # 0.033 against 0.1: within five times
    # whole years 1 to 4: step 1 and four times, declared or not
    years = pd.DataFrame(
        {
            "days": np.random.default_rng(3).integers(1, 5, 400).astype(float),
            "died": (np.random.default_rng(4).random(400) < 0.5).astype(int),
        }
    )
    for unit in (None, year):
        curve, told = _curve_with_unit(years, unit)
        assert curve.number("time_step").value == 1 and curve.number("times").value == [1, 2, 3, 4]
        assert told == ()
    # hours declared over whole days: a unit finer than the record lowers nothing
    plain, _ = _curve_with_unit(ninety)
    curve, told = _curve_with_unit(ninety, TimeUnit("hour", 1 / 24))
    assert curve.number("time_step").value == plain.number("time_step").value == 5
    assert curve.number("times").value == plain.number("times").value
    assert (
        curve.number("time_unit_from").value == "data" and curve.number("time_unit").value == "hour"
    )
    assert len(told) == 1 and "more than 5 times finer" in told[0]
    # a day declared over visits every 30 days: the step comes from the 30-day record
    visits = pd.DataFrame(
        {
            "days": 30.0 * rng.integers(1, 41, 600),
            "died": (rng.random(600) < 0.5).astype(int),
        }
    )
    plain, _ = _curve_with_unit(visits)
    curve, told = _curve_with_unit(visits, day)
    assert curve.number("time_step").value == plain.number("time_step").value >= 30
    assert curve.number("time_unit_from").value == "data" and len(told) == 1
    # a unit longer than the follow-up (nine months, declared in years): no time, no "followed until 0"
    short = years.assign(days=np.random.default_rng(6).integers(1, 10, 400) / 12)
    curve, told = _curve_with_unit(short, year)
    assert curve.number("time_step").value == 1 and curve.number("times").value == []
    assert curve.number("followed_until").value is None
    assert any("longer than the follow-up shown" in said for said in told)
    # a closed list of names, and a size above 0
    assert TIME_UNITS == ("hour", "day", "week", "month", "year")
    assert day.to_dict() == {"name": "day", "size": 1.0}
    for name, size in (
        ("fortnight", 1),
        ("day", 0),
        ("day", -1),
        ("day", float("inf")),
        ("day", True),
    ):
        with pytest.raises(ValueError):
            TimeUnit(name, size)


def test_a_limit_of_the_median_below_the_first_step_is_not_written() -> None:
    """A lower limit of 3 days under a step of 5 was rounded down and written as 0: a median survival of at least
    no time at all."""
    from biostats.plan import DETERMINISTIC, EXACT, Number, Part, ResultRecord

    days = [float(d) for d in range(1, 201)]
    curve = Part(
        "curve#0",
        (
            Number("rows", 200, EXACT),
            Number("events", 100, EXACT),
            Number("median", 150.0, EXACT, fine=True),
            Number("median_lower", 3.0, EXACT, fine=True),
            Number("median_upper", 172.0, EXACT, fine=True),
            Number("times", days[::2], EXACT, fine=True),
            Number("survival", [1 - (j + 1) / 200 for j in range(100)], DETERMINISTIC, scale=1.0),
            Number("events_at", [1] * 100, EXACT, fine=True),
            Number("follow_times", days, EXACT, fine=True),
            Number("followed", [200 - j for j in range(200)], EXACT, fine=True),
        ),
        {"rows": 200, "events": 100, "censored": 100},
        {"group": "all"},
    )
    shown = _disclose(ResultRecord("survival", parts=(curve,))).record.part("curve#0")
    assert shown.number("time_step").value == 5
    assert shown.number("median").value == 150 and shown.number("median_upper").value == 175
    assert shown.number("median_lower").value is None  # and not named as later than the follow-up
    assert shown.number("later_than_followed_until").value == []


def test_the_time_step_is_read_on_the_groups_that_are_shown() -> None:
    """A group followed 300 days, beside a small group and its cover, which is followed 3,000 days and not shown:
    the step was read over every group, and the curve shown had 6 times at a step of 50 where its own follow-up
    gives it a step of 5."""
    rng = np.random.default_rng(7)

    def followed(name: str, rows: int, days: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "group": name,
                "days": rng.integers(1, days + 1, rows).astype(float),
                "died": (rng.random(rows) < 0.5).astype(int),
            }
        )

    small = followed("small", 5, 300)
    df = pd.concat(
        [followed("main", 300, 300), followed("long", 100, 3000), small], ignore_index=True
    )
    (main,) = _curve_shown(df, "group")
    assert main.labels == {"group": "main"} and main.number("time_step").value == 5
    assert len(main.number("times").value) >= 10
    # shown together, the two large groups share the step of the longer follow-up
    both = _curve_shown(df[df["group"] != "small"], "group")
    assert [p.number("time_step").value for p in both] == [50, 50]


def test_a_curve_s_values_take_the_decimals_that_the_rows_still_followed_allow() -> None:
    """With every digit, a value and its lower limit give the rows at risk at each time, and so the rows whose
    follow-up ended between two times: 3 of them, in a study with a common closing date. Three decimals still gave
    them back late in a curve, where 30 to 60 rows are at risk."""
    from biostats import plan
    from biostats.plan import ITERATIVE_LOOSE
    from biostats.survival import compare_survival
    from core.disclosure import DECIMALS_BY_ROWS, decimals_at

    assert DECIMALS_BY_ROWS == ((2000, 4), (400, 3), (0, 2))
    held = (0, 10, 399, 400, 1999, 2000, 50_000)
    assert [decimals_at(rows) for rows in held] == [2, 2, 2, 3, 3, 4, 4]
    record = plan.record("survival", compare_survival(_cohort(), "days", "died", "arm"))
    disclosed = _disclose(record)
    for part in disclosed.record.parts[:2]:
        # about 200 rows in each arm: two decimals at every time
        assert part.number("decimals").value == [2] * len(part.number("times").value)
        for name in ("survival", "lower", "upper"):
            number = part.number(name)
            assert number.value and all(v == round(v, 2) for v in number.value), name
            # a value computed again agrees to one unit of the last decimal, and is compared so
            assert number.cls == ITERATIVE_LOOSE and number.scale == 10.0, name
        full = record.part(part.id).number("survival").value
        assert not all(v == round(v, 3) for v in full)  # the record itself holds every digit
    # 3,000 rows whose follow-up ends on any day of 2,000: four decimals while 2,000 rows are followed, three
    # while 400 are, then two; the curve says which, time by time
    rng = np.random.default_rng(11)
    long = pd.DataFrame(
        {
            "days": rng.integers(1, 2001, 3000).astype(float),
            "died": (rng.random(3000) < 0.5).astype(int),
        }
    )
    (curve,) = _curve_shown(long)
    times, digits = curve.number("times").value, curve.number("decimals").value
    assert digits == sorted(digits, reverse=True) and set(digits) == {4, 3, 2}
    followed = [int((long["days"] >= t).sum()) for t in times]
    assert digits == [decimals_at(rows) for rows in followed]
    assert followed[digits.index(3) - 1] >= 2000 > followed[digits.index(3)]
    assert followed[digits.index(2) - 1] >= 400 > followed[digits.index(2)]
    whole = plan.record("survival", compare_survival(long, "days", "died", None))
    steps = whole.part(curve.id).number("times").value
    for name in ("survival", "lower", "upper"):
        full = dict(zip(steps, whole.part(curve.id).number(name).value, strict=True))
        written = curve.number(name)
        for t, value, places in zip(times, written.value, digits, strict=True):
            assert value == round(full[max(s for s in steps if s <= t)], places), name
        assert written.scale == 10.0  # one unit of the fewest decimals written
    # 5,000 rows, 500 of them with the event in the first 100 days and the others followed to day 1000: four
    # decimals throughout, and the time the flat end of the curve repeats is written as that time is
    plateau = pd.DataFrame(
        {
            "days": np.r_[rng.integers(1, 101, 500), np.full(4500, 1000)].astype(float),
            "died": np.r_[np.ones(500), np.zeros(4500)].astype(int),
        }
    )
    (curve,) = _curve_shown(plateau)
    assert set(curve.number("decimals").value) == {4}
    assert curve.number("times").value[-1] == curve.number("followed_until").value == 1000
    assert curve.number("survival").value[-1] == curve.number("survival").value[-2] == 0.9
    assert curve.number("survival").scale == 0.1
    assert any(v != round(v, 3) for v in curve.number("lower").value)


def _greenwood(value: float, limit: float, z: float) -> float | None:
    """Greenwood's sum, from a value of a curve and one limit of its log-log interval (None outside its domain)."""
    if not (0 < limit < 1 and 0 < value < 1) or limit == value:
        return None
    se = abs(math.log(-math.log(limit)) - math.log(-math.log(value))) / z
    return (se * abs(math.log(value))) ** 2


def _greenwood_range(
    value: float, low: float, high: float, half: float, z: float
) -> tuple[float, float] | None:
    """The range of Greenwood's sum that the written digits of a value and of its two limits allow."""
    bounds = []
    for limit in (low, high):
        corners = [
            _greenwood(a, b, z)
            for a in (value - half, value + half)
            for b in (limit - half, limit + half)
        ]
        if all(c is not None for c in corners):
            bounds.append((min(corners), max(corners)))
    if not bounds:
        return None
    return max(b[0] for b in bounds), min(b[1] for b in bounds)


def _read_stretch(
    written: dict[str, list], halves: list[float], j: int, most_rows: int, z: float, k: int = 10
) -> dict[int, tuple[int, int]] | None:
    """
    What a reader gets from the digits of a curve for the stretch that ends at its place j, when no follow-up ends
    inside it: the whole pairs (rows at risk, events) that agree with the value and the limits written at both of
    its ends, as {rows: (fewest events, most events)}. The stretch multiplies the curve by (n - d) / n and adds
    d / (n (n - d)) to Greenwood's sum. None when the digits give no reading.
    """
    values, low, high = written["survival"], written["lower"], written["upper"]
    half, before = halves[j], halves[j - 1]
    least = (values[j] - half) / (values[j - 1] + before)
    most = (values[j] + half) / max(values[j - 1] - before, 1e-12)
    now = _greenwood_range(values[j], low[j], high[j], half, z)
    earlier = (
        (0.0, 0.0)
        if j == 1
        else _greenwood_range(values[j - 1], low[j - 1], high[j - 1], before, z)
    )
    if now is None or earlier is None:
        return None
    g_low, g_high = max(now[0] - earlier[1], 0.0), now[1] - earlier[0]
    found: dict[int, tuple[int, int]] = {}
    if g_high <= 0:
        return found
    for rows in range(2 * k, most_rows + 1):
        fewest = max(
            k,
            math.ceil(rows * (1 - most) - 1e-9),
            math.ceil(g_low * rows * rows / (1 + g_low * rows) - 1e-9),
        )
        largest = min(
            rows - k,
            math.floor(rows * (1 - least) + 1e-9),
            math.floor(g_high * rows * rows / (1 + g_high * rows) + 1e-9),
        )
        if fewest <= largest:
            found[rows] = (fewest, largest)
    return found


def _lost_between(
    first: dict[int, tuple[int, int]], second: dict[int, tuple[int, int]]
) -> list[int]:
    """The values the rows whose follow-up ended between two stretches can take: the rows left at the end of the
    first, less the rows at risk in the second."""
    left: set[int] = set()
    for rows, (fewest, largest) in first.items():
        left.update(range(rows - largest, rows - fewest + 1))
    return sorted({a - b for a in left for b in second if a - b >= 0})


def test_three_rows_lost_late_in_a_curve_do_not_come_back_from_its_digits() -> None:
    """75 events before day 1500, 15 rows followed to day 2000, and 3 rows whose follow-up ends late, after the last
    event before a time given (93 rows). The stretches on either side hold 39 and 26 rows at risk. A reader who
    keeps every whole pair of rows at risk and events that agrees with the digits of each stretch got one count for
    the rows lost between the two: 3, at three decimals."""
    from statistics import NormalDist

    from biostats import plan
    from biostats.survival import compare_survival

    z = NormalDist().inv_cdf(0.975)
    for seed in (1, 3):
        rng = np.random.default_rng(seed)
        days = np.sort(rng.choice(np.arange(1, 1500), 75, replace=False))
        late = pd.DataFrame(
            {
                "days": np.r_[days, np.full(15, 2000)].astype(float),
                "died": np.r_[np.ones(75), np.zeros(15)].astype(int),
            }
        )
        (before,) = _curve_shown(late)
        third_from_the_end = before.number("times").value[-3]
        gap = float(days[days <= third_from_the_end].max() + 1)
        lost = pd.DataFrame({"days": [gap] * 3, "died": [0] * 3})
        frame = pd.concat([late, lost], ignore_index=True)
        record = plan.record("survival", compare_survival(frame, "days", "died", None))
        (curve,) = [p for p in _disclose(record).record.parts if p.id.startswith("curve#")]
        times, digits = curve.number("times").value, curve.number("decimals").value
        assert third_from_the_end in times and set(digits) == {2}, seed
        where = times.index(third_from_the_end)
        full = record.part(curve.id)
        steps = full.number("times").value
        place = [max(i for i, s in enumerate(steps) if s <= t) for t in times]

        def read(decimals: list[int], full=full, place=place, where=where, rows=len(frame)):
            """The counts of rows lost after the stretch that ends at ``where``, as the reader gets them from the
            curve written to these decimals."""
            written = {
                name: [
                    1.0,
                    *(
                        round(full.number(name).value[i], d)
                        for i, d in zip(place, decimals, strict=True)
                    ),
                ]
                for name in ("survival", "lower", "upper")
            }
            halves = [0.0, *(0.5 * 10.0**-d for d in decimals)]
            first, second = (
                _read_stretch(written, halves, j, rows, z) for j in (where + 1, where + 2)
            )
            assert first and second
            return _lost_between(first, second)

        # the values the layer writes are those of the record, to the decimals it says
        for name in ("survival", "lower", "upper"):
            assert curve.number(name).value == [
                round(full.number(name).value[i], d) for i, d in zip(place, digits, strict=True)
            ], name
        assert read([3] * len(times)) == [3], seed  # what three decimals gave back
        as_written = read(digits)
        assert 3 in as_written and len(as_written) >= 20, (seed, as_written)


def test_a_pooled_result_is_compared_with_the_rows_its_datasets_read() -> None:
    """400 rows; the event is missing for 20 of them and no imputation fills it. Every dataset's curves and models
    read 380 rows, and the flow said 400 analysed and none missing: a pooled result's rows were not compared with
    its line of the flow."""
    from biostats import plan
    from biostats.mi import MIData
    from biostats.plan import specify
    from core.disclosure import DisclosedAnalysis, disclose_record

    rng = np.random.default_rng(2)
    truth = _cohort()
    base = truth.copy()
    gap = rng.random(len(base)) < 0.15
    base.loc[gap, "age"] = np.nan
    base["died"] = base["died"].astype(float)
    base.loc[base.index[:20], "died"] = np.nan
    completed = []
    for _ in range(5):
        one = base.copy()
        one.loc[gap, "age"] = truth.loc[gap, "age"] + rng.normal(0, 5, gap.sum())
        completed.append(one)
    mi = MIData.from_completed(base, completed, methods={"age": "pmm"})
    for kind, roles in (
        ("survival", {"time": "days", "event": "died", "group": "arm"}),
        ("cox", {"time": "days", "event": "died", "compare": "arm", "adjust": ["age"]}),
        ("logistic", {"outcome": "died", "compare": "arm", "adjust": ["age"]}),
    ):
        done = plan.run(specify(kind, pooled=True, **roles), mi=mi)
        record = plan.record(f"{kind}_mi", done.result)
        assert record is not None
        assert _in_the_flow(record).n == 380, kind
        read = DisclosedAnalysis("analysis", 380, missing=20, imputations=5)
        assert disclose_record(record, K10, spec=done.spec, analysis=read).record is not None, kind
        every = DisclosedAnalysis("analysis", 400, imputations=5)
        said = disclose_record(record, K10, spec=done.spec, analysis=every)
        assert said.record is None and said.withheld == (NOT_ITS_ROWS,), kind


def test_a_pooled_poisson_model_says_whether_it_is_overdispersed() -> None:
    """A pooled count model recorded no dispersion: the application warned of one above 2 in each of the imputed
    datasets, and the package wrote the arm's standard error without a caution. The record holds the smallest and
    the largest over the datasets, and the fact is read on the largest."""
    from biostats import plan
    from biostats.plan import specify
    from biostats.regression import count_regression
    from core.disclosure import disclose_record

    truth = _cohort()
    rng = np.random.default_rng(1)
    truth["visits"] = rng.poisson(2 * rng.gamma(0.5, 2.0, len(truth)))
    mi = _imputed_age(truth)
    done = plan.run(
        specify("count", pooled=True, count="visits", compare="arm", adjust=["age"]), mi=mi
    )
    record = plan.record("count_mi", done.result)
    assert record is not None
    # as each dataset's own model gives it
    own = [
        count_regression(one, "visits", ["arm", "age"], family="poisson").dispersion
        for _, one in mi.completed()  # type: ignore[attr-defined]
    ]
    held = record.part("model")
    assert held.number("dispersion_min").value == pytest.approx(min(own))
    assert held.number("dispersion_max").value == pytest.approx(max(own))
    assert min(own) > 1.5
    disclosed = disclose_record(record, K10, spec=done.spec, analysis=_in_the_flow(record))
    assert disclosed.record is not None
    model = disclosed.record.part("model")
    assert model.number("overdispersed").value is True
    assert model.number("dispersion_max").value == pytest.approx(max(own))  # no term is withheld
    # a steady count: said too, as "not overdispersed"
    even = _cohort()
    even["visits"] = np.random.default_rng(2).poisson(2, len(even))
    mi = _imputed_age(even)
    done = plan.run(
        specify("count", pooled=True, count="visits", compare="arm", adjust=["age"]), mi=mi
    )
    record = plan.record("count_mi", done.result)
    steady = disclose_record(record, K10, spec=done.spec, analysis=_in_the_flow(record))
    assert steady.record.part("model").number("overdispersed").value is False


def test_two_cautions_travel_as_facts() -> None:
    """With a term withheld, a Poisson model lost its dispersion (2.28 on the tutorial: its standard errors are 1.5
    times too small); and at k = 5 a ratio can rest on 5 to 9 events beside a model that is not sparse as a
    whole."""
    from biostats import plan
    from biostats.regression import count_regression
    from biostats.survival import cox_regression

    df = _cohort()
    rng = np.random.default_rng(1)
    df["visits"] = rng.poisson(2 * rng.gamma(0.5, 2.0, len(df)))
    poisson = _disclose(
        plan.record("count", count_regression(df, "visits", ["arm", "age"], family="poisson")),
        compare="arm",
    ).record.part("model")
    assert (
        poisson.number("dispersion").value > 1.5 and poisson.number("overdispersed").value is True
    )
    # with a term withheld (12 rows, 3 of them with a visit), the dispersion goes with the fit's numbers, and the
    # fact stays
    rare = df.assign(rare=0)
    rare.loc[rare.index[:12], "rare"] = 1
    rare.loc[rare.index[:12], "visits"] = [4, 2, 1] + [0] * 9
    partly = _disclose(
        plan.record(
            "count", count_regression(rare, "visits", ["rare", "arm", "age"], family="poisson")
        ),
        compare="arm",
    )
    model = partly.record.part("model")
    assert "term:rare" not in _ids(partly) and "term:arm#1" in _ids(partly)
    assert "dispersion" not in [n.name for n in model.numbers]
    assert model.number("overdispersed").value is True
    even = df.assign(visits=np.random.default_rng(2).poisson(2, len(df)))
    steady = _disclose(
        plan.record("count", count_regression(even, "visits", ["arm", "age"], family="poisson")),
        compare="arm",
    ).record.part("model")
    assert steady.number("overdispersed").value is False
    # a negative binomial model's dispersion is a parameter of its fit: no such fact
    fitted = _disclose(
        plan.record(
            "count", count_regression(df, "visits", ["arm", "age"], family="negative_binomial")
        ),
        compare="arm",
    ).record.part("model")
    assert "overdispersed" not in [n.name for n in fitted.numbers]
    # a site of 20 rows, 7 of whom died: written at k = 5, and marked as resting on fewer than 10
    df = _cohort()
    df["site"] = "Beta"
    df.loc[df.index[:20], "site"] = "Gamma"
    df.loc[df.index[:20], "died"] = [1] * 7 + [0] * 13
    record = plan.record(
        "cox", cox_regression(df, "days", "died", ["site", "age"], diagnostics=False)
    )
    at_five = _disclose(record, DisclosurePolicy(5)).record
    assert at_five.part("model").number("sparse").value is False
    assert at_five.part("term:site#1").number("sparse").value is True
    assert "sparse" not in [n.name for n in at_five.part("term:age").numbers]  # no cell behind it
    assert "term:site#1" not in _ids(_disclose(record))  # at k = 10 it is withheld


def test_a_shared_spec_names_no_row_and_no_withheld_level() -> None:
    import hashlib

    from biostats import plan
    from biostats.plan import ComputedOn, canonical, specify
    from core.disclosure import DisclosedAnalysis, DisclosedRecord, disclose_record, disclose_spec

    df = _cohort()
    df["site"] = np.where(
        np.arange(len(df)) % 2 == 0, "Alpha", df["site"]
    )  # the reference: the largest
    df.loc[df.index[:12], "site"] = "CANARY-SITE"
    df.loc[df.index[:12], "died"] = [1, 1] + [0] * 10
    done = plan.run(specify("logistic", outcome="died", compare="arm", adjust=["site", "age"]), df)
    spec = done.spec.on(
        ComputedOn(
            "observed",
            "datafp",
            400,
            400,
            rows_fp="ROWSFP",
            protocol_hash="P" * 64,
            trace_head="HEAD",
        )
    )
    in_the_flow = DisclosedAnalysis("Logistic regression", 400)
    disclosed = disclose_record(
        plan.record("logistic", done.result), K10, spec=spec, analysis=in_the_flow
    )
    shared = disclose_spec(spec, disclosed, analysis=in_the_flow)
    assert shared["request_id"] == spec.request_id and shared["kind"] == "logistic"
    assert shared["variables"] == {"outcome": "died", "compare": "arm", "adjust": ["site", "age"]}
    # the request as it was asked: its id can be computed again from what is shown
    asked = {
        key: shared[key]
        for key in ("kind", "kind_version", "pooled", "variables", "options", "random")
    }
    assert shared["options"]["event_level"] is None
    assert hashlib.sha256(canonical(asked).encode("utf-8")).hexdigest() == shared["request_id"]
    # the level that was the event was decided on the data: beside the request, with a model that is shown
    assert shared["decided"]["event_level"] == "1"
    # the site of 12 rows, two of whom died, is withheld, and the smaller of the two others with it
    sizes = df["site"].value_counts()
    larger = "Beta" if sizes["Gamma"] < sizes["Beta"] else "Gamma"
    assert shared["decided"]["levels"] == {
        "arm": {"shown": ["control", "drug"], "withheld": 0},
        "site": {"shown": ["Alpha", larger], "withheld": 2},
    }
    assert shared["data"] == {
        "mode": "observed",
        "data_fp": "datafp",
        "n_population": 400,
        "n_analysed": 400,
        "rules_version": None,
        "trace_steps": None,
    }
    # an analysis the flow withholds (it left a few rows out), or the line of another analysis: the two numbers
    # beside the flow's would be a subtraction
    without = {
        "mode": "observed",
        "data_fp": "datafp",
        "rules_version": None,
        "trace_steps": None,
    }
    assert disclose_spec(spec, disclosed, analysis=None)["data"] == without
    another = DisclosedAnalysis("Kaplan-Meier", 399)
    assert disclose_spec(spec, disclosed, analysis=another)["data"] == without
    assert (
        disclose_spec(spec, disclosed, analysis=DisclosedAnalysis("x", 390, left_out=12))["data"]
        == without
    )  # another population: 402 rows
    text = json.dumps(shared)
    assert not any(
        secret in text for secret in ("CANARY", "ROWSFP", "HEAD", "PPPP", spec.fingerprint)
    )
    # a result that is withheld says nothing of its spec
    withheld = DisclosedRecord("logistic", False, None, ("withheld",))
    assert disclose_spec(spec, withheld, analysis=in_the_flow) is None


def test_the_notes_a_package_reads_its_results_with_are_fixed_words() -> None:
    from core.disclosure import reading_notes

    notes = reading_notes(DisclosurePolicy(12))
    assert len(notes) == 11 and all(isinstance(note, str) and note.endswith(".") for note in notes)
    # which unit a curve's step was read on
    assert "time_unit_from" in notes[10] and "the unit the study declares" in notes[10]
    # a fact that travels has its sentence: "overdispersed" stood in a package without a word on what it means
    assert "overdispersed" in notes[9] and "standard errors are too small" in notes[9]
    assert "above 1.5" in notes[9] and "at least one of its imputed datasets" in notes[9]
    assert "fewer than 12" in notes[0] and "selected by their outcome" in notes[1]
    # what a model with a withheld term is, and when nothing of a model is written
    assert (
        "adjusted for every variable and every level" in notes[2] and "withheld whole" in notes[2]
    )
    assert "per unit" in notes[3] and "the reference written beside it" in notes[3]
    assert "the rarer outcome of a logistic model" in notes[4] and "Below 10 (sparse)" in notes[4]
    # what a time of a curve stands for, and what follows its last time
    assert (
        "written to 2 decimals where fewer than 400 participants are still followed at that time, 3 from 400 "
        "and 4 from 2,000 (decimals gives the number at each time)"
    ) in notes[5]
    assert "whose follow-up ended between two times" in notes[5] and "is rough" in notes[5]
    assert "between 1 and the first value" in notes[5]
    assert "followed_until" in notes[6] and "12 events or more follow" in notes[6]
    assert "stays flat" not in " ".join(
        notes
    )  # it does not, when 12 events or more follow the last time
    assert "later_than_followed_until" in notes[7] and "before the first step" in notes[7]
    assert "_min" in notes[8] and "do not add up" in notes[8]
    assert reading_notes(DisclosurePolicy(12)) == notes and reading_notes(K10) != notes


# ----------------------------------------------------------------------------- rare values in a package's texts


def _sites() -> pd.DataFrame:
    """60 participants: a site of 3, an identifier held as text and one held as a number, one participant aged 97,
    a diagnosis that two of them hold, two enrolled on a day of their own."""
    return pd.DataFrame(
        {
            "patient": [f"P-{i:04d}" for i in range(60)],
            "site": ["Luxembourg"] * 45 + ["Esch"] * 12 + ["Wiltz-CANARY"] * 3,
            "sex": ["F", "M"] * 30,
            "age": [60.0] * 30 + [71.0] * 29 + [97.0],
            "grade": ["II"] * 57 + ["IV"] * 3,
            "stage": ["early"] * 56 + ["site"] * 2 + ["True"] + ["12,5"],
            "diagnosis": ["heart failure"] * 40
            + ["hypertension"] * 18
            + ["sarcoidosis CANARY"] * 2,
            "enrolled": pd.to_datetime(["2021-03-04"] * 2 + ["2021-05-01"] * 58),
            "transplanted": [True] * 2 + [False] * 58,
            "record_no": [100200 + i for i in range(60)],
        }
    )


def test_the_rare_values_of_a_dataset() -> None:
    """Held by 1 to k - 1 rows, in their normal form (no case, no accent, anything but letters and digits read as a
    space): the values of a text column, of an identifier held as a number, and the days of a date column."""
    from core.disclosure import one_hot_names, rare_values

    rare = rare_values(_sites(), K10)
    assert set(rare) == {"patient", "site", "diagnosis", "enrolled", "record_no"}
    assert rare["site"] == {"wiltz canary"} and rare["diagnosis"] == {"sarcoidosis canary"}
    assert len(rare["patient"]) == 60 and "p 0007" in rare["patient"]
    assert len(rare["record_no"]) == 60 and "100207" in rare["record_no"]
    assert rare["enrolled"] == {
        "2021 03 04", "04 03 2021", "03 04 2021", "2021 3 4", "4 3 2021", "3 4 2021",
    }  # fmt: skip
    assert rare_values(_sites(), DisclosurePolicy(15))["site"] == {"wiltz canary", "esch"}
    assert set(rare_values(_sites(), K10, columns=["site", "age"])) == {"site"}
    # counted in that form too: a spelling in capitals held by one row is no rare value of a word held 45 times
    spelt = _sites()
    spelt.loc[0, "site"] = "LUXEMBOURG"
    assert rare_values(spelt, K10)["site"] == {"wiltz canary"}
    # not looked for on their own: two characters ("IV"), a short number ("12,5"), True, a column's name ("site")
    assert "grade" not in rare and "stage" not in rare
    # but after their column's name any value is a level: the name a column per value would have
    names = one_hot_names(_sites(), K10)
    assert names["grade"] == {"grade iv"} and names["transplanted"] == {"transplanted true"}
    assert names["stage"] == {"stage site", "stage true", "stage 12 5"}
    assert names["age"] == {"age 97", "age 97 0"} and names["site"] == {"site wiltz canary"}
    assert "record_no" not in names and "enrolled" not in names  # 60 values; a date


def test_a_rare_value_is_found_where_it_stands() -> None:
    from core.disclosure import RareFinder

    finder = RareFinder([_sites()], K10)
    assert finder.named("df['site'] == 'Wiltz-CANARY'") == ("site",)
    assert finder.named("site_wiltz-canary") == ("site",)  # a column named after the level
    # in another case, with a space for the hyphen, with a no-break space
    assert finder.named("WILTZ canary") == ("site",) and finder.named("Wiltz Canary") == ("site",)
    assert (
        finder.named("Wiltz-CANARYs") == () and finder.named("xWiltz-CANARY") == ()
    )  # in a longer word
    assert finder.named("Luxembourg and Esch, early") == ()  # held by 10 rows or more
    # two characters: in every text on their own, a level after their column's name
    assert finder.named("IV") == () and finder.named("grade_IV") == ("grade",)
    assert finder.named("grade IV, early") == ("grade",)
    assert finder.named("participant P-0007, with sarcoidosis canary") == ("diagnosis", "patient")
    assert finder.named("P-00070") == () and finder.named("") == ()
    assert finder.named("record 100207") == ("record_no",)  # an identifier held as a number
    for written in ("2021-03-04", "enrolled on 04/03/2021", "4.3.2021", "03/04/2021"):
        assert finder.named(written) == ("enrolled",)  # a day two participants hold, in any order
    assert finder.named("2021-05-01") == ()  # the day of the 58 others
    assert finder.named("age 97") == ("age",) and finder.named("age 60") == ()


def test_an_accent_is_one_character_or_two() -> None:
    """Some exports write an accented letter as the letter and a combining accent: the value stored one way and
    written the other was not found."""
    import unicodedata

    from core.disclosure import RareFinder, normal

    name = "Zoé-CANARY"
    apart = unicodedata.normalize("NFD", name)
    assert len(apart) == len(name) + 1 and normal(name) == normal(apart) == "zoe canary"
    finder = RareFinder([pd.DataFrame({"site": [name] * 2 + ["Esch"] * 20})], K10)
    for written in (name, apart, "ZOE CANARY", "zoe_canary"):
        assert finder.named(written) == ("site",)
    stored_apart = RareFinder([pd.DataFrame({"site": [apart] * 2 + ["Esch"] * 20})], K10)
    assert stored_apart.named(f"the {name} site") == ("site",)


def test_a_hit_names_its_place_and_its_column_never_the_value() -> None:
    from core.disclosure import LITERAL, VALUE, RareFinder, RareHit

    finder = RareFinder([_sites()], K10)
    own = {
        "mask_families", "operator", "numeric", "expression", "description", "lower", "upper", "population",
        "ticked",
    }  # fmt: skip
    protocol = {
        "mask_families": {
            "study_inclusion": {
                "operator": "all",
                "not Wiltz-CANARY": {
                    "numeric": False,
                    "expression": "df['site'] != 'Wiltz-CANARY'",
                    "description": "Adults",
                },
                "adults": {"numeric": True, "lower": 18, "upper": 110},
            }
        },
        "population": {"ticked": ["study_inclusion::not Wiltz-CANARY", "study_inclusion::adults"]},
    }
    hits = finder.document("protocol.json", protocol, own=own)
    inside = "protocol.json: mask_families > name 1 > name 2"
    assert hits == [
        RareHit(inside, "site", VALUE),  # the criterion's own name holds the value
        RareHit(f"{inside} > expression", "site", VALUE),
        RareHit(f"{inside} > expression", "site", LITERAL),
        RareHit("protocol.json: population > ticked > item 1", "site", VALUE),
    ]
    assert "canary" not in repr(hits).lower()
    # a name the user or the data gave is written by its position: it may hold a value the finder does not see
    assert not any("study_inclusion" in hit.place or "adults" in hit.place for hit in hits)
    assert finder.text("title", "A study of the Wiltz-CANARY site") == [RareHit("title", "site")]
    assert finder.document("results.json", {"rows": 60, "labels": ["Luxembourg", "Esch"]}) == []


def test_a_column_whose_name_holds_a_value_is_named_by_its_position() -> None:
    from core.disclosure import RareFinder, RareHit

    df = _sites().rename(
        columns={"diagnosis": "dx_P-0007"}
    )  # the seventh column, named after a participant
    hits = RareFinder([df], K10).text("title", "sarcoidosis CANARY")
    assert hits == [RareHit("title", "column 7")]


@pytest.mark.parametrize(
    ("expression", "columns"),
    [
        ("df['site'] == 'Wiltz-CANARY'", ("site",)),
        ("df['site'] != 'Wiltz-CANARY'", ("site",)),  # the 3 rows it leaves hold the value
        ("df['site'].isin(['Luxembourg', 'Wiltz-CANARY'])", ("site",)),  # each value of the list
        ("df['site'].str.lower() == 'wiltz-canary'", ("site",)),
        ("df['site'].str.contains('wiltz', case=False)", ("site",)),
        ("df['site'].str[:5] == 'Wiltz'", ("site",)),
        ("df['site'] == 'Luxembourg'", ()),  # 45 rows
        ("(df['age'] > 18) & (df['site'] == 'Esch')", ()),  # 12 rows
        ("df['site'] == 'Nowhere'", ()),  # no row
        ("df['age'] == 97", ("age",)),  # one participant
        ("df['age'].astype(str) != '97.0'", ("age",)),
        ("df['age'].round() == 97", ("age",)),
        ("df['age'].between(97, 97)", ("age",)),  # a range whose two ends are one value
        ("(df['age'] >= 97) & (df['age'] <= 97)", ("age",)),
        ("df['age'].isin([60, 97])", ("age",)),
        ("df['age'] == 60", ()),  # 30 rows
        ("df['age'] >= 97", ()),  # a bound: the flow's counts cover it
        ("df['age'].between(60, 97)", ()),
        ("df['grade'] == 'IV'", ("grade",)),
        ("where(grade == 'IV', 1, 0)", ("grade",)),  # a formula names its columns bare
        ("df['patient'] == 'P-0003'", ("patient",)),
        ("df['patient'] == 'nobody'", ("patient",)),  # an identifier compared with anything
        ("df['record_no'] != 100999", ("record_no",)),  # a participant who is no longer in the data
        (
            "df['patient'].str.startswith('P-')",
            (),
        ),  # every row: a test on an identifier counts its rows
        ("df['patient'].str.endswith('07')", ("patient",)),  # one row
        ("df['diagnosis'].str.contains('sarcoid')", ("diagnosis",)),  # the test accepts 2 rows
        ("df['diagnosis'].str.contains('heart')", ()),  # 40 rows
        ("df['diagnosis'].str.contains('sarcoid(')", ()),  # no pattern: the evaluator refuses it
        ("df['enrolled'] == '2021-03-04'", ("enrolled",)),  # 2 rows that day
        ("df['enrolled'] == pd.Timestamp('2021-03-04')", ("enrolled",)),
        ("df['enrolled'].dt.year == 2021", ()),  # every row
        ("df['enrolled'] == '2021-05-01'", ()),  # the 2 rows it leaves are not named by it
        ("df['transplanted'] == 1", ("transplanted",)),
        ("df['unknown'] == 'x'", ()),
        ("(df['site'] == 'Wiltz-CANARY') | (df['age'] == 97)", ("site", "age")),
        ("'Wiltz-CANARY' != df.site", ("site",)),  # df.site is read as df['site']
        ("site ==", ()),
        ("Adults of the Wiltz-CANARY site", ()),
    ],
)
def test_an_expression_that_names_a_value_few_rows_hold(expression: str, columns: tuple) -> None:
    """Each comparison is run by the application's own evaluator and the rows that hold the value are counted."""
    from core.disclosure import RareFinder

    assert RareFinder([_sites()], K10).compared(expression) == columns


def test_a_value_is_rare_when_the_source_or_the_analysis_base_holds_it_in_a_few_rows() -> None:
    from core.disclosure import RareFinder

    source = _sites()
    base = source.iloc[50:]  # 7 rows of Esch, where the source holds 12
    assert RareFinder([source], K10).named("Esch") == ()
    both = RareFinder([source, base], K10)
    assert both.named("the Esch site") == ("site",)
    assert both.compared("df['site'] == 'Esch'") == ("site",)
    assert RareFinder([source], K10).compared("df['site'] == 'Esch'") == ()
    assert RareFinder([], K10).document("results.json", {"a": ["Esch"]}) == []


def test_what_a_result_shows_holds_no_rare_value() -> None:
    """A site of 3 participants: the record as computed names it (a curve's label, a term's, the levels and the
    groups the run recorded); what the package shows of the record and of its request does not."""
    from biostats import plan
    from biostats.plan import specify
    from core.disclosure import RareFinder, disclose_record, disclose_spec

    df = _cohort()
    df["site"] = np.where(np.arange(len(df)) % 3 == 0, "Alpha", df["site"])
    df.loc[df.index[:3], "site"] = "CANARY-SITE"
    finder = RareFinder([df], K10)
    curves = plan.run(specify("survival", time="days", event="died", group="site"), df)
    model = plan.run(specify("logistic", outcome="died", compare="arm", adjust=["site", "age"]), df)
    shown_groups = 0
    for done in (curves, model):
        record = plan.record(done.spec.kind, done.result)
        as_computed = finder.document(
            "results.json", {"record": record.to_dict(), "spec": done.spec.to_dict()}
        )
        assert as_computed and {hit.column for hit in as_computed} == {"site"}
        assert "canary" not in repr(as_computed).lower()
        line = _in_the_flow(record)
        disclosed = disclose_record(record, K10, spec=done.spec, analysis=line)
        shown_of_it = {
            "record": None if disclosed.record is None else disclosed.record.to_dict(),
            "spec": disclose_spec(done.spec, disclosed, analysis=line),
            "withheld": list(disclosed.withheld),
        }
        assert finder.document("results.json", shown_of_it) == []
        if disclosed.record is not None:
            shown_groups += len([i for i in _ids(disclosed) if i.startswith("curve#")])
    # two sites are shown by the curves, and named; the model, which a site of 3 rows drops out of, is not written
    assert shown_groups == 2
    assert disclosed.record is None and shown_of_it["spec"] is None


def test_the_scan_of_a_large_file_stays_fast() -> None:
    """20,000 identifiers that share a prefix and 20,000 notes that share their words, against a file of 20,000
    texts that hold those words: each text is read once, and tried only against the values whose rarest word it
    holds."""
    import time

    from core.disclosure import RareFinder

    df = pd.DataFrame(
        {
            "patient": [f"P-{i:06d}" for i in range(20_000)],
            "arm": ["drug", "control"] * 10_000,
            "note": [f"patient seen at visit {i}" for i in range(20_000)],
        }
    )
    finder = RareFinder([df], K10)
    document = {
        "parts": [
            {"id": f"term:{i}", "p": "P", "rows": "rows", "label": f"patient {i:06d} seen"}
            for i in range(5_000)
        ]
    }
    started = time.perf_counter()
    assert finder.document("results.json", document) == []
    found = finder.document("x.json", {"a": {"note": "see P-012345."}}, own={"a", "note"})
    assert [(hit.place, hit.column) for hit in found] == [("x.json: a > note", "patient")]
    assert time.perf_counter() - started < 5


def test_the_scan_never_changes_the_datasets_it_is_given() -> None:
    """The texts of a package were run on the caller's own dataset: a title holding
    ``df.replace(97.0, 60.0, inplace=True)`` changed the data, outside the trace, and the next reading of
    ``df['age'] == 97`` found nobody."""
    from core.disclosure import UNCHECKED, RareFinder, RareHit

    df = _sites()
    before = df.copy(deep=True)
    finder = RareFinder([df], K10)
    in_place = "df['age'] == df.replace(97.0, 60.0, inplace=True)"
    for text in (
        in_place,
        "df['age'].isna() & (df.fillna(0, inplace=True) is None)",
        "df['site'] == df['site'].where(df['age'] > 96, 'x', inplace=True)",
    ):
        finder.text("title", text)
    assert df.equals(before)
    # a test the evaluator refuses is said not to be checked: it is not passed
    assert finder.compared(in_place) == () and finder.unchecked(in_place) == ("age",)
    assert finder.text("title", in_place) == [RareHit("title", "age", UNCHECKED)]
    assert finder.compared("df['age'] == 97") == ("age",)


def test_a_pattern_is_run_only_by_the_engine_that_cannot_backtrack() -> None:
    """``(.|.)*(?=#)`` takes Python's own engine minutes on a value of 30 characters and days on one of 40. pandas
    chooses its engine on the data as well as on the pattern (a letter with an accent in one value, a newline, a
    column kept as Python objects): a pattern is run by the engine that cannot backtrack, called directly, when
    that engine takes it, and is not checked otherwise."""
    import time

    from core.disclosure import UNCHECKED, RareFinder, RareHit

    pytest.importorskip("pyarrow")
    df = _sites().astype({"diagnosis": "string[pyarrow]"})
    finder = RareFinder([df], K10)
    started = time.perf_counter()
    assert finder.compared("df['diagnosis'].str.contains('sarc.id|granul')") == ("diagnosis",)
    assert finder.compared("df['diagnosis'].str.contains('heart|hyper')") == ()  # 58 rows
    slow = "df['diagnosis'].str.contains('(.|.)*(?=#)')"
    assert finder.compared(slow) == () and finder.unchecked(slow) == ("diagnosis",)
    assert finder.text("criterion", slow) == [RareHit("criterion", "diagnosis", UNCHECKED)]
    # flags are not read, given by name or by position
    assert finder.unchecked("df['diagnosis'].str.contains('sarc.id', flags=2)") == ("diagnosis",)
    assert finder.unchecked("df['diagnosis'].str.contains('sarc.id', True, 2)") == ("diagnosis",)
    # a column kept as Python objects is searched by the same engine, whatever pandas would do with it
    objects = RareFinder([_sites().astype({"diagnosis": object})], K10)
    assert objects.compared("df['diagnosis'].str.contains('sarc.id')") == ("diagnosis",)
    assert objects.unchecked("df['diagnosis'].str.contains('sarc.id')") == ()
    assert objects.compared("df['diagnosis'].str.match('(a|a)*b')") == ()
    assert objects.unchecked("df['diagnosis'].str.match('(a|a)*b')") == ()
    # from the start of a value, the whole of it, without the case of letters, after another method
    assert objects.compared("df['diagnosis'].str.match('sarc')") == ("diagnosis",)
    assert objects.compared("df['diagnosis'].str.match('oidosis')") == ()
    assert objects.compared("df['diagnosis'].str.fullmatch('sarc.*')") == ("diagnosis",)
    assert objects.compared("df['diagnosis'].str.fullmatch('sarc')") == ()
    assert objects.compared("df['diagnosis'].str.contains('SARC.ID', case=False)") == ("diagnosis",)
    assert objects.compared("df['diagnosis'].str.contains('SARC.ID')") == ()
    assert objects.compared("df['diagnosis'].str.upper().str.contains('SARC.ID')") == ("diagnosis",)
    # values that are not all texts, and a pattern that is not written in the text: nothing can be said
    mixed = _sites().astype({"diagnosis": object})
    mixed.loc[0, "diagnosis"] = 7
    of_mixed = RareFinder([mixed], K10)
    assert of_mixed.unchecked("df['diagnosis'].str.contains('sarc.id')") == ("diagnosis",)
    computed = "df['diagnosis'].str.contains(df['site'][0])"
    assert objects.compared(computed) == () and set(objects.unchecked(computed)) == {
        "diagnosis",
        "site",
    }
    # a text searched as it is written is run whatever the column: there is no pattern to try
    assert objects.compared("df['diagnosis'].str.contains('sarcoid')") == ("diagnosis",)
    assert objects.compared("df['diagnosis'].str.contains('sarc.id', regex=False)") == ()
    assert objects.unchecked("df['diagnosis'].str.contains('sarc.id', regex=False)") == ()
    assert time.perf_counter() - started < 5


@pytest.mark.parametrize("stored", ["str", "object", "string[pyarrow]"])
def test_no_text_of_a_package_reaches_the_engine_that_backtracks(stored: str) -> None:
    """pandas sent each of these to Python's own engine: a class of letters beside a value with an accent, a column
    kept as objects, flags given by position, an end of text beside a value with a newline, and a pattern inside
    the text that is tested (``split``, ``replace``). Each took about four times longer for every two characters
    of a value: 20 seconds at 22 characters, hours at 30."""
    import time

    from core.disclosure import RareFinder

    pytest.importorskip("pyarrow")
    values = ["a" * 22] * 9 + ["é" + "a" * 21, "a" * 20 + "\nb", "b" * 22]
    frame = pd.DataFrame({"site": pd.Series(values, dtype=stored), "age": np.arange(12.0)})
    finder = RareFinder([frame], K10, identifying=[])
    run, not_checked = ((), ()), ((), ("site",))
    cases = {
        # a class of letters beside a value with an accent: the two engines do not read it alike, so not checked
        r"df['site'].str.contains('(\\w|\\w)*#')": not_checked,
        "df['site'].astype('object').str.contains('(.|.)*#')": run,
        "df['site'].str.contains('(.|.)*#$')": run,
        "df['site'].str.contains('(.|.)*#', True, 2)": not_checked,
        "df['site'].str.split('(.|.)*#').str.len() == 1": not_checked,
        "df['site'].str.replace('(.|.)*#', 'x', regex=True, case=False) == 'x'": not_checked,
        "df['site'].replace('(.|.)*#', 'x', regex=True) == 'x'": not_checked,
        # and what is run still counts its rows: the one value made of b
        "df['site'].str.fullmatch('b{22}')": (("site",), ()),
        "df['site'].str.lower().str.contains('^b+$')": (("site",), ()),
        "df['site'].str.replace('b', 'c') == 'c' * 22": (("site",), ()),
        "df['site'].str.split('-').str.len() == 2": run,
    }
    started = time.perf_counter()
    for text, expected in cases.items():
        assert (finder.compared(text), finder.unchecked(text)) == expected, text
    assert time.perf_counter() - started < 5


def test_a_pattern_is_counted_as_pandas_counts_it_or_not_at_all() -> None:
    """Two texts the scan counted differently from pandas, so that a criterion that names a rare value passed
    unflagged. A class of letters beside a value outside ASCII: pandas takes the 3 rows of "Genève" with
    ``Gen\\wve``, the engine without backtracking none (its classes know ASCII only). And ``regex=0``, which pandas
    reads as "searched as written": 3 rows hold "A+B", and the pattern A+B takes 57 others."""
    from core.disclosure import UNCHECKED, RareFinder, RareHit

    pytest.importorskip("pyarrow")
    frame = pd.DataFrame(
        {
            "town": ["Genève"] * 3 + ["Luxembourg"] * 57,
            "code": ["A+B"] * 3 + ["AAB"] * 27 + ["AB"] * 30,
        }
    )
    finder = RareFinder([frame], K10, identifying=[])
    accented = r"df['town'].str.fullmatch('Gen\\wve')"
    assert frame["town"].str.fullmatch(r"Gen\wve").sum() == 3  # what pandas takes
    assert finder.compared(accented) == () and finder.unchecked(accented) == ("town",)
    assert finder.text("criterion", accented) == [RareHit("criterion", "town", UNCHECKED)]
    for classed in (r"contains('\\d')", r"contains('ve\\b')", r"match('\\S+ve')"):
        assert finder.unchecked(f"df['town'].str.{classed}") == ("town",), classed
    # the same classes on values of ASCII alone are counted, as before
    plain = r"df['code'].str.fullmatch('A\\wB')"
    assert finder.compared(plain) == () and finder.unchecked(plain) == ()  # the 27 rows of AAB
    assert finder.compared(r"df['code'].str.fullmatch('A\\WB')") == ("code",)  # the 3 of A+B
    # a setting read by its truth, as pandas reads it
    assert frame["code"].str.contains("A+B", regex=0).sum() == 3  # type: ignore[call-overload]
    for written in ("regex=0", "regex=False", "regex=None", "regex=''"):
        text = f"df['code'].str.contains('A+B', {written})"
        assert finder.compared(text) == ("code",) and finder.unchecked(text) == (), written
    for pattern in ("", "regex=1", "regex=True", "regex='yes'"):
        text = f"df['code'].str.contains('A+B'{', ' if pattern else ''}{pattern})"
        assert finder.compared(text) == () and finder.unchecked(text) == (), pattern  # 57 rows
    lower = "df['code'].str.contains('a+b', case=0, regex=0)"
    assert finder.compared(lower) == ("code",) and finder.unchecked(lower) == ()
    assert finder.compared("df['code'].str.contains('a+b', regex=0)") == ()  # no row holds it


def test_a_text_too_deep_to_be_read_is_not_checked() -> None:
    """250 comparisons joined by ``&`` raised RecursionError out of the scan, where it promises "not checked" for
    what it cannot read."""
    from core.disclosure import UNCHECKED, VALUE, RareFinder, RareHit

    finder = RareFinder([_sites()], K10)

    def joined(comparisons: int) -> str:
        return " & ".join(f"(df['age'] > {i})" for i in range(comparisons))

    assert finder.text("criterion", joined(250)) == []  # read: bounds, which name no value
    assert finder.text("criterion", joined(3000)) == [RareHit("criterion", "age", UNCHECKED)]
    # what such a text names is still found as a text, and every column it reads is said not to be checked
    named = finder.text("criterion", joined(3000) + " & (df['site'] == 'Wiltz-CANARY')")
    assert {(hit.column, hit.kind) for hit in named} == {
        ("site", VALUE),
        ("age", UNCHECKED),
        ("site", UNCHECKED),
    }
    # brackets as deep as Python reads them
    nested = "(" * 150 + "df['age'] == 97" + ")" * 150
    assert finder.compared(nested) == ("age",)


def test_a_column_named_after_a_level_is_named_by_its_position() -> None:
    """A column ``grade_IV`` of 0 and 1: a hit on it was labelled by its name, which holds the level that 3
    participants have (two characters: not a value that is looked for on its own)."""
    from core.disclosure import LITERAL, VALUE, RareFinder

    df = _sites().assign(grade_IV=lambda d: (d["grade"] == "IV").astype(int))
    assert list(df.columns).index("grade_IV") == 10
    hits = RareFinder([df], K10).text("criterion", "df['grade_IV'] == 1")
    assert {(hit.column, hit.kind) for hit in hits} == {
        ("grade", VALUE),
        ("column 11", VALUE),
        ("column 11", LITERAL),
    }
    assert "grade_IV" not in repr(hits)


def test_a_criterion_stored_as_numbers_is_read_as_the_comparison_it_stands_for() -> None:
    """A criterion that excludes ``record_no`` 100234 to 100234, or ``age`` 97 to 97, holds its values as numbers:
    the finder reads texts, and gave no hit."""
    from core.disclosure import LITERAL, UNCHECKED, RareFinder, RareHit

    finder = RareFinder([_sites()], K10)
    assert finder.bounds("criterion 3", "age", 97, 97) == [RareHit("criterion 3", "age", LITERAL)]
    # a bound is read as the criterion's own mask reads it: a number written as a text, a number of numpy
    one = [RareHit("c", "age", LITERAL)]
    assert finder.bounds("c", "age", "97", "97") == one
    assert finder.bounds("c", "age", "96.5", "97.5") == one
    assert finder.bounds("c", "age", np.float64(97), np.float64(97)) == one
    assert finder.bounds("c", "record_no", "100234", "100234") == [
        RareHit("c", "record_no", LITERAL)
    ]
    assert (
        finder.bounds("c", "age", "18", "110") == [] and finder.bounds("c", "age", None, "97") == []
    )
    # one that cannot be read is said not to be checked; an infinite one is no bound on its side
    for unread in ("abc", [1], float("nan")):
        assert finder.bounds("c", "age", unread, 97) == [RareHit("c", "age", UNCHECKED)], unread
    assert finder.bounds("c", "age", float("-inf"), 97) == []
    assert finder.bounds("criterion 3", "record_no", 100234, 100234) == [
        RareHit("criterion 3", "record_no", LITERAL)
    ]
    # an identifier the data no longer hold: a hit whatever it selects
    assert finder.bounds("c", "record_no", 999999, 999999) == [RareHit("c", "record_no", LITERAL)]
    # a range that takes the one participant aged 97, and ranges that take many
    assert finder.bounds("c", "age", 96.5, 97.5) == [RareHit("c", "age", LITERAL)]
    assert finder.bounds("c", "age", 18, 110) == [] and finder.bounds("c", "age", 60, 60) == []
    assert finder.bounds("c", "age", 18, None) == [] and finder.bounds("c", "age", None, None) == []


def test_a_bound_at_a_rare_level_is_not_the_name_of_a_level() -> None:
    """A score from 0 to 8 that 3 rows hold at 8: ``df['score'] >= 8`` was a hit, the text reading "score 8" once
    its signs were dropped, and the usual thresholds (a score of 8 or less, a class of 3 or more) with it."""
    from core.disclosure import RareFinder

    df = pd.DataFrame({"score": [0, 1, 2, 3, 4, 5, 6, 7] * 12 + [8] * 3})
    finder = RareFinder([df], K10)
    for bound in (
        "df['score'] >= 8",
        "df['score'] > 7",
        "score >= 8",
        "8 <= df['score']",
        "a score >= 8 at admission",
        # as a text writes them: the signs of Unicode, and "8+" for 8 or more
        "score \u2265 8",
        "GCS \u2264 8 or score \u2265 8",
        "score \u2a7e 8",
        "score \uff1e 7",
        "score 8+",
        "a score of 8+ at admission",
        "(score 8+)",
    ):
        assert finder.text("criterion", bound) == [], bound
    # the level itself, after its column's name, is one
    for level in (
        "score_8",
        "df['score_8'] == 1",
        "score 8",
        "score = 8",
        "score-8",
        "score_8 + 1",  # the 8 ends a column's name: no "8 or more"
    ):
        assert [hit.kind for hit in finder.text("title", level)] == ["value"], level
    # and an equality with it is found by the rows it takes
    assert {hit.kind for hit in finder.text("criterion", "df['score'] == 8")} == {
        "value",
        "literal",
    }


def test_the_identifying_columns_are_those_the_person_who_shares_marks() -> None:
    """Read from the data, an identifier is a guess: a laboratory value in whole numbers is one in a small cohort
    (its missing-value code was then a hit), and an identifier repeated at each visit is none (a participant named
    by it was missed)."""
    from core.disclosure import RareFinder, identifying_columns

    assert identifying_columns(_sites()) == ["patient", "record_no"]
    rng = np.random.default_rng(4)
    lab = pd.DataFrame(
        {
            "patient": [f"P-{i:04d}" for i in range(120)],
            "creatinine": rng.permutation(np.arange(40, 160)),
        }
    )
    assert identifying_columns(lab) == ["patient", "creatinine"]
    assert RareFinder([lab], K10).compared("df['creatinine'] != 9999") == ("creatinine",)
    marked = RareFinder([lab], K10, identifying=["patient"])
    assert marked.identifying == {"patient"}
    assert marked.compared("df['creatinine'] != 9999") == ()
    # on a marked column an equality is a hit whatever it selects, but with the empty text, which names nobody
    assert marked.compared("df['patient'] == 'P-9999'") == ("patient",)
    assert marked.compared("df['patient'] != ''") == ()
    # three visits per participant
    visits = pd.DataFrame(
        {"patient_no": [100000 + i // 3 for i in range(90)], "visit": [1, 2, 3] * 30}
    )
    assert identifying_columns(visits) == []
    guessed = RareFinder([visits], K10)
    assert guessed.compared("df['patient_no'] != 100999") == ()
    assert guessed.named("patient 100004 withdrew") == ()
    told = RareFinder([visits], K10, identifying=["patient_no"])
    assert told.compared("df['patient_no'] != 100999") == ("patient_no",)
    assert told.named("patient 100004 withdrew") == ("patient_no",)


def test_a_range_that_takes_one_value_names_it() -> None:
    """One participant aged 97: ``(df['age'] > 96) & (df['age'] < 98)`` and ``between(96.5, 97.5)`` were missed,
    a range counting only when its two ends were the same value; so was ``df.age == 97``, which the application's
    evaluator does not read."""
    from core.disclosure import RareFinder

    finder = RareFinder([_sites()], K10)
    for narrow in (
        "(df['age'] > 96) & (df['age'] < 98)",
        "df['age'].between(96.5, 97.5)",
        "df.age == 97",
        "97 == df.age",
    ):
        assert finder.compared(narrow) == ("age",), narrow
    assert finder.compared("(df['age'] > 50) & (df['age'] < 98)") == ()  # 60 rows, of three values
    assert (
        finder.compared("(df['age'] > 96) & (df['grade'] == 'II')") == ()
    )  # no range of one column
    # an identifier the data no longer hold, searched as it is written
    assert finder.compared("df['patient'].str.fullmatch('P-0999')") == ("patient",)
    assert finder.compared("df['patient'].str.startswith('P-')") == ()  # every row


def test_a_value_that_is_a_column_s_name_is_looked_for_unless_the_package_shows_that_column() -> (
    None
):
    """The site "Wiltz" is held by 3 rows, and a column of 0 and 1 is named Wiltz: "the patients of Wiltz" was not
    a hit, a value equal to a column's name being left out, and a hit on that column named it by the value."""
    from core.disclosure import RareFinder

    df = pd.DataFrame(
        {
            "site": ["Esch"] * 40 + ["Wiltz"] * 3,
            "Wiltz": [0] * 40 + [1] * 3,
            "age": [60.0] * 43,
        }
    )
    # every column is shown: the name is in every table of the package, and says nothing of a row
    assert RareFinder([df], K10).named("the patients of Wiltz") == ()
    # the package shows the site and the age only
    finder = RareFinder([df], K10, shown=["site", "age"])
    assert finder.named("the patients of Wiltz") == ("site",)
    hits = finder.text("criterion", "df['Wiltz'] == 1")
    assert {hit.column for hit in hits} == {"site", "column 2"}
    assert "wiltz" not in repr(hits).lower()
