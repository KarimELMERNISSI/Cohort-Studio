"""
The small-cell rule of a shared package (M8): a count of participants is exact or "fewer than k", k is never below
5, and the participant flow is shown so that no withheld count comes back by subtraction (the reasons that are too
small on one line, an exit that is too small merged with the next transition, and only the population when every
exclusion together is too small).
"""

from __future__ import annotations

import json
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
    assert DisclosurePolicy(5).to_dict() == {"k": 5, "version": 1}
    # a package says why its k is not the study's
    raised = DisclosurePolicy.for_study(5, floor=10)
    assert raised.to_dict() == {"k": 10, "asked": 5, "floor": 10, "version": 1}
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
    "median_upper", "later_than_followed_until", "times", "survival", "lower", "upper",
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
    if record.kind == "survival":
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


def test_a_term_of_one_participant_coded_1_is_withheld() -> None:
    """One participant is coded 1 and died: the coefficient (+22.9 in the review's probe) gives their outcome, and
    the term carried the whole model's counts."""
    from biostats import plan
    from biostats.regression import logistic_regression

    df = _cohort()
    df["transplanted"] = 0
    df.loc[df.index[7], ["transplanted", "died"]] = [1, 1]
    record = plan.record(
        "logistic", logistic_regression(df, "died", ["transplanted", "arm", "age"])
    )
    assert record is not None and record.part("term:transplanted").support["rows"] == 1
    disclosed = _disclose(record, compare="arm")
    assert disclosed.record is not None
    ids = _ids(disclosed)
    assert "term:transplanted" not in ids and "reference:transplanted" not in ids
    assert {"model", "reference:arm", "term:arm#1", "term:age"} <= set(ids)
    assert disclosed.withheld == (f"1 term of transplanted withheld: {CELL}", FIT)
    # at k = 5 too: one row is below any k
    assert "term:transplanted" not in _ids(_disclose(record, DisclosurePolicy(5), compare="arm"))
    # as the comparison of the model: what is left is the adjustment, whose coefficients are no effects
    alone = _disclose(record, compare="transplanted")
    assert alone.record is None and alone.withheld == (NO_COMPARISON,)


def test_a_rare_reference_level_withholds_every_term_of_its_factor() -> None:
    """The first level in sorted order is the reference: held by 2 rows who both died, it made every other level's
    coefficient (-22.8, -23.0) give their outcome, while those terms passed on their own counts."""
    from biostats import plan
    from biostats.regression import logistic_regression

    df = _cohort()
    df.loc[df.index[:2], ["site", "died"]] = ["Aaa-CANARY", 1]
    record = plan.record("logistic", logistic_regression(df, "died", ["site", "age"]))
    assert record is not None and record.part("reference:site").support["rows"] == 2
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
    """A site of 3 participants who all died: its term was withheld, and the model's rows, events and rows without
    the event, less the reference's and the shown term's, gave 3, 3 and 0: the outcome of each of the three."""
    from biostats import plan
    from biostats.regression import logistic_regression
    from biostats.survival import cox_regression

    df = _three_sites()
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
        assert left == {"rows": 3, "events": 3, without: 0}
        disclosed = _disclose(record)  # the site is an adjustment here
        # no count travels beside a term, and the other level is withheld with the small one: nothing says what
        # the 3 rows hold
        assert disclosed.withheld == (
            f"1 term of site withheld: {CELL}",
            f"1 more term of site withheld: {COVER}",
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


def test_a_model_of_categories_alone_gives_no_cell_back() -> None:
    """In a model of one factor, the coefficient of a level is the log odds ratio of its cell and the reference's,
    and the square of its standard error the sum of their four reciprocals: with one term shown, a single set of
    whole numbers gave both cells, and the model's totals then left the 3 participants of Delta, who all died."""
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
    # with a fourth site, the smaller of the two others covers Delta: Beta's term is written, and what the totals
    # leave is Gamma and Delta together, at least k of each count
    four = _four_sites()
    for key, result, counted in (
        ("logistic", logistic_regression(four, "died", ["site"]), ("rows", "events", "non_events")),
        ("cox", cox_regression(four, "days", "died", ["site"]), ("rows", "events", "censored")),
        (
            "count",
            count_regression(four, "visits", ["site"]),
            ("rows", "rows_with_events", "rows_without_events"),
        ),
    ):
        record = plan.record(key, result)
        assert record is not None
        disclosed = _disclose(record, compare="site")
        assert [i for i in _ids(disclosed) if i.startswith(("reference:", "term:"))] == [
            "reference:site",
            "term:site#1",
        ], key
        assert disclosed.withheld == (
            f"1 term of site withheld: {CELL}",
            f"1 more term of site withheld: {COVER}",
            FIT,
        )
        left = {
            name: record.part("model").support[name]
            - record.part("reference:site").support[name]
            - record.part("term:site#1").support[name]
            for name in counted
        }
        assert left["rows"] == 153 and min(left.values()) >= 10, key
        assert not FIT_NAMES & {n.name for n in disclosed.record.part("model").numbers}, key
        assert "Delta" not in _written(disclosed) and "Gamma" not in _written(disclosed)


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
    # adjusted for age, the model is written with its totals and its age term, and none of the fit's numbers
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


def test_a_numeric_covariate_that_a_few_rows_carry_is_read_as_a_cell() -> None:
    """A dose that is 0 for 398 rows, 1 and 3 for two participants who both died: its term carried the model's
    counts and was written, +22.8: everybody with a dose died."""
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
    assert dose.labels == {"variable": "dose"}  # a slope, not a level
    assert dose.support == {
        "rows": 2,
        "events": 2,
        "non_events": 0,
    }  # the rows off its commonest value
    reference = record.part("reference:dose")
    assert (
        reference.labels == {"variable": "dose", "level": "0"} and reference.support["rows"] == 398
    )
    disclosed = _disclose(record)
    assert _ids(disclosed) == ["model", "term:age"]
    assert disclosed.withheld == (f"1 term of dose withheld: {CELL}", FIT)
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
        assert record is not None and record.part("term:prior").support["rows"] == 6, key
        assert _ids(_disclose(record)) == ["model", "term:age"], key
    # a covariate that half of the rows hold at 0 while the others spread over seven values: its term is written
    df["packs"] = np.where(np.arange(len(df)) % 2 == 0, 0, 1 + np.arange(len(df)) % 7)
    spread = plan.record("logistic", logistic_regression(df, "died", ["packs", "age"]))
    assert spread is not None and spread.part("term:packs").support["rows"] == 200
    assert "term:packs" in _ids(_disclose(spread))
    # and age, which no value holds for half of the rows, has the model's counts behind it
    whole = {k: n for k, n in spread.part("model").support.items() if k != "df_resid"}
    assert spread.part("term:age").support == whole
    assert "reference:age" not in [p.id for p in spread.parts]


def test_a_term_without_a_maximum_is_not_written() -> None:
    """The application itself says that such a ratio cannot be read; and beside a withheld term, "not converged"
    made a reader doubt the sound terms and told them that a withheld level had no maximum."""
    from biostats.plan import EXACT, ITERATIVE, Number, Part, ResultRecord

    counts = {"rows": 400, "events": 160, "non_events": 240}

    def term(name: str, extreme: bool) -> Part:
        numbers = (Number("coef", 0.2, ITERATIVE), Number("extreme", extreme, EXACT))
        return Part(f"term:{name}", numbers, counts, {"variable": name})

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
    assert _ids(disclosed) == ["model", "term:age"]
    assert disclosed.withheld == (f"1 term of marker withheld: {CELL}", FIT)
    assert [n.name for n in disclosed.record.part("model").numbers] == [
        "rows",
        "parameters",
        "events_per_parameter",
        "sparse",
    ]
    assert disclosed.record.part("model").number("events_per_parameter").value == 80.0


def test_a_level_or_a_group_where_everybody_had_the_event_is_withheld() -> None:
    """A site of 15 participants who all died: its Cox term and its curve were shown with "15 rows, 15 events",
    which says of each of the 15 that they died. Every kind counts the rows without the event."""
    from biostats import plan
    from biostats.survival import compare_survival, cox_regression

    df = _three_sites(delta=15)
    record = plan.record("cox", cox_regression(df, "days", "died", ["site", "age"]))
    assert record is not None
    assert record.part("term:site#2").support == {"rows": 15, "events": 15, "censored": 0}
    disclosed = _disclose(record)
    assert [i for i in _ids(disclosed) if "site" in i] == []
    assert disclosed.withheld[0] == f"1 term of site withheld: {CELL}"
    curves = _disclose(plan.record("survival", compare_survival(df, "days", "died", "site")))
    assert [p.labels["group"] for p in curves.record.parts if p.id.startswith("curve#")] == [
        "Alpha"
    ]
    assert curves.withheld == (GROUP_WITHHELD, COVER_WITHHELD, TOGETHER, LOG_RANK)
    assert "Delta" not in _written(curves) and "Beta" not in _written(curves)


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
    # the two terms without their counts, and Beta with them: nothing says what they hold together
    assert _ids(disclosed) == ["model", "term:age"]
    assert disclosed.withheld == (
        f"2 terms of site withheld: {CELL}",
        f"1 more term of site withheld: {COVER}",
        FIT,
    )
    # with both cells counted, Beta's term is written
    counted = ResultRecord("logistic", parts=record.parts[:4])
    assert _ids(_disclose(counted)) == ["model", "reference:site", "term:age", "term:site#1"]
    # a level without its reference's counts cannot be judged
    orphan = ResultRecord("logistic", parts=(model, age, record.parts[3]))
    assert _ids(_disclose(orphan)) == ["model", "term:age"]
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
    monkeypatch.setitem(disclosure.RULES, "survival", lambda record, policy, spec: careless)
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
        # the value at a time given is the full curve's at that time
        for t, value in zip(times, survival, strict=True):
            assert value == full[max(s for s in steps if s <= t)]
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
    assert [round(s, 3) for s in plateau.number("survival").value] == [
        0.968,
        0.925,
        0.877,
        0.85,
        0.85,
    ]
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


def _imputed_age() -> object:
    from biostats.mi import MIData

    truth = _cohort()
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


def test_a_shared_spec_names_no_row_and_no_withheld_level() -> None:
    import hashlib

    from biostats import plan
    from biostats.plan import ComputedOn, canonical, specify
    from core.disclosure import DisclosedAnalysis, DisclosedRecord, disclose_record, disclose_spec

    df = _cohort()
    df["site"] = np.where(np.arange(len(df)) % 3 == 0, "Alpha", df["site"])
    df.loc[df.index[:3], "site"] = "CANARY-SITE"
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
    # the site of 3 rows is withheld, and the smaller of the two others with it
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
    assert len(notes) == 6 and all(isinstance(note, str) and note.endswith(".") for note in notes)
    assert "fewer than 12" in notes[0] and "selected by their outcome" in notes[1]
    assert "followed_until" in notes[3] and "below 10" in notes[5]


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
    """A site of 3 participants: the record as computed names it (a term's label, the levels the run recorded);
    what the package shows of the record and of its request does not."""
    from biostats import plan
    from biostats.plan import specify
    from core.disclosure import RareFinder, disclose_record, disclose_spec

    df = _cohort()
    df["site"] = np.where(np.arange(len(df)) % 3 == 0, "Alpha", df["site"])
    df.loc[df.index[:3], "site"] = "CANARY-SITE"
    done = plan.run(specify("logistic", outcome="died", compare="arm", adjust=["site", "age"]), df)
    record = plan.record("logistic", done.result)
    finder = RareFinder([df], K10)
    as_computed = finder.document(
        "results.json", {"record": record.to_dict(), "spec": done.spec.to_dict()}
    )
    assert as_computed and {hit.column for hit in as_computed} == {"site"}
    assert "canary" not in repr(as_computed).lower()
    line = _in_the_flow(record)
    disclosed = disclose_record(record, K10, spec=done.spec, analysis=line)
    shown_of_it = {
        "record": disclosed.record.to_dict(),
        "spec": disclose_spec(done.spec, disclosed, analysis=line),
        "withheld": list(disclosed.withheld),
    }
    assert any(
        i.startswith("term:site#") for i in _ids(disclosed)
    )  # a level of the site is shown, and named
    assert finder.document("results.json", shown_of_it) == []


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
    """``(.|.)*(?=#)`` takes Python's own engine minutes on a value of 30 characters and days on one of 40: a
    pattern is run by the engine behind text stored by Arrow, when that engine takes it, and is not checked
    otherwise."""
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
    # flags send pandas to Python's own engine, and a column kept as Python objects is searched by it
    assert finder.unchecked("df['diagnosis'].str.contains('sarc.id', flags=2)") == ("diagnosis",)
    objects = RareFinder([_sites().astype({"diagnosis": object})], K10)
    assert objects.unchecked("df['diagnosis'].str.contains('sarc.id')") == ("diagnosis",)
    assert objects.unchecked("df['diagnosis'].str.match('(a|a)*b')") == ("diagnosis",)
    # a text searched as it is written is run whatever the column: there is no pattern to try
    assert objects.compared("df['diagnosis'].str.contains('sarcoid')") == ("diagnosis",)
    assert objects.compared("df['diagnosis'].str.contains('sarc.id', regex=False)") == ()
    assert objects.unchecked("df['diagnosis'].str.contains('sarc.id', regex=False)") == ()
    assert time.perf_counter() - started < 5


def test_a_criterion_stored_as_numbers_is_read_as_the_comparison_it_stands_for() -> None:
    """A criterion that excludes ``record_no`` 100234 to 100234, or ``age`` 97 to 97, holds its values as numbers:
    the finder reads texts, and gave no hit."""
    from core.disclosure import LITERAL, RareFinder, RareHit

    finder = RareFinder([_sites()], K10)
    assert finder.bounds("criterion 3", "age", 97, 97) == [RareHit("criterion 3", "age", LITERAL)]
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
    ):
        assert finder.text("criterion", bound) == [], bound
    # the level itself, after its column's name, is one
    for level in ("score_8", "df['score_8'] == 1", "score 8", "score = 8", "score-8"):
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
