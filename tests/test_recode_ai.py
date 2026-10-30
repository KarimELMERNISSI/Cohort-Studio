"""
The AI engine's proposals on Data > Clean > Categories: what may be sent (manage.ai_payload.category_values), the
engine's request (manage.rag_computed_vars.suggest_category_groups), and how its answer is read (enrich.recode.
ai_offers): only a value sent, never two categories joined, never to missing; offered, never applied.
"""

from __future__ import annotations

import json
import logging

import pytest

from enrich.recode import AI, CHOSEN, Offer, ai_offers, free_text, groupings, new_choices, tick
from manage.ai_payload import category_values

COUNTS = {"Female": 400, "Male": 380, "Femme": 12, "Homme": 9, "W": 2}


class _Answer:
    def __init__(self, content: str) -> None:
        self.content = content


class _Model:
    """A language model that answers what it is given, and keeps the prompts."""

    model_name = "fake-model"

    def __init__(self, *answers: str, fails: bool = False) -> None:
        self.answers, self.prompts, self.fails = list(answers), [], fails

    def invoke(self, prompt: str) -> _Answer:
        self.prompts.append(prompt)
        if self.fails:
            raise ConnectionError("CANARY-ERROR-TEXT Female")
        return _Answer(self.answers.pop(0))


def _engine(*answers: str, fails: bool = False):
    from manage.rag_computed_vars import ComputedVarsMixin

    class Engine(ComputedVarsMixin):
        initialized = True

        def __init__(self) -> None:
            self.llm = _Model(*answers, fails=fails)

        @staticmethod
        def _clean_json_response(text: str) -> str:
            return text

    return Engine()


@pytest.mark.parametrize(
    ("provider", "url", "sent"),
    [
        (
            "ollama",
            "http://localhost:11434",
            ["Female", "Male", "Femme", "Homme", "W"],
        ),  # this machine: every value
        (
            "ollama",
            "http://192.168.1.20:11434",
            ["Female", "Male", "Femme", "Homme"],
        ),  # the private network
        ("anthropic", None, ["Female", "Male", "Femme", "Homme"]),  # a cloud provider
    ],
)
def test_a_value_held_by_fewer_than_5_rows_reaches_only_an_ollama_on_this_machine(
    provider, url, sent
) -> None:
    got, held = category_values(COUNTS, provider, url)
    assert got == sent and held == len(COUNTS) - len(sent)


def test_the_engine_is_sent_the_name_and_the_values_never_a_count_and_its_answer_is_read() -> None:
    answer = {
        "proposals": [
            {"value": "Femme", "clean": "Female", "reason": "French for female"},
            {"value": "Homme", "clean": "Male", "reason": "French for male"},
            {"value": 3, "clean": "x"},  # not text: left out
        ]
    }
    engine = _engine(json.dumps(answer))
    found, error = engine.suggest_category_groups("sex", ["Female", "Male", "Femme", "Homme"])
    assert error is None
    assert [(p["value"], p["clean"]) for p in found] == [("Femme", "Female"), ("Homme", "Male")]
    (prompt,) = engine.llm.prompts
    assert '"sex"' in prompt and "- Femme" in prompt and "- W" not in prompt
    assert "400" not in prompt and "12" not in prompt  # no count


def test_an_engine_error_is_named_by_its_type_and_logs_no_value(
    caplog: pytest.LogCaptureFixture,
) -> None:
    engine = _engine(fails=True)
    with caplog.at_level(logging.DEBUG):
        found, error = engine.suggest_category_groups("sex", ["Female", "CANARY-VALUE-7Q"])
    assert found == [] and error is not None and "ConnectionError" in error
    assert "CANARY" not in error and "CANARY" not in caplog.text


def test_the_answer_is_read_only_for_values_sent_and_never_joins_two_categories() -> None:
    counts = {"NYHA II": 400, "NYHA III": 300, "Femme": 12, "Female": 400, "W": 2, "Femme ": 1}
    proposals = [
        {"value": "Femme", "clean": "Female", "reason": "French\nfor   female"},
        {"value": "Femme ", "clean": "Female"},  # the same spelling: one offer
        {"value": "W", "clean": "Female"},  # not sent
        {"value": "Unknown value", "clean": "Female"},  # not a value of the column
        {"value": "NYHA III", "clean": "NYHA II", "reason": "close"},  # two categories
        {"value": "Female", "clean": ""},  # no clean value
        {"value": "Female", "clean": "x" * 61},  # too long
    ]
    offers = ai_offers(proposals, counts, sent=["NYHA II", "NYHA III", "Femme", "Female", "Femme "])
    assert offers == (Offer("Femme", "Female", AI, reason="French for female"),)
    assert (
        ai_offers(proposals, counts, sent=["Femme"], taken={"Femme"}) == ()
    )  # a rule's offer names it


def test_a_proposed_merge_of_two_large_values_is_a_grouping_its_reason_pre_filled() -> None:
    counts = {"Oui": 300, "Yes": 400, "Non": 200}
    (offer,) = ai_offers(
        [{"value": "Oui", "clean": "Yes", "reason": "French for yes"}], counts, sent=list(counts)
    )
    choices = new_choices()
    tick(choices, counts, offer)
    assert choices["reasons"] == {"Yes": "French for yes"}
    assert groupings(counts, choices["mapping"], choices["kinds"]) == {"Yes": ["Oui", "Yes"]}
    assert groupings({"F": 400, "Femme": 3}, {"Femme": "F"}, {"Femme": CHOSEN}) == {}


def test_free_text_is_not_sent() -> None:
    assert not free_text({"Female": 3, "Male": 2})
    assert free_text(
        {"the patient reported chest pain at rest": 1, "dyspnoea on exertion for three weeks": 1}
    )


def test_a_translation_or_an_abbreviation_is_read_and_a_number_never_crossed() -> None:
    counts = {
        "myocardial infarction": 300,
        "MI": 20,
        "Yes": 300,
        "Oui": 30,
        "Type 1": 50,
        "Type 2": 300,
    }
    proposals = [
        {
            "value": "MI",
            "clean": "myocardial infarction",
        },  # "MI" reads as a numeral only beside another
        {"value": "Oui", "clean": "Yes"},
        {"value": "Type 1", "clean": "Type 2"},
    ]
    got = {o.value: o.clean for o in ai_offers(proposals, counts, sent=list(counts))}
    assert got == {"MI": "myocardial infarction", "Oui": "Yes"}
