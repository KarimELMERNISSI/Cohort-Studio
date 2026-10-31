"""
The AI engine's proposals on Data > Clean > Categories: what may be sent (manage.ai_payload.category_values), the
engine's request (manage.rag_computed_vars.suggest_category_groups), and how its answer is read (enrich.recode.
ai_offers): only a value sent, never two categories joined, never to missing; offered, never applied.
"""

from __future__ import annotations

import json
import logging

import pytest

from enrich.recode import (
    AI,
    CHOSEN,
    Offer,
    ai_offers,
    choices_of,
    free_text,
    groupings,
    new_choices,
    recorded,
    report_params,
    tick,
)
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
    ("provider", "url", "model", "sent"),
    [
        # this machine: every value, A to Z (their order says nothing of their rows)
        ("ollama", "http://localhost:11434", "llama3.2", ["Female", "Femme", "Homme", "Male", "W"]),
        (
            "ollama",
            "http://192.168.1.20:11434",
            "llama3.2",
            ["Female", "Femme", "Homme", "Male"],
        ),  # the network
        # an Ollama cloud model runs on ollama.com through the local server
        (
            "ollama",
            "http://localhost:11434",
            "gpt-oss:120b-cloud",
            ["Female", "Femme", "Homme", "Male"],
        ),
        ("ollama", "http://localhost:11434", "glm-4.6:cloud", ["Female", "Femme", "Homme", "Male"]),
        ("anthropic", None, "claude", ["Female", "Femme", "Homme", "Male"]),  # a cloud provider
    ],
)
def test_a_value_held_by_fewer_than_5_rows_reaches_only_a_model_run_on_this_machine(
    provider, url, model, sent
) -> None:
    got, held = category_values(COUNTS, provider, url, model)
    assert got == sent and held == len(COUNTS) - len(sent)


def test_free_text_mostly_rare_values_or_an_identifier_is_never_sent() -> None:
    long_text = {"the patient reported chest pain at rest": 9, "dyspnoea on exertion for weeks": 9}
    assert category_values(long_text, "ollama", "http://localhost:11434") == ([], 2)
    names = {f"Name {i}": 1 for i in range(40)} | {"Doe": 5}
    assert category_values(names, "anthropic", None) == ([], 41)


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


def test_a_proposal_is_an_offer_its_reason_never_pre_filled_and_who_proposed_is_recorded() -> None:
    """A synonym the user ticks is one category written two ways: no grouping, and the model's words never stand
    for the user's reason (the report printed "AVC is French for stroke" beside a 3-row value)."""
    counts = {"Oui": 300, "Yes": 400, "Non": 200}
    (offer,) = ai_offers(
        [{"value": "Oui", "clean": "Yes", "reason": "French for yes"}], counts, sent=list(counts)
    )
    by = {
        "provider": "ollama",
        "model": "gpt-oss:120b-cloud",
        "where": "cloud",
        "values_sent": 3,
        "values_held": 0,
    }
    choices = new_choices()
    tick(choices, counts, offer, by=by)
    assert choices["reasons"] == {} and choices["ai_by"] == {"Oui": by}
    mapping, kinds, kept, _ = recorded(counts, choices)
    assert kinds == {"Oui": AI} and groupings(counts, mapping, kinds, kept=kept) == {}
    assert groupings(counts, mapping, {"Oui": CHOSEN}, kept=kept) == {
        "Yes": ["Oui", "Yes"]
    }  # typed by hand
    # kept when the copy is cleaned again, and reported by counts, never by value
    again = choices_of({"mapping": mapping, "kinds": kinds, "ai_by": {"Oui": by, "Gone": by}})
    assert again["ai_by"] == {"Oui": by}
    shown = report_params({"mapping": mapping, "kinds": kinds, "ai_by": again["ai_by"]})
    assert shown["proposed by the AI engine"] == ["ollama gpt-oss:120b-cloud: 1 value"]
    assert "ai_by" not in shown and "Oui" not in repr(shown)


def test_a_proposal_both_ways_or_along_a_chain_keeps_one_direction() -> None:
    """Oui → Yes and Yes → Oui, both ticked, swapped the two names and merged nothing."""
    counts = {"Oui": 40, "Yes": 300, "Non": 30, "No": 200, "MALE": 6, "Male": 300, "M": 20}
    proposals = [
        {"value": "Yes", "clean": "Oui"},
        {"value": "Oui", "clean": "Yes"},
        {"value": "Non", "clean": "No"},
        {"value": "MALE", "clean": "Male"},
        {"value": "Male", "clean": "M"},
    ]
    got = {o.value: o.clean for o in ai_offers(proposals, counts, sent=list(counts))}
    assert got == {"Oui": "Yes", "Non": "No", "MALE": "Male"}


@pytest.mark.parametrize(
    ("value", "clean"),
    [
        ("Stage IIIB", "Stage IIIA"),
        ("Stage IIIA", "Stage III"),
        ("CKD 3b", "CKD 3a"),
        ("pT1b", "pT1a"),
        ("Stage IB", "Stage IA"),
        ("NSTEMI", "STEMI"),
        ("STEMI", "MI"),
        ("Unstable angina", "Stable angina"),
        ("HFmrEF", "HFrEF"),
        ("Ex-smoker", "Current smoker"),
        ("Hepatitis C", "Hepatitis B"),
        ("Hepatitis B", "Viral hepatitis"),
        ("Influenza B", "Influenza A"),
        ("Persistent AF", "Permanent AF"),
        ("Hyperthyroidism", "Hypothyroidism"),
        ("Not tested", "Negative"),
        ("Not applicable", "No"),
        ("Refused", "Unknown"),
        ("Homme", "unknown"),
        ("mg/dL", "mmol/L"),
        ("Negative", "Positive"),
        ("IgG lambda", "IgG kappa"),
        ("NYHA II", "NYHA III"),
        ("Male", "Female"),
        ("IgM", "IgG"),
    ],
)
def test_a_proposal_that_joins_two_categories_is_refused(value: str, clean: str) -> None:
    counts = {value: 80, clean: 300, "Other value": 50}
    assert ai_offers([{"value": value, "clean": clean}], counts, sent=list(counts)) == ()


@pytest.mark.parametrize(
    ("value", "clean"),
    [
        ("Femme", "Female"),
        ("MI", "Myocardial infarction"),
        ("Oui", "Yes"),
        ("OUI", "YES"),
        ("NYHA 3", "NYHA III"),
        ("Grade 2", "Grade II"),
        ("BPCO", "COPD"),
        ("Diabetes melitus", "Diabetes mellitus"),
    ],
)
def test_a_translation_a_synonym_or_an_abbreviation_is_offered(value: str, clean: str) -> None:
    counts = {value: 40, clean: 300, "Other value": 50}
    (offer,) = ai_offers(
        [{"value": value, "clean": clean, "reason": "same"}], counts, sent=list(counts)
    )
    assert (offer.value, offer.clean, offer.why) == (value, clean, AI)


def test_a_long_reason_is_cut_after_a_whole_word() -> None:
    counts = {"Femme": 40, "Female": 300}
    reason = "the French word for a woman, which this column writes in English in most of its rows, as Female"
    (offer,) = ai_offers(
        [{"value": "Femme", "clean": "Female", "reason": reason}], counts, sent=list(counts)
    )
    assert len(offer.reason) <= 80 and offer.reason.endswith("…")
    assert reason.startswith(offer.reason[:-1]) and reason[len(offer.reason) - 1] == " "


def test_free_text_mostly_rare_values_and_identifiers_are_never_asked_about() -> None:
    assert not free_text({"Female": 300, "Male": 200, "F": 3})
    assert free_text(
        {"the patient reported chest pain at rest": 9, "dyspnoea on exertion for three weeks": 9}
    )
    assert free_text({"Female": 3, "Male": 2})  # rare values hold half the rows or more
    assert free_text({f"Site {i}": 10 for i in range(40)})  # many values, none frequent


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
