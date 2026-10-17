"""A language model that answers with an error is reported as such, never as an empty result.

The chat connectors answer a failure with the error as their message (manage/rag/providers), so that chains keep
running. A missing model once gave "Suggestions received!" over an empty list: Google no longer serves
gemini-2.5-flash to new keys.
"""

import json

import pytest

pytest.importorskip("langchain_core", reason="the AI stack is in the optional 'ai' extra")
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from manage.rag_computed_vars import ComputedVarsMixin
from utils.llm_utils import clean_json_response, provider_error

GONE = "API Error (Gemini): 404 NOT_FOUND. This model models/gemini-2.5-flash is no longer available to new users."
COLUMNS = ["patient_id", "height_cm", "weight_kg"]
# the shape of a Google key, built here so that no secret scanner sees one
KEY = "AIza" + "Sy0abcDEF_ghi-JKL" * 2 + "m"


class Engine(ComputedVarsMixin):
    """The suggestion pipeline of RAGManager on a scripted chat model."""

    def __init__(self, answers: list[str]) -> None:
        self.initialized = True
        self.llm = FakeListChatModel(responses=answers)
        self.vector_store = None  # no document library: the retrieval step finds nothing
        self.concept_cache: dict = {}
        self.adherence_score = 0.5
        self.adherence_guidance = ""
        self.current_role = "Medical Researcher"
        self.variable_taxonomy = None

    @staticmethod
    def _clean_json_response(text: str) -> str:
        return clean_json_response(text)


def test_an_error_answer_is_told_from_a_completion() -> None:
    assert provider_error(GONE) == GONE
    refused = "API Error (Gemini): Consumer 'api_key:{}' has been suspended."
    assert provider_error(refused.format(KEY)) == refused.format("[redacted]")
    for completion in ('[{"name": "bmi"}]', '{"suggestions": []}', "Body surface area", ""):
        assert provider_error(completion) is None


def test_suggestions_report_a_model_that_answers_with_an_error() -> None:
    suggestions, error = Engine([GONE]).suggest_computed_variables_with_validation(
        COLUMNS, search_hint="BSA"
    )
    assert suggestions is None
    assert error.startswith("The AI engine did not answer:") and "no longer available" in error


def test_a_failed_answer_is_not_kept_for_the_next_request() -> None:
    engine = Engine([GONE])
    assert engine.suggest_computed_variables_with_validation(COLUMNS)[1]

    concepts = [{"name": "bsa_mosteller", "formula": "sqrt(height * weight / 3600)"}]
    mapped = {
        "suggestions": [
            {
                "name": "bsa_mosteller",
                "title": "Mosteller",
                "formula": "np.sqrt(height_cm * weight_kg / 3600)",
            }
        ]
    }
    engine.llm = FakeListChatModel(responses=[json.dumps(concepts), json.dumps(mapped), "{}"])
    suggestions, error = engine.suggest_computed_variables_with_validation(COLUMNS)
    assert error is None
    assert [s["name"] for s in json.loads(suggestions)["suggestions"]] == ["bsa_mosteller"]


def test_a_connector_error_answer_hides_the_refused_key() -> None:
    pytest.importorskip("google.genai", reason="the Gemini SDK is in the optional 'ai' extra")
    from manage.rag.providers.gemini import GeminiChatConnector
    from manage.rag.schemas import ProviderConfig, ProviderType

    class Refusing:
        class models:
            @staticmethod
            def generate_content(**kwargs):
                raise RuntimeError(
                    f"403 PERMISSION_DENIED. Consumer 'api_key:{KEY}' has been suspended."
                )

    connector = GeminiChatConnector(
        ProviderConfig(provider=ProviderType.GEMINI, api_key="x" * 12, max_retries=1)
    )
    connector.client = Refusing()
    answer = connector.invoke("Which formula gives the body surface area?").content
    assert KEY not in answer and provider_error(answer) and "has been suspended" in answer
