"""
Unit Tests for RAG Evaluator Module (manage/rag_evaluator.py).

Tests retrieval quality, semantic similarity, RAGAS generation metrics
(faithfulness, answer relevance, hallucination detection), JSON output validation,
and metric logging.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from manage.rag_evaluator import RAGEvaluator


@pytest.fixture
def eval_log_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for evaluation logs."""
    log_dir = tmp_path / "rag_eval_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


@pytest.fixture
def mock_llm() -> MagicMock:
    """Create a mock LLM connector returning simulated judge outputs."""
    llm = MagicMock()
    return llm


class TestRAGEvaluator:
    """Test suite for RAGEvaluator."""

    def test_evaluator_initialization(self, eval_log_dir: Path) -> None:
        """Evaluator creates log directory and metrics summary file on start."""
        evaluator = RAGEvaluator(log_dir=str(eval_log_dir))
        assert evaluator.log_dir.exists()
        assert evaluator.summary_file.exists()

        summary = evaluator._load_summary()
        assert "metrics" in summary
        assert "context_relevance" in summary["metrics"]
        assert "faithfulness" in summary["metrics"]
        assert summary["total_evaluations"] == 0

    def test_semantic_similarity_computation(self, eval_log_dir: Path) -> None:
        """Calculates cosine similarity metrics across query and chunk embeddings."""
        evaluator = RAGEvaluator(log_dir=str(eval_log_dir))

        query_emb = [1.0, 0.0, 0.0]
        chunk_embs = [
            [1.0, 0.0, 0.0],  # Cosine sim = 1.0
            [0.0, 1.0, 0.0],  # Cosine sim = 0.0
            [0.7071, 0.7071, 0.0],  # Cosine sim ~ 0.7071
        ]

        metrics = evaluator.calculate_semantic_similarity(query_emb, chunk_embs)
        assert metrics["max_similarity"] == pytest.approx(1.0, rel=1e-3)
        assert metrics["min_similarity"] == pytest.approx(0.0, rel=1e-3)
        assert metrics["avg_similarity"] > 0.5
        assert "std_similarity" in metrics

    def test_semantic_similarity_empty_inputs(self, eval_log_dir: Path) -> None:
        """Empty vectors return default zeroes without exception."""
        evaluator = RAGEvaluator(log_dir=str(eval_log_dir))
        res = evaluator.calculate_semantic_similarity([], [])
        assert res["avg_similarity"] == 0
        assert res["max_similarity"] == 0

    def test_evaluate_retrieval_quality_with_mock_llm(
        self, eval_log_dir: Path, mock_llm: MagicMock
    ) -> None:
        """Retrieval quality parses LLM-as-judge relevance score."""
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "score": 8.5,
            "reasoning": "Chunks contain direct clinical dosage guidelines.",
        })
        mock_llm.invoke.return_value = mock_response

        evaluator = RAGEvaluator(llm=mock_llm, log_dir=str(eval_log_dir))
        res = evaluator.evaluate_retrieval_quality(
            query="What is the recommended dosage for ticagrelor?",
            retrieved_chunks=["Chunk 1 text", "Chunk 2 text"],
            k=2,
        )

        assert res.get("context_relevance") == 8.5
        assert "dosage" in res.get("reasoning", "").lower()
        assert res.get("chunks_evaluated") == 2
        assert mock_llm.invoke.called

    def test_evaluate_retrieval_quality_missing_llm(self, eval_log_dir: Path) -> None:
        """Retrieval quality handles None LLM gracefully."""
        evaluator = RAGEvaluator(llm=None, log_dir=str(eval_log_dir))
        res = evaluator.evaluate_retrieval_quality(
            query="Query", retrieved_chunks=["Chunk"]
        )
        assert "error" in res
        assert res["context_relevance"] == 0

    def test_evaluate_generation_quality(
        self, eval_log_dir: Path, mock_llm: MagicMock
    ) -> None:
        """Tests faithfulness, answer relevance, and hallucination scoring."""
        # 1st call for faithfulness, 2nd for answer_relevance, 3rd for hallucination
        faith_resp = MagicMock(content=json.dumps({"score": 9.2, "reasoning": "Fully grounded"}))
        rel_resp = MagicMock(content=json.dumps({"score": 9.5, "reasoning": "Direct answer"}))
        hall_resp = MagicMock(content=json.dumps({"detected": False, "claims": []}))
        mock_llm.invoke.side_effect = [faith_resp, rel_resp, hall_resp]

        evaluator = RAGEvaluator(llm=mock_llm, log_dir=str(eval_log_dir))
        results = evaluator.evaluate_generation_quality(
            query="What is the stroke risk?",
            context="The prospective trial demonstrated a 1.2% stroke risk.",
            response="The stroke risk is 1.2% according to the trial.",
        )

        assert results.get("faithfulness") == 9.2
        assert results.get("answer_relevance") == 9.5
        assert results.get("has_hallucination") is False
        assert mock_llm.invoke.call_count == 3

    def test_validate_json_output(self, eval_log_dir: Path) -> None:
        """Validates JSON structure extraction from LLM response text."""
        evaluator = RAGEvaluator(log_dir=str(eval_log_dir))

        valid_json_text = 'Here is the result: {"variable": "systolic_bp", "unit": "mmHg"}'
        res_valid = evaluator.validate_json_output(valid_json_text)
        assert res_valid.get("is_valid") is True
        assert res_valid.get("json_type") == "object"
        assert "variable" in res_valid.get("keys", [])

        invalid_text = "There is no JSON structure here at all."
        res_invalid = evaluator.validate_json_output(invalid_text)
        assert res_invalid.get("is_valid") is False

    def test_log_evaluation_and_summary_persistence(self, eval_log_dir: Path) -> None:
        """Logging an evaluation appends to results file and updates running summary."""
        evaluator = RAGEvaluator(log_dir=str(eval_log_dir))

        eval_data = {
            "eval_id": "test_id_123",
            "timestamp": "2026-10-10T12:00:00",
            "function_name": "trial_qa",
            "query_preview": "Clinical query A",
            "context_relevance": 8.0,
            "faithfulness": 9.0,
            "answer_relevance": 9.0,
            "json_validity": {"is_valid": True},
        }

        evaluator.log_result(eval_data)

        assert evaluator.results_file.exists()
        with open(evaluator.results_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 1
            record = json.loads(lines[0])
            assert record["function_name"] == "trial_qa"

        summary = evaluator._load_summary()
        assert summary["total_evaluations"] == 1
        assert summary["metrics"]["faithfulness"]["avg"] == 9.0
        assert summary["metrics"]["context_relevance"]["avg"] == 8.0
