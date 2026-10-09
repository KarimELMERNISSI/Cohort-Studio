"""
RAG Evaluator Module

Provides comprehensive evaluation metrics for the RAG system including:
- Retrieval quality (context relevance, semantic similarity)
- Generation quality (faithfulness, answer relevance, hallucination detection)
- Embedding quality assessment
- Logging and tracking of evaluation results
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

# Import prompt functions
from prompts.evaluation import (
    detect_hallucination,
    evaluate_answer_relevance,
    evaluate_context_relevance,
    evaluate_faithfulness,
)
from utils.data_paths import get_rag_eval_logs_dir


class RAGEvaluator:
    """
    Evaluates RAG system outputs for quality monitoring.
    
    Uses LLM-as-judge approach for semantic evaluation metrics
    and direct validation for structural metrics.
    """
    
    def __init__(self, llm=None, log_dir: str = None):
        """
        Initialize the RAG Evaluator.
        
        Args:
            llm: The LLM instance to use for evaluation (same as RAG system)
            log_dir: Directory to store evaluation logs (defaults to DATA_ROOT/rag_eval_logs)
        """
        self.llm = llm
        self.log_dir = Path(log_dir) if log_dir else Path(get_rag_eval_logs_dir())
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Log files
        self.results_file = self.log_dir / "evaluation_results.jsonl"
        self.summary_file = self.log_dir / "metrics_summary.json"
        self.test_suite_file = self.log_dir / "test_suite.json"
        
        # Initialize summary if not exists
        if not self.summary_file.exists():
            self._init_summary()
    
    def _init_summary(self):
        """Initialize the metrics summary file."""
        summary = {
            "created_at": datetime.now().isoformat(),
            "total_evaluations": 0,
            "metrics": {
                "context_relevance": {"sum": 0, "count": 0, "avg": 0},
                "faithfulness": {"sum": 0, "count": 0, "avg": 0},
                "answer_relevance": {"sum": 0, "count": 0, "avg": 0},
                "json_validity": {"success": 0, "total": 0, "rate": 0},
                "hallucination_rate": {"detected": 0, "total": 0, "rate": 0},
            },
            "by_function": {},
            "recent_scores": []
        }
        self._save_summary(summary)
    
    def _save_summary(self, summary: dict):
        """Save the metrics summary to file."""
        with open(self.summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    
    def _load_summary(self) -> dict:
        """Load the metrics summary from file."""
        if self.summary_file.exists():
            with open(self.summary_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
    
    def _generate_eval_id(self, query: str, timestamp: str) -> str:
        """Generate a unique evaluation ID."""
        content = f"{query}{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    # ==================== RETRIEVAL QUALITY METRICS ====================
    
    def evaluate_retrieval_quality(
        self,
        query: str,
        retrieved_chunks: list[str],
        k: int = 5
    ) -> dict[str, Any]:
        """
        Evaluate the quality of retrieved context.
        
        Args:
            query: The original query
            retrieved_chunks: List of retrieved text chunks
            k: Number of top chunks to consider
        
        Returns:
            Dictionary with relevance scores and analysis
        """
        if not self.llm or not retrieved_chunks:
            return {"error": "LLM or chunks not available", "context_relevance": 0}
        
        # Combine chunks for evaluation
        context = "\n\n---\n\n".join(retrieved_chunks[:k])
        
        # Use LLM-as-judge for context relevance
        prompt = evaluate_context_relevance(query=query, context=context)
        
        try:
            response = self.llm.invoke(prompt)
            result = self._parse_evaluation_response(response.content, "context_relevance")
            return {
                "context_relevance": result.get("score", 0),
                "reasoning": result.get("reasoning", ""),
                "chunks_evaluated": len(retrieved_chunks[:k])
            }
        except Exception as e:
            return {"error": str(e), "context_relevance": 0}
    
    def calculate_semantic_similarity(
        self,
        query_embedding: list[float],
        chunk_embeddings: list[list[float]]
    ) -> dict[str, float]:
        """
        Calculate semantic similarity between query and chunks.
        
        Args:
            query_embedding: The query's embedding vector
            chunk_embeddings: List of chunk embedding vectors
        
        Returns:
            Dictionary with similarity metrics
        """
        import numpy as np
        
        if not query_embedding or not chunk_embeddings:
            return {"avg_similarity": 0, "max_similarity": 0, "min_similarity": 0}
        
        query_vec = np.array(query_embedding)
        similarities = []
        
        for chunk_emb in chunk_embeddings:
            chunk_vec = np.array(chunk_emb)
            # Cosine similarity
            similarity = np.dot(query_vec, chunk_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(chunk_vec)
            )
            similarities.append(similarity)
        
        return {
            "avg_similarity": float(np.mean(similarities)),
            "max_similarity": float(np.max(similarities)),
            "min_similarity": float(np.min(similarities)),
            "std_similarity": float(np.std(similarities))
        }
    
    # ==================== GENERATION QUALITY METRICS ====================
    
    def evaluate_generation_quality(
        self,
        query: str,
        context: str,
        response: str,
        function_name: str = "generic"
    ) -> dict[str, Any]:
        """
        Evaluate the quality of generated response.
        
        Args:
            query: The original query
            context: The retrieved context used
            response: The generated response
            function_name: Name of the RAG function being evaluated
        
        Returns:
            Dictionary with quality scores
        """
        if not self.llm:
            return {"error": "LLM not available"}
        
        results = {}
        
        # 1. Faithfulness (groundedness)
        try:
            prompt = evaluate_faithfulness(context=context, response=response)
            llm_response = self.llm.invoke(prompt)
            faith_result = self._parse_evaluation_response(llm_response.content, "faithfulness")
            results["faithfulness"] = faith_result.get("score", 0)
            results["faithfulness_reasoning"] = faith_result.get("reasoning", "")
        except Exception as e:
            results["faithfulness"] = 0
            results["faithfulness_error"] = str(e)
        
        # 2. Answer Relevance
        try:
            prompt = evaluate_answer_relevance(query=query, response=response)
            llm_response = self.llm.invoke(prompt)
            rel_result = self._parse_evaluation_response(llm_response.content, "answer_relevance")
            results["answer_relevance"] = rel_result.get("score", 0)
            results["relevance_reasoning"] = rel_result.get("reasoning", "")
        except Exception as e:
            results["answer_relevance"] = 0
            results["relevance_error"] = str(e)
        
        # 3. Hallucination Detection
        try:
            prompt = detect_hallucination(context=context, response=response)
            llm_response = self.llm.invoke(prompt)
            hall_result = self._parse_evaluation_response(llm_response.content, "hallucination")
            results["has_hallucination"] = hall_result.get("detected", False)
            results["hallucination_claims"] = hall_result.get("claims", [])
        except Exception as e:
            results["has_hallucination"] = None
            results["hallucination_error"] = str(e)
        
        return results
    
    def validate_json_output(self, response: str) -> dict[str, Any]:
        """
        Validate if the response contains valid JSON.
        
        Args:
            response: The generated response
        
        Returns:
            Dictionary with validation results
        """
        # Try to find JSON in the response
        import re
        
        # Look for JSON object or array
        json_patterns = [
            r'\{[\s\S]*\}',  # Object
            r'\[[\s\S]*\]',  # Array
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, response)
            for match in matches:
                try:
                    parsed = json.loads(match)
                    return {
                        "is_valid": True,
                        "json_type": "object" if isinstance(parsed, dict) else "array",
                        "keys": list(parsed.keys()) if isinstance(parsed, dict) else None
                    }
                except json.JSONDecodeError:
                    continue
        
        return {"is_valid": False, "error": "No valid JSON found"}
    
    # ==================== FULL EVALUATION ====================
    
    def evaluate(
        self,
        query: str,
        context: str,
        response: str,
        function_name: str = "generic",
        retrieved_chunks: list[str] | None = None
    ) -> dict[str, Any]:
        """
        Run full evaluation on a RAG query-response pair.
        
        Args:
            query: The original query
            context: The retrieved context
            response: The generated response
            function_name: Name of the RAG function
            retrieved_chunks: Optional list of individual chunks
        
        Returns:
            Complete evaluation results
        """
        timestamp = datetime.now().isoformat()
        eval_id = self._generate_eval_id(query, timestamp)
        
        results = {
            "eval_id": eval_id,
            "timestamp": timestamp,
            "function_name": function_name,
            "query_preview": query[:100] + "..." if len(query) > 100 else query,
        }
        
        # Retrieval quality
        if retrieved_chunks:
            retrieval_results = self.evaluate_retrieval_quality(query, retrieved_chunks)
            results.update(retrieval_results)
        
        # Generation quality
        generation_results = self.evaluate_generation_quality(
            query, context, response, function_name
        )
        results.update(generation_results)
        
        # JSON validity
        json_results = self.validate_json_output(response)
        results["json_validity"] = json_results
        
        # Calculate overall score
        scores = [
            results.get("context_relevance", 0),
            results.get("faithfulness", 0),
            results.get("answer_relevance", 0),
        ]
        valid_scores = [s for s in scores if s > 0]
        results["overall_score"] = sum(valid_scores) / len(valid_scores) if valid_scores else 0
        
        return results
    
    def log_result(self, result: dict[str, Any]):
        """
        Log evaluation result to file and update summary.
        
        Args:
            result: Evaluation result dictionary
        """
        # Append to JSONL file
        with open(self.results_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
        
        # Update summary
        summary = self._load_summary()
        summary["total_evaluations"] += 1
        
        # Update metric averages
        for metric in ["context_relevance", "faithfulness", "answer_relevance"]:
            if metric in result and isinstance(result[metric], (int, float)):
                summary["metrics"][metric]["sum"] += result[metric]
                summary["metrics"][metric]["count"] += 1
                summary["metrics"][metric]["avg"] = (
                    summary["metrics"][metric]["sum"] / summary["metrics"][metric]["count"]
                )
        
        # Update JSON validity rate
        if "json_validity" in result:
            summary["metrics"]["json_validity"]["total"] += 1
            if result["json_validity"].get("is_valid"):
                summary["metrics"]["json_validity"]["success"] += 1
            summary["metrics"]["json_validity"]["rate"] = (
                summary["metrics"]["json_validity"]["success"] / 
                summary["metrics"]["json_validity"]["total"]
            )
        
        # Update hallucination rate
        if "has_hallucination" in result and result["has_hallucination"] is not None:
            summary["metrics"]["hallucination_rate"]["total"] += 1
            if result["has_hallucination"]:
                summary["metrics"]["hallucination_rate"]["detected"] += 1
            summary["metrics"]["hallucination_rate"]["rate"] = (
                summary["metrics"]["hallucination_rate"]["detected"] / 
                summary["metrics"]["hallucination_rate"]["total"]
            )
        
        # Track by function
        func_name = result.get("function_name", "generic")
        if func_name not in summary["by_function"]:
            summary["by_function"][func_name] = {"count": 0, "avg_score": 0, "sum_score": 0}
        summary["by_function"][func_name]["count"] += 1
        summary["by_function"][func_name]["sum_score"] += result.get("overall_score", 0)
        summary["by_function"][func_name]["avg_score"] = (
            summary["by_function"][func_name]["sum_score"] / 
            summary["by_function"][func_name]["count"]
        )
        
        # Keep last 100 scores for trend
        summary["recent_scores"].append({
            "timestamp": result["timestamp"],
            "score": result.get("overall_score", 0),
            "function": func_name
        })
        summary["recent_scores"] = summary["recent_scores"][-100:]
        
        self._save_summary(summary)
    
    def _parse_evaluation_response(self, response: str, metric_type: str) -> dict[str, Any]:
        """Parse LLM evaluation response into structured format."""
        try:
            # Try to parse as JSON
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
        
        # Fallback: extract score from text
        import re
        score_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:/\s*10|out of 10)?', response)
        score = float(score_match.group(1)) if score_match else 0
        
        return {
            "score": min(score, 10),  # Cap at 10
            "reasoning": response[:200]
        }
    
    # ==================== TEST SUITE ====================
    
    def load_test_suite(self) -> list[dict[str, Any]]:
        """Load the test suite from file."""
        if self.test_suite_file.exists():
            with open(self.test_suite_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("test_cases", [])
        return []
    
    def save_test_suite(self, test_cases: list[dict[str, Any]]):
        """Save test cases to file."""
        data = {
            "created_at": datetime.now().isoformat(),
            "test_cases": test_cases
        }
        with open(self.test_suite_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    
    def add_test_case(
        self,
        query: str,
        expected_concepts: list[str],
        ground_truth: str | None = None,
        expected_sources: list[str] | None = None
    ):
        """Add a new test case to the suite."""
        test_cases = self.load_test_suite()
        test_id = f"test_{len(test_cases) + 1:03d}"
        
        test_cases.append({
            "id": test_id,
            "query": query,
            "expected_concepts": expected_concepts,
            "ground_truth": ground_truth,
            "expected_sources": expected_sources or []
        })
        
        self.save_test_suite(test_cases)
        return test_id
    
    def run_test_suite(self, rag_manager) -> dict[str, Any]:
        """
        Run all test cases against the RAG system.
        
        Args:
            rag_manager: The RAGManager instance to test
        
        Returns:
            Test suite results with pass/fail for each case
        """
        test_cases = self.load_test_suite()
        results = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(test_cases),
            "passed": 0,
            "failed": 0,
            "details": []
        }
        
        for test in test_cases:
            # This would call the appropriate RAG function based on the test
            # For now, we'll just mark it as a placeholder
            test_result = {
                "id": test["id"],
                "query": test["query"],
                "status": "pending",
                "message": "Test execution not implemented"
            }
            results["details"].append(test_result)
        
        return results
    
    # ==================== METRICS RETRIEVAL ====================
    
    def get_summary(self) -> dict[str, Any]:
        """Get the current metrics summary."""
        return self._load_summary()
    
    def get_recent_evaluations(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent evaluation results."""
        if not self.results_file.exists():
            return []
        
        evaluations = []
        with open(self.results_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    evaluations.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        
        return evaluations[-limit:]
    
    def get_metrics_by_date(self, days: int = 7) -> dict[str, Any]:
        """Get metrics aggregated by date for the last N days."""
        from datetime import timedelta
        
        evaluations = self.get_recent_evaluations(limit=1000)
        cutoff = datetime.now() - timedelta(days=days)
        
        by_date = {}
        for eval_result in evaluations:
            try:
                eval_date = datetime.fromisoformat(eval_result["timestamp"]).date()
                if datetime.combine(eval_date, datetime.min.time()) >= cutoff:
                    date_str = eval_date.isoformat()
                    if date_str not in by_date:
                        by_date[date_str] = {"scores": [], "count": 0}
                    by_date[date_str]["scores"].append(eval_result.get("overall_score", 0))
                    by_date[date_str]["count"] += 1
            except (KeyError, ValueError):
                continue
        
        # Calculate averages
        for date_str in by_date:
            scores = by_date[date_str]["scores"]
            by_date[date_str]["avg_score"] = sum(scores) / len(scores) if scores else 0
        
        return by_date
