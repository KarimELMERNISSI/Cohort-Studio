"""Evaluation prompts for RAG quality assessment using LLM-as-judge."""


def evaluate_context_relevance(query: str, context: str) -> str:
    """
    Evaluate how relevant the retrieved context is to the query.
    
    Args:
        query: The original user query
        context: The retrieved context chunks
    
    Returns:
        Prompt for LLM-as-judge
    """
    return f"""Role: Evaluation Judge

Task: Evaluate the relevance of the retrieved context to the given query.

Query:
{query}

Retrieved Context:
{context}

Instructions:
1. Assess how well the context addresses the query.
2. Consider: Does the context contain information needed to answer the query?
3. Rate on a scale of 1-10 where:
   - 1-3: Context is irrelevant or off-topic
   - 4-6: Context is partially relevant, some useful information
   - 7-9: Context is highly relevant, contains key information
   - 10: Context is perfectly relevant, comprehensive coverage

Return JSON:
{{
    "score": <1-10>,
    "reasoning": "Brief explanation of the score"
}}"""


def evaluate_faithfulness(context: str, response: str) -> str:
    """
    Evaluate if the response is faithful/grounded in the provided context.
    
    Args:
        context: The source context used for generation
        response: The generated response
    
    Returns:
        Prompt for LLM-as-judge
    """
    return f"""Role: Faithfulness Judge

Task: Evaluate if the response is factually grounded in the provided context.

Context (Source of Truth):
{context}

Generated Response:
{response}

Instructions:
1. Check if ALL claims in the response can be verified from the context.
2. Identify any statements that go beyond what the context supports.
3. Rate on a scale of 1-10 where:
   - 1-3: Response contains significant unsupported claims
   - 4-6: Response is partially grounded, some extrapolation
   - 7-9: Response is well-grounded, minimal extrapolation
   - 10: Response is fully supported by the context

Return JSON:
{{
    "score": <1-10>,
    "reasoning": "Brief explanation",
    "unsupported_claims": ["list of claims not in context, if any"]
}}"""


def evaluate_answer_relevance(query: str, response: str) -> str:
    """
    Evaluate how relevant the response is to the original query.
    
    Args:
        query: The original user query
        response: The generated response
    
    Returns:
        Prompt for LLM-as-judge
    """
    return f"""Role: Relevance Judge

Task: Evaluate how well the response addresses the original query.

Query:
{query}

Response:
{response}

Instructions:
1. Assess if the response directly answers the query.
2. Check for completeness - are all aspects of the query addressed?
3. Rate on a scale of 1-10 where:
   - 1-3: Response is off-topic or doesn't address the query
   - 4-6: Response partially addresses the query
   - 7-9: Response addresses the query well
   - 10: Response perfectly and completely addresses the query

Return JSON:
{{
    "score": <1-10>,
    "reasoning": "Brief explanation"
}}"""


def detect_hallucination(context: str, response: str) -> str:
    """
    Detect if the response contains hallucinated (fabricated) information.
    
    Args:
        context: The source context
        response: The generated response
    
    Returns:
        Prompt for hallucination detection
    """
    return f"""Role: Hallucination Detector

Task: Identify any fabricated or hallucinated claims in the response.

Context (Ground Truth):
{context}

Response to Check:
{response}

Instructions:
1. Identify specific claims or facts in the response.
2. For each claim, verify if it is supported by the context.
3. Flag claims that are:
   - Completely fabricated (not in context at all)
   - Partially wrong (misrepresents context)
   - Invented statistics or numbers
   - Made-up sources or citations

Return JSON:
{{
    "detected": true/false,
    "claims": [
        {{
            "claim": "The specific claim text",
            "issue": "Why this is a hallucination",
            "severity": "high/medium/low"
        }}
    ]
}}"""


def evaluate_json_structure(expected_keys: list, response: str) -> str:
    """
    Evaluate if the JSON response has the expected structure.
    
    Args:
        expected_keys: List of expected top-level keys
        response: The response containing JSON
    
    Returns:
        Prompt for structure evaluation
    """
    keys_str = ", ".join(expected_keys)
    return f"""Role: JSON Structure Validator

Task: Validate the JSON structure in the response.

Expected Keys: [{keys_str}]

Response:
{response}

Instructions:
1. Extract the JSON from the response.
2. Check if all expected keys are present.
3. Check if values have appropriate types (not null/empty when shouldn't be).

Return JSON:
{{
    "is_valid": true/false,
    "missing_keys": ["list of missing keys"],
    "empty_values": ["keys with null/empty values"],
    "extra_keys": ["unexpected keys found"]
}}"""


def evaluate_formula_correctness(formula: str, expected_variables: list) -> str:
    """
    Evaluate if a generated formula is mathematically correct.
    
    Args:
        formula: The generated formula string
        expected_variables: List of variables that should be in the formula
    
    Returns:
        Prompt for formula evaluation
    """
    vars_str = ", ".join(expected_variables)
    return f"""Role: Medical Formula Validator

Task: Validate the mathematical formula.

Formula: {formula}
Expected Variables: [{vars_str}]

Instructions:
1. Check if the formula is syntactically correct (can be computed).
2. Verify it uses the expected variables.
3. Check for common errors: division by zero risks, wrong operators, etc.

Return JSON:
{{
    "is_valid": true/false,
    "syntax_errors": ["list of syntax issues"],
    "missing_variables": ["expected but not used"],
    "unknown_variables": ["used but not in expected"],
    "warnings": ["potential issues"]
}}"""
