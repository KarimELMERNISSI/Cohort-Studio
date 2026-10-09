"""
LLM Utilities for Reliable Structured Output.

Provides robust JSON parsing, schema validation, and structured LLM invocation
with retry and repair capabilities.
"""
import json
import logging
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


def clean_json_response(text: str) -> str:
    """
    Clean LLM response to extract valid JSON.
    
    Uses multiple strategies:
    1. Extract from markdown code blocks
    2. Find JSON by brace/bracket matching
    3. Return cleaned text
    """
    if not text:
        return "{}"
    
    text = text.strip()
    
    # Strategy 1: Extract from markdown code blocks
    if "```" in text:
        match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    
    # Strategy 2: Find JSON structure
    first_brace = text.find("{")
    first_bracket = text.find("[")
    
    start = -1
    end = -1
    
    if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
        start = first_brace
        end = text.rfind("}") + 1
    elif first_bracket != -1:
        start = first_bracket
        end = text.rfind("]") + 1
    
    if start != -1 and end > start:
        return text[start:end]
    
    return text


def repair_json(broken_json: str) -> str:
    """
    Attempt to repair common JSON issues.
    
    Fixes:
    - Trailing commas
    - Single quotes to double quotes (outside of values)
    - Missing closing brackets
    """
    text = broken_json.strip()
    
    # Fix trailing commas before closing brackets/braces
    text = re.sub(r',\s*([}\]])', r'\1', text)
    
    # Balance brackets
    open_braces = text.count('{')
    close_braces = text.count('}')
    open_brackets = text.count('[')
    close_brackets = text.count(']')
    
    # Add missing closing characters
    text += '}' * (open_braces - close_braces)
    text += ']' * (open_brackets - close_brackets)
    
    return text


def parse_json_safe(text: str, default: Any = None) -> Any:
    """
    Safely parse JSON with multiple fallback strategies.
    
    Args:
        text: Raw text containing JSON
        default: Default value if parsing fails completely
        
    Returns:
        Parsed JSON or default value
    """
    if not text:
        return default if default is not None else {}
    
    # Clean the response
    cleaned = clean_json_response(text)
    
    # Strategy 1: Direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Repair and parse
    try:
        repaired = repair_json(cleaned)
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Use ast.literal_eval for Python dict syntax
    try:
        import ast
        # Convert Python-style to JSON-style first
        py_text = cleaned.replace('True', 'true').replace('False', 'false').replace('None', 'null')
        return ast.literal_eval(py_text)
    except (ValueError, SyntaxError):
        pass
    
    logger.warning(f"Failed to parse JSON: {text[:200]}...")
    return default if default is not None else {}


def validate_and_parse(
    text: str,
    schema: type[T],
    strict: bool = False
) -> tuple[T | None, str | None]:
    """
    Parse JSON and validate against Pydantic schema.
    
    Args:
        text: Raw LLM response text
        schema: Pydantic model class to validate against
        strict: If True, raise on validation error
        
    Returns:
        Tuple of (parsed_model, error_message)
    """
    try:
        parsed = parse_json_safe(text)
        if parsed is None:
            return None, "Failed to parse JSON"
        
        # Handle both dict and list inputs based on schema
        model = schema.model_validate(parsed)
        return model, None
        
    except ValidationError as e:
        error_msg = f"Schema validation failed: {e.error_count()} errors"
        logger.warning(f"{error_msg}: {e.errors()}")
        
        if strict:
            raise
        
        # Try to create with partial data
        try:
            # Get minimal required fields
            parsed = parse_json_safe(text, {})
            model = schema.model_construct(**parsed)
            return model, f"Partial parse: {error_msg}"
        except Exception:
            return None, error_msg
            
    except Exception as e:
        error_msg = f"Unexpected error: {e!s}"
        logger.error(error_msg)
        return None, error_msg


def create_json_repair_prompt(broken_json: str, expected_schema: str) -> str:
    """
    Create a prompt to ask LLM to repair broken JSON.
    
    Args:
        broken_json: The malformed JSON string
        expected_schema: Description of expected schema
        
    Returns:
        Prompt for LLM to fix the JSON
    """
    return f"""
Role: JSON Repair Expert.

The following JSON is malformed and failed to parse:
```
{broken_json[:2000]}
```

Expected format:
{expected_schema}

Task: Return ONLY the corrected, valid JSON. Do not explain, just output valid JSON.
"""


def extract_json_from_text(text: str) -> list[dict]:
    """
    Extract all JSON objects from text.
    
    Useful when LLM returns multiple JSON blocks or explanatory text.
    
    Returns:
        List of parsed JSON objects
    """
    results = []
    
    # Find all potential JSON blocks
    patterns = [
        r'\{[^{}]*\}',  # Simple objects
        r'\[[^\[\]]*\]',  # Simple arrays
        r'```(?:json)?\s*(.*?)```',  # Code blocks
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                parsed = json.loads(match)
                results.append(parsed)
            except json.JSONDecodeError:
                continue
    
    return results


class StructuredOutputHelper:
    """
    Helper class for managing structured LLM outputs with schemas.
    
    Usage:
        helper = StructuredOutputHelper(llm)
        result, error = helper.invoke_with_schema(prompt, MySchema)
    """
    
    def __init__(self, llm, max_retries: int = 2):
        self.llm = llm
        self.max_retries = max_retries
    
    def invoke_with_schema(
        self,
        prompt: str,
        schema: type[T],
        repair_on_fail: bool = True
    ) -> tuple[T | None, str | None]:
        """
        Invoke LLM and parse response with schema validation.
        
        Args:
            prompt: The prompt to send
            schema: Pydantic model for validation
            repair_on_fail: If True, attempt LLM-based repair on parse failure
            
        Returns:
            Tuple of (parsed_model, error_message)
        """
        for attempt in range(self.max_retries + 1):
            try:
                response = self.llm.invoke(prompt)
                content = response.content if hasattr(response, 'content') else str(response)
                
                result, error = validate_and_parse(content, schema)
                
                if result is not None:
                    return result, None
                
                # Attempt repair on last retry
                if repair_on_fail and attempt < self.max_retries:
                    repair_prompt = create_json_repair_prompt(
                        content[:2000],
                        schema.model_json_schema().__str__()
                    )
                    response = self.llm.invoke(repair_prompt)
                    repaired = response.content if hasattr(response, 'content') else str(response)
                    result, error = validate_and_parse(repaired, schema)
                    
                    if result is not None:
                        logger.info("JSON repair successful")
                        return result, None
                        
            except Exception as e:
                error = str(e)
                logger.warning(f"Attempt {attempt + 1} failed: {error}")
                
                if attempt == self.max_retries:
                    return None, f"All attempts failed: {error}"
        
        return None, "Failed to get valid response"
    
    def invoke_json(
        self,
        prompt: str,
        default: Any = None
    ) -> tuple[Any, str | None]:
        """
        Invoke LLM and parse as generic JSON without schema.
        
        Args:
            prompt: The prompt to send
            default: Default value if parsing fails
            
        Returns:
            Tuple of (parsed_json, error_message)
        """
        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            result = parse_json_safe(content, default)
            return result, None
        except Exception as e:
            return default, str(e)
