"""
Computed Variables Mixin for RAGManager.

Contains all computed variable suggestion and validation methods.
This is part of the RAGManager class composition pattern.
"""
import ast
import hashlib
import json
import logging
import re

from utils.llm_utils import parse_json_safe, validate_and_parse

# Import Pydantic schemas for structured output
from .rag_schemas import (
    AlternativeFormula,
    ProxyVariable,
)

logger = logging.getLogger(__name__)


def parse_formula_vars(formula_str: str) -> tuple[set[str], bool]:
    """
    Parses variable identifiers from a formula expression using Python AST.
    
    Handles:
    - Standard variable identifiers (e.g., `weight`, `height`)
    - Power operator normalization (`^` to `**`)
    - Escaped column names enclosed in double quotes (e.g., `""Systolic BP""`)
    - Column names enclosed in backticks (e.g., `` `Diastolic BP` ``)
    
    Args:
        formula_str: Mathematical expression string to parse.
        
    Returns:
        tuple[set[str], bool]: Set of extracted variable names and syntax validity flag.
    """
    found: set[str] = set()
    try:
        temp_fmt = formula_str.strip()
        # Normalize power operator if model output ^ instead of **
        temp_fmt = re.sub(r'(?<=\w)\s*\^\s*(?=\w|\d|\()', ' ** ', temp_fmt)

        q_vars: dict[str, str] = {}
        # Match ""var"" or `var`
        q_matches = re.findall(r'""([^"]+)""|`([^`]+)`', temp_fmt)
        idx = 0
        for m_double, m_backtick in q_matches:
            match_val = m_double if m_double else m_backtick
            placeholder = f"__quoted_var_{idx}__"
            q_vars[placeholder] = match_val
            if m_double:
                temp_fmt = temp_fmt.replace(f'""{m_double}""', placeholder)
            elif m_backtick:
                temp_fmt = temp_fmt.replace(f'`{m_backtick}`', placeholder)
            idx += 1
        
        tree = ast.parse(temp_fmt, mode='eval')
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                found.add(q_vars.get(node.id, node.id))
        return found, True
    except Exception:
        return set(), False


class ComputedVarsMixin:
    """
    Mixin class containing computed variable functionality for RAGManager.
    
    Methods:
        - _expand_search_hint
        - suggest_computed_variables
        - _get_theoretical_formulas
        - _map_formulas_to_data
        - suggest_computed_variables_with_validation
        - suggest_proxy_variable
        - suggest_alternative_formula
    """

    def _expand_search_hint(self, search_hint):
        """Pre-processing: Expand acronyms or ambiguous terms to standard medical concepts."""
        from prompts import expand_search_hint
        prompt = expand_search_hint(search_hint=search_hint)
        try:
            expanded = self.llm.invoke(prompt).content.strip().strip('"').strip("'")
            if len(expanded) < 60:
                return expanded
            return search_hint
        except Exception:
            return search_hint

    def suggest_computed_variables(self, columns, search_hint=None, num_suggestions=5, suggestion_mode="Comprehensive", allow_missing_variables=False, use_taxonomy=True, existing_categories=None, progress_callback=None):
        """Suggests computed variables based on dataset columns and medical knowledge."""
        if not self.initialized:
            return None, "RAG system not initialized."
        
        original_hint = search_hint
        if search_hint:
            if progress_callback: progress_callback(5, "Analyzing search term...")
            search_hint = self._expand_search_hint(search_hint)

        if search_hint:
            retrieval_query = f"{search_hint} formula calculation clinical score"
        else:
            retrieval_query = f"clinical scores formulas using {', '.join(columns[:50])}"
            
        try:
            docs = self.vector_store.similarity_search(retrieval_query, k=5)
            context_text = "\n\n".join([f"SOURCE: {d.metadata.get('source')} | CONTENT: {d.page_content}" for d in docs])
        except Exception:
            context_text = ""

        if progress_callback: progress_callback(30, "Generating theoretical concepts...")
        
        columns_hash = hashlib.md5(json.dumps(sorted(columns)).encode()).hexdigest()
        cache_key = hashlib.md5(f"{search_hint}_{suggestion_mode}_{self.adherence_score}_{columns_hash}_{use_taxonomy}".encode()).hexdigest()
        
        if cache_key in self.concept_cache:
            theoretical_concepts = self.concept_cache[cache_key]
        else:
            theoretical_concepts = self._get_theoretical_formulas(search_hint, suggestion_mode, context_text, num_suggestions * 2, columns, use_taxonomy)
            self.concept_cache[cache_key] = theoretical_concepts

        if progress_callback: progress_callback(50, "Mapping formulas to dataset...")
        final_suggestions_json = self._map_formulas_to_data(theoretical_concepts, columns, allow_missing_variables, num_suggestions, use_taxonomy, existing_categories)
        
        return final_suggestions_json, None

    def _get_theoretical_formulas(self, search_hint, suggestion_mode, context_text, count, available_columns=None, use_taxonomy=True):
        """Step 1: Identify WHAT to calculate (Standard Medical Knowledge)."""
        
        if suggestion_mode == "Go To Target" and search_hint:
            task_desc = f"Identify standard medical formulas that result in '{search_hint}' (Target is OUTPUT). If necessary, rearrange formulas to solve for it."
        elif suggestion_mode == "Go From Target" and search_hint:
            task_desc = f"Identify standard medical scores or indices that use '{search_hint}' as a required INPUT parameter. If necessary, rearrange formulas to solve for it. (Example: If input is BMI, suggest Weight = BMI*Height^2, If input is Height, suggest BMI = Weight/Height^2)."
        elif suggestion_mode == "Around Target" and search_hint:
            task_desc = f"Identify clinical proxies, surrogates, or alternative metrics for '{search_hint}'. Focus on variables that are strongly correlated or used as substitutes."
        else:
            task_desc = "Identify standard medical indices and scores relevant to the retrieved context."

        scoping_instruction = ""
        if available_columns:
            # Use all columns, as modern LLMs have large context windows
            cols_str = ", ".join(available_columns)
            
            taxonomy_context = ""
            if self.variable_taxonomy and use_taxonomy:
                taxonomy_str = json.dumps(self.variable_taxonomy, indent=2)
                taxonomy_context = f"\nUse this Taxonomy to understand the available variables:\n{taxonomy_str}\n"

            scoping_instruction = f"""
            **Scoping Constraint:**
            The suggested formulas MUST involve at least one variable from the following list (or a close synonym):
            [{cols_str}]
            {taxonomy_context}
            Do NOT suggest formulas where ALL inputs are missing from this list.
            """

        from prompts import theoretical_formulas
        prompt = theoretical_formulas(
            current_role=self.current_role,
            adherence_guidance=self.adherence_guidance,
            context_text=context_text[:3000],
            task_desc=task_desc,
            count=count,
            scoping_instruction=scoping_instruction
        )
        try:
            response = self.llm.invoke(prompt)
            # Use robust JSON parsing
            concepts = parse_json_safe(response.content, default=[])
            if not isinstance(concepts, list):
                # Handle case where model returns object with concepts key
                concepts = concepts.get('concepts', []) if isinstance(concepts, dict) else []
            logger.debug(f"Parsed {len(concepts)} theoretical concepts")
            return concepts
        except Exception as e:
            logger.warning(f"Failed to get theoretical formulas: {e}")
            return []

    def _map_formulas_to_data(self, concepts, columns, allow_missing, limit, use_taxonomy=True, existing_categories=None):
        """Step 2: Map theoretical inputs to actual dataset columns."""
        columns_str = ", ".join(columns)
        concepts_str = json.dumps(concepts, indent=2)
        missing_instr = "If a variable is missing, do NOT suggest the formula." if not allow_missing else "If a variable is missing, list it in 'missing_variables' and keep the standard name."

        taxonomy_context = ""
        if self.variable_taxonomy and use_taxonomy:
            taxonomy_str = json.dumps(self.variable_taxonomy, indent=2)
            taxonomy_context = f"""
            VARIABLE TAXONOMY (Use this to understand cryptic column names):
            {taxonomy_str}
            """
            
        category_instruction = ""
        if existing_categories:
            cats_str = ", ".join(existing_categories)
            category_instruction = f"""
            5. **CATEGORY CLASSIFICATION**: 
               - You MUST classify the new variable into one of these Existing Categories: [{cats_str}].
               - Only create a new category if absolutely necessary (e.g. "Computed", "Derived"). 
               - Return this in the 'suggestion_category' field.
            """

        from prompts import map_formulas
        prompt = map_formulas(
            columns_str=columns_str,
            concepts_str=concepts_str,
            missing_instr=missing_instr,
            limit=limit,
            taxonomy_context=taxonomy_context,
            category_instruction=category_instruction
        )
        try:
            response = self.llm.invoke(prompt)
            cleaned = self._clean_json_response(response.content)
            
            # Validate basic structure before returning
            parsed = parse_json_safe(cleaned, default={"suggestions": []})
            if not isinstance(parsed, dict) or 'suggestions' not in parsed:
                parsed = {"suggestions": parsed if isinstance(parsed, list) else []}
            
            logger.debug(f"Mapped {len(parsed.get('suggestions', []))} formulas to data")
            return json.dumps(parsed)
        except Exception as e:
            logger.warning(f"Failed to map formulas to data: {e}")
            return json.dumps({"suggestions": [], "error": str(e)})

    def suggest_computed_variables_with_validation(self, columns, search_hint=None, num_suggestions=5, suggestion_mode="Comprehensive", allow_missing_variables=False, use_taxonomy=True, existing_categories=None, progress_callback=None):
        """Wrapper that calls the pipeline and performs final AST validation with Auto-Correction."""
        
        json_str, error = self.suggest_computed_variables(columns, search_hint, num_suggestions, suggestion_mode, allow_missing_variables, use_taxonomy, existing_categories, progress_callback)
        if error: 
            return None, error
        
        # Use robust parsing with fallback
        try:
            data = parse_json_safe(json_str, default={"suggestions": []})
            if not isinstance(data, dict):
                data = {"suggestions": data if isinstance(data, list) else []}
            suggestions = data.get("suggestions", [])
        except Exception as e:
            logger.warning(f"Failed to parse suggestions: {e}")
            return None, f"Failed to parse suggestions: {e}"

        if progress_callback: progress_callback(60, "Validating and aligning variables...")
        validated = []
        
        known_globals = {'np', 'pd', 'log', 'exp', 'sqrt', 'abs', 'min', 'max', 'constant'} 


        total_sugg = len(suggestions)
        for idx, sugg in enumerate(suggestions):
            if progress_callback:
                current_prog = 60 + int((idx / max(1, total_sugg)) * 25)
                progress_callback(current_prog, f"Validating suggestion {idx+1}/{total_sugg}...")

            formula = sugg.get("formula", "")
            
            if not formula.strip() or formula.strip() in columns: 
                continue 

            found_vars, is_valid_syntax = parse_formula_vars(formula)
            
            if not is_valid_syntax:
                continue

            missing = [v for v in found_vars if v not in columns and v not in known_globals and v != 'np']
            
            if missing:
                if progress_callback:
                    progress_callback(current_prog, f"Aligning variables for suggestion {idx+1}...")
                
                from prompts import fix_formula_variables
                fix_query = fix_formula_variables(
                    formula=formula,
                    missing=missing,
                    columns=columns
                )
                try:
                    corrected_formula = self.llm.invoke(fix_query).content.strip().strip('"').strip("'")
                    
                    found_vars_2, is_valid_2 = parse_formula_vars(corrected_formula)
                    
                    if not is_valid_2:
                        if allow_missing_variables:
                            sugg['missing_variables'] = missing
                            validated.append(sugg)
                        continue

                    missing_2 = [v for v in found_vars_2 if v not in columns and v not in known_globals and v != 'np']
                    
                    if not missing_2:
                        sugg['formula'] = corrected_formula
                        sugg['original_formula'] = formula
                        validated.append(sugg)
                    else:
                        if allow_missing_variables:
                            sugg['formula'] = corrected_formula
                            sugg['missing_variables'] = missing_2
                            validated.append(sugg)
                except Exception:
                    if allow_missing_variables:
                        sugg['missing_variables'] = missing
                        validated.append(sugg)
            else:
                validated.append(sugg)

        if validated:
            if progress_callback: progress_callback(90, "Generating mathematical notation...")
            try:
                formulas_to_convert = {s['name']: s['formula'] for s in validated}
                
                from prompts import markdown_formula
                markdown_prompt = markdown_formula(
                    formulas_json=json.dumps(formulas_to_convert, indent=2)
                )
                
                md_response = self.llm.invoke(markdown_prompt)
                md_map = parse_json_safe(md_response.content, default={})
                
                for s in validated:
                    if s['name'] in md_map:
                        s['markdown_formula'] = md_map[s['name']]
            except Exception as e:
                logger.debug(f"Markdown formula conversion failed: {e}")

        data['suggestions'] = validated
        return json.dumps(data), None

    def suggest_proxy_variable(self, target_variable, available_columns):
        """Suggests a proxy variable when the target is missing."""
        query = f"Suggest a proxy for '{target_variable}' using [{', '.join(available_columns)}]. Return JSON with keys: proxy_found (bool), proxy_name, formula, explanation."
        try:
            response = self.llm.invoke(query)
            # Use Pydantic validation for structured output
            result, error = validate_and_parse(response.content, ProxyVariable)
            if result:
                return result.model_dump()
            return parse_json_safe(response.content, default={"proxy_found": False})
        except Exception as e:
            logger.warning(f"suggest_proxy_variable failed: {e}")
            return {"proxy_found": False}

    def suggest_alternative_formula(self, target_concept, missing_variable, available_columns):
        """Suggests an alternative formula avoiding a missing variable."""
        query = f"Suggest alternative formula for '{target_concept}' avoiding '{missing_variable}' using [{', '.join(available_columns)}]. Return JSON with keys: alternative_found (bool), alternative_name, formula, explanation."
        try:
            response = self.llm.invoke(query)
            # Use Pydantic validation for structured output
            result, error = validate_and_parse(response.content, AlternativeFormula)
            if result:
                return result.model_dump()
            return parse_json_safe(response.content, default={"alternative_found": False})
        except Exception as e:
            logger.warning(f"suggest_alternative_formula failed: {e}")
            return {"alternative_found": False}

    def suggest_imputation_formulas(self, target_variable, available_columns, search_hint=None, context_columns=None, context_stats=None, num_suggestions=3):
        """
        Suggests formulas to impute missing values for a specific target variable 
        based on relationships with other available columns.
        
        Parameters:
        - target_variable (str): Name of the variable to impute.
        - available_columns (list): List of all available column names.
        - search_hint (str): Optional user-provided hint (e.g. "Reverse BMI").
        - context_columns (list): Optional list of specific columns selected by the user.
        - context_stats (dict): Optional dictionary of statistics for the context columns.
        - num_suggestions (int): Number of suggestions to generate.
        """
        if not self.initialized:
            return None, "RAG system not initialized."
            
        # Determine which columns to show in the prompt
        if context_columns:
            columns_scope = context_columns
            scope_desc = "User-Selected Context Columns"
        else:
            columns_scope = available_columns[:100]
            scope_desc = "Available Columns"
            
        columns_str = ", ".join(columns_scope)
        
        # Prepare context stats string if available
        stats_context = ""
        if context_stats:
            stats_context = f"\n\nContext Column Statistics (Use this to match units or ranges):\n{json.dumps(context_stats, indent=2)}"

        # Prepare hint string
        hint_instruction = ""
        if search_hint:
            hint_instruction = f"\nUSER HINT: {search_hint}\nFocus specifically on relationships related to this hint."

        from prompts import imputation_formulas
        prompt = imputation_formulas(
            scope_desc=scope_desc,
            columns_str=columns_str,
            target_variable=target_variable,
            stats_context=stats_context,
            hint_instruction=hint_instruction,
            num_suggestions=num_suggestions
        )
        try:
            response = self.llm.invoke(prompt)
            cleaned = self._clean_json_response(response.content)
            parsed = parse_json_safe(cleaned, default={"suggestions": []})
            
            if not isinstance(parsed, dict) or 'suggestions' not in parsed:
                parsed = {"suggestions": parsed if isinstance(parsed, list) else []}
                
            return parsed.get("suggestions", []), None
            
        except Exception as e:
            logger.warning(f"Failed to suggest imputation formulas: {e}")
            return [], str(e)
