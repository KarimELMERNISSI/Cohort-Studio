"""
Taxonomy Mixin for RAGManager.

Contains all variable taxonomy generation and enrichment methods.
This is part of the RAGManager class composition pattern.
"""
import json


class TaxonomyMixin:
    """
    Mixin class containing taxonomy-related functionality for RAGManager.
    
    Methods:
        - suggest_column_renaming
        - generate_variable_taxonomy
        - _generate_simple_taxonomy
        - _generate_advanced_taxonomy
        - _merge_stats
        - _identify_standard_concepts
        - _enrich_with_formulas
        - _contextualize_variables
        - _resolve_formula_links
        - _unify_synonyms
        - _enrich_graph_metadata
        - enrich_variable_taxonomy
        - refine_variable_taxonomy
    """

    def suggest_column_renaming(self, columns, strategy="literature", progress_callback=None):
        """Suggests standardized column names based on medical literature."""
        from prompts import column_renaming
        
        if not self.initialized:
            return None, "RAG system not initialized."

        if progress_callback: progress_callback(10, "Analyzing columns...")

        # Handle columns input (list of strings or dicts)
        if isinstance(columns, (list, tuple)) and len(columns) > 0 and isinstance(columns[0], dict):
            columns_str = json.dumps(columns, indent=2)
            search_query = ", ".join([c.get("name", "") for c in columns])
        else:
            columns_list = [str(c) for c in columns]
            columns_str = ", ".join(columns_list)
            search_query = columns_str

        # Retrieve Context
        if progress_callback: progress_callback(30, "Retrieving context...")
        try:
            docs = self.vector_store.similarity_search(f"Dataset variables: {search_query}", k=5)
            context_text = "\n\n".join([d.page_content for d in docs])
        except Exception:
            context_text = "No specific documentation found."

        if strategy == "standardization":
            task = "Map these columns to standard medical terminology (UMLS, SNOMED CT, LOINC)."
        else:
            task = "Suggest scientifically accurate and standard variable names based on the medical literature."

        prompt = column_renaming(
            current_role=self.current_role,
            adherence_guidance=self.adherence_guidance,
            task=task,
            columns_str=columns_str,
            context_text=context_text[:3000]
        )

        if progress_callback: progress_callback(60, "Generating suggestions...")
        try:
            response = self.llm.invoke(prompt)
            if progress_callback: progress_callback(90, "Parsing results...")
            cleaned_response = self._clean_json_response(response.content)
            return json.loads(cleaned_response), None
        except json.JSONDecodeError:
            return None, f"Failed to parse AI response. Raw output: {response.content[:100]}..."
        except Exception as e:
            return None, str(e)

    def generate_variable_taxonomy(self, columns_info, existing_mapping=None, deep_analysis=False, progress_callback=None):
        """
        Generates a taxonomy mapping for the provided columns.
        If deep_analysis is True, uses a multi-step pipeline for richer results.
        """
        if deep_analysis:
            res = self._generate_advanced_taxonomy(columns_info, existing_mapping, progress_callback)
        else:
            res = self._generate_simple_taxonomy(columns_info, existing_mapping, deep_analysis=False, progress_callback=progress_callback)
        return res

    def _merge_stats(self, taxonomy, columns_info):
        """Merges statistical info from columns_info into the taxonomy."""
        col_map = {c["name"]: c for c in columns_info if isinstance(c, dict)}
        
        for var_name, var_data in taxonomy.items():
            if var_name in col_map:
                col_info = col_map[var_name]
                if "stats" in col_info and col_info["stats"]:
                    var_data["stats"] = col_info["stats"]
                elif "top_values" in col_info and col_info["top_values"]:
                    var_data["stats"] = col_info["top_values"]
        return taxonomy

    def _generate_simple_taxonomy(self, columns_info, existing_mapping=None, deep_analysis=False, progress_callback=None):
        """
        Generates a taxonomy mapping for the provided columns.
        columns_info: List of dicts with 'name', 'type', 'stats', etc.
        existing_mapping: Dict of manual renames (original -> new) to use as ground truth.
        """
        from prompts import taxonomy_simple
        
        if not self.initialized:
            return None, "RAG system not initialized."

        if progress_callback: progress_callback(10, "Analyzing variable structure...")
        
        if isinstance(columns_info[0], str):
            columns_input = [{"name": c} for c in columns_info]
        else:
            columns_input = columns_info

        columns_str = json.dumps(columns_input, indent=2) 
        
        if progress_callback: progress_callback(25, "Retrieving documentation context...")
        try:
            search_query = ", ".join([c.get("name", "") for c in columns_input[:20]])
            docs = self.vector_store.similarity_search(f"Dataset variables definition: {search_query}", k=5)
            context_text = "\n\n".join([d.page_content for d in docs])
        except Exception:
            context_text = "No specific documentation found."

        manual_renames_str = ""
        if existing_mapping:
            meaningful_renames = {k: v for k, v in existing_mapping.items() if k != v}
            if meaningful_renames:
                manual_renames_str = f"""
                USER PROVIDED RENAMINGS (Ground Truth):
                {json.dumps(meaningful_renames, indent=2)}
                Use these as the definitive standard names for these variables.
                """

        deep_instructions = ""
        json_structure_extra = ""
        if deep_analysis:
            deep_instructions = """
            6. **DEEP ANALYSIS**: For each variable, identify:
               - **Related Formulas**: Standard medical formulas where this variable is used as a parameter or result.
               - **Clinical Usage**: How this variable is typically used in clinical practice or research.
            """
            json_structure_extra = """,
                "related_formulas": ["List of standard formulas (e.g. 'BMI = Weight/Height^2')"],
                "clinical_usage": "Brief explanation of clinical relevance."
            """

        prompt = taxonomy_simple(
            columns_str=columns_str,
            context_text=context_text[:4000],
            manual_renames_str=manual_renames_str,
            deep_instructions=deep_instructions,
            json_structure_extra=json_structure_extra
        )

        if progress_callback: progress_callback(50, "Generating taxonomy...")
        try:
            response = self.llm.invoke(prompt)
            if progress_callback: progress_callback(80, "Parsing taxonomy...")
            cleaned_response = self._clean_json_response(response.content)
            try:
                taxonomy = json.loads(cleaned_response)
            except json.JSONDecodeError as e:
                return None, f"JSON Parsing Error: {str(e)}. Raw output: {response.content[:500]}..."
            
            taxonomy = self._merge_stats(taxonomy, columns_info)
            self.variable_taxonomy = taxonomy
            return taxonomy, None
        except Exception as e:
            return None, f"Generation Error: {str(e)}"

    def _generate_advanced_taxonomy(self, columns_info, existing_mapping, progress_callback):
        """Orchestrates the multi-step taxonomy generation pipeline."""
        if not self.initialized:
            return None, "RAG system not initialized."
            
        taxonomy = {}
        
        if progress_callback: progress_callback(10, "Step 1/5: Standardizing variables...")
        standard_mapping, error = self._identify_standard_concepts(columns_info, existing_mapping)
        if error: return None, error
        
        if progress_callback: progress_callback(30, "Step 2/5: Adding clinical context...")
        taxonomy = self._contextualize_variables(standard_mapping, columns_info)
        
        if progress_callback: progress_callback(50, "Step 3/5: Identifying related formulas...")
        taxonomy = self._enrich_with_formulas(taxonomy, columns_info)
        
        if progress_callback: progress_callback(70, "Step 4/5: Resolving external links...")
        taxonomy = self._resolve_formula_links(taxonomy)

        if progress_callback: progress_callback(90, "Step 5/5: Enriching graph metadata...")
        taxonomy = self._enrich_graph_metadata(taxonomy, columns_info)
        
        taxonomy = self._merge_stats(taxonomy, columns_info)
        self.variable_taxonomy = taxonomy
        return taxonomy, None

    def _enrich_graph_metadata(self, taxonomy, columns_info):
        """Step 4: Identify node types and relationships for graph visualization."""
        formula_outputs = set()
        for f in self.formulas_registry.values():
            if f.get('output_variable'):
                formula_outputs.add(f['output_variable'])

        for k in taxonomy.keys():
            if k in formula_outputs:
                taxonomy[k]['node_type'] = "Derived-Internal"
            else:
                taxonomy[k]['node_type'] = "Input-Internal"

        vars_desc_list = []
        for k, v in taxonomy.items():
            desc = f"{k} ({v.get('standard_name', '')}): {v.get('description', '')}"
            vars_desc_list.append(desc)
            
        vars_desc = "\n".join(vars_desc_list[:50])
        
        prompt = f"""
        Role: Medical Data Scientist.
        
        Task: Analyze the variables to identify semantic relationships (correlations, risk factors) for a Knowledge Graph.
        
        Variables:
        {vars_desc}
        
        Instructions:
        For each variable, identify OTHER variables from the list that are directly related (e.g., risk factors, co-morbidities).
        Do NOT analyze formulas (we already have those). Focus on clinical associations.
        
        JSON Output Format:
        {{
            "original_var_name": ["related_var_1", "related_var_2"],
            ...
        }}
        """
        
        try:
            response = self.llm.invoke(prompt)
            cleaned_response = self._clean_json_response(response.content)
            metadata = json.loads(cleaned_response)
            
            for var, relationships in metadata.items():
                if var in taxonomy and isinstance(relationships, list):
                    current_rels = taxonomy[var].get('relationships', [])
                    taxonomy[var]['relationships'] = list(set(current_rels + relationships))
                    
        except Exception:
            pass  # Silently handle errors
            
        return taxonomy

    def _identify_standard_concepts(self, columns_info, existing_mapping):
        """Step 1: Map variables to standard medical concepts."""
        return self._generate_simple_taxonomy(columns_info, existing_mapping, deep_analysis=False)

    def _enrich_with_formulas(self, taxonomy, columns_info):
        """Step 2: Identify formula relationships (Structured)."""
        from prompts import formula_enrichment
        
        updated_taxonomy = taxonomy.copy()
        
        vars_desc_list = []
        for k, v in taxonomy.items():
            desc = f"{k} ({v.get('standard_name', '')})"
            col_info = next((c for c in columns_info if c['name'] == k), None)
            if col_info and 'top_values' in col_info:
                desc += f" [Values: {', '.join(list(col_info['top_values'].keys())[:5])}]"
            vars_desc_list.append(desc)
            
        vars_desc = "\n".join(vars_desc_list)
        
        prompt = formula_enrichment(
            current_role=self.current_role,
            vars_desc=vars_desc
        )

        try:
            response = self.llm.invoke(prompt)
            cleaned_response = self._clean_json_response(response.content)
            try:
                data = json.loads(cleaned_response)
                
                formulas = data.get('formulas', [])
                for f in formulas:
                    f_id = f.get('id')
                    if f_id:
                        self.formulas_registry[f_id] = f
                
                var_updates = data.get('variable_updates', {})
                std_to_orig = {v.get('standard_name', '').lower(): k for k, v in taxonomy.items()}
                
                for var_key, updates in var_updates.items():
                    target_key = None
                    
                    if var_key in updated_taxonomy:
                        target_key = var_key
                    elif var_key.lower() in std_to_orig:
                        target_key = std_to_orig[var_key.lower()]
                    
                    if target_key:
                        current_formulas = updated_taxonomy[target_key].get('related_formula_ids', [])
                        new_formulas = updates.get('involved_in_formulas', [])
                        updated_taxonomy[target_key]['related_formula_ids'] = list(set(current_formulas + new_formulas))
                        
            except json.JSONDecodeError:
                pass
            
        except Exception:
            pass
            
        return updated_taxonomy

    def _contextualize_variables(self, taxonomy, columns_info):
        """Step 3: Add clinical usage and interpretation context."""
        updated_taxonomy = taxonomy.copy()
        
        vars_desc_list = []
        for k, v in taxonomy.items():
            desc = f"{k} ({v.get('standard_name', '')})"
            col_info = next((c for c in columns_info if c['name'] == k), None)
            if col_info and 'top_values' in col_info:
                desc += f" [Values: {', '.join(list(col_info['top_values'].keys())[:5])}]"
            vars_desc_list.append(desc)
            
        vars_desc = "\n".join(vars_desc_list)
        
        prompt = f"""
        Role: Medical Expert.
        
        Task: Provide clinical context and detailed descriptions for the provided variables.
        
        Variables:
        {vars_desc}
        
        Instructions:
        For each variable, provide:
        1. "description": A clear, medical definition.
        2. "clinical_usage": How this variable is used in clinical practice (e.g., diagnosis, monitoring, prognosis).
        3. "category": A broad category (e.g., Demographics, Vitals, Lab Test, Comorbidity).
        4. "topic": A specific medical topic (e.g., "Cardiovascular Health", "Renal Function", "Diabetes Management").
        5. "proxy_variables": A list of potential proxy variables or synonyms often used interchangeably or as surrogates.
        
        Return JSON:
        {{
            "original_var_name": {{
                "description": "...",
                "clinical_usage": "...",
                "category": "...",
                "topic": "...",
                "proxy_variables": ["...", "..."]
            }}
        }}
        }}
        """
        try:
            response = self.llm.invoke(prompt)
            cleaned_response = self._clean_json_response(response.content)
            try:
                context_data = json.loads(cleaned_response)
            except json.JSONDecodeError:
                context_data = {}
            
            for var, data in context_data.items():
                if var in updated_taxonomy:
                    updated_taxonomy[var]['clinical_usage'] = data.get('clinical_usage', "")
        except Exception:
            pass
            
        return updated_taxonomy

    def _resolve_formula_links(self, taxonomy):
        """Step 4: Resolve 'External' formula inputs to internal keys via LLM."""
        updated_taxonomy = taxonomy.copy()
        
        unresolved_vars = set()
        for f in self.formulas_registry.values():
            for inp in f.get('input_variables', []):
                if inp not in taxonomy:
                    unresolved_vars.add(inp)
            out = f.get('output_variable')
            if out and out not in taxonomy:
                unresolved_vars.add(out)
        
        if not unresolved_vars:
            return updated_taxonomy

        dataset_desc = "\n".join([f"{k}: {v.get('standard_name', '')}" for k, v in taxonomy.items()])
        
        prompt = f"""
        Role: Data Mapping Expert.
        
        Task: Map 'External' variables found in formulas to existing variables in the dataset.
        
        External Variables (Unresolved):
        {list(unresolved_vars)}
        
        Dataset Dictionary (Key: Standard Name):
        {dataset_desc}
        
        Instructions:
        For each External Variable, determine if it corresponds to an existing Dataset Key (likely via synonym or abbreviation).
        
        Return JSON mapping:
        {{
            "External_Name_1": "Dataset_Key_X", 
            "External_Name_2": null  // if no match found
        }}
        """
        
        try:
            response = self.llm.invoke(prompt)
            cleaned_response = self._clean_json_response(response.content)
            mapping = json.loads(cleaned_response)
            
            for f_id, f_data in self.formulas_registry.items():
                new_inputs = []
                for inp in f_data.get('input_variables', []):
                    mapped_key = mapping.get(inp)
                    if mapped_key and mapped_key in taxonomy:
                        new_inputs.append(mapped_key)
                        current_links = updated_taxonomy[mapped_key].get('related_formula_ids', [])
                        if f_id not in current_links:
                            updated_taxonomy[mapped_key]['related_formula_ids'] = current_links + [f_id]
                    else:
                        new_inputs.append(inp)
                
                self.formulas_registry[f_id]['input_variables'] = new_inputs
                
                out = f_data.get('output_variable')
                if out:
                    mapped_out = mapping.get(out)
                    if mapped_out and mapped_out in taxonomy:
                        self.formulas_registry[f_id]['output_variable'] = mapped_out
                        current_links = updated_taxonomy[mapped_out].get('related_formula_ids', [])
                        if f_id not in current_links:
                            updated_taxonomy[mapped_out]['related_formula_ids'] = current_links + [f_id]
                            
        except Exception:
            pass
            
        return updated_taxonomy

    def _unify_synonyms(self, new_candidates, existing_keys):
        """
        Uses LLM to identify and merge synonyms within the new candidates 
        and against existing taxonomy keys.
        Returns a mapping { 'alias_id': 'canonical_id' }.
        """
        if not new_candidates:
            return {}
            
        new_keys_str = ", ".join(list(new_candidates.keys()))
        exist_sample = ", ".join(list(existing_keys)[:100]) 
        
        prompt = f"""
        Role: Clinical Data Standardizer.
        Task: Identify synonyms in a list of variable IDs and map them to a single canonical ID.
        
        New Variables: {new_keys_str}
        Existing Variables (Context): {exist_sample}
        
        Instructions:
        1. Look for synonyms among "New Variables" (e.g. 'bmi', 'body_mass_index').
        2. Look for synonyms between "New Variables" and "Existing Variables".
        3. If a synonym exists, choose the MOST STANDARD medical acronym or name as the 'canonical_id'.
        4. If the canonical ID is already in "Existing Variables", map to that.
        5. If duplicates found (e.g. 'bsa_dubois' and 'bsa_mosteller' are NOT synonyms, keep both), do NOT map distinct variants.
        
        Return JSON mapping {{ "alias_id": "canonical_id" }}
        Only include entries that need re-mapping.
        """
        try:
            resp = self.llm.invoke(prompt)
            clean = self._clean_json_response(resp.content)
            return json.loads(clean)
        except Exception:
            return {}
