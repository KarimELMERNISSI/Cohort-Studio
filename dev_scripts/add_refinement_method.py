    def refine_variable_taxonomy(self, current_taxonomy, feedback_dict, columns_info):
        """
        Refines specific variables in the taxonomy based on user feedback.
        
        Args:
            current_taxonomy (dict): The current taxonomy dictionary.
            feedback_dict (dict): A dictionary mapping variable names to user feedback strings.
                                  Example: {'var1': 'This is actually a date', 'var2': 'Unit is mg/dL'}
            columns_info (list): List of column info dicts.
            
        Returns:
            dict: The updated taxonomy.
        """
        if not self.initialized:
            return current_taxonomy, "RAG system not initialized."
            
        updated_taxonomy = current_taxonomy.copy()
        variables_to_refine = [v for v in feedback_dict.keys() if v in updated_taxonomy]
        
        if not variables_to_refine:
            return updated_taxonomy, "No valid variables to refine."
            
        # Prepare context for the LLM
        refinement_context = []
        for var in variables_to_refine:
            current_data = updated_taxonomy[var]
            feedback = feedback_dict[var]
            
            # Get column info for context
            col_info = next((c for c in columns_info if c['name'] == var), {})
            top_values = col_info.get('top_values', {})
            stats = col_info.get('stats', {})
            
            context_str = f"""
            Variable: {var}
            Current Taxonomy: {json.dumps(current_data)}
            Data Stats: {json.dumps(stats)}
            Top Values: {json.dumps(top_values)}
            USER FEEDBACK: "{feedback}"
            """
            refinement_context.append(context_str)
            
        context_block = "\n---\n".join(refinement_context)
        
        prompt = f"""
        Role: Medical Data Expert & Taxonomy Refiner.
        
        Task: Refine the taxonomy for the following variables based on specific USER FEEDBACK.
        
        Instructions:
        1. Review the "Current Taxonomy" and "USER FEEDBACK" for each variable.
        2. Update the taxonomy fields (standard_name, description, category, clinical_usage, related_formulas, topic, proxy_variables) to address the feedback.
        3. If the user corrects a unit, update the description and related_formulas (conversion) accordingly.
        4. If the user corrects the meaning, update the standard_name and description.
        5. Keep existing valid information if it doesn't conflict with the feedback.
        
        Variables to Refine:
        {context_block}
        
        Return JSON:
        {{
            "variable_name": {{
                "standard_name": "...",
                "description": "...",
                "category": "...",
                "clinical_usage": "...",
                "related_formulas": ["..."],
                "topic": "...",
                "proxy_variables": ["..."]
            }}
        }}
        """
        
        try:
            response = self.llm.invoke(prompt)
            cleaned_response = self._clean_json_response(response.content)
            try:
                refinements = json.loads(cleaned_response)
            except json.JSONDecodeError as e:
                return updated_taxonomy, f"JSON Error in refinement: {str(e)}"
            
            # Update the taxonomy
            for var, new_data in refinements.items():
                if var in updated_taxonomy:
                    # Merge new data into existing, overwriting keys
                    updated_taxonomy[var].update(new_data)
                    
            # Re-merge stats to ensure they are preserved/updated
            updated_taxonomy = self._merge_stats(updated_taxonomy, columns_info)
            self.variable_taxonomy = updated_taxonomy
            
            return updated_taxonomy, None
            
        except Exception as e:
            return updated_taxonomy, f"Refinement Error: {str(e)}"
