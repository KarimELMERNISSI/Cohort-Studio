import os
import shutil
import streamlit as st
from pathlib import Path
import ast
import json

# Try importing RAG dependencies
try:
    import google.generativeai as genai
    from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    RAG_AVAILABLE = True
except ImportError as e:
    RAG_AVAILABLE = False
    MISSING_LIBS_ERROR = str(e)
    # Define a dummy genai to prevent NameError if referenced later in __init__ before checking availability
    genai = None

class RAGManager:
    def __init__(self, documents_dir="DOCUMENTS", api_key=None):
        self.documents_dir = documents_dir
        self.api_key = api_key
        self.vector_store = None
        self.qa_chain = None
        self.initialized = False
        self.global_context = "General Medical Domain"
        
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            if genai:
                genai.configure(api_key=api_key)

    def is_available(self):
        return RAG_AVAILABLE

    def get_available_models(self):
        if not self.api_key or not genai:
            return []
        try:
            genai.configure(api_key=self.api_key)
            models = []
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    models.append(m.name)
            return models
        except Exception as e:
            return []

    def _analyze_global_context(self, texts, dataset_columns=None):
        """Analyzes a sample of the documents and dataset columns to determine the global context."""
        if not texts or not hasattr(self, 'llm'):
            return {
                "domain": "General Medical", 
                "context_description": "General medical context.", 
                "roles": {
                    "strict": "Document Analyst",
                    "balanced": "Medical Researcher",
                    "creative": "Medical Consultant"
                }
            }
        
        # Take a sample of text (e.g., first 3 chunks)
        sample_text = "\n\n".join([t.page_content for t in texts[:3]])
        
        columns_context = ""
        if dataset_columns:
            # Limit to first 50 columns to avoid token limits if many
            columns_str = ", ".join(dataset_columns[:50]) 
            columns_context = f"\nDataset Variables Sample: [{columns_str}]"

        prompt = f"""
        Analyze the following text samples from a set of documents{ " and the provided dataset variable names" if dataset_columns else ""}. 
        
        Your task is to identify the context and define expert personas that would be best suited to answer questions about this data, depending on how strictly they must adhere to the documents.

        Return the result strictly as a JSON object with the following structure:
        {{
            "domain": "The specific medical or scientific domain (e.g. Cardiology, Oncology)",
            "context_description": "A concise description (max 2 sentences) of the study type and document nature.",
            "roles": {{
                "strict": "A role title for high adherence (e.g. Clinical Data Auditor)",
                "balanced": "A role title for balanced adherence (e.g. Principal Investigator)",
                "creative": "A role title for low adherence/high creativity (e.g. Senior Medical Consultant)"
            }}
        }}
        
        Text Samples:
        {sample_text[:4000]}
        {columns_context}
        """
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            # Clean up markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            import json
            return json.loads(content)
        except Exception:
            return {
                "domain": "General Medical", 
                "context_description": "General medical context.", 
                "roles": {
                    "strict": "Document Analyst",
                    "balanced": "Medical Researcher",
                    "creative": "Medical Consultant"
                }
            }

    def initialize_system(self, model_name="models/gemini-1.5-flash", adherence_score=0.5, dataset_columns=None):
        if not self.is_available():
            return False, f"Missing dependencies: {MISSING_LIBS_ERROR}. Please install `chromadb`, `pypdf`, `langchain-community`, `langchain-google-genai`."
        
        if not self.api_key:
            return False, "Google API Key is required."

        try:
            # 1. Load Documents
            if not os.path.exists(self.documents_dir):
                os.makedirs(self.documents_dir)
                return False, f"Documents directory '{self.documents_dir}' created. Please add PDF files."

            loader = DirectoryLoader(self.documents_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
            documents = loader.load()
            
            if not documents:
                return False, "No PDF documents found in the DOCUMENTS folder."

            # Fix 0-based page indexing from PyPDFLoader
            for doc in documents:
                if 'page' in doc.metadata:
                    doc.metadata['page'] += 1

            # 2. Split Text
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(documents)

            # 3. Create Embeddings & Vector Store
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            
            # Persist directory for Chroma
            persist_directory = "data/chroma_db"
            
            # Clear existing vector store to ensure fresh indexing with corrected metadata
            if os.path.exists(persist_directory):
                # Force garbage collection to release locks on Windows
                self.vector_store = None
                import gc
                import time
                import stat
                gc.collect()
                time.sleep(0.5)
                
                def remove_readonly(func, path, excinfo):
                    os.chmod(path, stat.S_IWRITE)
                    func(path)
                    
                try:
                    shutil.rmtree(persist_directory, onerror=remove_readonly)
                except Exception as e:
                    print(f"Warning: Could not delete {persist_directory}: {e}")
                    # If deletion fails, we proceed. Chroma will append to the existing store.
                    # This is better than crashing, though it may lead to duplicate embeddings 
                    # until the application is fully restarted.
            
            self.vector_store = Chroma.from_documents(
                documents=texts, 
                embedding=embeddings,
                persist_directory=persist_directory
            )

            # 4. Setup LLM & Chain
            # Use the selected model. Strip 'models/' prefix if present as langchain might handle it differently
            clean_model_name = model_name.replace("models/", "") if model_name.startswith("models/") else model_name
            self.llm = ChatGoogleGenerativeAI(model=clean_model_name, temperature=0.3)
            
            # 5. Analyze Global Context (One-time)
            self.global_context = self._analyze_global_context(texts, dataset_columns)

            # Construct prompt based on adherence score
            if adherence_score >= 0.7:
                self.current_role = self.global_context['roles'].get('strict', 'Document Analyst') # Save to self
                role = self.current_role
                guidance = (
                    f"**Constraint:** HIGH ADHERENCE (Score: {adherence_score}).\n"
                    "- You must strictly stick to the provided documents.\n"
                    "- Do NOT use outside knowledge.\n"
                    "- If the answer is not in the context, state it clearly."
                )
            elif adherence_score <= 0.3:
                self.current_role = self.global_context['roles'].get('creative', 'Medical Consultant') # Save to self
                role = self.current_role
                guidance = (
                    f"**Constraint:** LOW ADHERENCE (Score: {adherence_score}).\n"
                    "- Use the documents as context, but prioritize your general medical knowledge.\n"
                    "- Feel free to suggest standard variables or formulas even if not explicitly mentioned."
                )
            else:
                self.current_role = self.global_context['roles'].get('balanced', 'Medical Researcher') # Save to self
                role = self.current_role
                guidance = (
                    f"**Constraint:** BALANCED ADHERENCE (Score: {adherence_score}).\n"
                    "- Use the documents as the primary source of truth.\n"
                    "- Use general knowledge to explain concepts or fill minor gaps, but do not contradict the documents."
                )

            prompt_template = f"""
            **Global Context:**
            Domain: {self.global_context.get('domain', 'General')}
            Description: {self.global_context.get('context_description', '')}

            **Role:**
            Act as a {role}.

            **Adherence Instructions:**
            {guidance}

            **Context:**
            {{context}}

            **Question:**
            {{question}}

            **Answer:**
            """
            
            PROMPT = PromptTemplate(
                template=prompt_template, input_variables=["context", "question"]
            )

            # Define document prompt to include metadata
            document_prompt = PromptTemplate(
                input_variables=["page_content", "source", "page"],
                template="Document: {source} | Page: {page}\nContent: {page_content}\n----------------"
            )

            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 8, "fetch_k": 20}
                ),
                chain_type_kwargs={
                    "prompt": PROMPT,
                    "document_prompt": document_prompt
                }
            )
            
            self.initialized = True
            return True, f"RAG System Initialized Successfully. Processed {len(documents)} pages and {len(texts)} chunks."

        except Exception as e:
            return False, f"Error initializing RAG: {str(e)}"

    def update_adherence_score(self, adherence_score):
        """Updates the adherence score and rebuilds the chain without full re-initialization."""
        if not self.initialized or not self.vector_store or not hasattr(self, 'llm'):
            return False, "RAG system not fully initialized."
        
        try:
            # Reconstruct prompt based on adherence score
            if adherence_score >= 0.7:
                role = self.global_context['roles'].get('strict', 'Document Analyst')
                guidance = (
                    f"**Constraint:** HIGH ADHERENCE (Score: {adherence_score}).\n"
                    "- You must strictly stick to the provided documents.\n"
                    "- Do NOT use outside knowledge.\n"
                    "- If the answer is not in the context, state it clearly."
                )
            elif adherence_score <= 0.3:
                role = self.global_context['roles'].get('creative', 'Medical Consultant')
                guidance = (
                    f"**Constraint:** LOW ADHERENCE (Score: {adherence_score}).\n"
                    "- Use the documents as context, but prioritize your general medical knowledge.\n"
                    "- Feel free to suggest standard variables or formulas even if not explicitly mentioned."
                )
            else:
                role = self.global_context['roles'].get('balanced', 'Medical Researcher')
                guidance = (
                    f"**Constraint:** BALANCED ADHERENCE (Score: {adherence_score}).\n"
                    "- Use the documents as the primary source of truth.\n"
                    "- Use general knowledge to explain concepts or fill minor gaps, but do not contradict the documents."
                )

            prompt_template = f"""
            **Global Context:**
            Domain: {self.global_context.get('domain', 'General')}
            Description: {self.global_context.get('context_description', '')}

            **Role:**
            Act as a {role}.

            **Adherence Instructions:**
            {guidance}

            **Context:**
            {{context}}

            **Question:**
            {{question}}

            **Answer:**
            """
            
            PROMPT = PromptTemplate(
                template=prompt_template, input_variables=["context", "question"]
            )

            # Define document prompt to include metadata
            document_prompt = PromptTemplate(
                input_variables=["page_content", "source", "page"],
                template="Document: {source} | Page: {page}\nContent: {page_content}\n----------------"
            )

            # Recreate the chain with the new prompt
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 8, "fetch_k": 20}
                ),
                chain_type_kwargs={
                    "prompt": PROMPT,
                    "document_prompt": document_prompt
                }
            )
            return True, "Adherence score updated."
        except Exception as e:
            return False, f"Error updating adherence score: {str(e)}"

    def suggest_column_renaming(self, columns, strategy="literature"):
        if not self.initialized:
            return None, "RAG system not initialized."

        # Handle both list of strings (legacy) and list of dicts (rich context)
        is_rich_context = False
        if isinstance(columns, (list, tuple)) and len(columns) > 0:
            if isinstance(columns[0], dict):
                is_rich_context = True

        if is_rich_context:
            import json
            columns_str = json.dumps(columns, indent=2)
            # Extract just names for similarity search to avoid noise
            column_names = [c.get("name", "") for c in columns]
            search_query = ", ".join(column_names)
        else:
            # Ensure all elements are strings to avoid TypeError
            columns_list = [str(c) for c in columns]
            columns_str = ", ".join(columns_list)
            search_query = columns_str
        
        if strategy == "standardization":
            # For standardization, we use the LLM directly to access external knowledge (UMLS/SNOMED)
            # We still retrieve context to help the model understand the variables
            try:
                docs = self.vector_store.similarity_search(f"Dataset variables: {search_query}", k=5)
                context_text = "\n\n".join([d.page_content for d in docs])
                
                prompt = f"""
                You are a Medical Data Standardization Expert.
                
                Global Context: {self.global_context if isinstance(self.global_context, str) else self.global_context.get('context_description', '')}
                
                Task: Map the following dataset column names to standard medical terminology (UMLS, SNOMED CT, LOINC).
                
                Input Columns (with stats/samples): 
                {columns_str}
                
                Context from Documents (to help identify variable meanings):
                {context_text}
                
                Instructions:
                1. Analyze the Input Columns and Context to understand what each column represents. Use the statistics and top values to infer the meaning (e.g., units, categorical values).
                2. Map each identified variable to its standard Preferred Term in SNOMED CT, LOINC, or UMLS.
                3. Return a JSON object where keys are the original column names and values are the standardized names.
                4. Only include columns where a clear standard mapping is found.
                
                JSON Output:
                """
                
                response = self.llm.invoke(prompt)
                content = response.content.strip()
                
                # Clean up markdown code blocks if present
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]
                
                import json
                return json.loads(content), None
                
            except Exception as e:
                return None, str(e)
        
        else:
            # Default literature-based suggestion using the QA chain (respects adherence score)
            query = f"""
            Given the following list of dataset columns with their statistics/samples: 
            {columns_str}
            
            Based on the provided medical literature, suggest more standard or scientifically accurate names for these columns if applicable.
            Use the provided statistics and values to better understand the content of the columns (e.g. distinguishing between values and units).
            Return the result as a JSON object where keys are original names and values are suggested names. 
            Only include columns that need renaming.
            """
            
            try:
                response = self.qa_chain.invoke(query)
                return response['result'], None
            except Exception as e:
                return None, str(e)


    def suggest_computed_variables(self, columns, search_hint=None, num_suggestions=5, suggestion_mode="Comprehensive", allow_missing_variables=False):
        if not self.initialized:
            return None, "RAG system not initialized."

        columns_str = ", ".join(columns)
        
        # 1. Retrieve Context Manually
        # We do this manually because passing the huge prompt as a query to RetrievalQA fails to find relevant docs.
        if search_hint:
            retrieval_query = f"{search_hint} formula calculation clinical score"
        else:
            # Limit query length for retrieval
            retrieval_query = f"clinical scores formulas using {columns_str[:500]}"
            
        try:
            docs = self.vector_store.similarity_search(retrieval_query, k=5)
            
            # Format context with metadata for the LLM to cite
            context_text = ""
            for doc in docs:
                source = doc.metadata.get('source', 'Unknown')
                page = doc.metadata.get('page', 'N/A')
                content = doc.page_content.replace("\n", " ")
                context_text += f"SOURCE: {source} | PAGE: {page}\nCONTENT: {content}\n\n"
                
        except Exception as e:
            # Fallback if retrieval fails
            context_text = "No documents retrieved due to error: " + str(e)

        # 2. Retrieve the AUTO-SELECTED Persona
        role = getattr(self, 'current_role', self.global_context.get('roles', {}).get('balanced', 'Expert Medical Data Scientist'))
        domain = self.global_context.get('domain', 'General Medical')
        context_desc = self.global_context.get('context_description', 'Medical research dataset')

        # 3. Define Mode-Specific Instructions
        if suggestion_mode == "Go To Target":
            focus_instruction = f"""
            FOCUS: The user wants to calculate "{search_hint}" using the Available Variables.
            
            EXECUTION LOGIC (2-Step Process):
            1. IDENTIFY: Find standard medical formulas that involve "{search_hint}" and the Available Variables.
               - The standard formula might not have "{search_hint}" as the subject (e.g. BMI = Weight/Height^2).
            2. DEDUCE: Algebraically rearrange the formula to isolate "{search_hint}" as the output.
               - Target = f(Available Variables).
               - Example: If target is "Weight" and you have "BMI" and "Height", deduce "Weight = BMI * Height^2".

            CONSTRAINTS:
            - The suggested 'formula' MUST result in "{search_hint}".
            - STRICT CONSTRAINT: Do NOT suggest related concepts (e.g. if hint is "BSA", do NOT suggest "BMI"). Only suggest variants of "{search_hint}".
            - EXPLANATION REQUIREMENT: In the 'description' field, you MUST explain this process: "Derived from [Original Formula] by solving for [Target]."
            """
        elif suggestion_mode == "Go From Target":
            focus_instruction = f"""
            FOCUS: The user wants to use "{search_hint}" to calculate OTHER variables.
            - Suggest formulas where "{search_hint}" is an INPUT variable.
            - Example: If hint is "Weight", suggest "BMI" (Weight/Height^2).
            - Example: If hint is "Creatinine", suggest "eGFR".
            """
        elif suggestion_mode == "Around Target":
            focus_instruction = f"""
            FOCUS: The user wants to find interesting metrics close to "{search_hint}" and clinically relevant from our data.
            - Suggest variables that are conceptually related or often analyzed together with "{search_hint}".
            - Example: If hint is "Blood Pressure", suggest "Pulse Pressure", "MAP".
            - CRITICAL: The suggestion MUST involve a transformation formula. Do NOT suggest simple aliases or renames.
            """
        else: # "Comprehensive" (Default for exploratory mode)
            focus_instruction = f"""
            FOCUS: Suggest standard medical indices and scores that can be calculated from the Available Variables.
            - Prioritize variables that add high predictive value.
            - Ensure suggestions are linked to the dataset variables by a formula.
            """

        # 4. Strict Formatting Rules
        if allow_missing_variables:
            validity_instruction = """2. VALIDITY: Prefer available variables. 
            - If a standard formula requires a missing variable, YOU MUST WRITE THE FORMULA using the standard variable name (e.g. 'Height').
            - Do NOT write "Cannot calculate" or text in the formula field.
            - List the missing variable names in the 'missing_variables' list."""
        else:
            validity_instruction = """2. VALIDITY: STRICTLY FORBIDDEN to use missing variables.
            - You MUST ONLY use variables explicitly listed in the 'Available Variables' list.
            - If a formula requires a variable that is not in the list, DO NOT SUGGEST IT.
            - If no formulas can be calculated with the available variables, return an empty list.
            - CHECK your variables against the list. If 'Weight_kg' is available, do NOT use 'Weight'.
            """

        formatting_instructions = f"""
        CRITICAL OUTPUT RULES:
        1. FORMULA SYNTAX: You MUST put a single space before and after every operator, number, and variable name.
           - WRONG: "Weight/(Height/100)**2"
           - CORRECT: " Weight / ( Height / 100 ) ** 2 "
        {validity_instruction}
        3. CITATIONS: 
           - If using a document: "Filename.pdf (Pages X, Y)".
           - If using Model Knowledge: "Model Knowledge".
           - If Hybrid: "Filename.pdf (Pages X, Y) + Model Knowledge".
           - Do NOT cite external websites (like Wikipedia) unless they are the actual source file name.
        4. EXPLANATION:
           - Clearly state what information came from the document.
           - Clearly state what was inferred by Model Knowledge (e.g. "Formula structure inferred from standard medical usage").
        5. EXCLUSION:
           - Do NOT suggest variables that are just renaming of existing variables. 
           - The formula MUST involve some mathematical operation or transformation. 
           - If the formula is just 'Variable_A', do not suggest it.
        6. OUTPUT FORMAT: Return ONLY a raw JSON object. No markdown.
        """

        if search_hint:
            # Scenario A: Targeted Search with User-Selected Mode
            query = f"""
            Role: Act as a {role} in the domain of {domain}.
            Context Description: {context_desc}.
            
            Task: The user is looking for "{search_hint}".
            Mode: {suggestion_mode}.
            
            {focus_instruction}
            
            Available Variables: [{columns_str}].
            
            RETRIEVED DOCUMENTS (Use these for citations):
            {context_text}

            Instructions:
            1. Generate {num_suggestions} suggestions matching the Mode.
            2. If exact variables are missing, look for valid proxies (e.g., 'Weight_kg' -> 'Weight_lbs * 0.45').
            3. Ensure every formula is syntactically valid (spaced).
            4. Use the RETRIEVED DOCUMENTS to fill the 'source_citation' field.

            {formatting_instructions}

            Return a JSON object with this structure:
            {{
                "domain_analysis": {{
                    "dataset_domain": "{domain}",
                    "document_domain": "Derived from document context",
                    "relevant_domains": ["Top 1", "Top 2", "Top 3"]
                }},
                "suggestions": [
                    {{
                        "name": "variable_name_snake_case",
                        "title": "Readable Title",
                        "suggestion_category": "One of ['Direct Variant', 'Component', 'Related/Derived']",
                        "markdown_formula": "Standard LaTeX format (e.g. \\frac{{a}}{{b}}) without $$ wrappers",
                        "formula": " STRICTLY SPACED FORMULA ",
                        "missing_variables": ["List", "of", "missing", "variables"],
                        "description": "Explanation of clinical relevance.",
                        "source_type": "Document" or "Model Knowledge" or "Hybrid",
                        "source_citation": "Filename.pdf (Pages X, Y) + Model Knowledge",
                        "source_explanation": "Document defines X. Model Knowledge provided Y."
                    }}
                ]
            }}
            """
        else:
            # Scenario B: Open Exploration (Hint is None)
            query = f"""
            Role: Act as a {role} in the domain of {domain}.
            Context Description: {context_desc}.

            Task: Suggest {num_suggestions} highly clinically relevant computed variables for this dataset.
            Available Variables: [{columns_str}].
            
            RETRIEVED DOCUMENTS (Use these for citations):
            {context_text}

            Instructions:
            1. Analyze the variable list to understand the medical sub-specialty.
            2. Suggest standard medical indices (e.g., BMI, MAP, Pulse Pressure, NLR).
            3. Prioritize variables that add high predictive value.
            4. Use the RETRIEVED DOCUMENTS to fill the 'source_citation' field.

            {formatting_instructions}

            Return a JSON object with this structure:
            {{
                "domain_analysis": {{
                    "dataset_domain": "{domain}",
                    "document_domain": "Derived from document context",
                    "relevant_domains": ["Top 1", "Top 2", "Top 3"]
                }},
                "suggestions": [
                    {{
                        "name": "variable_name_snake_case",
                        "title": "Readable Title",
                        "suggestion_category": "Clinical Score / Ratio / Index",
                        "markdown_formula": "Standard LaTeX format (e.g. \\frac{{a}}{{b}}) without $$ wrappers",
                        "formula": " STRICTLY SPACED FORMULA ",
                        "missing_variables": ["List", "of", "missing", "variables"],
                        "description": "Clinical utility.",
                        "source_type": "Document" or "Model Knowledge",
                        "source_citation": "Filename.pdf (Pages X, Y) + Model Knowledge",
                        "source_explanation": "Document defines X. Model Knowledge provided Y."
                    }}
                ]
            }}
            """
        
        try:
            # Use LLM directly instead of QA chain to ensure prompt integrity
            response = self.llm.invoke(query)
            result_text = response.content.strip()
            
            # JSON Cleanup
            if result_text.startswith("```"):
                lines = result_text.splitlines()
                if lines[0].strip().startswith("```"): lines = lines[1:]
                if lines[-1].strip().startswith("```"): lines = lines[:-1]
                result_text = "\n".join(lines)
            
            return result_text, None

        except Exception as e:
            return None, str(e)

    def suggest_computed_variables_with_validation(self, columns, search_hint=None, num_suggestions=5, suggestion_mode="Comprehensive", allow_missing_variables=False):
        """
        Generates suggestions and runs a self-correction loop to ensure 
        all variables in the formula actually exist in the dataset.
        """

        # 1. Get initial suggestions (Pass ALL parameters including suggestion_mode)
        response_text, error = self.suggest_computed_variables(
            columns, 
            search_hint, 
            num_suggestions, 
            suggestion_mode=suggestion_mode,
            allow_missing_variables=allow_missing_variables
        )
        if error: return None, error

        try:
            # Handle potential markdown wrappers if the main function didn't catch them
            clean_text = response_text.strip()
            if clean_text.startswith("```"):
                lines = clean_text.splitlines()
                if lines[0].startswith("```"): lines = lines[1:]
                if lines[-1].startswith("```"): lines = lines[:-1]
                clean_text = "\n".join(lines)
            
            data = json.loads(clean_text)
            suggestions = data.get("suggestions", [])
        except Exception as e:
            return None, f"JSON Parse Error in Validation: {str(e)}"

        # 2. Validation Loop
        validated_suggestions = []
        
        # known_globals are things like np, pd, log that are valid but not columns
        known_globals = {'np', 'pd', 'log', 'exp', 'sqrt', 'abs', 'round', 'min', 'max', 'constant'}
        
        for suggestion in suggestions:
            formula = suggestion.get("formula", "")
            
            # --- FILTER IDENTITY TRANSFORMATIONS ---
            # If the formula is just a single variable (no transformation), skip it.
            # This is especially important for "Around Target" mode to avoid "Weight -> Weight" suggestions.
            
            # 1. Check if formula is exactly a column name (even with spaces)
            if formula.strip() in columns:
                continue

            # 2. Check for absence of mathematical operators (heuristic)
            # If no operators are present, it's likely just a variable reference or a constant
            has_operator = any(op in formula for op in ['+', '-', '*', '/', '**', '>', '<', '=', 'np.', 'pd.'])
            if not has_operator:
                # It might be a single variable like "Weight_kg" or "Total mass"
                # We skip it unless it's a function call (which usually has parens, covered by operators check if we include parens)
                # Let's include parens in operators check
                if not any(op in formula for op in ['(', ')']):
                     continue

            try:
                tree = ast.parse(formula, mode='eval')
                if isinstance(tree.body, ast.Name):
                    # It's just a variable name (e.g. "Weight")
                    # We skip this suggestion as it adds no value (just an alias)
                    continue
            except SyntaxError:
                pass # Let the robust parsing below handle syntax errors

            # --- ROBUST PARSING (AST) ---
            # Extract variable names safely using Python's Abstract Syntax Tree
            found_vars = set()
            try:
                tree = ast.parse(formula, mode='eval')
                for node in ast.walk(tree):
                    if isinstance(node, ast.Name):
                        found_vars.add(node.id)
            except SyntaxError:
                # If formula is syntactically invalid, mark it for fixing
                found_vars = set() # Force a fix

            # Identify Missing Columns (Hallucinations)
            # A variable is missing if it's not in columns AND not a known python/numpy global
            missing_cols = [var for var in found_vars if var not in columns and var not in known_globals]
            
            if missing_cols:
                # 3. SELF-CORRECTION: Ask LLM to fix it
                print(f"🔧 Fixing hallucination in: {formula} | Missing: {missing_cols}")
                
                fix_query = f"""
                You provided a formula that uses variables NOT present in the dataset.
                
                Invalid Formula: "{formula}"
                Missing Variables: {missing_cols}
                
                AVAILABLE VARIABLES: {columns}
                
                TASK: Rewrite the formula using ONLY the Available Variables. 
                If a variable is missing, find a valid proxy from the list or mathematically approximate it.
                
                CRITICAL FORMATTING RULE: 
                You MUST put a single space before and after every operator, number, and variable name.
                Example: " Weight / ( Height / 100 ) ** 2 "
                
                Return ONLY the corrected formula string. No text, no quotes.
                """
                
                try:
                    corrected_formula = self.llm.invoke(fix_query).content.strip()
                    # Clean up quotes if the LLM added them
                    corrected_formula = corrected_formula.strip('"').strip("'")
                    
                    suggestion['formula'] = corrected_formula
                    suggestion['correction_applied'] = True
                    suggestion['original_hallucination'] = str(missing_cols)
                    
                    # Re-validate the corrected formula
                    found_vars_corrected = set()
                    try:
                        tree_corrected = ast.parse(corrected_formula, mode='eval')
                        for node in ast.walk(tree_corrected):
                            if isinstance(node, ast.Name):
                                found_vars_corrected.add(node.id)
                        missing_cols_corrected = [var for var in found_vars_corrected if var not in columns and var not in known_globals]
                        
                        if missing_cols_corrected:
                            # Still missing variables after correction
                            if not allow_missing_variables:
                                print(f"❌ Dropping suggestion due to persistent missing variables: {missing_cols_corrected}")
                                continue # SKIP this suggestion
                            else:
                                suggestion['missing_variables'] = missing_cols_corrected
                    except:
                        if not allow_missing_variables:
                            continue
                        pass

                except Exception as e:
                    # If fixing fails, keep original but flag it
                    suggestion['error'] = "Could not auto-correct missing variables"
                    if not allow_missing_variables:
                        continue # Skip if we can't fix it and missing vars aren't allowed
            
            validated_suggestions.append(suggestion)

        data['suggestions'] = validated_suggestions
        return json.dumps(data), None

    def suggest_proxy_variable(self, target_variable, available_columns):
        """
        Suggests a way to approximate a missing variable using available columns.
        """
        columns_str = ", ".join(available_columns)
        
        query = f"""
        Role: Expert Medical Data Scientist.
        Task: The user needs the variable "{target_variable}" but it is missing.
        Available Variables: [{columns_str}].
        
        Instructions:
        1. Suggest a scientifically valid way to approximate or proxy "{target_variable}" using the available variables.
        2. If a direct calculation is possible (e.g. Weight_lbs * 0.45 for Weight_kg), provide the formula.
        3. If a clinical proxy is possible (e.g. using 'Total Mass' for 'Weight'), explain the validity and limitations.
        4. If no valid proxy exists, state that clearly.
        
        Return a JSON object:
        {{
            "proxy_found": true/false,
            "proxy_name": "Name of proxy variable",
            "formula": "Calculation formula if applicable",
            "explanation": "Medical justification for this proxy."
        }}
        """
        
        try:
            response = self.llm.invoke(query)
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            import json
            return json.loads(content)
        except Exception as e:
            return {"proxy_found": False, "explanation": str(e)}

    def suggest_alternative_formula(self, target_concept, missing_variable, available_columns):
        """
        Suggests an alternative formula for a concept that avoids a specific missing variable.
        """
        columns_str = ", ".join(available_columns)
        
        query = f"""
        Role: Expert Medical Data Scientist.
        Task: The user wants to calculate "{target_concept}" but is missing the variable "{missing_variable}".
        Available Variables: [{columns_str}].
        
        Instructions:
        1. Suggest a DIFFERENT formula or method for "{target_concept}" that does NOT use "{missing_variable}".
        2. Example: If "BSA (DuBois)" requires Height (missing), suggest "BSA (Boyd)" if it uses different inputs, or a different index entirely.
        3. If no alternative exists without that variable, state it.
        
        Return a JSON object:
        {{
            "alternative_found": true/false,
            "alternative_name": "Name of alternative method",
            "formula": "Calculation formula",
            "explanation": "Why this alternative is valid and how it differs."
        }}
        """
        
        try:
            response = self.llm.invoke(query)
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            import json
            return json.loads(content)
        except Exception as e:
            return {"alternative_found": False, "explanation": str(e)}