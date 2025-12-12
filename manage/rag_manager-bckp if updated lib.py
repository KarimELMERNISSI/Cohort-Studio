import os
import shutil
import streamlit as st
from pathlib import Path
import ast
import json
import hashlib
import time
import logging
import re

# Disable ChromaDB telemetry to prevent errors
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_IMPL"] = "false"

# Suppress specific ChromaDB telemetry errors
logging.getLogger('chromadb.telemetry.product.posthog').setLevel(logging.CRITICAL)

# Try importing RAG dependencies
try:
    import google.genai as genai # Explicit import for the SDK
    from google.genai import types # Useful for typing
    from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
    # NEW: Splitters live in their own package now
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
    from utils.custom_gemini import CustomGeminiChat, CustomGeminiEmbeddings

    # for our QA chains
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
    RAG_AVAILABLE = True
except ImportError as e:
    RAG_AVAILABLE = False
    MISSING_LIBS_ERROR = str(e)
    # Define a dummy genai to prevent NameError
    genai = None

class RAGManager:
    def __init__(self, documents_dir="DOCUMENTS", api_key=None):
        self.documents_dir = documents_dir
        self.api_key = api_key
        self.client = None  # Initialize client placeholder
        self.vector_store = None
        self.qa_chain = None
        self.initialized = False
        self.global_context = "General Medical Domain"
        
        # Cache for theoretical concepts to avoid reprocessing the "Knowledge" step
        self.concept_cache = {} 
        
        # Variable Taxonomy (Mapping of cryptic names to standard concepts)
        self.variable_taxonomy = {}
        
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            if genai:
                self.client = genai.Client(api_key=api_key)

    def is_available(self):
        return RAG_AVAILABLE

    def get_available_models(self):
        if not self.api_key or not genai:
            return []
        try:
            # Ensure client is ready
            if not self.client:
                self.client = genai.Client(api_key=self.api_key)
                
            models = []
            # NEW: client.models.list() returns an iterator
            for m in self.client.models.list():
                # The new SDK model object usually has a 'name' attribute like 'models/gemini-1.5-flash'
                # We filter for 'gemini' to ensure it's a chat model
                if "gemini" in m.name.lower(): 
                    models.append(m.name)
            return models
        except Exception as e:
            print(f"Error listing models: {e}")
            return []

    def _clean_json_response(self, text):
        """Helper to clean JSON output from LLM."""
        text = text.strip()
        if not text: return "{}"
        
        # Try to find JSON block within markdown
        if "```" in text:
            import re
            match = re.search(r"```(?:json)?(.*?)```", text, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # If no markdown, try to find the first { and last }
        if "{" in text and "}" in text:
            start = text.find("{")
            end = text.rfind("}") + 1
            return text[start:end]
            
        return text

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
            content = self._clean_json_response(response.content)
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

    def initialize_system(self, model_name="models/gemini-1.5-flash", adherence_score=0.5, dataset_columns=None, use_existing_db=False):
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
            embeddings = CustomGeminiEmbeddings(
                api_key=self.api_key, 
                model="models/text-embedding-004"
            ) # GoogleGenerativeAIEmbeddings(model="models/embedding-001"), tbc when langchain-google-genai is updated
            
            # Persist directory for Chroma
            persist_directory = "data/chroma_db"
            
            # Logic to use existing DB or rebuild
            if use_existing_db and os.path.exists(persist_directory):
                try:
                    self.vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
                except Exception as e:
                    return False, f"Failed to load existing database: {e}. Try disabling 'Use Existing Database'."
            else:
                # Clear existing vector store logic
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
                
                # Batch processing
                if not texts:
                    self.vector_store = Chroma(embedding_function=embeddings, persist_directory=persist_directory)
                else:
                    batch_size = 5
                    total_texts = len(texts)
                    
                    try:
                        self.vector_store = Chroma.from_documents(
                            documents=texts[:batch_size], 
                            embedding=embeddings, 
                            persist_directory=persist_directory
                        )
                    except Exception as e:
                        pass

                    start_index = batch_size if self.vector_store._collection.count() > 0 else 0
                    
                    for i in range(start_index, total_texts, batch_size):
                        batch = texts[i : i + batch_size]
                        retry_count = 0
                        max_retries = 3
                        
                        while retry_count < max_retries:
                            try:
                                self.vector_store.add_documents(batch)
                                time.sleep(1.5)
                                break
                            except Exception as e:
                                retry_count += 1
                                time.sleep(2 * retry_count)
                                if retry_count == max_retries:
                                    print(f"Failed to embed batch starting at {i} after {max_retries} retries: {e}")

            # 4. Setup LLM & Chain
            # Use the selected model. Strip 'models/' prefix if present as langchain might handle it differently
            clean_model_name = model_name.replace("models/", "") if model_name.startswith("models/") else model_name
            self.llm = CustomGeminiChat(api_key=self.api_key, model=clean_model_name, temperature=0.2) #ChatGoogleGenerativeAI(model=clean_model_name, temperature=0.3) #to be changed when langchain-google-genai is updated
            
            # 5. Analyze Global Context (One-time)
            self.global_context = self._analyze_global_context(texts, dataset_columns)

            # 6. Initialize Roles and QA Chain
            self._update_internal_state(adherence_score)
            
            self.initialized = True
            return True, f"RAG System Initialized Successfully. Processed {len(documents)} pages and {len(texts)} chunks."

        except Exception as e:
            return False, f"Error initializing RAG: {str(e)}"

    def update_adherence_score(self, adherence_score):
        """Updates the adherence score and rebuilds the chain without full re-initialization."""
        if not self.initialized or not self.vector_store or not hasattr(self, 'llm'):
            return False, "RAG system not fully initialized."
        
        try:
            self._update_internal_state(adherence_score)
            return True, "Adherence score updated."
        except Exception as e:
            return False, f"Error updating adherence score: {str(e)}"

    def _update_internal_state(self, adherence_score):
        """Updates internal state using the Modern LangChain 0.3 Architecture."""
        self.adherence_score = adherence_score
        roles = self.global_context.get('roles', {})
        
        # 1. Determine Role & Guidance (Logic preserved exactly)
        if adherence_score >= 0.7:
            self.current_role = roles.get('strict', 'Document Analyst')
            guidance = (
                f"**Constraint:** HIGH ADHERENCE (Score: {adherence_score}).\n"
                "- You must strictly stick to the provided documents.\n"
                "- Do NOT use outside knowledge.\n"
                "- If the answer is not in the context, state it clearly."
            )
        elif adherence_score <= 0.3:
            self.current_role = roles.get('creative', 'Medical Consultant')
            guidance = (
                f"**Constraint:** LOW ADHERENCE (Score: {adherence_score}).\n"
                "- Use the documents as context, but prioritize your general medical knowledge.\n"
                "- Feel free to suggest standard variables or formulas."
            )
        else:
            self.current_role = roles.get('balanced', 'Medical Researcher')
            guidance = (
                f"**Constraint:** BALANCED ADHERENCE (Score: {adherence_score}).\n"
                "- Use the documents as the primary source of truth.\n"
                "- Use general knowledge to fill gaps, but do not contradict documents."
            )
        
        self.adherence_guidance = guidance

        # 2. Modern Chain Construction
        
        # OPTIMIZATION: Use System Message for instructions.
        # This prevents the model from getting confused between instructions and user questions.
        system_prompt = f"""
        **Global Context:**
        Domain: {self.global_context.get('domain', 'General')}
        Description: {self.global_context.get('context_description', '')}

        **Role:**
        Act as a {self.current_role}.

        **Adherence Instructions:**
        {guidance}

        **Context:**
        {{context}}
        """

        # We use ChatPromptTemplate to separate System (Instructions) from Human (Input)
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),  # Standard key is now 'input', not 'question'
        ])

        # OPTIMIZATION: Precise Document Formatting
        # This preserves your exact "Document: {source} | Page: {page}" format
        document_prompt = PromptTemplate(
            input_variables=["page_content", "source", "page"],
            template="Document: {source} | Page: {page}\nContent: {page_content}\n----------------"
        )

        # 3. Create the Chains (LCEL style)
        question_answer_chain = create_stuff_documents_chain(
            self.llm,
            prompt,
            document_prompt=document_prompt,
            document_variable_name="context" # This automatically feeds into {context} above
        )

        self.qa_chain = create_retrieval_chain(
            self.vector_store.as_retriever(
                search_type="mmr", 
                search_kwargs={"k": 8, "fetch_k": 20} # Preserving high precision settings
            ),
            question_answer_chain
        )

    def suggest_column_renaming(self, columns, strategy="literature", progress_callback=None):
        if not self.initialized:
            return None, "RAG system not initialized."

        if progress_callback: progress_callback(10, "Analyzing columns...")

        # Handle columns input (list of strings or dicts)
        if isinstance(columns, (list, tuple)) and len(columns) > 0 and isinstance(columns[0], dict):
            import json
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

        prompt = f"""
        Role: {self.current_role}
        Constraint: {self.adherence_guidance}

        Task: {task}
        
        Input Data (Columns & Stats): 
        {columns_str}
        
        Context from Documents:
        {context_text[:3000]}
        
        Instructions:
        1. Analyze the Input Data and Context.
        2. Suggest a new name ONLY if the current name is ambiguous, non-standard, or can be improved.
        3. Return a JSON object where keys are the ORIGINAL names and values are the NEW names.
        4. Return ONLY valid JSON. No markdown formatting, no explanations outside the JSON.
        
        JSON Output:
        """

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

    def generate_variable_taxonomy(self, columns_info, existing_mapping=None, progress_callback=None):
        """
        Generates a taxonomy mapping for the provided columns.
        columns_info: List of dicts with 'name', 'type', 'stats', etc.
        existing_mapping: Dict of manual renames (original -> new) to use as ground truth.
        """
        if not self.initialized:
            return None, "RAG system not initialized."

        if progress_callback: progress_callback(10, "Analyzing variable structure...")
        
        # Prepare input for LLM
        # We assume columns_info is a list of dicts or strings. If strings, convert to simple dicts.
        if isinstance(columns_info[0], str):
             columns_input = [{"name": c} for c in columns_info[:100]]
        else:
             columns_input = columns_info[:100]

        columns_str = json.dumps(columns_input, indent=2) 
        
        # Retrieve Context (Inspired by suggest_column_renaming)
        if progress_callback: progress_callback(25, "Retrieving documentation context...")
        try:
            # Create a query based on the variable names to find relevant data dictionaries or protocols
            search_query = ", ".join([c.get("name", "") for c in columns_input[:20]]) # Use first 20 vars for search
            docs = self.vector_store.similarity_search(f"Dataset variables definition: {search_query}", k=5)
            context_text = "\n\n".join([d.page_content for d in docs])
        except Exception:
            context_text = "No specific documentation found."

        manual_renames_str = ""
        if existing_mapping:
             # Filter out identity mappings
             meaningful_renames = {k: v for k, v in existing_mapping.items() if k != v}
             if meaningful_renames:
                 manual_renames_str = f"""
                 USER PROVIDED RENAMINGS (Ground Truth):
                 {json.dumps(meaningful_renames, indent=2)}
                 Use these as the definitive standard names for these variables.
                 """

        prompt = f"""
        Role: Medical Data Standardizer.
        
        Task: Create a taxonomy mapping for the provided dataset variables.
        The goal is to map potentially cryptic or non-standard variable names to their Standard Medical Concept.
        
        Input Variables:
        {columns_str}
        
        Context from Documents (Data Dictionaries / Protocols):
        {context_text[:4000]}
        
        {manual_renames_str}
        
        Instructions:
        1. Analyze each variable name and its statistics/values to infer its meaning.
        2. USE THE CONTEXT from documents to find exact definitions if available.
        3. Map it to a Standard Medical Concept (e.g. "sbp_val" -> "Systolic Blood Pressure").
        4. Provide a brief description.
        5. Return a JSON object where keys are the ORIGINAL variable names.
        
        JSON Output Format:
        {{
            "original_var_name": {{
                "standard_name": "Standard Concept Name",
                "description": "Brief description of what this variable represents.",
                "category": "Demographics/Vitals/Labs/etc"
            }},
            ...
        }}
        """
        
        if progress_callback: progress_callback(50, "Generating taxonomy...")
        try:
            response = self.llm.invoke(prompt)
            if progress_callback: progress_callback(80, "Parsing taxonomy...")
            taxonomy = json.loads(self._clean_json_response(response.content))
            
            # Store in instance
            self.variable_taxonomy = taxonomy
            return taxonomy, None
        except Exception as e:
            return None, str(e)

    def suggest_computed_variables(self, columns, search_hint=None, num_suggestions=5, suggestion_mode="Comprehensive", allow_missing_variables=False, use_taxonomy=True, progress_callback=None):
        if not self.initialized:
            return None, "RAG system not initialized."
        
        # --- STEP 0: RETRIEVE CONTEXT ---
        if search_hint:
            retrieval_query = f"{search_hint} formula calculation clinical score"
        else:
            retrieval_query = f"clinical scores formulas using {', '.join(columns[:50])}"
            
        try:
            docs = self.vector_store.similarity_search(retrieval_query, k=5)
            context_text = "\n\n".join([f"SOURCE: {d.metadata.get('source')} | CONTENT: {d.page_content}" for d in docs])
        except:
            context_text = ""

        # --- STEP 1: THEORETICAL GENERATION (Cached) ---
        if progress_callback: progress_callback(30, "Generating theoretical concepts...")
        
        # Include columns in cache key for scoping
        columns_hash = hashlib.md5(json.dumps(sorted(columns)).encode()).hexdigest()
        # Include use_taxonomy in cache key
        cache_key = hashlib.md5(f"{search_hint}_{suggestion_mode}_{self.adherence_score}_{columns_hash}_{use_taxonomy}".encode()).hexdigest()
        
        if cache_key in self.concept_cache:
            theoretical_concepts = self.concept_cache[cache_key]
        else:
            theoretical_concepts = self._get_theoretical_formulas(search_hint, suggestion_mode, context_text, num_suggestions * 2, columns, use_taxonomy)
            self.concept_cache[cache_key] = theoretical_concepts

        # --- STEP 2: DATA MAPPING (Real-time) ---
        if progress_callback: progress_callback(50, "Mapping formulas to dataset...")
        final_suggestions_json = self._map_formulas_to_data(theoretical_concepts, columns, allow_missing_variables, num_suggestions, use_taxonomy)
        
        return final_suggestions_json, None

    def _get_theoretical_formulas(self, search_hint, suggestion_mode, context_text, count, available_columns=None, use_taxonomy=True):
        """Step 1: Identify WHAT to calculate (Standard Medical Knowledge)."""
        
        if suggestion_mode == "Go To Target" and search_hint:
            task_desc = f"Identify standard medical formulas that result in '{search_hint}'. If necessary, rearrange formulas to solve for it."
        elif suggestion_mode == "Go From Target" and search_hint:
            task_desc = f"Identify standard medical scores or indices that use '{search_hint}' as an INPUT."
        elif suggestion_mode == "Around Target" and search_hint:
            task_desc = f"Identify clinical metrics conceptually related to '{search_hint}'."
        else:
            task_desc = "Identify standard medical indices and scores relevant to the retrieved context."

        # Add scoping context
        scoping_instruction = ""
        if available_columns:
            # Limit to first 100 columns to avoid token limit issues if dataset is huge
            cols_str = ", ".join(available_columns[:100])
            
            taxonomy_context = ""
            if self.variable_taxonomy and use_taxonomy:
                 # We assume the taxonomy is a dict. We'll dump a simplified version to save tokens if needed
                 # For now, just dump it.
                 taxonomy_str = json.dumps(self.variable_taxonomy, indent=2)
                 taxonomy_context = f"\nUse this Taxonomy to understand the available variables:\n{taxonomy_str}\n"

            scoping_instruction = f"""
            **Scoping Constraint:**
            The suggested formulas MUST involve at least one variable from the following list (or a close synonym):
            [{cols_str}]
            {taxonomy_context}
            Do NOT suggest formulas where ALL inputs are missing from this list.
            """

        prompt = f"""
        Role: {self.current_role}
        Constraint: {self.adherence_guidance}
        
        Context from Documents:
        {context_text[:3000]}
        
        {scoping_instruction}
        
        Task: {task_desc}
        
        Generate {count} distinct medical concepts. For each, provide the standard mathematical formula using standard variable names (e.g. 'Weight', 'Height').
        
        Return JSON List:
        [
            {{
                "concept_name": "Name of the variable",
                "standard_formula": "Mathematical formula using standard terms",
                "required_inputs": ["List", "of", "standard", "inputs"],
                "clinical_relevance": "Why is this useful?",
                "source": "Exact filename and page number if from context, else 'Standard Medical Knowledge'",
                "logic": "Brief explanation of the formula derivation"
            }}
        ]
        """
        try:
            response = self.llm.invoke(prompt)
            return json.loads(self._clean_json_response(response.content))
        except:
            return []

    def _map_formulas_to_data(self, concepts, columns, allow_missing, limit, use_taxonomy=True):
        """Step 2: Map theoretical inputs to actual dataset columns."""
        columns_str = ", ".join(columns)
        concepts_str = json.dumps(concepts, indent=2)
        missing_instr = "If a variable is missing, do NOT suggest the formula." if not allow_missing else "If a variable is missing, list it in 'missing_variables' and keep the standard name."

        # Include Taxonomy if available
        taxonomy_context = ""
        if self.variable_taxonomy and use_taxonomy:
            # Limit taxonomy size to avoid context overflow if huge
            # We take the first 200 entries or just dump it if it's reasonable
            taxonomy_str = json.dumps(self.variable_taxonomy, indent=2)
            taxonomy_context = f"""
            VARIABLE TAXONOMY (Use this to understand cryptic column names):
            {taxonomy_str}
            """

        prompt = f"""
        Role: Expert Data Engineer.
        Task: Implement the following Theoretical Concepts using the Available Dataset Columns.
        
        Available Columns: [{columns_str}]
        {taxonomy_context}
        Theoretical Concepts: {concepts_str}
        
        CRITICAL SYNTAX RULES (Interpreter Constraints):
        1. **AUTHORIZED OPERATORS**: You may use: +, -, *, /, ** (for power), (, ).
        2. **FUNCTIONS**: You MUST use the 'np.' prefix for mathematical functions.
           - Correct: np.sqrt(x), np.log(x), np.exp(x), np.abs(x)
           - Wrong: sqrt(x), log(x), ln(x), square_root(x)
        3. **SPACING**: You MUST put a single space around every operator.
           - Correct: " ( Weight / Height ) ** 2 "
           - Wrong: "Weight/Height**2"
        4. **VARIABLE NAMES**: 
           - If a variable name contains spaces or special characters, you MUST enclose it in double double-quotes.
           - Example: ""Weight (kg)"" / ""Height (m)""
           - If it is a simple name, you can use it directly: Weight / Height
        
        Instructions:
        1. Map 'required_inputs' to 'Available Columns'.
        2. Handle Unit Conversions (e.g. m to cm, lbs to kg) directly in the formula.
        3. {missing_instr}
        4. Select the top {limit} feasible suggestions.
        5. **CRITICAL**: You MUST preserve the 'source' and 'logic' information from the Theoretical Concepts into 'source_citation' and 'source_explanation'.
        
        Return JSON Object:
        {{
            "suggestions": [
                {{
                    "name": "snake_case_name",
                    "title": "Readable Title",
                    "formula": " Spaced Formula ",
                    "missing_variables": ["list", "if", "any"],
                    "description": "Clinical relevance",
                    "source_type": "Document" or "Model Knowledge" or "Hybrid",
                    "source_citation": "Filename.pdf (Pages X, Y) or 'Model Knowledge'",
                    "source_explanation": "Briefly explain the logic or source of the formula."
                }}
            ]
        }}
        """
        try:
            response = self.llm.invoke(prompt)
            return self._clean_json_response(response.content)
        except Exception as e:
            return json.dumps({"suggestions": [], "error": str(e)})

    def suggest_computed_variables_with_validation(self, columns, search_hint=None, num_suggestions=5, suggestion_mode="Comprehensive", allow_missing_variables=False, use_taxonomy=True, progress_callback=None):
        """Wrapper that calls the pipeline and performs final AST validation with Auto-Correction."""
        
        # 1. Get initial suggestions from the Smart Pipeline
        json_str, error = self.suggest_computed_variables(columns, search_hint, num_suggestions, suggestion_mode, allow_missing_variables, use_taxonomy, progress_callback)
        if error: return None, error
        
        try:
            data = json.loads(json_str)
            suggestions = data.get("suggestions", [])
        except:
            return None, "Failed to parse suggestions."

        if progress_callback: progress_callback(60, "Validating and aligning variables...")
        validated = []
        
        # IMPORTANT: 'np' and 'pd' must be in known_globals so AST doesn't flag them as missing variables
        known_globals = {'np', 'pd', 'log', 'exp', 'sqrt', 'abs', 'min', 'max', 'constant'} 

        def parse_formula_vars(formula_str):
            """Helper to parse variables from formula, handling quoted names."""
            found = set()
            try:
                temp_fmt = formula_str
                q_vars = {}
                # Find ""Variable Name"" pattern
                q_matches = re.findall(r'""([^"]+)""', temp_fmt)
                for i, match in enumerate(q_matches):
                    placeholder = f"__quoted_var_{i}__"
                    q_vars[placeholder] = match
                    temp_fmt = temp_fmt.replace(f'""{match}""', placeholder)
                
                tree = ast.parse(temp_fmt, mode='eval')
                for node in ast.walk(tree):
                    if isinstance(node, ast.Name):
                        found.add(q_vars.get(node.id, node.id))
                return found, True
            except:
                return set(), False

        total_sugg = len(suggestions)
        for idx, sugg in enumerate(suggestions):
            # Update progress dynamically
            if progress_callback:
                # Map progress from 60% to 85%
                current_prog = 60 + int((idx / max(1, total_sugg)) * 25)
                progress_callback(current_prog, f"Validating suggestion {idx+1}/{total_sugg}...")

            formula = sugg.get("formula", "")
            
            # Skip empty or identity formulas
            if not formula.strip() or formula.strip() in columns: 
                continue 

            # --- AST Analysis ---
            found_vars, is_valid_syntax = parse_formula_vars(formula)
            
            if not is_valid_syntax:
                continue # Skip invalid syntax

            # Check for missing variables (Hallucinations or Typos)
            # We exclude 'np' from the missing check logic explicitly here as well just in case
            missing = [v for v in found_vars if v not in columns and v not in known_globals and v != 'np']
            
            if missing:
                # Attempt to fix/align variables first (Auto-Alignment)
                if progress_callback:
                    progress_callback(current_prog, f"Aligning variables for suggestion {idx+1}...")
                print(f"🔧 Aligning variables for: {formula} | Missing: {missing}")
                
                fix_query = f"""
                Role: Code Fixer.
                The formula "{formula}" contains variables that do not match the dataset: {missing}.
                
                Available Variables: {columns}
                
                Task: Rewrite the formula by replacing the missing variables with their exact counterparts from the Available Variables list.
                - Fix typos, case sensitivity, or slight naming variations (e.g. 'Weight' -> 'Weight_kg').
                - Ensure Python/NumPy syntax (use 'np.' for functions).
                - If a variable name contains spaces, enclose it in double double-quotes (e.g. ""Variable Name"").
                - If a variable is truly missing and has no match, keep it as is.
                
                Return ONLY the corrected formula string.
                """
                try:
                    corrected_formula = self.llm.invoke(fix_query).content.strip().strip('"').strip("'")
                    
                    # Re-validate the corrected formula
                    found_vars_2, is_valid_2 = parse_formula_vars(corrected_formula)
                    
                    if not is_valid_2:
                        # If correction resulted in invalid syntax, try to keep original if allowed, else drop
                        if allow_missing_variables:
                             sugg['missing_variables'] = missing
                             validated.append(sugg)
                        continue

                    missing_2 = [v for v in found_vars_2 if v not in columns and v not in known_globals and v != 'np']
                    
                    if not missing_2:
                        # Fix successful!
                        sugg['formula'] = corrected_formula
                        sugg['original_formula'] = formula
                        validated.append(sugg)
                    else:
                        # Fix failed or partial (still missing vars)
                        if allow_missing_variables:
                            sugg['formula'] = corrected_formula # Use the best effort
                            sugg['missing_variables'] = missing_2
                            validated.append(sugg)
                        else:
                            # Drop it if we can't fix it and missing vars aren't allowed
                            pass 
                except Exception:
                     # If fix crashes
                    if allow_missing_variables:
                        sugg['missing_variables'] = missing
                        validated.append(sugg)
            else:
                # No missing vars -> Perfect suggestion
                validated.append(sugg)

        # --- Post-Process: Generate Markdown Formulas ---
        if validated:
            if progress_callback: progress_callback(90, "Generating mathematical notation...")
            try:
                formulas_to_convert = {s['name']: s['formula'] for s in validated}
                
                markdown_prompt = f"""
                Task: Convert these Python formulas into standard LaTeX/Markdown mathematical notation.
                
                Input Formulas:
                {json.dumps(formulas_to_convert, indent=2)}
                
                Instructions:
                1. Return a JSON object where keys are the variable names and values are the LaTeX strings.
                2. Use standard LaTeX notation (e.g. \\frac{{}}, \\sqrt{{}}, \\times).
                3. Remove 'np.' prefixes.
                4. Use readable variable names (remove underscores if it improves readability).
                5. Do NOT wrap in $$ or $.
                
                Example:
                Input: "np.sqrt( Weight_kg / (Height_m ** 2) )"
                Output: "\\sqrt{{\\frac{{Weight}}{{Height^2}}}}"
                """
                
                md_response = self.llm.invoke(markdown_prompt)
                md_map = json.loads(self._clean_json_response(md_response.content))
                
                for s in validated:
                    if s['name'] in md_map:
                        s['markdown_formula'] = md_map[s['name']]
            except Exception as e:
                print(f"Markdown generation failed: {e}")

        data['suggestions'] = validated
        return json.dumps(data), None

    # --- PROXY & ALTERNATIVE HELPERS ---

    def suggest_proxy_variable(self, target_variable, available_columns):
        query = f"Suggest a proxy for '{target_variable}' using [{', '.join(available_columns)}]. Return JSON with keys: proxy_found (bool), proxy_name, formula, explanation."
        try:
            return json.loads(self._clean_json_response(self.llm.invoke(query).content))
        except: return {"proxy_found": False}

    def suggest_alternative_formula(self, target_concept, missing_variable, available_columns):
        query = f"Suggest alternative formula for '{target_concept}' avoiding '{missing_variable}' using [{', '.join(available_columns)}]. Return JSON with keys: alternative_found (bool), alternative_name, formula, explanation."
        try:
            return json.loads(self._clean_json_response(self.llm.invoke(query).content))
        except: return {"alternative_found": False}