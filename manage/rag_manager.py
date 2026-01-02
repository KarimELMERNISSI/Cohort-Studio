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

# Import prompt functions
from prompts import (
    context_analysis,
    column_renaming,
    taxonomy_simple,
    formula_enrichment,
)


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
    # for our QA chains
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
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
        self.current_role = "Medical Researcher" # Default role
        self.adherence_score = 0.5 # Default score
        
        # Cache for theoretical concepts to avoid reprocessing the "Knowledge" step
        self.concept_cache = {} 
        
        # Variable Taxonomy (Mapping of cryptic names to standard concepts)
        self.variable_taxonomy = {}
        # Formulas Registry (Detailed formula definitions)
        self.formulas_registry = {}
        
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
        
        # If no markdown, find outer brackets
        first_brace = text.find("{")
        first_bracket = text.find("[")
        
        start = -1
        end = -1
        
        # Determine if we should look for { or [
        if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
             # It's likely an object
             start = first_brace
             end = text.rfind("}") + 1
        elif first_bracket != -1:
             # It's likely a list
             start = first_bracket
             end = text.rfind("]") + 1
             
        if start != -1 and end > start:
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

        columns_suffix = " and the provided dataset variable names" if dataset_columns else ""
        prompt = context_analysis(
            sample_text=sample_text[:4000],
            columns_context=columns_context,
            columns_suffix=columns_suffix
        )


        
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

    def initialize_system(self, model_name="models/gemini-flash-latest", adherence_score=0.5, temperature=0.3, dataset_columns=None, use_existing_db=False, progress_callback=None, selected_files=None):
        if not self.is_available():
            return False, f"Missing dependencies: {MISSING_LIBS_ERROR}. Please install `chromadb`, `pypdf`, `langchain-community`, `langchain-google-genai`."
        
        if not self.api_key:
            return False, "Google API Key is required."

        try:
            # 1. Load Documents
            if progress_callback: progress_callback(10, "Loading documents...")
            if not os.path.exists(self.documents_dir):
                os.makedirs(self.documents_dir)
                return False, f"Documents directory '{self.documents_dir}' created. Please add PDF files."

            # If specific files are selected, load only those files
            if selected_files:
                documents = []
                for file_name in selected_files:
                    # Search for the file in DOCUMENTS directory and subdirectories
                    file_path = None
                    for root, dirs, files in os.walk(self.documents_dir):
                        if file_name in files:
                            file_path = os.path.join(root, file_name)
                            break
                    
                    if file_path and file_path.endswith('.pdf'):
                        try:
                            loader = PyPDFLoader(file_path)
                            file_docs = loader.load()
                            documents.extend(file_docs)
                        except Exception as e:
                            print(f"Warning: Could not load {file_name}: {e}")
            else:
                # Load all PDFs if no specific files selected
                loader = DirectoryLoader(self.documents_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
                documents = loader.load()
            
            if not documents:
                return False, "No PDF documents found in the DOCUMENTS folder (or none selected)."

            # Fix 0-based page indexing from PyPDFLoader
            for doc in documents:
                if 'page' in doc.metadata:
                    doc.metadata['page'] += 1

            # 2. Split Text
            if progress_callback: progress_callback(30, "Splitting text...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(documents)

            # 3. Create Embeddings & Vector Store
            if progress_callback: progress_callback(50, "Initializing embeddings...")
            embeddings = CustomGeminiEmbeddings(
                api_key=self.api_key, 
                model="models/text-embedding-004"
            ) # GoogleGenerativeAIEmbeddings(model="models/embedding-001"), tbc when langchain-google-genai is updated
            
            # Persist directory for Chroma
            persist_directory = "data/chroma_db"
            
            # Logic to use existing DB or rebuild
            if use_existing_db and os.path.exists(persist_directory):
                if progress_callback: progress_callback(60, "Loading existing database...")
                try:
                    self.vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
                except Exception as e:
                    return False, f"Failed to load existing database: {e}. Try disabling 'Use Existing Database'."
            else:
                if progress_callback: progress_callback(60, "Building new vector database...")
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
                        if progress_callback: 
                            current_prog = 60 + int((i / total_texts) * 30)
                            progress_callback(current_prog, f"Embedding batch {i // batch_size + 1}...")
                            
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
            if progress_callback: progress_callback(95, "Setting up LLM chains...")
            # Use the selected model. Strip 'models/' prefix if present as langchain might handle it differently
            clean_model_name = model_name.replace("models/", "") if model_name.startswith("models/") else model_name
            self.llm = CustomGeminiChat(api_key=self.api_key, model=clean_model_name, temperature=temperature) #ChatGoogleGenerativeAI(model=clean_model_name, temperature=0.3) #to be changed when langchain-google-genai is updated
            
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
        
        retriever = self.vector_store.as_retriever(
            search_type="mmr", 
            search_kwargs={"k": 8, "fetch_k": 20} # Preserving high precision settings
        )

        def format_docs(docs):
            return "\n\n".join(f"Document: {d.metadata.get('source', 'Unknown')} | Page: {d.metadata.get('page', 'N/A')}\nContent: {d.page_content}\n----------------" for d in docs)

        self.qa_chain = (
            {
                "context": (lambda x: x["input"]) | retriever | format_docs,
                "input": lambda x: x["input"]
            }
            | prompt
            | self.llm
            | StrOutputParser()
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
            
        print(f"DEBUG: generate_variable_taxonomy result type: {type(res)}")
        return res

    def _merge_stats(self, taxonomy, columns_info):
        """Merges statistical info from columns_info into the taxonomy."""
        # Create a lookup for column info by name
        col_map = {c["name"]: c for c in columns_info if isinstance(c, dict)}
        
        for var_name, var_data in taxonomy.items():
            if var_name in col_map:
                col_info = col_map[var_name]
                # If numeric stats exist, use them
                if "stats" in col_info and col_info["stats"]:
                    var_data["stats"] = col_info["stats"]
                # If top_values exist (categorical), use them as stats
                elif "top_values" in col_info and col_info["top_values"]:
                    var_data["stats"] = col_info["top_values"]
        return taxonomy

    def _generate_simple_taxonomy(self, columns_info, existing_mapping=None, deep_analysis=False, progress_callback=None):
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
             columns_input = [{"name": c} for c in columns_info]
        else:
             columns_input = columns_info

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

        # Deep Analysis Instructions
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
            
            # Store in instance
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
        
        # 1. Standardization Step
        if progress_callback: progress_callback(10, "Step 1/5: Standardizing variables...")
        standard_mapping, error = self._identify_standard_concepts(columns_info, existing_mapping)
        if error: return None, error
        
        # 2. Contextualization Step (Moved UP)
        if progress_callback: progress_callback(30, "Step 2/5: Adding clinical context...")
        taxonomy = self._contextualize_variables(standard_mapping, columns_info)
        
        # 3. Formula Enrichment Step (Moved DOWN)
        if progress_callback: progress_callback(50, "Step 3/5: Identifying related formulas...")
        taxonomy = self._enrich_with_formulas(taxonomy, columns_info)
        
        # 4. Resolution Step (NEW)
        if progress_callback: progress_callback(70, "Step 4/5: Resolving external links...")
        taxonomy = self._resolve_formula_links(taxonomy)

        # 5. Graph Metadata Enrichment Step
        if progress_callback: progress_callback(90, "Step 5/5: Enriching graph metadata...")
        taxonomy = self._enrich_graph_metadata(taxonomy, columns_info)
        
        # Store in instance
        taxonomy = self._merge_stats(taxonomy, columns_info)
        self.variable_taxonomy = taxonomy
        print(f"DEBUG: _generate_advanced_taxonomy returning taxonomy with {len(taxonomy)} items")
        return taxonomy, None

    def _enrich_graph_metadata(self, taxonomy, columns_info):
        """Step 4: Identify node types and relationships for graph visualization."""
        # 1. Deterministic Node Typing (Internal vs Derived)
        # Identify confirmed formula outputs
        formula_outputs = set()
        for f in self.formulas_registry.values():
            if f.get('output_variable'):
                formula_outputs.add(f['output_variable'])

        # Apply deterministic types
        for k in taxonomy.keys():
            if k in formula_outputs:
                taxonomy[k]['node_type'] = "Derived-Internal"
            else:
                taxonomy[k]['node_type'] = "Input-Internal"

        vars_desc_list = []
        for k, v in taxonomy.items():
            desc = f"{k} ({v.get('standard_name', '')}): {v.get('description', '')}"
            vars_desc_list.append(desc)
            
        vars_desc = "\n".join(vars_desc_list[:50]) # Limit to 50 for now
        
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
            
            # Merge relationships into taxonomy
            for var, relationships in metadata.items():
                if var in taxonomy and isinstance(relationships, list):
                    current_rels = taxonomy[var].get('relationships', [])
                    # Append unique
                    taxonomy[var]['relationships'] = list(set(current_rels + relationships))
                    
        except Exception as e:
            print(f"Graph Enrichment Error: {e}")
            
        return taxonomy

    def _identify_standard_concepts(self, columns_info, existing_mapping):
        """Step 1: Map variables to standard medical concepts."""
        # Reuse the simple logic but with a stricter persona
        return self._generate_simple_taxonomy(columns_info, existing_mapping, deep_analysis=False)

    def _enrich_with_formulas(self, taxonomy, columns_info):
        """Step 2: Identify formula relationships (Structured)."""
        updated_taxonomy = taxonomy.copy()
        
        # Prepare a summary of variables to ask about formulas
        vars_desc_list = []
        for k, v in taxonomy.items():
            desc = f"{k} ({v.get('standard_name', '')})"
            # Add top values/modalities if available to help identify units
            # We need to look up the original column info to get top values
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
                print(f"DEBUG: Formulas Data Keys: {data.keys()}") # DEBUG
                
                # 1. Update Registry
                formulas = data.get('formulas', [])
                print(f"DEBUG: Found {len(formulas)} formulas") # DEBUG
                for f in formulas:
                    # Use ID as key
                    f_id = f.get('id')
                    if f_id:
                        self.formulas_registry[f_id] = f
                
                # 2. Update Taxonomy Links
                var_updates = data.get('variable_updates', {})
                print(f"DEBUG: Found updates for {len(var_updates)} variables") # DEBUG
                
                # Create reverse lookup for standard names
                std_to_orig = {v.get('standard_name', '').lower(): k for k, v in taxonomy.items()}
                
                for var_key, updates in var_updates.items():
                    target_key = None
                    
                    # 1. Try direct match
                    if var_key in updated_taxonomy:
                        target_key = var_key
                    # 2. Try standard name match
                    elif var_key.lower() in std_to_orig:
                        target_key = std_to_orig[var_key.lower()]
                    
                    if target_key:
                        # Append to existing list or create new
                        current_formulas = updated_taxonomy[target_key].get('related_formula_ids', [])
                        new_formulas = updates.get('involved_in_formulas', [])
                        
                        # Merge unique
                        updated_taxonomy[target_key]['related_formula_ids'] = list(set(current_formulas + new_formulas))
                    else:
                        print(f"DEBUG: Could not match update key '{var_key}' to any variable.")
                        
            except json.JSONDecodeError as e:
                print(f"JSON Error in formulas: {e}. Raw: {response.content[:200]}...")
            
        except Exception as e:
            print(f"Error in formula enrichment: {e}")
            
        print(f"DEBUG: Registry Size: {len(self.formulas_registry)}") # DEBUG
        return updated_taxonomy

    def _contextualize_variables(self, taxonomy, columns_info):
        """Step 3: Add clinical usage and interpretation context."""
        updated_taxonomy = taxonomy.copy()
        
        # Prepare a summary of variables to ask about formulas
        vars_desc_list = []
        for k, v in taxonomy.items():
            desc = f"{k} ({v.get('standard_name', '')})"
            # Add top values/modalities if available to help identify units
            # We need to look up the original column info to get top values
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
            except json.JSONDecodeError as e:
                print(f"JSON Error in context: {e}. Raw: {response.content[:200]}...")
                context_data = {}
            
            for var, data in context_data.items():
                if var in updated_taxonomy:
                    updated_taxonomy[var]['clinical_usage'] = data.get('clinical_usage', "")
        except Exception as e:
            print(f"Error in contextualization: {e}")
            
        return updated_taxonomy

    def _resolve_formula_links(self, taxonomy):
        """Step 4: Resolve 'External' formula inputs to internal keys via LLM."""
        updated_taxonomy = taxonomy.copy()
        
        # 1. Identify Unresolved Variables in Formulas
        unresolved_vars = set()
        for f in self.formulas_registry.values():
            for inp in f.get('input_variables', []):
                if inp not in taxonomy:
                    unresolved_vars.add(inp)
            out = f.get('output_variable')
            if out and out not in taxonomy:
                unresolved_vars.add(out)
        
        if not unresolved_vars:
            print("DEBUG: No unresolved formula variables found.")
            return updated_taxonomy

        # 2. Prepare Prompt
        dataset_keys = list(taxonomy.keys())
        # Summarize dataset roughly
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
            
            print(f"DEBUG: Resolution Mapping: {mapping}")
            
            # 3. Apply Mapping to Registry
            for f_id, f_data in self.formulas_registry.items():
                # Inputs
                new_inputs = []
                for inp in f_data.get('input_variables', []):
                    # Check mapping
                    mapped_key = mapping.get(inp)
                    if mapped_key and mapped_key in taxonomy:
                        new_inputs.append(mapped_key)
                        
                        # Also update Variable Taxonomy links
                        current_links = updated_taxonomy[mapped_key].get('related_formula_ids', [])
                        if f_id not in current_links:
                            updated_taxonomy[mapped_key]['related_formula_ids'] = current_links + [f_id]
                    else:
                        new_inputs.append(inp) # Keep original if no match
                
                self.formulas_registry[f_id]['input_variables'] = new_inputs
                
                # Output
                out = f_data.get('output_variable')
                if out:
                    mapped_out = mapping.get(out)
                    if mapped_out and mapped_out in taxonomy:
                        self.formulas_registry[f_id]['output_variable'] = mapped_out
                        # Link
                        current_links = updated_taxonomy[mapped_out].get('related_formula_ids', [])
                        if f_id not in current_links:
                            updated_taxonomy[mapped_out]['related_formula_ids'] = current_links + [f_id]
                            
        except Exception as e:
            print(f"Error in resolution: {e}")
            
        return updated_taxonomy

    def _unify_synonyms(self, new_candidates, existing_keys):
        """
        Uses LLM to identify and merge synonyms within the new candidates 
        and against existing taxonomy keys.
        Returns a mapping { 'alias_id': 'canonical_id' }.
        """
        if not new_candidates:
            return {}
            
        # Context
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
        except Exception as e:
            print(f"Unification Error: {e}")
            return {}

    # =========================================================================
    # DOCUMENT GRAPH KNOWLEDGE (NATIVE FILE API)
    # =========================================================================

    def get_available_documents(self):
        """Returns a list of PDF files in the documents directory."""
        if not os.path.exists(self.documents_dir):
            return []
        return [f for f in os.listdir(self.documents_dir) if f.lower().endswith('.pdf')]

    def extract_custom_graph_from_doc(self, file_name, progress_callback=None):
        """
        Uses Gemini Native File API to extract a knowledge graph from a specific document.
        """
        if not self.initialized:
            return None, "RAG System not initialized."
            
        file_path = os.path.join(self.documents_dir, file_name)
        if not os.path.exists(file_path):
            return None, f"File {file_name} not found."
            
        if progress_callback: progress_callback(10, f"Uploading {file_name} to Gemini...")
        
        try:
            # 1. Upload File
            # Access underlying client from CustomGeminiChat wrapper
            client = self.llm.client 
            
            # The 'files' module is on the client instance in v1.0 SDK
            with open(file_path, "rb") as f:
                # Need to read logs? No, client.files.upload accepts path directly usually
                # But SDK v1.0 might want path or file-like. Let's use path argument if supported.
                # Assuming client.files.upload(path=...) is correct based on general SDK usage.
                uploaded_file = client.files.upload(file=file_path)
            
            # 2. Wait for Processing
            while uploaded_file.state.name == "PROCESSING":
                if progress_callback: progress_callback(20, "Processing file...")
                time.sleep(2)
                uploaded_file = client.files.get(name=uploaded_file.name)
                
            if uploaded_file.state.name == "FAILED":
                return None, "File processing failed by Google."
                
            if progress_callback: progress_callback(40, "Generating Knowledge Graph (Deep Analysis)...")
            
            # 3. Generate Content
            prompt_text = """
            Role: Expert Information Architect.
            Task: Analyze this document and extract a Knowledge Graph of key entities and their relationships.
            
            Instructions:
            1. Identify core entities (Concepts, Methods, Metrics, Findings, Diseases, Treatments).
            2. Identify relationships between them.
            3. NAMING CONVENTION: Use the **Canonical/Standard** name for each entity. 
               - E.g., Use "Heart Failure" instead of "HF". 
               - Deduplicate within the document (do not create separate nodes for acronyms).
            4. EXHAUSTIVE EXTRACTION: For each entity, scan the ENTIRE document.
               - Collect ALL page numbers.
               - Select the BEST definition and representative quote.
            5. SCIENTIFIC SUMMARY: Analyze the document type (e.g. Clinical Study, Review, Protocol) and generate a structured summary.
            
            Return JSON:
            {
                "summary": {
                    "title": "Inferred Document Title",
                    "doc_type": "Study Type (e.g. Cohort Study, Review)",
                    "objective": "Primary goal/hypothesis of the study",
                    "methods": "Key methodology, population, study design",
                    "key_findings": "Primary results and outcomes",
                    "significance": "Clinical or scientific implications",
                    "top_concepts": ["List of 3-5 most important concepts"]
                },
                "nodes": [
                    {
                        "id": "Canonical Name",
                        "type": "Concept/Metric/Finding/etc",
                        "description": "Comprehensive Definition",
                        "source_text": "Representative quote...",
                        "page_reference": "1, 3, 5"
                    }
                ],
                "edges": [
                    {
                        "source": "Source Node ID",
                        "target": "Target Node ID",
                        "relation": "relationship_type",
                        "description": "Context of relationship"
                    }
                ],
                "formulas": [
                    {
                        "name": "Formula Name",
                        "expression": "Math expression",
                        "page": "Page X",
                        "description": "Explanation"
                    }
                ]
            }
            """
            
            # Prepare contents
            # Using types.Content for structure
            
            # Note: We need to import types locally if not available, or assume it's there. 
            # We imported types globally at the top.
            
            response = client.models.generate_content(
                model=self.llm.model_name,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_uri(
                                file_uri=uploaded_file.uri,
                                mime_type=uploaded_file.mime_type
                            ),
                            types.Part.from_text(text=prompt_text)
                        ]
                    )
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2
                )
            )
            
            # 4. Cleanup
            try:
                client.files.delete(name=uploaded_file.name)
            except:
                pass

            # 5. Parse Response
            json_str = response.text
            # Use our robust cleaner
            cleaned_json = self._clean_json_response(json_str)
            parsed_data = json.loads(cleaned_json)
            
            # Robustness: Handle if LLM returned a list [ { "nodes": ... } ]
            if isinstance(parsed_data, list):
                if len(parsed_data) > 0 and isinstance(parsed_data[0], dict):
                    parsed_data = parsed_data[0]
                else:
                    # Fallback or error?
                    return None, "LLM returned an unexpected list format."
            
            if not isinstance(parsed_data, dict):
                 return None, "LLM did not return a valid JSON object."
                 
            return parsed_data, None
            
        except Exception as e:
            return None, str(e)



    def extract_merged_graph_from_docs(self, doc_list, progress_callback=None):
        """
        Extracts and merges graphs from multiple documents.
        Returns (merged_json, error).
        """
        if not doc_list:
            return None, "No documents specified."
            
        merged_nodes = {} # id_lower -> node_data
        merged_edges = [] # list of edge dicts
        merged_formulas = []
        merged_summaries = []
        
        seen_edges = set() # (src_lower, tgt_lower, rel_lower)
        
        total = len(doc_list)
        
        for idx, doc in enumerate(doc_list):
            if progress_callback: progress_callback(int((idx/total)*100), f"Processing {doc}...")
            
            # Reuse single doc extraction
            g_json, err = self.extract_custom_graph_from_doc(doc)
            if err:
                print(f"Error processing {doc}: {err}")
                continue # Skip bad docs but keep partial result
            
            # Collect Summary
            if "summary" in g_json:
                s = g_json["summary"]
                s["doc"] = doc # Link to source
                merged_summaries.append(s)

            # Merge Nodes
            for n in g_json.get("nodes", []):
                nid = n.get("id", "").strip()
                if not nid: continue
                nid_lower = nid.lower()
                
                # Create Citation
                citation = {
                    "doc": doc,
                    "page": n.get("page_reference", "Unknown"),
                    "text": n.get("source_text", "")
                }
                
                # ID Matching Logic
                match_id = None
                
                # 1. Exact Match
                if nid_lower in merged_nodes:
                    match_id = nid_lower
                else:
                    # 2. Fuzzy Match (Robustness for LLM variations)
                    # Iterate existing keys to find close match
                    import difflib
                    existing_keys = list(merged_nodes.keys())
                    # Using get_close_matches for speed, cutoff 0.85 (high similarity)
                    matches = difflib.get_close_matches(nid_lower, existing_keys, n=1, cutoff=0.85)
                    if matches:
                        match_id = matches[0]
                
                if not match_id:
                    # New Node
                    n["citations"] = [citation]
                    # Clean up single doc fields to avoid confusion
                    n.pop("source_text", None)
                    n.pop("page_reference", None)
                    merged_nodes[nid_lower] = n
                else:
                    # Merge Logic
                    existing = merged_nodes[match_id]
                    existing["citations"].append(citation)
                    # Keep longest description? or concatenation?
                    if len(n.get("description", "")) > len(existing.get("description", "")):
                        existing["description"] = n.get("description", "")
            
            # Merge Edges
            for e in g_json.get("edges", []):
                s = str(e.get("source"))
                t = str(e.get("target"))
                r = e.get("relation", "relates to")
                key = (s.lower(), t.lower(), r.lower())
                
                if key not in seen_edges:
                    merged_edges.append(e)
                    seen_edges.add(key)
            
            # Merge Formulas
            for f in g_json.get("formulas", []):
                f["doc"] = doc # Add source
                merged_formulas.append(f)

        return {
            "nodes": list(merged_nodes.values()),
            "edges": merged_edges,
            "formulas": merged_formulas,
            "summaries": merged_summaries
        }, None


    def match_columns_to_graph(self, columns, graph_json):
        """
        Matches dataset columns to Knowledge Graph nodes using fuzzy string matching.
        Returns dict: {column_name: {match_found: bool, node: data, confidence: float}}
        """
        import difflib
        
        results = {}
        nodes = graph_json.get("nodes", [])
        if not nodes:
            return {c: {"match_found": False} for c in columns}
            
        # Create lookups
        node_lookup = {n.get("id", "").lower(): n for n in nodes}
        node_ids_lower = list(node_lookup.keys())
        
        for col in columns:
            col_lower = col.lower().replace("_", " ")
            
            # 1. Exact Match (Best)
            if col_lower in node_lookup:
                results[col] = {
                    "match_found": True, 
                    "node": node_lookup[col_lower], 
                    "confidence": 1.0,
                    "method": "Exact"
                }
                continue
                
            # 2. Fuzzy Match
            matches = difflib.get_close_matches(col_lower, node_ids_lower, n=1, cutoff=0.6)
            if matches:
                 match_id = matches[0]
                 # Calculate similarity score
                 ratio = difflib.SequenceMatcher(None, col_lower, match_id).ratio()
                 results[col] = {
                     "match_found": True, 
                     "node": node_lookup[match_id], 
                     "confidence": ratio,
                     "method": "Fuzzy"
                 }
            else:
                results[col] = {"match_found": False}
                
        return results


    def chat_with_specific_doc(self, doc_names, query):
        """
        Chat with specific document(s) using the vector store.
        doc_names: string or list of strings.
        """
        if not self.initialized or not self.vector_store:
            return "System not initialized or no database available."

        if isinstance(doc_names, str):
            doc_names = [doc_names]
            
        full_paths = [os.path.join(self.documents_dir, d) for d in doc_names]
        
        # Retrieval Filter
        filter_dict = {}
        if len(full_paths) == 1:
            filter_dict = {"source": full_paths[0]}
        else:
             # ChromaDB $in operator
             filter_dict = {"source": {"$in": full_paths}}
        
        retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "filter": filter_dict
            }
        )
        
        doc_list_str = ", ".join(doc_names)
        
        prompt = ChatPromptTemplate.from_template(f"""
        Role: Document Assistant.
        Context (From {doc_list_str}):
        {{context}}
        
        User Question: {{question}}
        
        Instruction: Answer the question based ONLY on the provided context from the documents.
        If the answer is not in the context, say "I cannot find this information in the documents."
        """)
        
        def format_docs(docs):
            return "\\n\\n".join(f"[Source: {{d.metadata.get('source','Unknown')}}] {{d.page_content}}" for d in docs)

        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        try:
            return chain.invoke(query)
        except Exception as e:
            return f"Error responding: {{str(e)}}"

    def enrich_variable_taxonomy(self, current_taxonomy, columns_info, distance=1, progress_callback=None):
        """
        Enriches an existing taxonomy using 'Wise Enrichment' strategy (Formula-centric).
        Returns (new_candidates, error) where new_candidates is a dict of proposed variables.
        Does NOT merge automatically.
        """
        if not self.initialized:
            return {}, "RAG system not initialized."
            
        new_candidates = {}
        new_formulas = {}
        
        # Helper for duplicate detection
        existing_keys_lower = {k.lower().strip() for k in current_taxonomy.keys()}
        
        def is_duplicate(key):
            return key.lower().strip() in existing_keys_lower

        # Dynamic Progress Calculation
        # We have 'distance' levels, and each level has 2 major steps (Harvest, Infer)
        total_steps = distance * 2
        step_increment = 90 // total_steps if total_steps > 0 else 20
        current_progress = 5 

        # Iterative Enrichment Loop
        current_pool = current_taxonomy.copy() # Start with base taxonomy
        new_items_history = [] # Track enrichment history for context window

        for level in range(1, distance + 1):
            level_candidates = {}
            
            # --- Step A: Harvest from Formulas Registry (Deterministic) ---
            current_progress += step_increment
            if progress_callback: 
                progress_callback(min(current_progress, 100), f"Level {level}: Harvesting formulas...")
                
            # Scan registry against CURRENT POOL (Base + Previous Candidates)
            for f in self.formulas_registry.values():
                # Check Inputs against pool
                # If we have SOME inputs in pool, propose the MISSING ones
                inputs = f.get('input_variables', [])
                inputs_in_pool = [i for i in inputs if i in current_pool or i.lower() in [k.lower() for k in current_pool.keys()]]
                
                if inputs_in_pool: # We have a connection
                    # Propose MISSING inputs
                    for inp in inputs:
                        if not is_duplicate(inp) and inp not in new_candidates and inp not in current_pool:
                            level_candidates[inp] = {
                                "standard_name": inp.replace('_', ' ').title(),
                                "node_type": "Input-External",
                                "description": f"Required input for formula: {f.get('name')}",
                                "category": "External Factor",
                                "relationships": [f.get('id')]
                            }
                    
                    # Propose Output if we have ALL inputs (or most?)
                    out = f.get('output_variable')
                    if out and not is_duplicate(out) and out not in new_candidates and out not in current_pool:
                         level_candidates[out] = {
                            "standard_name": out.replace('_', ' ').title(),
                            "node_type": "Derived-External",
                            "description": f"Calculated result of formula: {f.get('name')}",
                            "category": "External Factor",
                            "relationships": [f.get('id')]
                        }
            
            # Update candidates for this level
            if level_candidates:
                new_candidates.update(level_candidates)
                current_pool.update(level_candidates)
                new_items_history.extend(list(level_candidates.keys()))

            # --- Step B: LLM Inference (Creative with RAG) ---
            current_progress += step_increment
            if progress_callback: 
                 progress_callback(min(current_progress, 100), f"Level {level}: Inferring new connections...")
            
            # Prepare context (Summary of current pool with Standard Names if avaialble)
            all_keys = list(current_pool.keys())
            
            def format_var(k):
                sn = current_pool[k].get('standard_name', k)
                return f'{k}: "{sn}"'
            
            # Prioritize listing the NEW stuff if we are deeper
            # Context Strategy: Latest 50 items + Random 50 base items
            if new_items_history:
                # Take recent history (e.g. from previous level or this level's harvest)
                recent_window = new_items_history[-60:] 
                base_window = [k for k in all_keys if k not in new_items_history][:40]
                focus_keys = recent_window + base_window
                vars_context = ", ".join([format_var(k) for k in focus_keys])
            else:
                vars_context = ", ".join([format_var(k) for k in all_keys[:100]])
            
            # RAG: Retrieve Relevant Documents
            rag_context = ""
            if self.vector_store:
                try:
                    search_query = f"Clinical formulas and scores related to: {vars_context[:200]}"
                    docs = self.vector_store.similarity_search(search_query, k=3)
                    rag_context = "\n\n".join([d.page_content for d in docs])
                except Exception as e:
                    print(f"Vector Search Error: {e}")

            # Define Adherence Logic
            if self.adherence_score > 0.7:
                adherence_instruction = "STRICTLY propose formulas explicitly mentioned in the 'Context from Documents'. Do NOT halluncinate or invent formulas."
            elif self.adherence_score < 0.4:
                adherence_instruction = "Use 'Context from Documents' as inspiration, but rely primarily on general medical knowledge to identify standard missing scores."
            else:
                adherence_instruction = "Prioritize formulas found in 'Context from Documents', but you may also suggest standard medical scores if clearly relevant."

            prompt = f"""
            Role: Clinical Knowledge Expert.
            
            Task: "Wise Enrichment" of a Clinical Knowledge Graph (Level {level}).
            
            Goal: Identify potential NEW formulas or scores that could be calculated from the Current Variables.
            
            Context from Documents:
            {rag_context}
            
            Current Variables (Format: ID: "Standard Name"):
            {vars_context}
            
            Instructions:
            1. {adherence_instruction}
            2. **Synonym Check**: Check the 'Current Variables' list CAREFULLY. If a concept exists (e.g. 'Body Height'), USE THAT ID. Do NOT create a duplicate (e.g. 'height_cm').
            3. **ID Format**: Use snake_case for IDs (e.g. `body_mass_index`, NOT `calc_bmi`). Avoid prefixes like `calc_` or `derived_`.
            4. **Formula Check**: Do not propose formulas that strictly duplicate existing ones. Variants are OK (e.g. BSA DuBois vs BSA Mosteller), but exact duplicates (BMI vs Body Mass Index) are NOT.
            5. Look for standard medical scores (e.g. BMI, eGFR) where we have SOME of the variables.
            6. Propose the MISSING external variables needed.
            
            Return JSON with the NEW variables and NEW formulas:
            {{
                "variables": {{
                    "new_variable_id_snake_case": {{
                        "standard_name": "Standard Name",
                        "description": "Why this is needed",
                        "node_type": "Input-External", 
                        "category": "Suggested Category",
                        "clinical_usage": "Reason for inclusion"
                    }}
                }},
                "formulas": {{
                    "new_formula_id": {{
                        "name": "Formula Name (e.g. BMI)",
                        "description": "Calculation logic",
                        "output_variable": "output_var_id",
                        "input_variables": ["input_var_id_1", "input_var_id_2"]
                    }}
                }}
            }}
            """
            
            try:
                response = self.llm.invoke(prompt)
                cleaned_response = self._clean_json_response(response.content)
                inferred_data = json.loads(cleaned_response)
                
                # Pre-Initialize unify_map for this batch to handle duplicates immediately
                unify_map = {}
                
                # Check structure
                if "variables" in inferred_data:
                    for k, v in inferred_data["variables"].items():
                        # Enhanced Deduplication
                        # Check 1: Exact ID Match
                        if k in current_taxonomy or k in new_candidates or k in current_pool:
                            continue
                            
                        # Check 2: Name/Standard Name Match (Case Insensitive)
                        is_var_dup = False
                        cand_std = v.get('standard_name', '').lower().strip()
                        
                        # Check against Current Taxonomy
                        target_remap = None
                        
                        for exist_k, exist_v in current_taxonomy.items():
                            exist_std = exist_v.get('standard_name', '').lower().strip()
                            exist_name = exist_v.get('name', exist_k).lower().strip()
                            
                            if cand_std == exist_std or cand_std == exist_name:
                                is_var_dup = True
                                target_remap = exist_k
                                break
                        
                        if not is_var_dup:
                             # Check against previously added candidates in this batch
                            for added_k, added_v in new_candidates.items():
                                added_std = added_v.get('standard_name', '').lower().strip()
                                if cand_std == added_std:
                                    is_var_dup = True
                                    target_remap = added_k
                                    break

                        if not is_var_dup:
                            new_candidates[k] = v
                            current_pool[k] = v
                        elif target_remap:
                            # Map duplicate ID to existing ID
                            # We use this to fix formula links later
                            unify_map = unify_map or {} # Ensure it's init
                            unify_map[k] = target_remap
                
                if "formulas" in inferred_data:
                    for k, v in inferred_data["formulas"].items():
                        # Remap IDs in Formulas based on Variable Deduplication
                        f_out = v.get('output_variable')
                        if f_out in unify_map:
                            v['output_variable'] = unify_map[f_out]
                        
                        f_inputs = v.get('input_variables', [])
                        f_inputs = [unify_map.get(i, i) for i in f_inputs]
                        v['input_variables'] = list(set(f_inputs))

                        # Strict Formula Deduplication
                        # Check if a formula with same output already exists in new_formulas or registry
                        output_var = v.get('output_variable')
                        
                        is_formula_dup = False
                        # Check against new formulas
                        for existing_f in new_formulas.values():
                             if existing_f.get('output_variable') == output_var:
                                 is_formula_dup = True
                                 break
                        # Check against registry
                        if not is_formula_dup:
                             for existing_f in self.formulas_registry.values():
                                  if existing_f.get('output_variable') == output_var:
                                       is_formula_dup = True
                                       break
                        
                        # Check 3: Name Similarity (e.g. "BSA DuBois" vs "Body Surface Area (DuBois)")
                        if not is_formula_dup:
                            cand_name = v.get('name', '').lower().replace(" ", "").replace("(", "").replace(")", "")
                            
                            # Check against new formulas
                            for existing_f in new_formulas.values():
                                ext_name = existing_f.get('name', '').lower().replace(" ", "").replace("(", "").replace(")", "")
                                if cand_name == ext_name:
                                    is_formula_dup = True
                                    break
                            
                            # Check against registry
                            if not is_formula_dup:
                                for existing_f in self.formulas_registry.values():
                                    ext_name = existing_f.get('name', '').lower().replace(" ", "").replace("(", "").replace(")", "")
                                    if cand_name == ext_name:
                                        is_formula_dup = True
                                        break

                        if not is_formula_dup and k not in new_formulas and k not in self.formulas_registry:
                            v["id"] = k
                            new_formulas[k] = v
                            v["id"] = k
                            new_formulas[k] = v

                # Fallback for old flat format (just in case LLM is stubborn)
                if "variables" not in inferred_data and "formulas" not in inferred_data:
                     for k, v in inferred_data.items():
                         # Assume variables
                         if not is_duplicate(k) and k not in new_candidates and k not in current_pool:
                            new_candidates[k] = v
                            current_pool[k] = v

            except Exception as e:
                print(f"Error in LLM enrichment: {e}")

        # --- Post-Processing: Semantic Unification ---
        if progress_callback:
            progress_callback(95, "Unifying synonyms...")

        unify_map = self._unify_synonyms(new_candidates, current_taxonomy.keys())
        
        if unify_map:
            # Remap Candidates
            final_candidates = {}
            for k, v in new_candidates.items():
                target = unify_map.get(k, k)
                if target in current_taxonomy: 
                    continue # Merged into existing taxonomy, drop candidate
                
                # If target is new canonical, ensure it exists
                if target not in final_candidates:
                    if k == target:
                        final_candidates[target] = v
                    else:
                        # Alias k -> target. Use v, but update ID info
                        v["standard_name"] = v.get("standard_name", "").replace(k, target) # naïve update
                        final_candidates[target] = v
                else:
                    # Target already exists (e.g. we processed the canonical one, or another alias)
                    # Merge metadata? (Keep 'v' logic simple: first writer or extend)
                    pass

            new_candidates = final_candidates
            
            # Remap Formulas
            for f in new_formulas.values():
                # Output
                out = f.get('output_variable')
                if out in unify_map:
                    f['output_variable'] = unify_map[out]
                
                # Inputs
                new_inputs = []
                for inp in f.get('input_variables', []):
                    new_inputs.append(unify_map.get(inp, inp))
                f['input_variables'] = list(set(new_inputs))

        # Post-Processing: Fill Metadata for ALL candidates
        if progress_callback:
            progress_callback(98, "Finalizing candidates...")

        for k, v in new_candidates.items():
            # Check if this candidate is an OUTPUT of any new formula
            is_derived = False
            for f in new_formulas.values():
                if f.get('output_variable') == k:
                    is_derived = True
                    break
            
            if is_derived:
                v['node_type'] = "Derived-External"
            else:
                # Default to Input-External if not explicitly derived
                if 'node_type' not in v or v['node_type'] not in ["Input-External", "Input-Internal"]:
                     v['node_type'] = "Input-External"

            if 'role' not in v:
                v['role'] = v.get('node_type', 'derived-external').lower()

        return new_candidates, new_formulas, None



    def _expand_search_hint(self, search_hint):
        """Pre-processing: Expand acronyms or ambiguous terms to standard medical concepts."""
        if not search_hint or len(search_hint) > 20: # Skip if too long, likely already descriptive
            return search_hint
            
        prompt = f"""
        Role: Medical Terminology Expert.
        Input: "{search_hint}"
        
        Task: 
        1. If the input is a medical acronym or abbreviation (e.g. BSA, BMI, eGFR, SBP), return the STANDARD FULL NAME (e.g. Body Surface Area).
        2. If it is already a full name or not a known medical acronym, return it exactly as is.
        
        Return ONLY the expanded/standard name. No bolding, no extra text.
        """
        try:
             # Quick call, low temperature for determinism
            expanded = self.llm.invoke(prompt).content.strip().strip('"').strip("'")
            # Basic sanity check: don't accept if it turned into a sentence
            if len(expanded) < 60:
                return expanded
            return search_hint
        except:
            return search_hint

    def suggest_computed_variables(self, columns, search_hint=None, num_suggestions=5, suggestion_mode="Comprehensive", allow_missing_variables=False, use_taxonomy=True, progress_callback=None):
        if not self.initialized:
            return None, "RAG system not initialized."
        
        # --- STEP PRE-0: EXPAND HINT ---
        original_hint = search_hint
        if search_hint:
             if progress_callback: progress_callback(5, "Analyzing search term...")
             search_hint = self._expand_search_hint(search_hint)
             if search_hint != original_hint:
                 print(f"Expanded search hint: '{original_hint}' -> '{search_hint}'")

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
