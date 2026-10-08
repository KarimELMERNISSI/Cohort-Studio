"""
RAG Manager - Core Module.

This is the main RAGManager class that composes functionality from mixin classes:
- TaxonomyMixin: Variable taxonomy generation and enrichment
- DocumentsMixin: Document processing and knowledge graph extraction
- ComputedVarsMixin: Computed variable suggestions

The mixin pattern allows splitting a large class into focused modules
while maintaining a single class interface for existing code.
"""
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
from rapidfuzz import process
import pandas as pd

# Import prompt functions
from prompts import (
    context_analysis,
    column_renaming,
    taxonomy_simple,
    formula_enrichment,
    anomaly_criteria_prompt,
)

# Import mixins
from .rag_taxonomy import TaxonomyMixin
from .rag_documents import DocumentsMixin
from .rag_computed_vars import ComputedVarsMixin

# Import LLM utilities for reliable parsing
from utils.llm_utils import (
    parse_json_safe,
    validate_and_parse,
    StructuredOutputHelper,
    clean_json_response,
)


os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_IMPL"] = "false"

# Suppress specific ChromaDB telemetry errors
logging.getLogger('chromadb.telemetry.product.posthog').setLevel(logging.CRITICAL)

# Try importing RAG dependencies
try:
    import google.genai as genai
    from google.genai import types
    from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
    from utils.custom_gemini import CustomGeminiChat, CustomGeminiEmbeddings

    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
    RAG_AVAILABLE = True
except ImportError as e:
    RAG_AVAILABLE = False
    MISSING_LIBS_ERROR = str(e)
    genai = None


class RAGManager(TaxonomyMixin, DocumentsMixin, ComputedVarsMixin):
    """
    Main RAG Manager class that composes functionality from mixins.
    
    Core Methods (defined here):
        - __init__
        - is_available
        - get_available_models
        - _clean_json_response
        - _analyze_global_context
        - initialize_system
        - update_adherence_score
        - _update_internal_state
    
    Taxonomy Methods (from TaxonomyMixin):
        - suggest_column_renaming
        - generate_variable_taxonomy
        - enrich_variable_taxonomy
        - refine_variable_taxonomy
    
    Document Methods (from DocumentsMixin):
        - get_available_documents
        - extract_custom_graph_from_doc
        - extract_merged_graph_from_docs
        - match_columns_to_graph
        - chat_with_specific_doc
    
    Computed Variable Methods (from ComputedVarsMixin):
        - suggest_computed_variables
        - suggest_computed_variables_with_validation
        - suggest_proxy_variable
        - suggest_alternative_formula
    """
    
    def __init__(self, documents_dir="DOCUMENTS", api_key=None):
        self.documents_dir = documents_dir
        self.api_key = api_key
        self.client = None
        self.vector_store = None
        self.qa_chain = None
        self.initialized = False
        self.global_context = "General Medical Domain"
        self.current_role = "Medical Researcher"
        self.adherence_score = 0.5
        
        # Cache for theoretical concepts
        self.concept_cache = {} 
        
        # Variable Taxonomy
        self.variable_taxonomy = {}
        # Formulas Registry
        self.formulas_registry = {}
        
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            if genai:
                self.client = genai.Client(api_key=api_key)

    def is_available(self):
        """Check if RAG dependencies are available."""
        return RAG_AVAILABLE

    def get_available_models(self):
        """Get list of available Gemini models."""
        if not self.api_key or not genai:
            return []
        try:
            if not self.client:
                self.client = genai.Client(api_key=self.api_key)
                
            models = []
            for m in self.client.models.list():
                if "gemini" in m.name.lower(): 
                    models.append(m.name)
            return models
        except Exception:
            return []

    def _clean_json_response(self, text):
        """
        Clean JSON output from LLM using robust parsing utilities.
        
        This method is a wrapper around llm_utils.clean_json_response
        for backward compatibility with existing code.
        """
        return clean_json_response(text)
    
    def _parse_json_safe(self, text, default=None):
        """
        Safely parse JSON with multiple fallback strategies.
        
        Uses llm_utils.parse_json_safe for robust parsing with:
        - Automatic JSON extraction from markdown
        - Trailing comma removal
        - Bracket balancing
        - AST literal_eval fallback
        """
        return parse_json_safe(text, default)
    
    def _get_structured_output_helper(self):
        """Get a StructuredOutputHelper instance for schema-validated LLM calls."""
        if not hasattr(self, '_output_helper') or self._output_helper is None:
            if hasattr(self, 'llm'):
                self._output_helper = StructuredOutputHelper(self.llm, max_retries=2)
            else:
                return None
        return self._output_helper

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
        
        sample_text = "\n\n".join([t.page_content for t in texts[:3]])
        
        columns_context = ""
        if dataset_columns:
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

    def initialize_system(self, model_name="models/gemini-flash-latest", adherence_score=0.5, temperature=0.3, dataset_columns=None, use_existing_db=False, progress_callback=None, selected_files=None, embedding_model="models/embedding-001"):
        """Initialize the RAG system with documents and LLM."""
        if not self.is_available():
            return False, f"Missing dependencies: {MISSING_LIBS_ERROR}. Please install required packages."
        
        if not self.api_key:
            return False, "Google API Key is required."

        try:
            # 1. Load Documents
            if progress_callback: progress_callback(10, "Loading documents...")
            if not os.path.exists(self.documents_dir):
                os.makedirs(self.documents_dir)
                return False, f"Documents directory '{self.documents_dir}' created. Please add PDF files."

            if selected_files:
                documents = []
                for file_name in selected_files:
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
                        except Exception:
                            pass
            else:
                loader = DirectoryLoader(self.documents_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
                documents = loader.load()
            
            if not documents:
                return False, "No PDF documents found in the DOCUMENTS folder (or none selected)."

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
                model=embedding_model
            )
            
            from utils.data_paths import get_chroma_dir
            persist_directory = get_chroma_dir()
            
            if use_existing_db and os.path.exists(persist_directory):
                if progress_callback: progress_callback(60, "Loading existing database...")
                try:
                    self.vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
                except Exception as e:
                    return False, f"Failed to load existing database: {e}. Try disabling 'Use Existing Database'."
            else:
                if progress_callback: progress_callback(60, "Building new vector database...")
                if os.path.exists(persist_directory):
                    self.vector_store = None
                    import gc
                    import stat
                    gc.collect()
                    time.sleep(0.5)
                    
                    def remove_readonly(func, path, excinfo):
                        os.chmod(path, stat.S_IWRITE)
                        func(path)
                        
                    try:
                        shutil.rmtree(persist_directory, onerror=remove_readonly)
                    except Exception:
                        pass
                
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
                        return False, f"Failed to create vector store: {e}"

                    if not self.vector_store:
                         return False, "Failed to initialize vector store (Unknown Error)."


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
                            except Exception:
                                retry_count += 1
                                time.sleep(2 * retry_count)

            # 4. Setup LLM & Chain
            if progress_callback: progress_callback(95, "Setting up LLM chains...")
            clean_model_name = model_name.replace("models/", "") if model_name.startswith("models/") else model_name
            self.llm = CustomGeminiChat(api_key=self.api_key, model=clean_model_name, temperature=temperature)
            
            # 5. Analyze Global Context
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

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        document_prompt = PromptTemplate(
            input_variables=["page_content", "source", "page"],
            template="Document: {source} | Page: {page}\nContent: {page_content}\n----------------"
        )

        retriever = self.vector_store.as_retriever(
            search_type="mmr", 
            search_kwargs={"k": 8, "fetch_k": 20}
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

    # =========================================================================
    # ENRICHMENT METHODS (Special handling - needs access to multiple mixins)
    # =========================================================================
    
    def enrich_variable_taxonomy(self, current_taxonomy, columns_info, distance=1, progress_callback=None):
        """
        Enriches an existing taxonomy using 'Wise Enrichment' strategy (Formula-centric).
        Returns (new_candidates, new_formulas, error) where new_candidates is a dict of proposed variables.
        Does NOT merge automatically.
        """
        if not self.initialized:
            return {}, {}, "RAG system not initialized."
            
        # Get existing variables for context
        existing_keys = set(current_taxonomy.keys())
        
        # Get format info from columns
        columns_str = ", ".join([c.get("name", str(c)) if isinstance(c, dict) else str(c) for c in columns_info])
        
        # Build variable context
        vars_desc_list = []
        for k, v in current_taxonomy.items():
            desc = f"{k}: {v.get('standard_name', '')} - {v.get('description', '')[:100]}"
            vars_desc_list.append(desc)
        vars_desc = "\n".join(vars_desc_list[:50])
        
        if progress_callback: progress_callback(10, "Retrieving formula context...")
        
        try:
            docs = self.vector_store.similarity_search("medical formulas clinical scores calculations", k=5)
            context_text = "\n\n".join([d.page_content for d in docs])
        except Exception:
            context_text = ""
        
        prompt = f"""
        Role: {self.current_role}
        Constraint: {self.adherence_guidance}
        
        Task: Suggest new computed variables that can be derived from the existing dataset.
        
        Existing Variables:
        {vars_desc}
        
        Context from Documents:
        {context_text[:3000]}
        
        Instructions:
        1. Identify standard medical formulas that use 2+ of the existing variables.
        2. Suggest at most {distance * 3} new variables.
        3. For each, provide formula using exact variable names from the list.
        
        Return JSON:
        {{
            "new_variables": {{
                "variable_id": {{
                    "standard_name": "...",
                    "description": "...",
                    "formula": "Python-syntax formula",
                    "input_variables": ["...", "..."],
                    "category": "...",
                    "node_type": "Derived-Internal"
                }}
            }},
            "new_formulas": {{
                "formula_id": {{
                    "name": "...",
                    "expression": "...",
                    "input_variables": ["...", "..."],
                    "output_variable": "..."
                }}
            }}
        }}
        """
        
        if progress_callback: progress_callback(50, "Generating enrichment suggestions...")
        
        try:
            response = self.llm.invoke(prompt)
            cleaned = self._clean_json_response(response.content)
            data = json.loads(cleaned)
            
            new_candidates = data.get("new_variables", {})
            new_formulas = data.get("new_formulas", {})
            
            # Set node types
            for v in new_candidates.values():
                v['node_type'] = v.get('node_type', 'Derived-Internal')
                v['role'] = v.get('node_type', 'derived-internal').lower()
            
            return new_candidates, new_formulas, None
            
        except Exception as e:
            return {}, {}, str(e)

    def refine_variable_taxonomy(self, current_taxonomy, feedback_dict, columns_info):
        """
        Refines specific variables in the taxonomy based on user feedback.
        """
        if not self.initialized:
            return current_taxonomy, "RAG system not initialized."
            
        updated_taxonomy = current_taxonomy.copy()
        variables_to_refine = [v for v in feedback_dict.keys() if v in updated_taxonomy]
        
        if not variables_to_refine:
            return updated_taxonomy, "No valid variables to refine."
            
        refinement_context = []
        for var in variables_to_refine:
            current_data = updated_taxonomy[var]
            feedback = feedback_dict[var]
            
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
        2. Update the taxonomy fields to address the feedback.
        3. Keep existing valid information if it doesn't conflict with the feedback.
        
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
            
            for var, new_data in refinements.items():
                if var in updated_taxonomy:
                    updated_taxonomy[var].update(new_data)
                    
            updated_taxonomy = self._merge_stats(updated_taxonomy, columns_info)
            self.variable_taxonomy = updated_taxonomy
            
            return updated_taxonomy, None
            
        except Exception as e:
            return updated_taxonomy, f"Refinement Eror: {str(e)}"

    def suggest_anomaly_criteria(self, description, columns, sample_data=None, mode="anomaly"):
        """
        Suggests anomaly or inclusion criteria based on natural language description.
        Returns a list of criteria dictionaries.
        """
        if not self.initialized:
            return [], "RAG system not initialized."
            
        columns_info = ", ".join(columns)
        
        # Format sample data as string if provided
        sample_str = ""
        if sample_data is not None:
            try:
                # If it's a dataframe
                if hasattr(sample_data, "to_markdown"):
                    sample_str = sample_data.head(3).to_markdown()
                    
                    # SMART DATE INFERENCE
                    # Iterate columns to find date-like strings and guess format
                    date_hints = []
                    for col in sample_data.columns:
                        if sample_data[col].dtype == 'object':
                            # check first non-null
                            head_vals = sample_data[col].dropna().head(5).astype(str).tolist()
                            if not head_vals: continue
                            
                            # Check for DD/MM/YYYY pattern
                            # If we see day > 12 at start, it's definitely DD/MM
                            is_dmy = any(re.match(r'(1[3-9]|2[0-9]|3[01])[-/]\d{2}[-/]\d{4}', v) for v in head_vals)
                            
                            if is_dmy:
                                date_hints.append(f"Column '{col}' appears to be DD/MM/YYYY. Use format='%d/%m/%Y'.")
                            else:
                                # Check generic date
                                try:
                                    pd.to_datetime(head_vals, dayfirst=True)
                                    # If no error and looks like a date, suggest dayfirst=True just in case
                                    if any(re.match(r'\d{2}[-/]\d{2}[-/]\d{4}', v) for v in head_vals):
                                         date_hints.append(f"Column '{col}' might be Day-First. Use dayfirst=True.")
                                except:
                                    pass
                    
                    if date_hints:
                        sample_str += "\n\nDate Parsing Hints:\n" + "\n".join(date_hints)
                        
                # If it's a list/dict
                elif isinstance(sample_data, (list, dict)):
                    sample_str = str(sample_data)
            except:
                sample_str = str(sample_data)

        prompt = anomaly_criteria_prompt(description, columns_info, sample_str, mode)
        
        try:
            response = self.llm.invoke(prompt)
            cleaned = self._clean_json_response(response.content)
            data = json.loads(cleaned)
            criteria_list = data.get("criteria", [])
            
            # Helper to extract variables from expression
            def extract_variables(expression):
                try:
                    tree = ast.parse(expression, mode='eval')
                    names = set()
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Name):
                            names.add(node.id)
                        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                            # Heuristic for df['Col'] where 'Col' is a string constant
                            names.add(node.value)
                    return names
                except:
                    return set()

            # Enhanced validation: Check for missing variables
            # Simple heuristic: Look for strings inside brackets ['...'] as potential column names
            for item in criteria_list:
                expr = item.get("expression", "")
                
                # Regex to find df['ColName'] or df["ColName"]
                found_cols = re.findall(r"df\[['\"](.*?)['\"]\]", expr)
                
                # Check for missing variables
                missing = [col for col in found_cols if col not in columns]
                
                # AUTO-CORRECTION ATTEMPT using fuzzy matching
                if missing:
                    still_missing = []
                    for m_col in missing:
                        # Find best match in existing columns
                        best_match, score, _ = process.extractOne(m_col, columns)
                        
                        # Threshold for auto-correction (e.g., 90% similarity)
                        if score >= 88:
                            # Auto-replace in expression
                            # Use regex substitutoin to avoid partial matches on other vars
                            # e.g. replacing 'Age' in 'Age_Group' -> risky, but here we replace explicit quoted string
                            # Simpler: string replace with quotes
                            expr = expr.replace(f"'{m_col}'", f"'{best_match}'").replace(f'"{m_col}"', f'"{best_match}"')
                            
                            # Add a note explaining the correction
                            if "explanation" in item:
                                item["explanation"] += f" (Auto-corrected '{m_col}' to '{best_match}')"
                        else:
                            still_missing.append(m_col)
                    
                    # Update expression in item
                    item["expression"] = expr
                    
                    # Only report variables that couldn't be auto-corrected
                    if still_missing:
                        item["missing_variables"] = still_missing
            
            return criteria_list, None
        except Exception as e:
            return [], str(e)
