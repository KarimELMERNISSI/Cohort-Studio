"""
RAG Manager - Core Module.

This is the main RAGManager class that composes functionality from mixin classes:
- TaxonomyMixin: Variable taxonomy generation and enrichment
- DocumentsMixin: Document processing and knowledge graph extraction
- ComputedVarsMixin: Computed variable suggestions

The mixin pattern allows splitting a large class into focused modules
while maintaining a single class interface for existing code.
"""
import ast
import json
import os
import re
from pathlib import Path

import pandas as pd
from rapidfuzz import process

# Import prompt functions
from prompts import (
    anomaly_criteria_prompt,
    context_analysis,
    refine_taxonomy,
    wise_enrichment,
)

# Import LLM utilities for reliable parsing
from utils.llm_utils import (
    StructuredOutputHelper,
    clean_json_response,
    parse_json_safe,
)

from .rag_computed_vars import ComputedVarsMixin
from .rag_documents import DocumentsMixin

# Import mixins
from .rag_taxonomy import TaxonomyMixin

CLOUD_PROVIDERS = {"gemini", "openai", "mistral", "anthropic"}
_ENV_KEY_NAMES = {
    "gemini": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}
# Providers that answer but have no embedding API: documents are embedded by another provider.
CHAT_ONLY_PROVIDERS = {"anthropic"}


def _env_api_key(provider: str | None) -> str | None:
    """Server-configured API key for a cloud provider; used only when the user typed none."""
    env_name = _ENV_KEY_NAMES.get((provider or "").lower())
    if not env_name:
        return None
    return os.getenv(env_name) or None

# Try importing RAG dependencies
try:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    from manage.rag import (
        ConnectorFactory,
        PDFDocumentLoader,
        ProviderConfig,
        ProviderType,
    )
    from manage.vectorstores import (
        DocumentIndex,
        VectorEngine,
        VectorStoreError,
        VectorStoreSettings,
        index_matches,
        open_index,
        probe_machine,
        recommend,
        resolve_engine,
    )
    RAG_AVAILABLE = True
except ImportError as e:
    RAG_AVAILABLE = False
    MISSING_LIBS_ERROR = str(e)



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
    
    def __init__(self, documents_dir=None, api_key=None, provider="gemini", base_url=None):
        if documents_dir is None:
            # Per-user document library (audit E4); the legacy shared folder is used outside a session.
            from utils.session_scope import get_current_scope

            scope = get_current_scope()
            documents_dir = str(scope.documents_dir()) if scope else "DOCUMENTS"
        self.documents_dir = documents_dir
        self.api_key = api_key
        self.provider = provider
        self.base_url = base_url
        self.client = None
        self.llm = None
        self.vector_store = None  # manage.vectorstores.DocumentIndex once initialised
        self.vector_engine: str | None = None  # "lancedb" | "qdrant"
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
        
        # Explicit per-session confirmation is required before data may be sent to a cloud provider.
        self.cloud_consent = False

    def is_available(self):
        """Check if RAG dependencies are available."""
        return RAG_AVAILABLE

    def get_available_models(self, provider: str = "gemini", base_url: str = None):
        """Get list of available models for the specified provider."""
        try:
            cfg = ProviderConfig(
                provider=ProviderType(provider.lower()),
                api_key=self.api_key or _env_api_key(provider),
                base_url=base_url or self.base_url,
            )
            connector = ConnectorFactory.get_llm_connector(cfg)
            return connector.get_available_models()
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

    def initialize_system(
        self,
        model_name="gemini-2.5-flash",
        adherence_score=0.5,
        temperature=0.3,
        dataset_columns=None,
        use_existing_db=False,
        progress_callback=None,
        selected_files=None,
        embedding_model="models/gemini-embedding-2",
        provider="gemini",
        base_url=None,
        vector_engine=None,
        qdrant_url=None,
        qdrant_api_key=None,
        embedding_provider=None,
        embedding_api_key=None,
        embedding_base_url=None,
    ):
        """Initialize the RAG system with documents, embeddings, the vector database and the LLM.

        ``embedding_provider`` embeds the documents when it differs from the answering ``provider``; Claude
        (``anthropic``) has no embedding API and uses local Ollama unless another provider is given.

        ``vector_engine`` is "lancedb" or "qdrant" (None: the server default ``CMP_VECTOR_STORE``, else
        the engine recommended for this machine); ``qdrant_url`` selects a Qdrant server instead of
        the embedded mode (default ``CMP_QDRANT_URL``).
        """
        if not self.is_available():
            return False, f"Missing dependencies: {MISSING_LIBS_ERROR}. Please install required packages."
        
        provider_clean = (provider or "gemini").lower()
        self.provider = provider_clean
        self.base_url = base_url

        effective_key = self.api_key or _env_api_key(provider_clean)
        embed_provider = (
            embedding_provider or ("ollama" if provider_clean in CHAT_ONLY_PROVIDERS else provider_clean)
        ).lower()
        if embed_provider in CHAT_ONLY_PROVIDERS:
            return False, f"{embed_provider.capitalize()} has no embedding API: choose another embedding provider."
        separate_embeddings = embed_provider != provider_clean
        embed_key = embedding_api_key or (_env_api_key(embed_provider) if separate_embeddings else effective_key)
        self.embedding_provider = embed_provider
        if (provider_clean in CLOUD_PROVIDERS or embed_provider in CLOUD_PROVIDERS) and not getattr(
            self, "cloud_consent", False
        ):
            return False, (
                "Cloud providers require explicit confirmation that data may leave this server. "
                "Confirm the data-sharing notice in Settings > AI engine first."
            )
        if provider_clean in CLOUD_PROVIDERS and not effective_key:
            return False, f"{provider_clean.capitalize()} API Key is required."
        if separate_embeddings and embed_provider in CLOUD_PROVIDERS and not embed_key:
            return False, f"{embed_provider.capitalize()} API Key is required for the embeddings."

        try:
            # 1. Load Documents using native PDFDocumentLoader
            if progress_callback: progress_callback(10, "Loading documents...")
            if not os.path.exists(self.documents_dir):
                os.makedirs(self.documents_dir)
                return False, f"Documents directory '{self.documents_dir}' created. Please add PDF files."

            loader = PDFDocumentLoader(self.documents_dir)
            if selected_files:
                documents = []
                for file_name in selected_files:
                    file_path = None
                    for root, dirs, files in os.walk(self.documents_dir):
                        if file_name in files:
                            file_path = os.path.join(root, file_name)
                            break
                    if file_path:
                        documents.extend(loader.load_single_pdf(file_path))
            else:
                documents = loader.load(recursive=True)
            
            if not documents:
                return False, "No PDF documents found in the DOCUMENTS folder (or none selected)."

            # 2. Split Text
            if progress_callback: progress_callback(30, "Splitting text...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(documents)

            # 3. Create Connectors (Embeddings & LLM)
            if progress_callback: progress_callback(50, "Initializing connectors & embeddings...")
            self.llm, embeddings = ConnectorFactory.create_connectors(
                provider=provider_clean,
                model_name=model_name,
                embedding_model=embedding_model,
                api_key=effective_key,
                base_url=base_url,
                temperature=temperature,
                embedding_provider=embed_provider if separate_embeddings else None,
                embedding_api_key=embed_key if separate_embeddings else None,
                embedding_base_url=embedding_base_url if separate_embeddings else None,
            )
            
            # 4. Vector database (engine-agnostic: manage.vectorstores)
            from utils.data_paths import get_vectors_dir
            from utils.session_scope import get_current_scope

            _scope = get_current_scope()
            base_dir = _scope.vectors_dir() if _scope else Path(get_vectors_dir())  # per-user indexes
            settings = VectorStoreSettings.from_env()
            url = qdrant_url or settings.qdrant_url
            api_key = qdrant_api_key or settings.qdrant_api_key
            advice = recommend(probe_machine(base_dir), chunks=len(texts), qdrant_server=bool(url))
            engine = resolve_engine(vector_engine, settings, advice)
            embedding_label = f"{embed_provider}:{embedding_model}"
            try:
                backend = open_index(
                    engine,
                    base_dir=base_dir,
                    username=_scope.username if _scope else None,
                    qdrant_url=url if engine is VectorEngine.QDRANT else None,
                    qdrant_api_key=api_key,
                )
                index = DocumentIndex(backend, embeddings)
                # the documents of this initialisation; an index built from other ones (a document added to or
                # removed from the library, another selection) is rebuilt instead of reused
                library = sorted({os.path.basename(str(t.metadata.get("source", ""))) for t in texts})
                existing = use_existing_db and backend.exists() and index.count() > 0
                if existing and index_matches(index.manifest, library):
                    if progress_callback: progress_callback(60, f"Loading the existing {engine.value} index...")
                    index.check_compatible(embedding_label)
                else:
                    if progress_callback:
                        changed = "The documents changed since the index was built: rebuilding" if existing else "Building"
                        progress_callback(60, f"{changed} the {engine.value} index...")
                    cloud = embed_provider in CLOUD_PROVIDERS  # cloud embedding APIs rate-limit bursts

                    def _embedding_progress(done, total):
                        if progress_callback:
                            progress_callback(60 + int(30 * done / max(total, 1)), f"Embedding chunks {done}/{total}...")

                    index.build(
                        texts,
                        embedding_label=embedding_label,
                        documents=library,
                        batch_size=8 if cloud else 16,
                        pause=0.5 if cloud else 0.0,  # cloud embedding APIs rate-limit bursts
                        progress=_embedding_progress,
                    )
            except VectorStoreError as e:
                return False, str(e)
            except Exception as e:
                return False, f"Vector database error ({engine.value}): {e}"
            self.vector_store = index
            self.vector_engine = engine.value

            # 4. Analyze Global Context
            if progress_callback: progress_callback(95, "Setting up LLM chains...")
            self.global_context = self._analyze_global_context(texts, dataset_columns)

            # 5. Initialize Roles and QA Chain
            self._update_internal_state(adherence_score)
            
            self.initialized = True
            return True, (
                f"RAG System Initialized Successfully ({provider_clean.upper()}, {engine.value} index). "
                f"Processed {len(documents)} pages and {len(texts)} chunks; the index holds {index.count()} chunks."
            )

        except Exception as e:
            return False, f"Error initializing RAG: {e!s}"


    def update_adherence_score(self, adherence_score):
        """Updates the adherence score and rebuilds the chain without full re-initialization."""
        if not self.initialized or not self.vector_store or not hasattr(self, 'llm'):
            return False, "RAG system not fully initialized."
        
        try:
            self._update_internal_state(adherence_score)
            return True, "Adherence score updated."
        except Exception as e:
            return False, f"Error updating adherence score: {e!s}"

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
        
        prompt = wise_enrichment(
            level=distance,
            rag_context=context_text[:3000],
            vars_context=vars_desc,
            adherence_instruction=self.adherence_guidance,
        )
        
        if progress_callback: progress_callback(50, "Generating enrichment suggestions...")
        
        try:
            response = self.llm.invoke(prompt)
            cleaned = self._clean_json_response(response.content)
            data = json.loads(cleaned)
            
            new_candidates = data.get("new_variables") or data.get("variables", {})
            new_formulas = data.get("new_formulas") or data.get("formulas", {})
            
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
        
        prompt = refine_taxonomy(context_block=context_block)
        
        try:
            response = self.llm.invoke(prompt)
            cleaned_response = self._clean_json_response(response.content)
            try:
                refinements = json.loads(cleaned_response)
            except json.JSONDecodeError as e:
                return updated_taxonomy, f"JSON Error in refinement: {e!s}"
            
            for var, new_data in refinements.items():
                if var in updated_taxonomy:
                    updated_taxonomy[var].update(new_data)
                    
            updated_taxonomy = self._merge_stats(updated_taxonomy, columns_info)
            self.variable_taxonomy = updated_taxonomy
            
            return updated_taxonomy, None
            
        except Exception as e:
            return updated_taxonomy, f"Refinement Error: {e!s}"

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
