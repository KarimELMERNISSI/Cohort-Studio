"""
Documents Mixin for RAGManager.

Contains all document processing, knowledge graph extraction and chat methods.
This is part of the RAGManager class composition pattern.
"""
import os

from manage.rag.providers import file_readers
from utils.logging_config import redact_secrets


class DocumentsMixin:
    """
    Mixin class containing document processing functionality for RAGManager.
    
    Methods:
        - get_available_documents
        - extract_custom_graph_from_doc
        - extract_merged_graph_from_docs
        - match_columns_to_graph
        - chat_with_specific_doc
    """

    def get_available_documents(self):
        """Returns a list of PDF files in the documents directory."""
        if not os.path.exists(self.documents_dir):
            return []
        return [f for f in os.listdir(self.documents_dir) if f.lower().endswith('.pdf')]

    def extract_custom_graph_from_doc(self, file_name, progress_callback=None):
        """
        Extract a Knowledge Graph of key entities and relationships from a specific document: sent as a file to a
        provider that declares ``sends_files`` (its connector's read_document), as its text to the others.
        """
        if not self.initialized:
            return None, "RAG System not initialized."
            
        file_path = os.path.join(self.documents_dir, file_name)
        if not os.path.exists(file_path):
            return None, f"File {file_name} not found."
            
        from prompts import document_graph
        prompt_text = document_graph()

        try:
            # The provider declares whether it is sent the PDF file itself (Gemini's file API, then a request to delete
            # it) or its text: never a files API the client merely has (Anthropic's has one, and a PDF uploaded there
            # stayed, though only the passages of a question are meant to reach Claude)
            if getattr(self, "provider", None) in file_readers():
                json_str = self.llm.read_document(
                    file_path, prompt_text, progress_callback, as_json=True, temperature=0.2
                )
            else:
                # Universal provider-agnostic text extraction (Ollama, OpenAI, Mistral)
                if progress_callback: progress_callback(20, f"Reading {file_name}...")
                from manage.rag import PDFDocumentLoader
                loader = PDFDocumentLoader(file_path)
                docs = loader.load_single_pdf(file_path)
                if not docs:
                    return None, f"Could not extract text from {file_name}."

                if progress_callback: progress_callback(40, "Extracting Knowledge Graph with AI...")
                doc_text = "\n\n".join([f"--- Page {d.metadata.get('page', 1)} ---\n{d.page_content}" for d in docs[:30]])
                
                full_prompt = f"{prompt_text}\n\nDocument text to analyze:\n{doc_text[:35000]}"
                response = self.llm.invoke(full_prompt)
                json_str = response.content if hasattr(response, 'content') else str(response)

            cleaned_json = self._clean_json_response(json_str)
            parsed_data = self._parse_json_safe(cleaned_json)
            
            if isinstance(parsed_data, list):
                if len(parsed_data) > 0 and isinstance(parsed_data[0], dict):
                    parsed_data = parsed_data[0]
                else:
                    return None, "LLM returned an unexpected list format."
            
            if not isinstance(parsed_data, dict):
                return None, "LLM did not return a valid JSON object."
                 
            return parsed_data, None
            
        except Exception as e:  # a provider's message: without the keys it can quote
            return None, redact_secrets(str(e))


    def extract_merged_graph_from_docs(self, doc_list, progress_callback=None, skipped=None):
        """
        Extracts and merges graphs from multiple documents.
        Entities are unified across documents and relation endpoints are remapped to the merged
        entities (graph.builders.document.merge_document_graphs). Returns (merged_json, error): an error when no
        document gave a graph. ``skipped``, a list, receives each document left out of the graph with the reason.
        """
        from graph.builders.document import merge_document_graphs

        if not doc_list:
            return None, "No documents specified."

        per_document, failed = [], []
        total = len(doc_list)
        for idx, doc in enumerate(doc_list):
            start, span = idx * 100 // total, max(1, 100 // total)
            if progress_callback:
                progress_callback(start, f"Processing {doc}...")

            def step(percent, message, doc=doc, start=start, span=span):
                # each document's progress, within its share of the bar
                if progress_callback:
                    progress_callback(min(99, start + percent * span // 100), f"{doc}: {message}")

            g_json, err = self.extract_custom_graph_from_doc(doc, progress_callback=step)
            if err or not isinstance(g_json, dict):
                failed.append((doc, err or "no graph was returned"))
                continue
            per_document.append((doc, g_json))

        if skipped is not None:
            skipped.extend(failed)
        if not per_document:
            return None, "No graph could be extracted: " + "; ".join(f"{doc}: {err}" for doc, err in failed)
        merged = merge_document_graphs(per_document)
        return merged.to_json_dict(), None

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
            
        node_lookup = {n.get("id", "").lower(): n for n in nodes}
        node_ids_lower = list(node_lookup.keys())
        
        for col in columns:
            col_lower = col.lower().replace("_", " ")
            
            if col_lower in node_lookup:
                results[col] = {
                    "match_found": True, 
                    "node": node_lookup[col_lower], 
                    "confidence": 1.0,
                    "method": "Exact"
                }
                continue
                
            matches = difflib.get_close_matches(col_lower, node_ids_lower, n=1, cutoff=0.6)
            if matches:
                match_id = matches[0]
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
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.runnables import RunnablePassthrough
        
        if not self.initialized or not self.vector_store:
            return "System not initialized or no database available."

        if isinstance(doc_names, str):
            doc_names = [doc_names]
            
        full_paths = [os.path.join(self.documents_dir, d) for d in doc_names]

        # Engine-agnostic filter: each vector database translates the list of sources itself.
        retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 20, "sources": full_paths},
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
            return "\n\n".join(f"[Source: {d.metadata.get('source','Unknown')}] {d.page_content}" for d in docs)

        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        try:
            return chain.invoke(query)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error("RAG Chat Error: %s", e, exc_info=True)
            return f"System Error: {e!s}"
