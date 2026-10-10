"""
Documents Mixin for RAGManager.

Contains all document processing, knowledge graph extraction and chat methods.
This is part of the RAGManager class composition pattern.
"""
import os
import time


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
        Extract a Knowledge Graph of key entities and relationships from a specific document.
        Supports both Gemini File API and universal text-based LLM extraction (Ollama, OpenAI, Mistral).
        """
        if not self.initialized:
            return None, "RAG System not initialized."
            
        file_path = os.path.join(self.documents_dir, file_name)
        if not os.path.exists(file_path):
            return None, f"File {file_name} not found."
            
        from prompts import document_graph
        prompt_text = document_graph()

        try:
            # Check if Gemini File API is available on the client
            has_gemini_file_api = (
                hasattr(self, 'llm') and 
                hasattr(self.llm, 'client') and 
                self.llm.client is not None and 
                hasattr(self.llm.client, 'files')
            )

            if has_gemini_file_api:
                from google.genai import types
                if progress_callback: progress_callback(10, f"Uploading {file_name} to Gemini...")
                
                client = self.llm.client
                with open(file_path, "rb"):
                    uploaded_file = client.files.upload(file=file_path)
                
                while uploaded_file.state.name == "PROCESSING":
                    if progress_callback: progress_callback(20, "Processing file...")
                    time.sleep(2)
                    uploaded_file = client.files.get(name=uploaded_file.name)
                    
                if uploaded_file.state.name == "FAILED":
                    return None, "File processing failed by Google."
                    
                if progress_callback: progress_callback(40, "Generating Knowledge Graph (Deep Analysis)...")
                
                clean_model = getattr(self.llm, "model_name", "gemini-1.5-flash")
                clean_model = clean_model.removeprefix("models/")

                response = client.models.generate_content(
                    model=clean_model,
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
                
                try:
                    client.files.delete(name=uploaded_file.name)
                except Exception:
                    pass

                json_str = response.text
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
            
        except Exception as e:
            return None, str(e)


    def extract_merged_graph_from_docs(self, doc_list, progress_callback=None):
        """
        Extracts and merges graphs from multiple documents.
        Returns (merged_json, error).
        """
        import difflib
        
        if not doc_list:
            return None, "No documents specified."
            
        merged_nodes = {}
        merged_edges = []
        merged_formulas = []
        merged_summaries = []
        
        seen_edges = set()
        
        total = len(doc_list)
        
        for idx, doc in enumerate(doc_list):
            if progress_callback: progress_callback(int((idx/total)*100), f"Processing {doc}...")
            
            g_json, err = self.extract_custom_graph_from_doc(doc)
            if err:
                continue
            
            if "summary" in g_json:
                s = g_json["summary"]
                s["doc"] = doc
                merged_summaries.append(s)

            for n in g_json.get("nodes", []):
                nid = n.get("id", "").strip()
                if not nid: continue
                nid_lower = nid.lower()
                
                page_ref = n.get("page_reference", "Unknown")
                if isinstance(page_ref, list):
                    page_str = ", ".join(str(p) for p in page_ref)
                else:
                    page_str = str(page_ref) if page_ref is not None else "Unknown"

                citation = {
                    "doc": doc,
                    "page": page_str,
                    "text": n.get("source_text", "")
                }
                
                match_id = None
                
                if nid_lower in merged_nodes:
                    match_id = nid_lower
                else:
                    existing_keys = list(merged_nodes.keys())
                    matches = difflib.get_close_matches(nid_lower, existing_keys, n=1, cutoff=0.85)
                    if matches:
                        match_id = matches[0]
                
                if not match_id:
                    n["citations"] = [citation]
                    n.pop("source_text", None)
                    n.pop("page_reference", None)
                    merged_nodes[nid_lower] = n
                else:
                    existing = merged_nodes[match_id]
                    existing["citations"].append(citation)
                    if len(n.get("description", "")) > len(existing.get("description", "")):
                        existing["description"] = n.get("description", "")
            
            for e in g_json.get("edges", []):
                s = str(e.get("source"))
                t = str(e.get("target"))
                r = e.get("relation", "relates to")
                key = (s.lower(), t.lower(), r.lower())
                
                if key not in seen_edges:
                    merged_edges.append(e)
                    seen_edges.add(key)
            
            for f in g_json.get("formulas", []):
                f["doc"] = doc
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
        
        filter_dict = {}
        if len(full_paths) == 1:
            filter_dict = {"source": full_paths[0]}
        else:
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
