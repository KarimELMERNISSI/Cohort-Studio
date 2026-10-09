"""
RAG Monitoring Dashboard

Streamlit page for monitoring and visualizing RAG system quality metrics.
Provides dashboards for:
- Quality overview with trends
- Evaluation details and drill-down
- Embedding health visualization
- Manual evaluation triggers
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


def render_rag_monitoring():
    """Main entry point for the RAG Monitoring page."""
    st.title("RAG Quality Monitor")
    st.markdown("Monitor and assess the quality of RAG outputs over time.")
    
    # Check if evaluator exists
    try:
        from manage.rag_evaluator import RAGEvaluator
        evaluator = RAGEvaluator()
    except ImportError:
        st.error("RAG Evaluator module not found. Please ensure `manage/rag_evaluator.py` exists.")
        return
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Quality Overview",
        "📋 Evaluation Details", 
        "🧬 Embedding Health",
        "🧪 Run Evaluation"
    ])
    
    with tab1:
        render_quality_overview(evaluator)
    
    with tab2:
        render_evaluation_details(evaluator)
    
    with tab3:
        render_embedding_health(evaluator)
    
    with tab4:
        render_run_evaluation(evaluator)


def render_quality_overview(evaluator):
    """Render the quality overview dashboard."""
    st.subheader("Quality Overview")
    
    summary = evaluator.get_summary()
    
    if not summary or summary.get("total_evaluations", 0) == 0:
        st.info("No evaluations recorded yet. Run some evaluations to see metrics.")
        return
    
    # Key metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = summary.get("metrics", {})
    
    with col1:
        context_rel = metrics.get("context_relevance", {}).get("avg", 0)
        st.metric(
            "Context Relevance",
            f"{context_rel:.1f}/10",
            delta=None,
            help="Average relevance of retrieved context to queries"
        )
    
    with col2:
        faithfulness = metrics.get("faithfulness", {}).get("avg", 0)
        st.metric(
            "Faithfulness",
            f"{faithfulness:.1f}/10",
            help="How well responses are grounded in context"
        )
    
    with col3:
        answer_rel = metrics.get("answer_relevance", {}).get("avg", 0)
        st.metric(
            "Answer Relevance",
            f"{answer_rel:.1f}/10",
            help="How well responses address the original query"
        )
    
    with col4:
        json_rate = metrics.get("json_validity", {}).get("rate", 0) * 100
        st.metric(
            "JSON Success Rate",
            f"{json_rate:.0f}%",
            help="Percentage of responses with valid JSON"
        )
    
    st.divider()
    
    # Trend chart
    if PLOTLY_AVAILABLE and PANDAS_AVAILABLE:
        st.subheader("Quality Trend (Last 7 Days)")
        
        metrics_by_date = evaluator.get_metrics_by_date(days=7)
        
        if metrics_by_date:
            dates = sorted(metrics_by_date.keys())
            scores = [metrics_by_date[d]["avg_score"] for d in dates]
            counts = [metrics_by_date[d]["count"] for d in dates]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=scores,
                mode='lines+markers',
                name='Avg Score',
                line=dict(color='#4CAF50', width=2),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                yaxis_title="Average Score (0-10)",
                xaxis_title="Date",
                height=300,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("Not enough data for trend chart yet.")
    
    st.divider()
    
    # Metrics by function
    st.subheader("Performance by Function")
    
    by_function = summary.get("by_function", {})
    
    if by_function:
        if PANDAS_AVAILABLE:
            df = pd.DataFrame([
                {
                    "Function": func,
                    "Evaluations": data["count"],
                    "Avg Score": round(data["avg_score"], 2)
                }
                for func, data in by_function.items()
            ])
            st.dataframe(df, width='stretch', hide_index=True)
        else:
            for func, data in by_function.items():
                st.write(f"**{func}**: {data['count']} evaluations, Avg: {data['avg_score']:.2f}")
    
    # Hallucination rate warning
    hall_rate = metrics.get("hallucination_rate", {}).get("rate", 0)
    if hall_rate > 0.1:
        st.warning(f"Hallucination rate is {hall_rate*100:.1f}%. Consider reviewing prompts.")


def render_evaluation_details(evaluator):
    """Render the evaluation details view."""
    st.subheader("Evaluation Details")
    
    # Filters
    col1, col2 = st.columns(2)
    with col1:
        limit = st.selectbox("Show last", [10, 25, 50, 100], index=1)
    with col2:
        min_score = st.slider("Minimum score", 0.0, 10.0, 0.0, 0.5)
    
    evaluations = evaluator.get_recent_evaluations(limit=limit)
    
    if not evaluations:
        st.info("No evaluations recorded yet.")
        return
    
    # Filter by score
    evaluations = [e for e in evaluations if e.get("overall_score", 0) >= min_score]
    
    if PANDAS_AVAILABLE:
        # Create summary table
        df = pd.DataFrame([
            {
                "ID": e.get("eval_id", "")[:8],
                "Function": e.get("function_name", "")[:20],
                "Query": e.get("query_preview", "")[:40] + "...",
                "Context": e.get("context_relevance", 0),
                "Faithful": e.get("faithfulness", 0),
                "Relevance": e.get("answer_relevance", 0),
                "Overall": round(e.get("overall_score", 0), 2),
                "Time": e.get("timestamp", "")[:19]
            }
            for e in evaluations
        ])
        
        st.dataframe(
            df,
            width='stretch',
            hide_index=True,
            column_config={
                "Context": st.column_config.ProgressColumn(min_value=0, max_value=10),
                "Faithful": st.column_config.ProgressColumn(min_value=0, max_value=10),
                "Relevance": st.column_config.ProgressColumn(min_value=0, max_value=10),
            }
        )
    
    # Expandable details
    st.subheader("Detailed View")
    
    for i, eval_result in enumerate(evaluations[:5]):
        with st.expander(f"#{i+1}: {eval_result.get('function_name', 'unknown')} - Score: {eval_result.get('overall_score', 0):.1f}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Query:**")
                st.text(eval_result.get("query_preview", ""))
                
                st.markdown("**Scores:**")
                st.write(f"- Context Relevance: {eval_result.get('context_relevance', 'N/A')}")
                st.write(f"- Faithfulness: {eval_result.get('faithfulness', 'N/A')}")
                st.write(f"- Answer Relevance: {eval_result.get('answer_relevance', 'N/A')}")
            
            with col2:
                st.markdown("**Reasoning:**")
                st.text(eval_result.get("faithfulness_reasoning", "")[:200])
                
                if eval_result.get("has_hallucination"):
                    st.error("Hallucination detected!")
                    claims = eval_result.get("hallucination_claims", [])
                    for claim in claims[:3]:
                        st.write(f"- {claim}")


def render_embedding_health(evaluator):
    """Render embedding health visualization."""
    st.subheader("Embedding Health")
    
    st.info("Embedding visualization requires a connected RAG system with embeddings.")
    
    # Test suite results
    st.subheader("Retrieval Test Suite")
    
    test_cases = evaluator.load_test_suite()
    
    if test_cases:
        st.write(f"**{len(test_cases)} test cases** in the test suite")
        
        if PANDAS_AVAILABLE:
            df = pd.DataFrame([
                {
                    "ID": t.get("id", ""),
                    "Query": t.get("query", "")[:50],
                    "Expected Concepts": ", ".join(t.get("expected_concepts", [])[:3])
                }
                for t in test_cases
            ])
            st.dataframe(df, width='stretch', hide_index=True)
    else:
        st.info("No test cases defined yet. Add test cases to track embedding quality.")
    
    # Add test case form
    with st.expander("Add Test Case"), st.form("add_test_case"):
        query = st.text_input("Query")
        concepts = st.text_input("Expected Concepts (comma-separated)")
        ground_truth = st.text_area("Ground Truth Answer (optional)")
        
        if st.form_submit_button("Add Test Case"):
            if query and concepts:
                concept_list = [c.strip() for c in concepts.split(",")]
                test_id = evaluator.add_test_case(
                    query=query,
                    expected_concepts=concept_list,
                    ground_truth=ground_truth or None
                )
                st.success(f"Added test case: {test_id}")
                st.rerun()
            else:
                st.error("Query and concepts are required")


def render_run_evaluation(evaluator):
    """Render the manual evaluation trigger with progress tracking."""
    st.subheader("Run Evaluation")
    
    st.markdown("""
    Run manual evaluations to assess RAG quality. This will:
    1. Execute test queries against the RAG system
    2. Calculate quality metrics using LLM-as-judge
    3. Log results for tracking
    """)
    
    # Check for RAG manager
    rag_available = False
    rag = None
    if "rag_manager" in st.session_state and st.session_state.rag_manager:
        rag = st.session_state.rag_manager
        rag_available = rag.initialized
    
    if not rag_available:
        st.warning("RAG system not initialized. Please initialize it first from the Data Insight page.")
        
        # Demo mode option
        st.info("You can run a demo evaluation without the full RAG system to test the monitoring pipeline.")
        if st.button("Run Demo Evaluation"):
            run_demo_evaluation(evaluator)
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Quick Evaluation")
        
        # Function selector
        eval_function = st.selectbox(
            "Select RAG Function to Test",
            ["suggest_computed_variables", "suggest_column_renaming", "query (Q&A)", "custom"],
            help="Choose which RAG function to evaluate"
        )
        
        with st.form("quick_eval"):
            if eval_function == "custom":
                query = st.text_input("Test Query", placeholder="Enter your query...")
            else:
                query = st.text_input("Search Hint / Query", placeholder="e.g., BMI, cardiovascular risk")
            
            run_eval = st.form_submit_button("▶️ Run Evaluation", type="primary")
            
            if run_eval and query:
                run_single_evaluation(evaluator, rag, query, eval_function)
    
    with col2:
        st.markdown("### Run Test Suite")
        
        test_cases = evaluator.load_test_suite()
        
        if test_cases:
            st.write(f"**{len(test_cases)} test cases** available")
            
            # Select which tests to run
            run_all = st.checkbox("Run all tests", value=True)
            if not run_all:
                selected = st.multiselect(
                    "Select tests",
                    [t["id"] for t in test_cases],
                    default=[test_cases[0]["id"]] if test_cases else []
                )
            else:
                selected = [t["id"] for t in test_cases]
            
            if st.button("Run Test Suite", type="secondary"):
                run_test_suite_with_progress(evaluator, rag, test_cases, selected)
        else:
            st.info("No test cases defined. Add test cases in the Embedding Health tab.")
    
    st.divider()
    
    # Export options
    render_export_options(evaluator)


def run_single_evaluation(evaluator, rag, query: str, eval_function: str):
    """Run a single evaluation with progress tracking."""
    progress_bar = st.progress(0, text="Initializing evaluation...")
    status_text = st.empty()
    
    try:
        # Step 1: Set up evaluator
        progress_bar.progress(10, text="Setting up evaluator...")
        evaluator.llm = rag.llm
        
        # Step 2: Retrieve context
        progress_bar.progress(25, text="Retrieving context from documents...")
        status_text.info("📚 Searching documents...")
        
        try:
            docs = rag.vector_store.similarity_search(query, k=3)
            context = "\n\n".join([d.page_content for d in docs])
            retrieved_chunks = [d.page_content for d in docs]
        except Exception:
            context = "No context retrieved"
            retrieved_chunks = []
        
        # Step 3: Generate response based on function
        progress_bar.progress(45, text=f"Running {eval_function}...")
        status_text.info(f"🤖 Executing {eval_function}...")
        
        if eval_function == "suggest_computed_variables":
            # Use actual RAG function
            columns = list(st.session_state.get("df", {}).keys())[:10] if "df" in st.session_state else ["weight", "height", "age"]
            response_json, error = rag.suggest_computed_variables(columns, search_hint=query, num_suggestions=3)
            response = str(response_json) if response_json else str(error)
        elif eval_function == "suggest_column_renaming":
            columns = list(st.session_state.get("df", {}).keys())[:10] if "df" in st.session_state else ["sbp", "dbp", "hr"]
            response_json, error = rag.suggest_column_renaming(columns)
            response = str(response_json) if response_json else str(error)
        else:
            # Generic Q&A
            response = rag.llm.invoke(f"Context: {context[:2000]}\n\nQuestion: {query}").content
        
        # Step 4: Evaluate context relevance
        progress_bar.progress(60, text="Evaluating context relevance...")
        status_text.info("🔍 Assessing context quality...")
        
        # Step 5: Evaluate faithfulness
        progress_bar.progress(75, text="Checking faithfulness...")
        status_text.info("✅ Checking response groundedness...")
        
        # Step 6: Run full evaluation
        progress_bar.progress(85, text="Computing final scores...")
        result = evaluator.evaluate(
            query=query,
            context=context,
            response=response,
            function_name=eval_function,
            retrieved_chunks=retrieved_chunks
        )
        
        # Step 7: Log and display results
        progress_bar.progress(95, text="Logging results...")
        evaluator.log_result(result)
        
        progress_bar.progress(100, text="Evaluation complete!")
        status_text.empty()
        
        # Display results
        st.success(f"Evaluation complete! Overall score: **{result.get('overall_score', 0):.1f}/10**")
        
        # Score breakdown
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Context Relevance", f"{result.get('context_relevance', 0)}/10")
        with col2:
            st.metric("Faithfulness", f"{result.get('faithfulness', 0)}/10")
        with col3:
            st.metric("Answer Relevance", f"{result.get('answer_relevance', 0)}/10")
        
        # Detailed results in expander
        with st.expander("View Full Evaluation Details"):
            st.json(result)
            
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Evaluation failed: {e!s}")


def run_test_suite_with_progress(evaluator, rag, test_cases: list, selected_ids: list):
    """Run test suite with progress tracking."""
    # Filter to selected tests
    tests_to_run = [t for t in test_cases if t["id"] in selected_ids]
    
    if not tests_to_run:
        st.warning("No tests selected")
        return
    
    evaluator.llm = rag.llm
    
    progress_bar = st.progress(0, text="Starting test suite...")
    results_container = st.container()
    
    results = []
    passed = 0
    failed = 0
    
    for i, test in enumerate(tests_to_run):
        progress = (i + 1) / len(tests_to_run)
        progress_bar.progress(progress, text=f"Running test {i+1}/{len(tests_to_run)}: {test['id']}")
        
        try:
            # Retrieve context
            docs = rag.vector_store.similarity_search(test["query"], k=3)
            context = "\n\n".join([d.page_content for d in docs])
            
            # Generate response
            response = rag.llm.invoke(f"Context: {context[:2000]}\n\nQuestion: {test['query']}").content
            
            # Evaluate
            result = evaluator.evaluate(
                query=test["query"],
                context=context,
                response=response,
                function_name=f"test_{test['id']}",
                retrieved_chunks=[d.page_content for d in docs]
            )
            
            # Check if concepts are in response
            expected = test.get("expected_concepts", [])
            found = sum(1 for c in expected if c.lower() in response.lower())
            concept_score = found / len(expected) if expected else 1
            
            result["concept_coverage"] = concept_score
            result["test_id"] = test["id"]
            
            # Log result
            evaluator.log_result(result)
            
            if result.get("overall_score", 0) >= 5 and concept_score >= 0.5:
                passed += 1
                result["status"] = "PASSED"
            else:
                failed += 1
                result["status"] = "FAILED"
            
            results.append(result)
            
        except Exception as e:
            failed += 1
            results.append({
                "test_id": test["id"],
                "status": "ERROR",
                "error": str(e)
            })
    
    progress_bar.progress(1.0, text="Test suite complete!")
    
    # Display results
    with results_container:
        st.markdown(f"### Results: {passed} passed, {failed} failed")
        
        if PANDAS_AVAILABLE:
            df = pd.DataFrame([
                {
                    "Test": r.get("test_id", ""),
                    "Status": r.get("status", ""),
                    "Score": r.get("overall_score", 0),
                    "Concepts": f"{r.get('concept_coverage', 0)*100:.0f}%" if "concept_coverage" in r else "N/A"
                }
                for r in results
            ])
            st.dataframe(df, width='stretch', hide_index=True)


def run_demo_evaluation(evaluator):
    """Run a demo evaluation without the full RAG system."""
    progress_bar = st.progress(0, text="Running demo evaluation...")
    
    # Simulated data
    demo_data = {
        "query": "What is Body Mass Index and how is it calculated?",
        "context": "Body Mass Index (BMI) is a measure of body fat based on height and weight. It is calculated as weight in kilograms divided by height in meters squared (kg/m²).",
        "response": '{"name": "bmi", "formula": "weight / (height ** 2)", "description": "Body Mass Index calculation"}'
    }
    
    progress_bar.progress(30, text="Simulating retrieval...")
    import time
    time.sleep(0.5)
    
    progress_bar.progress(60, text="Simulating generation...")
    time.sleep(0.5)
    
    progress_bar.progress(90, text="Computing demo scores...")
    
    # Create demo result
    result = {
        "eval_id": "demo_001",
        "timestamp": datetime.now().isoformat(),
        "function_name": "demo_test",
        "query_preview": demo_data["query"][:100],
        "context_relevance": 8.5,
        "faithfulness": 9.0,
        "answer_relevance": 8.0,
        "json_validity": {"is_valid": True},
        "has_hallucination": False,
        "overall_score": 8.5
    }
    
    evaluator.log_result(result)
    
    progress_bar.progress(100, text="Demo complete!")
    
    st.success(f"Demo evaluation logged! Score: **{result['overall_score']}/10**")
    st.info("This was a simulated evaluation. Initialize the RAG system for real evaluations.")


def render_export_options(evaluator):
    """Render export buttons."""
    st.subheader("Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        summary = evaluator.get_summary()
        st.download_button(
            label="📥 Download Summary (JSON)",
            data=json.dumps(summary, indent=2),
            file_name=f"rag_eval_summary_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )
    
    with col2:
        evaluations = evaluator.get_recent_evaluations(limit=1000)
        st.download_button(
            label="📥 Download All Evaluations",
            data=json.dumps(evaluations, indent=2),
            file_name=f"rag_evaluations_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )



# Entry point for Streamlit multipage app
if __name__ == "__main__":
    render_rag_monitoring()
