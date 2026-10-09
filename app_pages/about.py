import streamlit as st
from PIL import Image

def app():
    # --- Header Section ---
    col1, col2 = st.columns([1, 4])
    
    with col1:
        st.image("assets/karim-app-logo.png", width=140)
        
    with col2:
        st.title("Cohort Monitoring Package")
        st.markdown("**A unified environment for integrated research data management.**")

    st.divider()

    # --- Vision Section ---
    st.markdown("""
    ### 🚀 Vision
    
    This application bridges the gap between complex data science and medical research. 
    It offers a **professional-grade environment** that empowers researchers to streamline the entire data lifecycle—from raw ingestion to publication-ready analysis.
    """)

    st.markdown("")  # Spacer

    # --- Key Features Grid ---
    col1, col2 = st.columns(2)

    with col1:
        st.info("**Data Quality & Monitoring**")
        st.markdown("""
        *   **Global Data Quality Score**: A unified metric aggregating key quality dimensions into a single health indicator.
        *   **Quality Scorecards**: Detailed breakdown of the global score across 6 pillars: **Completeness**, **Consistency**, **Statistical Validity**, **Uniformity**, **Clinical Validity**, and **Uniqueness**.
        *   **Granular Inspection**: Drill-down tools to investigate scorecard flags, including **Advanced Outlier Detection** (Z-Score, Isolation Forest), **Integrity Checks**, and **Cohort Version Comparison**.
        """)

        st.success("**Descriptive Analytics**")
        st.markdown("""
        *   **Univariate Statistics**: Comprehensive summary statistics for quantitative (Mean, Median, SD, Normality tests) and qualitative variables (Frequencies).
        *   **Correlation Reports**: Interactive correlation matrices and cluster maps to identify variable relationships.
        *   **Distribution Analysis**: Visual exploration of variable distributions via histograms, violin plots, and density curves.
        """)
        
        st.warning("**AI & Knowledge Discovery**")
        st.markdown("""
        *   **RAG-Powered Insights**: Chat with documents and extract key concepts automatically.
        *   **Taxonomy Generation**: Discover variable relationships and build knowledge graphs.
        *   **Smart Suggestions**: AI-driven variable renaming and formula generation.
        """)
    
    with col2:
        st.info("**Data Enrichment**")
        st.markdown("""
        *   **Dataset Aggregation**: Seamlessly merge and harmonize multiple datasets into a single master cohort with one click.
        *   **Custom Variable Creation**: specific computed variables and derived scoring systems, manually or with **AI assistance**.
        *   **Advanced Imputation**: Handle missing data with statistical and ML-based methods (e.g., MissForest).
        *   **Dimensionality Reduction**: Apply PCA, UMAP, and t-SNE for high-dimensional feature exploration.
        """)

        st.success("**Epidemiology & Inferential Statistics**")
        st.markdown("""
        *   **Hypothesis Testing**: Automated selection of appropriate parametric/non-parametric tests (T-test, ANOVA, Chi-Square, etc.).
        *   **Multivariate Analysis**: Perform ANCOVA and regression models to adjust for confounding variables.
        *   **Epidemiological Metrics**: Calculation of Odds Ratios (OR), Relative Risks (RR), and survival analysis metrics.
        *   **Power Analysis**: Sample size calculation and statistical power estimation.
        """)
        
        st.warning("**Reproducibility & Reporting**")
        st.markdown("""
        *   **Version Control**: Built-in dataset versioning (DuckDB) and trace management.
        *   **Analysis Replay**: Reproduce exact analysis states and settings.
        *   **Publication-Ready Exports**: Generate high-resolution figures and formated Excel tables.
        """)

    st.divider()
    
    # --- Footer ---
    st.markdown("### 📬 Contact Information")
    
    col_contact, col_links = st.columns([1, 3])
    with col_contact:
        st.markdown("**Karim ELMERNISSI**")
    
    with col_links:
        st.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/karimelmernissi/)")


