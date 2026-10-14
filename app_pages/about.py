"""About page body: what the platform does, third-party notices and contact."""

import streamlit as st

FEATURES = (
    (":material/fact_check:", "Data quality & monitoring", (
        "**Global quality score** across completeness, consistency, statistical validity, uniformity, clinical validity and uniqueness.",
        "**Drill-down** on every flag: outlier detection (robust Z, IQR, Isolation Forest), integrity checks, version comparison.",
        "**Clinical anomalies and inclusion criteria** as declarative, traced rules.",
    )),
    (":material/auto_fix_high:", "Data enrichment", (
        "**Merge and harmonise** external datasets into one cohort.",
        "**Derived variables** from formulas evaluated in a sandbox, manually or with AI suggestions.",
        "**Imputation**: single imputation for exploration; a multiple-imputation diagnostic that pools the "
        "imputed means with Rubin's rules (pooled analyses are planned).",
    )),
    (":material/analytics:", "Descriptive analytics", (
        "**Univariate statistics** by variable type with normality diagnostics.",
        "**Mixed-type associations** with confidence intervals and an interactive correlation network.",
        "**Distributions, group comparisons, ROC and Bland-Altman** plots ready for publication.",
    )),
    (":material/science:", "Epidemiology & inference", (
        "**Welch-first group comparisons** with effect sizes, confidence intervals and every pairwise contrast.",
        "**2x2 measures** (RR, OR, RD, NNT), diagnostic accuracy, power and sample size.",
        "**Survival and regression**: Kaplan-Meier, log-rank, Cox, logistic and Poisson; rates, Mantel-Haenszel, kappa and ICC.",
    )),
    (":material/account_tree:", "AI & knowledge discovery", (
        "**Document intelligence**: chat with your literature and extract a knowledge graph.",
        "**Variable taxonomy** and formula suggestions grounded in your documents.",
        "**Local first**: Ollama by default; cloud providers only after explicit consent.",
    )),
    (":material/history:", "Reproducibility & reporting", (
        "**Versioned datasets** and a trace of every transformation, replayable step by step.",
        "**Word reports** of the full transformation history.",
        "**Exports**: high-resolution figures and formatted Excel tables.",
    )),
)


def app():
    st.markdown(
        "A unified environment for clinical cohort research: from raw ingestion to publication-ready analysis, "
        "with every step validated, traced and reproducible."
    )

    for start in range(0, len(FEATURES), 3):
        for column, (icon, title, bullets) in zip(st.columns(3, gap="medium"), FEATURES[start:start + 3], strict=False):
            with column, st.container(border=True, height="stretch"):
                st.markdown(f"#### {icon} {title}")
                st.markdown("\n".join(f"- {b}" for b in bullets))

    st.subheader("Third-party components and notices", icon=":material/gavel:", anchor=False)
    with st.container(border=True):
        st.markdown("""
- **Graph rendering** uses a library-agnostic connector. The default engine is **Plotly + NetworkX** (MIT / BSD-3), rendered
  locally with no network access and no telemetry.
- The optional **yFiles Graphs for Streamlit** engine is a proprietary component of **yWorks GmbH**, used under the
  *yFiles Streamlit Component Free License*. When it is selected, the graph runtime is loaded from `yworks.com` at render
  time (an internet connection is required) and anonymised, non-textual information about the structure of the graphs you
  create is reported to yWorks. yWorks may change or discontinue the component. Choose the offline engine in Settings
  (or set `CMP_GRAPH_RENDERER=plotly` / `CMP_OFFLINE=1`) to avoid any outbound connection.
- **AI providers**: Ollama runs on your server; Google Gemini, OpenAI and Mistral are cloud services that only receive data
  after an explicit confirmation in Settings.
- Full dependency licences: `documentation/developer/License_Dependencies.md`.
""")

    st.subheader("Contact", icon=":material/contact_mail:", anchor=False)
    with st.container(horizontal=True, vertical_alignment="center"):
        st.markdown("**Karim ELMERNISSI**")
        st.link_button("LinkedIn", "https://www.linkedin.com/in/karimelmernissi/", icon=":material/open_in_new:")
