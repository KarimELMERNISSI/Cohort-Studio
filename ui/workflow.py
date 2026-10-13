"""
The research workflow shown on the Start page: a core path (load, validate, explore and analyse,
report) and optional branches (enrich, map variables to the field, learn from documents).

Pure Python: the diagram is DOT text with explicit fills, so it reads the same in light and dark themes.
"""

from __future__ import annotations

from dataclasses import dataclass

from ui.theme import BRAND

CORE_FILL = BRAND["teal"]
OPTIONAL_FILL = "#E6F6F7"
OPTIONAL_BORDER = BRAND["teal"]
DONE_FILL = BRAND["navy"]
EDGE = "#8DA3AD"


@dataclass(frozen=True)
class Branch:
    key: str  # navigation key of the page that runs it
    icon: str
    title: str
    when: str  # where the branch attaches to the core path
    text: str
    links: tuple[tuple[str, str], ...]
    needs_ai: bool = False


BRANCHES: tuple[Branch, ...] = (
    Branch(
        "enrich",
        ":material/auto_fix_high:",
        "Enrich the cohort",
        "after validation",
        "Merge external sources, derive variables from formulas or AI suggestions, impute missing values "
        "(single imputation for exploration, multiple imputation for inference).",
        (("enrich", "Open Enrich"),),
    ),
    Branch(
        "variable_map",
        ":material/account_tree:",
        "Map your variables to your field",
        "any time · uses the AI engine",
        "Build a taxonomy of your variables grounded in your clinical domain and your documents: definitions, "
        "units, relationships and derived formulas, saved as versions.",
        (("variable_map", "Open Variable map"),),
        needs_ai=True,
    ),
    Branch(
        "documents",
        ":material/description:",
        "Learn from your documents",
        "any time · uses the AI engine",
        "Index your literature, protocols or data dictionaries, chat with them and extract a knowledge graph that "
        "also informs the variable map and the enrichment suggestions.",
        (("documents", "Open Documents"), ("settings", "Document library")),
        needs_ai=True,
    ),
    Branch(
        "reproduce",
        ":material/history:",
        "Report and reproduce",
        "at the end, or any time",
        "Every step is traced: reopen saved versions, replay an analysis step by step and export the Word report.",
        (("reproduce", "Open Versions & traces"),),
    ),
)


def workflow_dot(has_data: bool) -> str:
    """Core path (filled) with optional branches (outlined, dashed links)."""

    def node(name: str, label: str, *, core: bool, done: bool = False) -> str:
        if core:
            fill = DONE_FILL if done else CORE_FILL
            return (
                f'{name} [label="{label}", shape=box, style="rounded,filled", fillcolor="{fill}", '
                f'color="{fill}", fontcolor="#FFFFFF"];'
            )
        return (
            f'{name} [label="{label}", shape=box, style="rounded,filled,dashed", fillcolor="{OPTIONAL_FILL}", '
            f'color="{OPTIONAL_BORDER}", fontcolor="{BRAND["navy"]}"];'
        )

    lines = [
        "digraph workflow {",
        '  graph [rankdir=LR, bgcolor="transparent", nodesep=0.35, ranksep=0.45, pad=0.1];',
        '  node [fontname="Helvetica", fontsize=11, margin="0.18,0.08", penwidth=1.4];',
        f'  edge [color="{EDGE}", arrowsize=0.7, penwidth=1.3, fontname="Helvetica", fontsize=9, fontcolor="{EDGE}"];',
        "  "
        + node("load", "1  Load data" + (" (done)" if has_data else ""), core=True, done=has_data),
        "  " + node("validate", "2  Validate", core=True),
        "  " + node("analyse", "3  Explore & analyse", core=True),
        "  " + node("report", "Report & reproduce", core=True),
        "  " + node("enrich", "Enrich: merge,\\nderive, impute", core=False),
        "  " + node("docs", "Your documents:\\nliterature, protocols", core=False),
        "  " + node("varmap", "Variable map:\\nyour field's taxonomy", core=False),
        "  load -> validate -> analyse -> report;",
        '  validate -> enrich [style=dashed, label="optional"];',
        "  enrich -> analyse [style=dashed];",
        '  docs -> varmap [style=dashed, label="grounds"];',
        '  varmap -> analyse [style=dashed, label="informs"];',
        '  docs -> enrich [style=dashed, label="suggests variables"];',
        "  { rank=same; validate; docs; }",
        "  { rank=same; enrich; varmap; }",
        "}",
    ]
    return "\n".join(lines)


__all__ = ["BRANCHES", "Branch", "workflow_dot"]
