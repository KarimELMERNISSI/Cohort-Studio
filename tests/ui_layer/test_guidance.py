"""Test chooser decision table, glossary, interpretation wording, and help-text coverage of the new UI modules."""

import ast
from pathlib import Path

import pytest

from biostats._types import CI, AssumptionCheck, EffectSize
from biostats._types import TestResult as StatResult  # alias: pytest must not try to collect it
from biostats.inference import compare_groups
from ui.guidance import GLOSSARY, define, how_to_read, recommend
from ui.guidance.test_chooser import all_combinations

ROOT = Path(__file__).resolve().parents[2]
INPUT_WIDGETS = {
    "text_input",
    "number_input",
    "selectbox",
    "multiselect",
    "slider",
    "checkbox",
    "radio",
    "file_uploader",
    "text_area",
    "date_input",
    "toggle",
    "select_slider",
    "segmented_control",
    "pills",
}
CHECKED_FILES = [
    *(ROOT / "app_pages" / "views").glob("*.py"),
    ROOT / "app_pages" / "analysis_diagnostic.py",
    ROOT / "ui" / "import_flow.py",
    ROOT / "ui" / "shell.py",
    ROOT / "ui" / "components" / "__init__.py",
]


def test_every_combination_yields_a_recommendation() -> None:
    combos = all_combinations()
    assert len(combos) == 60
    available = 0
    for combo in combos:
        rec = recommend(*combo)
        assert rec.test and rec.why and rec.effect_size and rec.page
        available += rec.available
        if not rec.available:
            assert rec.caveats
    assert (
        available >= 40
    )  # paired, one-sample and survival routes exist since Tier C; Cochran's Q, Bowker and goodness-of-fit are planned


def test_specific_routes() -> None:
    welch = recommend(
        "Continuous (measurement)",
        "Two groups",
        "Independent groups",
        "Roughly symmetric, at least 15 per group",
    )
    assert welch.test.startswith("Welch's t-test") and welch.available
    kw = recommend(
        "Ordinal (ranked levels)",
        "Three or more groups",
        "Independent groups",
        "Roughly symmetric, at least 15 per group",
    )
    assert "Kruskal" in kw.test and "Dunn" in kw.test
    two_by_two = recommend(
        "Binary (yes/no)",
        "Two groups",
        "Independent groups",
        "Skewed, outliers or fewer than 15 per group",
    )
    assert "2x2" in two_by_two.where
    survival = recommend(
        "Time to event",
        "Two groups",
        "Independent groups",
        "Roughly symmetric, at least 15 per group",
    )
    assert survival.available and "Kaplan" in survival.test and survival.page == "survival"
    with pytest.raises(ValueError):
        recommend(
            "Weird", "Two groups", "Independent groups", "Roughly symmetric, at least 15 per group"
        )


def test_glossary_is_substantial() -> None:
    assert len(GLOSSARY) >= 35
    assert define("p-value") and define("HEDGES' G")
    assert define("nope") is None
    for text in GLOSSARY.values():
        assert len(text) > 40


def test_interpretation_is_ci_first_and_never_overclaims() -> None:
    res = compare_groups(
        {
            "a": [4.1, 5.6, 5.2, 6.1, 4.5, 4.6, 5.2, 4.5, 5.3, 5.1],
            "b": [6.3, 5.1, 5.5, 5.5, 5.4, 5.3, 4.9, 6.2, 5.8, 5.3],
        }
    )
    text = how_to_read(res)
    assert "95% CI" in text and "Hedges' g" in text
    for banned in ("no effect exists", "proves", "confirmed", "validated", "true negative"):
        assert banned not in text.lower()
    manual = StatResult(
        name="x",
        p_value=0.3,
        p_adjusted=0.6,
        adjustment="Holm",
        effect_sizes=[
            EffectSize(name="Hedges' g", kind="g", estimate=0.1, ci=CI(lower=-0.3, upper=0.5))
        ],
        assumptions=[AssumptionCheck(name="normality (a)", status="violated", detail="p = 0.01")],
        family="nonparametric",
    )
    text2 = how_to_read(manual)
    assert (
        "adjusted p" in text2
        and "not evidence of absence" in text2
        and "Rank-based" in text2
        and "Diagnostics" in text2
    )


def _widget_calls(tree: ast.AST):
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in INPUT_WIDGETS
        ):
            yield node


@pytest.mark.parametrize("path", CHECKED_FILES, ids=lambda p: p.name)
def test_every_input_widget_has_help_text(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    missing = []
    for call in _widget_calls(tree):
        keywords = {k.arg for k in call.keywords}
        if "help" not in keywords and "label_visibility" not in keywords:
            missing.append(f"{path.name}:{call.lineno} st.{call.func.attr}")
    assert not missing, missing
