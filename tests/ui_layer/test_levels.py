"""
The lists of groups (ui/components/levels.py, M5). The tutorial cohort's ``site`` holds "Luxembourg" and
"Luxembourg ", which a browser shows identically, so "Groups to include" seemed to list each site twice. The lists
now mark the hidden spaces, and a warning names the look-alike groups with the formula of a cleaned copy.
"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

from ingest.loaders import read_path
from ui.components.levels import alike_levels, cleaning_formula, show_level
from ui.state import TUTORIAL_COHORT_PATH


def test_a_hidden_space_is_shown_and_no_two_levels_read_the_same() -> None:
    assert show_level("Luxembourg") == "Luxembourg"
    assert show_level("Luxembourg ") == "Luxembourg␣"
    assert show_level(" M") == "␣M" and show_level("New York") == "New York"
    assert (
        show_level("a  b") == "a␣␣b"
        and show_level("M\t") == "M\\t"
        and show_level("M\xa0") == "M\\xa0"
    )
    assert show_level(2) == "2" and show_level(True) == "True"  # other values read as before
    tutorial = read_path(str(TUTORIAL_COHORT_PATH)).df
    for column in ("site", "sex"):
        # the multiselect tells options apart by their labels
        levels = tutorial[column].dropna().unique()
        assert len({show_level(v) for v in levels}) == len(levels), column


def test_look_alike_levels_and_the_formula_that_merges_them() -> None:
    sites = ["Esch-sur-Alzette", "Esch-sur-Alzette ", "Luxembourg", "Metz", "Metz "]
    assert alike_levels(sites) == [["Esch-sur-Alzette", "Esch-sur-Alzette "], ["Metz", "Metz "]]
    assert cleaning_formula("site", alike_levels(sites)) == "site.str.strip()"
    sexes = ["F", "F ", "Female", "M", "MALE", "f", "female", "male"]
    assert alike_levels(sexes) == [["F", "F ", "f"], ["Female", "female"], ["MALE", "male"]]
    assert cleaning_formula("sex", alike_levels(sexes)) == "sex.str.strip().str.lower()"
    assert cleaning_formula("study site", alike_levels(sites)) == '""study site"".str.strip()'
    assert alike_levels(["No", "Yes"]) == [] and alike_levels([0, 1, 2]) == []


def _groups_script() -> None:
    from app_pages.analysis_config import render_analysis_configuration
    from ingest.analyzer import DataAnalyzer
    from ingest.loaders import read_path
    from ui.state import TUTORIAL_COHORT_PATH

    df = read_path(str(TUTORIAL_COHORT_PATH)).df
    analyzer = DataAnalyzer(df)
    render_analysis_configuration(
        df,
        analyzer.numeric_cols,
        analyzer.categorical_cols,
        analyzer.binary_cols,
        default_group_col="site",
        key_prefix="t_",
    )


def test_groups_to_include_tells_the_tutorial_sites_apart() -> None:
    at = AppTest.from_function(_groups_script, default_timeout=60)
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    groups = at.multiselect(key="t_selected_groups::site")
    assert len(groups.options) == len(set(groups.options)) == 8
    assert {"Luxembourg", "Luxembourg␣", "Metz", "Metz␣"} <= set(groups.options)
    warning = at.warning[0].value
    assert "differ only by spaces" in warning and "`site_clean` = `site.str.strip()`" in warning
    groups.unselect("Luxembourg ").run()  # the label maps back to its own value
    chosen = at.multiselect(key="t_selected_groups::site").value
    assert "Luxembourg " not in chosen and "Luxembourg" in chosen and len(chosen) == 7


def _power_script() -> None:
    from app_pages.epidemiology import run_power_analysis
    from ingest.analyzer import DataAnalyzer
    from ingest.loaders import read_path
    from ui.state import TUTORIAL_COHORT_PATH

    df = read_path(str(TUTORIAL_COHORT_PATH)).df
    run_power_analysis(df, DataAnalyzer(df))


def test_the_power_analysis_lists_the_groups_the_same_way() -> None:
    at = AppTest.from_function(_power_script, default_timeout=60)
    at.run()
    at.selectbox(key="power_group_col").set_value("site").run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    groups = at.multiselect(key="power_selected_groups::site")
    assert len(set(groups.options)) == 8 and "Luxembourg␣" in groups.options
    assert any("`site_clean` = `site.str.strip()`" in w.value for w in at.warning)
    at.selectbox(key="power_group_col").set_value("sex").run()
    assert len(at.multiselect(key="power_selected_groups::sex").value) == 10


def test_another_grouping_variable_starts_with_all_its_groups() -> None:
    # one key for every variable kept the groups of site, none of which is a level of sex: the list was empty
    at = AppTest.from_function(_groups_script, default_timeout=60)
    at.run()
    at.multiselect(key="t_selected_groups::site").unselect("Metz").run()
    at.selectbox(key="t_group_col").set_value("sex").run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    assert len(at.multiselect(key="t_selected_groups::sex").value) == 10
    assert "`sex_clean` = `sex.str.strip().str.lower()`" in at.warning[0].value
