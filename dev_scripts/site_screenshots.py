"""
Screenshots of the real application for the documentation site (docs/), on the tutorial cohort.

Start the application with a throwaway data folder and no sign-in, for example:

    CMP_IAM_PROVIDER=disabled DATA_ROOT=/tmp/site_data uv run streamlit run main.py --server.port 8769

then, with Playwright from a temporary environment (Microsoft Edge or Chromium installed):

    CMP_SITE_BASE=http://localhost:8769 uv run --no-project --python 3.14 --with playwright         python dev_scripts/site_screenshots.py <out-dir> [shot names...]

and build the page images with dev_scripts/make_site_assets.py <out-dir> and the README images with
dev_scripts/make_readme_assets.py <out-dir>; dev_scripts/keep_screenshots.py <out-dir> adds them to the
screenshot library (assets/screenshots). The 1440x900 viewport is captured at 2x; the script ticks the
plausibility rules and study-inclusion criteria, captures the Explore statistics and correlations, draws six
Visualize figures (each also captured on its own, as plot_<name>.png), runs a multiple imputation (m = 10), the
pooled survival analysis and Cox model, the group comparison, adds two results to the study dossier, and ends on
the Start page, whose workflow map (workflow.png) then shows the steps done.
"""

import os
import re
import sys
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

BASE = os.environ.get("CMP_SITE_BASE", "http://localhost:8501")
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "site_shots")
OUT.mkdir(parents=True, exist_ok=True)
ONLY = set(sys.argv[2:])  # optional subset of shot names
log: list[str] = []


def settle(page: Page, ms: int = 700) -> None:
    page.wait_for_timeout(ms)
    for _ in range(1200):
        busy = page.locator('[data-testid="stStatusWidget"]').filter(
            has_text=re.compile("Running|Connecting|Stop")
        )
        if busy.count() == 0:
            break
        page.wait_for_timeout(250)
    page.wait_for_timeout(500)


def go(page: Page, section: str, title: str) -> None:
    page.locator('[data-testid="stTopNavSection"]').filter(has_text=section).first.click()
    page.wait_for_timeout(500)
    page.locator('[data-testid="stTopNavDropdownLink"]').filter(has_text=title).first.click()
    settle(page, 1800)


def choose(page: Page, label: str, option: str) -> None:
    page.get_by_label(label, exact=True).first.click()
    page.keyboard.type(option)
    page.wait_for_timeout(400)
    exact = page.get_by_role("option", name=option, exact=True)
    if exact.count():
        exact.first.click()
    else:
        page.keyboard.press("Enter")
    page.keyboard.press("Escape")
    settle(page, 900)


def choose_checked(page: Page, label: str, option: str) -> None:
    """choose(), then check that the selectbox holds the option: a rerun can swallow a selection."""
    for _ in range(3):
        choose(page, label, option)
        if page.get_by_label(label, exact=True).first.input_value() == option:  # the combobox input holds it
            return
    raise RuntimeError(f"{label!r} did not take {option!r}")


def pick(page: Page, label: str, options: list[str]) -> None:
    """Replace the selection of a multiselect."""
    page.get_by_label(label, exact=True).first.click()
    for _ in range(40):  # Backspace removes the last chip
        page.keyboard.press("Backspace")
    page.keyboard.press("Escape")
    settle(page, 900)
    for option in options:
        choose(page, label, option)


def set_checkbox(page: Page, text: str, value: bool) -> None:
    box = page.locator('[data-testid="stCheckbox"]').filter(has_text=re.compile(rf"^{re.escape(text)}$")).last
    for _ in range(3):
        if box.get_by_role("checkbox").is_checked() == value:
            return
        box.locator("label").first.click()
        settle(page, 900)
    raise RuntimeError(f"checkbox {text!r} did not become {value}")


def style(page: Page, theme: str, palette: str | None = None) -> None:
    """Theme and colour scheme of the Visualize page (sidebar, Visual Settings)."""
    sidebar = page.locator('[data-testid="stSidebar"]')
    if not sidebar.get_by_label("Theme", exact=True).first.is_visible():
        sidebar.get_by_text("Visual Settings").first.click()
        settle(page, 800)
    choose_checked(page, "Theme", theme)
    if palette:
        choose_checked(page, "Color Scheme", palette)


def figure(page: Page, name: str) -> None:
    """The last visible Plotly chart of the page on its own, as plot_<name>.png."""
    if ONLY and f"plot_{name}" not in ONLY:
        return
    chart = page.locator('[data-testid="stPlotlyChart"]:visible').last
    chart.locator(".main-svg").first.wait_for(timeout=120000)
    page.wait_for_timeout(1500)
    chart.scroll_into_view_if_needed()
    page.mouse.move(5, 5)  # no hover label and no mode bar
    page.wait_for_timeout(600)
    chart.screenshot(path=str(OUT / f"plot_{name}.png"), animations="disabled", timeout=120000)
    log.append(f"plot_{name}")


def start_page_top(page: Page) -> None:
    """The Start page scrolled to the top, where the workflow map is."""
    link = page.locator('[data-testid="stTopNavLink"]').filter(has_text="Start")
    if not link.count():
        link = page.get_by_role("link", name="Start")
    link.first.click()
    settle(page, 2500)
    page.mouse.move(900, 450)
    page.mouse.wheel(0, -8000)
    page.wait_for_timeout(800)


def workflow(page: Page, name: str) -> None:
    """The workflow map of the Start page on its own (its bordered card), as <name>.png."""
    if ONLY and name not in ONLY:
        return
    card = page.locator('[data-testid="stLayoutWrapper"]').filter(has=page.locator("div.wf")).last
    page.mouse.move(5, 880)  # off the map: no box is highlighted
    page.wait_for_timeout(600)
    card.screenshot(path=str(OUT / f"{name}.png"), animations="disabled", timeout=120000)
    log.append(name)


def scroll_to(page: Page, text: str, offset: int = -90) -> None:
    """Put the first element with this text at the top of the view, then scroll by offset (negative = up)."""
    target = page.get_by_text(text, exact=False).first
    target.evaluate("e => e.scrollIntoView({block: 'start'})")
    page.wait_for_timeout(300)
    page.mouse.move(900, 450)
    page.mouse.wheel(0, offset)
    page.wait_for_timeout(700)


def shot(page: Page, name: str) -> None:
    if ONLY and name not in ONLY:
        return
    page.mouse.move(1100, 20)  # off any button, so no tooltip is open
    page.wait_for_timeout(600)
    page.screenshot(path=str(OUT / f"{name}.png"), animations="disabled", timeout=120000)
    log.append(name)


with sync_playwright() as p:
    browser = p.chromium.launch(channel="msedge", headless=True)
    context = browser.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
    page = context.new_page()
    page.goto(BASE + "/")
    page.get_by_role("button", name="Load the tutorial cohort").wait_for(timeout=90000)
    settle(page)
    page.get_by_role("button", name="Load the tutorial cohort").click()
    settle(page, 2000)
    page.wait_for_timeout(4500)  # the "loaded" toast goes away
    page.mouse.move(900, 450)
    page.mouse.wheel(0, -8000)
    page.wait_for_timeout(800)
    shot(page, "start")
    workflow(page, "workflow_start")

    sidebar = page.locator('[data-testid="stSidebar"]')
    sidebar.get_by_text(re.compile(r"Population: all 1,206 rows")).first.click()
    settle(page, 800)
    page.locator("label").filter(has_text="Exclude: implausible values").first.click()
    settle(page, 1500)
    label = page.locator("label").filter(has_text="Require: study inclusion").first
    if not label.is_visible():
        sidebar.get_by_text(re.compile(r"Population: ")).first.click()
        settle(page, 800)
    label.click()
    settle(page, 1500)

    go(page, "Data", "Validate")
    settle(page, 2500)
    shot(page, "quality")
    page.get_by_role("tab", name="Analysis population").click(force=True)
    settle(page, 2500)
    scroll_to(page, "928 of 1,206 rows", offset=-140)
    shot(page, "population")

    # Explore > Statistics & correlations: column types, descriptive statistics, correlations
    go(page, "Explore", "Statistics & correlations")
    settle(page, 3000)
    scroll_to(page, "Columns Information", offset=-110)
    shot(page, "explore_types")
    scroll_to(page, "Quantitative Data Statistics", offset=-100)
    settle(page, 1500)
    shot(page, "explore_stats")
    scroll_to(page, "Qualitative Data Statistics", offset=-100)
    shot(page, "explore_qual")
    scroll_to(page, "Correlation Network Graph", offset=-60)
    settle(page, 2500)
    shot(page, "explore_graph")
    figure(page, "network")
    tab = page.get_by_role("tab", name="Correlation Matrix")
    tab.evaluate("e => e.scrollIntoView({block: 'center'})")  # not under the sticky top bar
    page.wait_for_timeout(500)
    tab.click()
    settle(page, 2500)
    tab.evaluate("e => e.scrollIntoView({block: 'start'})")
    page.mouse.move(900, 450)
    page.mouse.wheel(0, -80)
    page.wait_for_timeout(900)
    shot(page, "explore_matrix")
    figure(page, "explore_matrix")

    # Explore > Visualize: six figures, each with its own theme and palette
    go(page, "Explore", "Visualize")
    choose_checked(page, "Plot Type", "Box Plot")
    pick(page, "Select Numeric Variables", ["lvef_pct"])
    choose_checked(page, "Group by", "nyha_class")
    set_checkbox(page, "Show Statistical Significance", True)
    choose_checked(page, "Multiple Testing Correction", "Bonferroni")
    style(page, "plotly_white", "Safe - Colorblind-friendly color palette")
    figure(page, "box")
    page.locator('[data-testid="stPlotlyChart"]').last.evaluate("e => e.scrollIntoView({block: 'start'})")
    page.mouse.move(900, 450)
    page.mouse.wheel(0, -75)
    page.wait_for_timeout(800)
    shot(page, "visualize")

    choose_checked(page, "Plot Type", "Violin Plot")
    pick(page, "Select Numeric Variables", ["hba1c_pct", "potassium_mmol_l"])
    choose_checked(page, "Group by", "diabetes")
    set_checkbox(page, "Show Statistical Significance", True)
    style(page, "seaborn", "Set2 - Alternative qualitative color palette")
    figure(page, "violin")

    choose_checked(page, "Plot Category", "Relationships")
    choose_checked(page, "Plot Type", "Scatter Plot")
    choose_checked(page, "X Variable", "sbp_baseline")
    choose_checked(page, "Y Variable", "sbp_6m")
    choose_checked(page, "Group by (optional)", "arm")
    choose_checked(page, "Plot Mode", "Regression (Linear)")
    choose_checked(page, "Marginal Distribution (Joint Plot)", "Box")
    style(page, "ggplot2", "T10 - 10 distinct colors for smaller datasets")
    figure(page, "scatter")

    choose_checked(page, "Plot Type", "Correlation Matrix")
    style(page, "plotly_white")
    pick(page, "Select Columns for Correlation", [
        "sbp_baseline", "sbp_ambulatory", "sbp_6m", "dbp_baseline", "lvef_pct", "lvef_reader2_pct",
        "nt_probnp_pg_ml", "weight_kg", "height_cm", "age", "creatinine_umol_l", "fu_days", "n_hospital_visits",
    ])
    choose_checked(page, "Method", "spearman")
    for box in ("Cluster Rows", "Cluster Cols", "Show Row Dendrogram", "Show Col Dendrogram"):
        set_checkbox(page, box, True)
    figure(page, "corr")

    choose_checked(page, "Plot Type", "Clustermap")
    pick(page, "Select Features (Numeric Values)", [
        "age", "lvef_pct", "nt_probnp_pg_ml", "sbp_baseline", "dbp_baseline", "heart_rate", "creatinine_umol_l",
        "potassium_mmol_l", "hba1c_pct", "ldl_mmol_l", "troponin_ng_l", "weight_kg",
    ])
    choose_checked(page, "Group Samples by (Optional Color)", "nyha_class")  # no sample id: it has duplicates
    if not page.get_by_label("Group Features by", exact=True).first.is_visible():
        page.get_by_text("Advanced Options").last.click()
        settle(page, 900)
    set_checkbox(page, "Show Sample Labels", False)
    set_checkbox(page, "Show Column Dendrogram", True)
    set_checkbox(page, "Show Row Dendrogram", True)
    choose_checked(page, "Group Features by", "P-value (Sample Group)")
    page.get_by_role("button", name="Generate Clustermap").click()
    settle(page, 6000)
    figure(page, "clustermap")

    choose_checked(page, "Plot Category", "Categorical")
    choose_checked(page, "Plot Type", "Sunburst Chart")
    pick(page, "Select Hierarchy (in order)", ["arm", "nyha_class", "smoking_status"])
    style(page, "plotly_white", "Pastel1 - Soft pastel color palette")
    figure(page, "sunburst")

    go(page, "Data", "Enrich")
    page.get_by_role("tab", name="Handle Missing Data").click(force=True)
    settle(page, 800)
    page.get_by_role("tab", name="Multiple imputation (for inference)").click(force=True)
    settle(page, 1500)
    shot(page, "mi_setup")
    m_input = page.get_by_label("Imputations (m)")
    m_input.fill("10")
    m_input.press("Enter")
    settle(page, 800)
    page.get_by_role("button", name="Run the multiple imputation").click()
    settle(page, 4000)
    scroll_to(page, "4. Diagnostics", offset=-110)
    shot(page, "mi_diagnostics")

    page.locator('[data-testid="stSidebar"]').get_by_text(
        "Imputed (m = 10)", exact=True
    ).first.click()
    settle(page, 1500)
    go(page, "Analyze", "Survival & regression")
    choose(page, "Compare by", "arm")
    choose(page, "Adjust for", "age")
    choose(page, "Adjust for", "hba1c_pct")
    page.get_by_role("button", name="Run analysis").click()
    settle(page, 6000)
    scroll_to(page, "Hazard ratio", offset=-120)
    shot(page, "survival")
    page.get_by_text("Model details: every term of the Cox model").first.click()
    settle(page, 1200)
    scroll_to(page, "Model details: every term of the Cox model", offset=-100)
    shot(page, "cox")
    page.get_by_role("button", name="Add to the study dossier").last.click()
    settle(page, 1200)

    page.locator('[data-testid="stSidebar"]').get_by_text("Observed", exact=True).first.click()
    settle(page, 1500)
    go(page, "Analyze", "Hypothesis tests & epidemiology")
    choose(page, "Select Grouping Variable (Factor)", "arm")
    settle(page, 1500)
    scroll_to(page, "Before you run", offset=-120)
    shot(page, "plan")
    page.get_by_role("button", name="Run analysis").click()
    settle(page, 6000)
    scroll_to(page, "Summary Results", offset=-100)
    shot(page, "comparison")
    page.get_by_role("button", name="Add to the study dossier").first.click()
    settle(page, 1200)

    go(page, "Reproduce", "Study dossier")
    settle(page, 2500)
    try:
        page.get_by_text("Participant flow").first.wait_for(timeout=60000)
        shot(page, "dossier")
        scroll_to(page, "Results in the dossier", offset=-100)
    except Exception:
        page.screenshot(path=str(OUT / "debug_dossier.png"))
        raise
    shot(page, "dossier_results")
    go(page, "Reproduce", "Versions & traces")
    settle(page, 2500)
    shot(page, "traces")

    # the Start page again: the workflow map now shows the steps done
    start_page_top(page)
    shot(page, "start_workflow")
    workflow(page, "workflow")
    browser.close()

print("captured:", ", ".join(log))
