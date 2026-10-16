"""
Screenshots of the real application for the documentation site (docs/), on the tutorial cohort.

Start the application with a throwaway data folder and no sign-in, for example:

    CMP_IAM_PROVIDER=disabled DATA_ROOT=/tmp/site_data uv run streamlit run main.py --server.port 8769

then, with Playwright from a temporary environment (Microsoft Edge or Chromium installed):

    CMP_SITE_BASE=http://localhost:8769 uv run --no-project --python 3.14 --with playwright         python dev_scripts/site_screenshots.py <out-dir> [shot names...]

and build the page images with dev_scripts/make_site_assets.py <out-dir>. The 1440x900 viewport is captured
at 2x; the script ticks the clinical-anomaly and study-inclusion criteria, runs a multiple imputation (m = 10),
the pooled survival analysis and Cox model, the group comparison, and adds two results to the study dossier.
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
    shot(page, "start")

    sidebar = page.locator('[data-testid="stSidebar"]')
    sidebar.get_by_text(re.compile(r"Population: all 1,206 rows")).first.click()
    settle(page, 800)
    page.locator("label").filter(has_text="Exclude: clinical anomalies").first.click()
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
    choose(page, "Group (optional)", "arm")
    page.get_by_role("button", name="Run survival analysis").click()
    settle(page, 3000)
    scroll_to(page, "Kaplan-Meier survival and the log-rank test", offset=-20)
    shot(page, "survival")
    choose(page, "Covariates", "arm")
    choose(page, "Covariates", "age")
    choose(page, "Covariates", "hba1c_pct")
    page.get_by_role("button", name="Fit Cox model").click()
    settle(page, 4000)
    scroll_to(page, "Cox proportional-hazards regression", offset=-100)
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
    browser.close()

print("captured:", ", ".join(log))
