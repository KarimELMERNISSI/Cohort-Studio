"""
Screenshots of the AI variable suggestions (Data > Enrich > Create New Variables), on the tutorial cohort.

They need the AI engine: a provider key on the server (.env) and a document library. Start the application with a
throwaway data folder, no sign-in, and an open-access review of the body-surface-area formulae in the library
(we used Redlarski, Palkowski & Krawczuk, "Body surface area formulae: an alarming ambiguity", Sci Rep 2016,
doi:10.1038/srep27966, CC BY 4.0, saved as <data>/users/local_researcher/documents/redlarski-2016-bsa-formulae.pdf):

    CMP_IAM_PROVIDER=disabled DATA_ROOT=/tmp/ai_data uv run streamlit run main.py --server.port 8775

then, with Playwright from a temporary environment (Microsoft Edge or Chromium installed):

    CMP_SITE_BASE=http://localhost:8775 uv run --no-project --python 3.14 --with playwright \
        python dev_scripts/ai_screenshots.py <out-dir> [chat model] [word of the suggestion to show]

The script loads the tutorial cohort (all 1,206 rows: no criterion is ticked, so the preview flags the values
computed from entry errors), initialises the AI engine with Google Gemini (the data-sharing notice is confirmed
for this session), asks for the formulas of a concept (BSA, Go To Target) and captures the request
(ai_search.png), the first suggestion with its source (ai_card.png, and the card on its own as
ai_suggestion.png) and its preview once applied to the editor and computed (ai_preview.png). It prints every suggestion with its formula and source: check each formula against its published
reference before the screenshots are used. dev_scripts/make_readme_assets.py builds the animation from them.
"""

import os
import re
import sys
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

BASE = os.environ.get("CMP_SITE_BASE", "http://localhost:8501")
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "ai_shots")
OUT.mkdir(parents=True, exist_ok=True)
MODEL = sys.argv[2] if len(sys.argv) > 2 else None  # the provider's first chat model otherwise
SHOWN = sys.argv[3] if len(sys.argv) > 3 else None  # a word of the title of the suggestion to show; the first one otherwise
PROVIDER = "Google Gemini (cloud)"
CONCEPT = "BSA"
# Tags each suggestion card (the largest block around its own "Apply to Editor" button that holds no other one)
# with data-ai-card="<index>" and returns the card titles.
CARDS = """() => {
    const apply = (e) => [...e.querySelectorAll('button')].filter((b) => b.innerText.trim() === 'Apply to Editor');
    return apply(document).map((button, i) => {
        let card = button.parentElement;
        while (card.parentElement && apply(card.parentElement).length === 1 && !card.parentElement.querySelector('h3')) {
            card = card.parentElement;
        }
        card.setAttribute('data-ai-card', String(i));
        return card.querySelector('h4') ? card.querySelector('h4').innerText.trim() : '?';
    });
}"""


def settle(page: Page, ms: int = 700) -> None:
    page.wait_for_timeout(ms)
    for _ in range(2400):  # an initialisation or a suggestion run can take minutes
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
    """Select an option, then check that the selectbox holds it: a rerun can swallow a selection."""
    for _ in range(3):
        page.get_by_label(label, exact=True).first.click()
        page.keyboard.type(option)
        page.wait_for_timeout(400)
        page.get_by_role("option", name=option, exact=True).first.click()
        settle(page, 900)
        if page.get_by_label(label, exact=True).first.input_value() == option:
            return
    raise RuntimeError(f"{label!r} did not take {option!r}")


def scroll_to(page: Page, text: str, offset: int = -90) -> None:
    """Put the first element with this text at the top of the view, then scroll by offset (negative = up)."""
    page.get_by_text(text, exact=False).first.evaluate("e => e.scrollIntoView({block: 'start'})")
    page.wait_for_timeout(300)
    page.mouse.move(900, 450)
    page.mouse.wheel(0, offset)
    page.wait_for_timeout(700)


def shot(page: Page, name: str) -> None:
    page.mouse.move(1100, 20)  # off any button, so no tooltip is open
    page.wait_for_timeout(600)
    page.screenshot(path=str(OUT / f"{name}.png"), animations="disabled", timeout=120000)
    print("captured", name)


with sync_playwright() as p:
    browser = p.chromium.launch(channel="msedge", headless=True)
    context = browser.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
    page = context.new_page()
    page.goto(BASE + "/")
    page.get_by_role("button", name="Load the tutorial cohort").wait_for(timeout=90000)
    settle(page)
    page.get_by_role("button", name="Load the tutorial cohort").click()
    settle(page, 2000)

    go(page, "Help", "Settings")  # the AI engine is the first tab
    choose(page, "AI provider", PROVIDER)
    if MODEL:
        choose(page, "Chat model", MODEL)
    consent = page.locator('[data-testid="stCheckbox"]').filter(has_text="I confirm this data may be sent")
    if not consent.get_by_role("checkbox").is_checked():
        consent.locator("label").first.click()
        settle(page, 900)
    page.get_by_role("button", name="Initialise / reindex").click()
    settle(page, 3000)
    if page.get_by_text("Initialisation failed").count():
        page.screenshot(path=str(OUT / "debug_init.png"))
        raise SystemExit("the AI engine did not initialise: see debug_init.png")
    status = page.locator('[data-testid="stExpander"] summary').filter(has_text=re.compile("index", re.I))
    print("engine:", status.first.inner_text().splitlines()[-1] if status.count() else "initialised")

    go(page, "Data", "Enrich")
    page.get_by_role("tab", name="Create New Variables").click(force=True)
    settle(page, 1200)
    page.get_by_text("AI Variable Suggestions (RAG)").first.click()
    settle(page, 900)
    concept = page.get_by_role("textbox", name="Concept Search:")
    concept.fill(CONCEPT)
    concept.press("Enter")
    settle(page, 900)
    page.get_by_text("Advanced Configuration").first.click()  # folded: the request reads at a glance
    settle(page, 600)
    page.get_by_role("button", name="Generate AI Suggestions").click()
    settle(page, 3000)
    page.get_by_text("Suggestions received!").first.wait_for(timeout=300000)
    page.wait_for_timeout(4000)  # toasts go away
    scroll_to(page, "Define New Variable", offset=-110)
    shot(page, "ai_search")

    titles = page.evaluate(CARDS)
    if not titles:
        page.screenshot(path=str(OUT / "debug_suggestions.png"))
        raise SystemExit("no suggestion came back: see debug_suggestions.png and the server log")
    details = [
        page.locator(f'[data-ai-card="{i}"] [data-testid="stExpander"]').filter(has_text="View Computation Details")
        for i in range(len(titles))
    ]
    for expander in details:
        expander.locator("summary").first.click()
        settle(page, 500)
    for i, (title, expander) in enumerate(zip(titles, details, strict=True), start=1):
        text = " | ".join(line for line in expander.inner_text().splitlines() if line.strip())
        print(f"suggestion {i}: {title}\n    {text}")
    shown = next((i for i, title in enumerate(titles) if SHOWN and SHOWN.lower() in title.lower()), 0)
    for i, expander in enumerate(details):  # only the suggestion shown stays open
        if i != shown:
            expander.locator("summary").first.click()
            settle(page, 400)

    card = page.locator(f'[data-ai-card="{shown}"]')
    card.evaluate("e => e.scrollIntoView({block: 'start'})")
    page.mouse.move(900, 450)
    page.mouse.wheel(0, -130)
    page.wait_for_timeout(700)
    shot(page, "ai_card")
    card.screenshot(path=str(OUT / "ai_suggestion.png"), animations="disabled")
    print("captured ai_suggestion:", titles[shown])

    card.get_by_role("button", name="Apply to Editor").click()
    settle(page, 1500)
    page.get_by_role("button", name="Compute & Preview").click()
    settle(page, 2500)
    scroll_to(page, "New Variable Preview", offset=-100)
    shot(page, "ai_preview")
    browser.close()
