"""
Navigation registry: the single list of pages, their files, titles, URL paths and sections.

``main.py`` builds ``st.navigation`` from it; components link to pages by *key* so a page can be
moved or renamed in one place. Pages are files under ``app_pages/views/`` so that Streamlit's
AppTest can switch between them.
"""

from __future__ import annotations

from dataclasses import dataclass

NAV_POSITION = (
    "top"  # "top" keeps the sidebar free for the dataset card; "sidebar" is the alternative.
)
VIEWS_DIR = "app_pages/views"
# the query parameter of a link that opens one tab of its page (PageDef.tabs), read by that page
TAB_PARAM = "tab"


@dataclass(frozen=True)
class PageDef:
    key: str
    file: str
    title: str
    url_path: str
    section: str
    icon: str
    admin: bool = False
    default: bool = False
    chapter: str | None = None  # handbook chapter (documentation/<chapter>.md)
    # the tabs a link can open, by the value of TAB_PARAM (an id kept in the links): their labels, outer tab first
    tabs: tuple[tuple[str, tuple[str, ...]], ...] = ()


PAGES: tuple[PageDef, ...] = (
    PageDef(
        "start",
        "start.py",
        "Start",
        "start",
        "",
        ":material/home:",
        default=True,
        chapter="00_Researcher_Handbook",
    ),
    PageDef(
        "import",
        "import_data.py",
        "Import",
        "import",
        "Data",
        ":material/upload_file:",
        chapter="01_Home",
    ),
    PageDef(
        "validate",
        "validate.py",
        "Validate",
        "validate",
        "Data",
        ":material/fact_check:",
        chapter="02_Data_Validation",
        tabs=(("date_checks", ("Date checks",)),),
    ),
    PageDef(
        "enrich",
        "enrich.py",
        "Enrich",
        "enrich",
        "Data",
        ":material/auto_fix_high:",
        chapter="03_Data_Enrichment",
        tabs=(("derived_variables", ("Handle Missing Data", "Derived variables")),),
    ),
    PageDef(
        "explore",
        "explore.py",
        "Statistics & correlations",
        "explore",
        "Explore",
        ":material/analytics:",
        chapter="01_Home",
    ),
    PageDef(
        "visualize",
        "visualize.py",
        "Visualize",
        "visualize",
        "Explore",
        ":material/bar_chart:",
        chapter="04_Visualization",
    ),
    PageDef(
        "which_test",
        "which_test.py",
        "Which test?",
        "which-test",
        "Analyze",
        ":material/help_center:",
        chapter="05_Epidemiology",
    ),
    PageDef(
        "analyze",
        "analyze.py",
        "Hypothesis tests & epidemiology",
        "analyze",
        "Analyze",
        ":material/science:",
        chapter="05_Epidemiology",
    ),
    PageDef(
        "survival",
        "survival.py",
        "Survival & regression",
        "survival",
        "Analyze",
        ":material/timeline:",
        chapter="05_Epidemiology",
    ),
    PageDef(
        "variable_map",
        "variable_map.py",
        "Variable map",
        "variable-map",
        "Knowledge",
        ":material/account_tree:",
        chapter="06_Data_Insight",
    ),
    PageDef(
        "documents",
        "documents.py",
        "Documents",
        "documents",
        "Knowledge",
        ":material/description:",
        chapter="07_Document_Insight",
    ),
    PageDef(
        "reproduce",
        "reproduce.py",
        "Versions & traces",
        "reproduce",
        "Reproduce",
        ":material/history:",
        chapter="08_Reproduction",
    ),
    PageDef(
        "dossier",
        "dossier.py",
        "Study dossier",
        "dossier",
        "Reproduce",
        ":material/description:",
        chapter="08_Reproduction",
    ),
    PageDef("handbook", "handbook.py", "Handbook", "handbook", "Help", ":material/menu_book:"),
    PageDef("settings", "settings.py", "Settings", "settings", "Help", ":material/settings:"),
    PageDef("about", "about.py", "About", "about", "Help", ":material/info:"),
    PageDef("users", "users.py", "Users", "users", "Admin", ":material/group:", admin=True),
    PageDef(
        "ai_quality",
        "ai_quality.py",
        "AI quality",
        "ai-quality",
        "Admin",
        ":material/monitoring:",
        admin=True,
    ),
)

_BY_KEY = {p.key: p for p in PAGES}


def page(key: str) -> PageDef:
    return _BY_KEY[key]


def page_path(key: str) -> str:
    """Path of the page file relative to ``main.py`` (what ``st.page_link`` and AppTest expect)."""
    return f"{VIEWS_DIR}/{page(key).file}"


def opened_tabs(key: str, value: object) -> tuple[str, ...]:
    """The labels of the tabs of page ``key`` that the value of TAB_PARAM opens, outer tab first; none for a value the
    page does not declare (an old or mistyped link opens the page as usual)."""
    return dict(page(key).tabs).get(str(value), ()) if value else ()


def tab_query(key: str, tab: str | None) -> dict[str, str] | None:
    """The query parameters of a link to page ``key`` opened on ``tab`` (none without one). A tab the page does not
    declare is refused, so that a mistyped link fails in the tests instead of opening the page as usual."""
    if tab is None:
        return None
    if tab not in dict(page(key).tabs):
        raise ValueError(f"the page {key!r} declares no tab {tab!r}")
    return {TAB_PARAM: tab}


def visible_pages(is_admin: bool) -> list[PageDef]:
    return [p for p in PAGES if is_admin or not p.admin]


def build_navigation(is_admin: bool) -> dict[str, list]:
    """Section -> ``st.Page`` objects, in registry order (the empty section has no header)."""
    import streamlit as st

    sections: dict[str, list] = {}
    for p in visible_pages(is_admin):
        sections.setdefault(p.section, []).append(
            st.Page(
                page_path(p.key), title=p.title, url_path=p.url_path, icon=p.icon, default=p.default
            )
        )
    return sections


__all__ = [
    "NAV_POSITION",
    "PAGES",
    "TAB_PARAM",
    "PageDef",
    "build_navigation",
    "opened_tabs",
    "page",
    "page_path",
    "tab_query",
    "visible_pages",
]
