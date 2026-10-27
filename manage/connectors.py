"""
The connector settings at start-up (M5b, step 4): a setting that names no connector of this version stops start-up.

``check_connector_settings(settings)`` runs in ``main.py`` right after ``utils.settings.load_settings``, inside the same
``try``: it raises SettingsError, so the page and the server log say what to correct before any connector is
chosen. The message names each wrong variable and the valid names, never the value: a mistyped value can be a pasted
key or an address, and the page shows it before sign-in. Each family's registry is asked for its names only, so no
connector module is imported. ``manage_users`` does not import this module: the lazy names of ``manage.rag`` and
``manage.vectorstores`` lead to the AI packages, which a command-line entry point never reaches (rule R5,
tests/test_layering.py).
"""

from __future__ import annotations

from graph.renderers import GRAPH_RENDERERS
from manage.iam.providers import IAM_PROVIDERS
from manage.rag.providers import AI_PROVIDERS
from manage.vectorstores import VECTOR_STORES
from utils.settings import Settings, SettingsError


def connector_setting_problems(settings: Settings) -> list[str]:
    """For each connector setting that names nothing this version offers, the variable and the valid names."""
    checks = (
        (IAM_PROVIDERS, settings.iam.provider),
        (AI_PROVIDERS, settings.ai.provider),
        (VECTOR_STORES, settings.vector_store.engine),
        (GRAPH_RENDERERS, settings.graph.renderer),
    )
    return [problem for registry, value in checks if (problem := registry.setting_problem(value))]


def check_connector_settings(settings: Settings) -> Settings:
    """``settings`` when every connector setting names a connector of this version; SettingsError otherwise."""
    problems = connector_setting_problems(settings)
    if problems:
        raise SettingsError(
            "These settings name nothing this version offers: " + "; ".join(problems) + "."
        )
    return settings


__all__ = ["check_connector_settings", "connector_setting_problems"]
