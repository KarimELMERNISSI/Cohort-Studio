"""
The services of the application, built in one place (M5b step 5, the composition root).

- :class:`Shared`, one per server process, holds nothing of a user's: the settings, the identity provider, the graph
  renderers and the multiple-imputation worker budget, each read from its process-wide instance at each access (a
  test that installs another finds it; a copy kept here would be stale). A command line passes the settings it
  loaded instead.
- :class:`Services`, one per session (a user in a browser tab, or one command-line run): the user's scope, the
  resolver of the files a trace names, and the dataset store and the trace, built on first use with that scope, so
  that a command line that saves nothing creates no folder.

The module is headless (rule R5): it reaches neither Streamlit nor the AI packages. The session builds the AI engine
(``ui.services.ai_engine``), which needs them, and ``main.py`` checks the connector settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

from core.workspace import UserScope
from manage.paths import PathResolver
from utils.settings import Settings

if TYPE_CHECKING:
    from graph.manager import GraphRendererManager
    from manage.compute import WorkerBudget
    from manage.db_manager import DBManager
    from manage.iam.manager import IAMManager
    from manage.transformation_manager import TransformationManager


@dataclass(frozen=True)
class Shared:
    """One per server process; never a user's data."""

    # the settings a command line loaded; None: the process's, read at each access
    given: Settings | None = None

    @property
    def settings(self) -> Settings:
        if self.given is not None:
            return self.given
        from utils.settings import get_settings

        return get_settings()

    @property
    def iam(self) -> IAMManager:
        from manage.iam import get_iam_manager

        return get_iam_manager()

    @property
    def graphs(self) -> GraphRendererManager:
        from graph.manager import get_graph_renderer_manager

        return get_graph_renderer_manager()

    @property
    def mi_workers(self) -> WorkerBudget:
        from manage.compute import mi_workers

        return mi_workers()


def build_shared(settings: Settings | None = None) -> Shared:
    """
    The process's shared services. Builds nothing: each one is built on its first use, as before. ``settings``: those
    a command line loaded (``utils.settings.load_settings``); without them, the process's current ones.
    """
    return Shared(settings)


@dataclass(frozen=True)
class Services:
    """
    One session's services. ``scope`` is the signed-in user's workspace; None for no user (tests, scripts, the command
    line), whose stores use the shared folders. Never passed to a worker process: the budget holds a lock.
    """

    shared: Shared
    scope: UserScope | None
    paths: PathResolver

    @cached_property
    def datasets(self) -> DBManager:
        """The dataset store (saved versions and statistics), in the user's workspace."""
        from manage.db_manager import DBManager

        return DBManager(scope=self.scope)

    @cached_property
    def trace(self) -> TransformationManager:
        """The session's trace, in the user's workspace."""
        from manage.transformation_manager import TransformationManager

        return TransformationManager(scope=self.scope)


def build_services(
    shared: Shared,
    scope: UserScope | None,
    paths: PathResolver,
    *,
    datasets: DBManager | None = None,
    trace: TransformationManager | None = None,
) -> Services:
    """
    One session's services. ``datasets`` and ``trace`` are stores the session already holds for the same user, kept
    rather than built again: a session that outlives a code change keeps its trace.
    """
    built = Services(shared, scope, paths)
    for name, store in (("datasets", datasets), ("trace", trace)):
        if store is not None:
            vars(built)[name] = store  # where the cached property keeps what it built
    return built


def same_user(first: UserScope | None, second: UserScope | None) -> bool:
    """
    Whether two scopes are the same user's workspace: compared by folder, which survives a reload of
    ``core.workspace`` (a reloaded class makes two equal scopes unequal) and a change of role.
    """
    return _folder(first) == _folder(second)


def _folder(scope: UserScope | None) -> str | None:
    return scope.username if scope is not None else None


__all__ = ["Services", "Shared", "build_services", "build_shared", "same_user"]
