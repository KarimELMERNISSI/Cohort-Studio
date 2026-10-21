"""
Renderer contract.

A renderer turns a :class:`~graph.model.GraphSpec` into a drawing produced by a specific library.
The contract separates *translation* (pure: semantic model -> library primitives, testable without
a browser) from *rendering* (Streamlit side effects), which is what makes the rendering library
swappable: adding a new backend means implementing this class in one module and registering it in
:mod:`graph.renderers` (a connector family on :mod:`core.registry`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from core.registry import Availability
from graph.model import GraphSpec, RenderOptions, SelectionResult


class RendererCapabilities(BaseModel):
    """What a renderer can do, so the UI can adapt (fallback selectors, notices, export buttons)."""

    model_config = ConfigDict(frozen=True)

    supports_selection: bool
    supports_edge_selection: bool
    supports_heat: bool
    supports_node_shapes: bool
    supports_edge_labels: bool
    supports_mixed_direction: bool
    supports_hierarchical: bool
    supports_drag: bool
    supports_html_export: bool
    requires_internet: bool
    telemetry: bool
    license_note: str


class RenderError(RuntimeError):
    """Raised when a renderer cannot draw a spec (the caller may fall back to another renderer)."""


class BaseGraphRenderer(ABC):
    """Abstract base class for graph renderers (mirrors ``manage.iam.base.BaseIAMProvider``)."""

    name: ClassVar[str] = "base"
    display_name: ClassVar[str] = "Base renderer"
    # what the renderer can do, read from the class without drawing (the offline mode, the notices)
    capabilities: ClassVar[RendererCapabilities]

    @classmethod
    def availability(cls) -> Availability:
        """Whether the backing library is installed and compatible, found without importing it; never raises."""
        return Availability(True)

    @abstractmethod
    def translate(self, spec: GraphSpec, options: RenderOptions) -> Any:
        """Translate the semantic model into the library's own payload (no side effects; it may load the library,
        and only here and in ``render``)."""

    @abstractmethod
    def render(self, spec: GraphSpec, options: RenderOptions) -> SelectionResult:
        """Draw the graph in the current Streamlit container and return the user's selection."""

    def export_html(self, spec: GraphSpec, options: RenderOptions) -> bytes | None:
        """Standalone HTML export when the backend supports it, else ``None``."""
        return None


__all__ = ["BaseGraphRenderer", "RenderError", "RendererCapabilities"]
