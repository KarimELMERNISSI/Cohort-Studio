"""Pure builders turning domain data into a :class:`~graph.model.GraphSpec` (no UI dependencies)."""

from graph.builders.correlation import build_correlation_graph, to_legacy_dicts
from graph.builders.document import (
    DocumentGraph,
    build_document_graph,
    filter_by_documents,
    merge_document_graphs,
)
from graph.builders.taxonomy import (
    VariableNodeType,
    build_taxonomy_graph,
    make_category_id,
    make_formula_id,
    normalize_node_type,
    parse_formula_id,
)

__all__ = [
    "DocumentGraph",
    "VariableNodeType",
    "build_correlation_graph",
    "build_document_graph",
    "build_taxonomy_graph",
    "filter_by_documents",
    "make_category_id",
    "make_formula_id",
    "merge_document_graphs",
    "normalize_node_type",
    "parse_formula_id",
    "to_legacy_dicts",
]
