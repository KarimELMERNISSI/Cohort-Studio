"""
Clinical knowledge graph builder (variable taxonomy + formulas).

Replaces the in-page ``get_graph_data`` of the Data Insight page with a pure, deterministic
function returning a :class:`~graph.model.GraphSpec`. Fixes carried over from the audit:

* one normalised vocabulary for variable types (:class:`VariableNodeType`) instead of seven
  drifted spellings, with a style for every type (including ``Outcome``);
* one idempotent :func:`make_formula_id` so a registry key that already carries the ``FORM_``
  prefix is never prefixed twice;
* legacy (free-text) formula ids are stable slugs instead of process-dependent ``hash()`` values;
* nodes are emitted in a defined order and edges are de-duplicated by id.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from graph.model import EdgeSpec, GraphKind, GraphSpec, LayoutHint, NodeSpec, edge_id
from graph.palette import edge_style_for, legend_entries, node_style_for


class VariableNodeType(StrEnum):
    INPUT_INTERNAL = "Input-Internal"
    DERIVED_INTERNAL = "Derived-Internal"
    INPUT_EXTERNAL = "Input-External"
    DERIVED_EXTERNAL = "Derived-External"
    OUTCOME = "Outcome"
    FORMULA = "Formula"
    CATEGORY = "Category"


_ALIASES: dict[str, VariableNodeType] = {
    "input": VariableNodeType.INPUT_INTERNAL,
    "input-internal": VariableNodeType.INPUT_INTERNAL,
    "internal-input": VariableNodeType.INPUT_INTERNAL,
    "input_internal": VariableNodeType.INPUT_INTERNAL,
    "derived": VariableNodeType.DERIVED_INTERNAL,
    "derived-internal": VariableNodeType.DERIVED_INTERNAL,
    "internal-derived": VariableNodeType.DERIVED_INTERNAL,
    "derived_internal": VariableNodeType.DERIVED_INTERNAL,
    "computed": VariableNodeType.DERIVED_INTERNAL,
    "input-external": VariableNodeType.INPUT_EXTERNAL,
    "external-input": VariableNodeType.INPUT_EXTERNAL,
    "external": VariableNodeType.INPUT_EXTERNAL,
    "derived-external": VariableNodeType.DERIVED_EXTERNAL,
    "external-derived": VariableNodeType.DERIVED_EXTERNAL,
    "outcome": VariableNodeType.OUTCOME,
    "output": VariableNodeType.OUTCOME,
    "formula": VariableNodeType.FORMULA,
    "category": VariableNodeType.CATEGORY,
}

ROLE_FOR: dict[VariableNodeType, str] = {
    VariableNodeType.INPUT_INTERNAL: "input",
    VariableNodeType.DERIVED_INTERNAL: "derived",
    VariableNodeType.INPUT_EXTERNAL: "input_external",
    VariableNodeType.DERIVED_EXTERNAL: "derived_external",
    VariableNodeType.OUTCOME: "outcome",
    VariableNodeType.FORMULA: "formula",
    VariableNodeType.CATEGORY: "category",
}

CATEGORY_LABELS: dict[str, str] = {
    "input": "Input variables",
    "derived": "Derived variables",
    "input_external": "External inputs",
    "derived_external": "External derived variables",
    "outcome": "Outcome variables",
    "formula": "Formulas",
    "category": "Categories",
}


class Relation(StrEnum):
    CATEGORY_MEMBER = "category_member"
    INPUT = "input"
    YIELDS = "yields"
    RELATED = "related"


RELATION_LABELS: dict[str, str] = {
    Relation.CATEGORY_MEMBER.value: "Category membership",
    Relation.INPUT.value: "Formula inputs",
    Relation.YIELDS.value: "Formula outputs",
    Relation.RELATED.value: "Related variables",
}

FORMULA_PREFIX = "FORM_"
CATEGORY_PREFIX = "CAT_"
UNCATEGORIZED = "Uncategorized"


def normalize_node_type(raw: str | VariableNodeType | None) -> VariableNodeType:
    """Map any historical spelling of a variable type to the canonical enum (default: internal input)."""
    if isinstance(raw, VariableNodeType):
        return raw
    key = str(raw or "").strip().lower()
    if not key:
        return VariableNodeType.INPUT_INTERNAL
    if key in _ALIASES:
        return _ALIASES[key]
    for member in VariableNodeType:
        if key == member.value.lower():
            return member
    if "external" in key:
        return (
            VariableNodeType.DERIVED_EXTERNAL
            if "derived" in key
            else VariableNodeType.INPUT_EXTERNAL
        )
    if "derived" in key:
        return VariableNodeType.DERIVED_INTERNAL
    return VariableNodeType.INPUT_INTERNAL


def make_formula_id(key: str) -> str:
    """Node id of a formula; idempotent so prefixed registry keys are not prefixed twice."""
    key = str(key)
    return key if key.startswith(FORMULA_PREFIX) else f"{FORMULA_PREFIX}{key}"


def parse_formula_id(node_id: str) -> str | None:
    """Registry key for a formula node id (strips exactly one prefix), else ``None``."""
    node_id = str(node_id)
    return node_id[len(FORMULA_PREFIX) :] if node_id.startswith(FORMULA_PREFIX) else None


def make_category_id(name: str) -> str:
    return f"{CATEGORY_PREFIX}{name}"


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_") or "formula"


def _json_safe(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, default=str))
    except TypeError, ValueError:
        return str(value)


def _variable_payload(
    var_id: str, attributes: Mapping[str, Any], node_type: VariableNodeType
) -> dict[str, Any]:
    return _json_safe(
        {
            "node_type": node_type.value,
            "original_name": var_id,
            "standard_name": attributes.get("standard_name", var_id),
            "category": attributes.get("category", UNCATEGORIZED),
            "description": attributes.get("description", ""),
            "clinical_usage": attributes.get("clinical_usage", ""),
            "related_formula_ids": list(attributes.get("related_formula_ids", []) or []),
            "source_type": attributes.get("source_type", ""),
        }
    )


class _Draft:
    """Mutable working set; emits an immutable, ordered GraphSpec at the end."""

    def __init__(self, kind: GraphKind, theme: str) -> None:
        self.kind = kind
        self.theme = theme
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: dict[str, EdgeSpec] = {}
        self.restored_hidden = 0

    def add_node(
        self,
        node_id: str,
        label: str,
        node_type: VariableNodeType,
        description: str = "",
        data: dict[str, Any] | None = None,
    ) -> None:
        node_id = str(node_id)
        if node_id in self.nodes:
            return
        self.nodes[node_id] = {
            "label": str(label) if label else node_id,
            "type": node_type,
            "tooltip": str(description or ""),
            "data": dict(data or {}),
        }

    def set_type(self, node_id: str, node_type: VariableNodeType) -> None:
        if node_id in self.nodes:
            self.nodes[node_id]["type"] = node_type
            self.nodes[node_id]["data"]["node_type"] = node_type.value

    def add_edge(
        self,
        relation: Relation,
        source: str,
        target: str,
        *,
        directed: bool,
        label: str = "",
        data: dict[str, Any] | None = None,
    ) -> None:
        if source == target or source not in self.nodes or target not in self.nodes:
            return
        eid = edge_id(relation.value, source, target, undirected=not directed)
        if eid in self.edges:
            return
        self.edges[eid] = EdgeSpec(
            id=eid,
            source=source if directed else min(source, target),
            target=target if directed else max(source, target),
            label=label,
            category=relation.value,
            directed=directed,
            style=edge_style_for(self.kind, relation.value, self.theme),
            data=dict(data or {}),
        )

    def finalize(self, title: str, meta: dict[str, Any]) -> GraphSpec:
        categories = sorted(
            (nid for nid, n in self.nodes.items() if n["type"] is VariableNodeType.CATEGORY),
            key=lambda nid: self.nodes[nid]["label"].lower(),
        )
        formulas = sorted(
            nid for nid, n in self.nodes.items() if n["type"] is VariableNodeType.FORMULA
        )
        variables = [
            nid
            for nid, n in self.nodes.items()
            if n["type"] not in (VariableNodeType.CATEGORY, VariableNodeType.FORMULA)
        ]
        ordered = categories + variables + formulas
        nodes = []
        for nid in ordered:
            n = self.nodes[nid]
            role = ROLE_FOR[n["type"]]
            nodes.append(
                NodeSpec(
                    id=nid,
                    label=n["label"],
                    category=role,
                    style=node_style_for(self.kind, role, self.theme),
                    tooltip=n["tooltip"],
                    data=n["data"],
                )
            )
        edges = sorted(self.edges.values(), key=lambda e: (e.category, e.source, e.target))
        meta = dict(meta)
        meta["restored_hidden_nodes"] = self.restored_hidden
        return GraphSpec(
            kind=self.kind,
            title=title,
            nodes=tuple(nodes),
            edges=tuple(edges),
            directed=True,
            layout=LayoutHint.HIERARCHICAL,
            legend=legend_entries(self.kind, self.theme),
            meta=meta,
        )


def build_taxonomy_graph(
    taxonomy: Mapping[str, Mapping[str, Any]] | None,
    formulas: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    full_taxonomy: Mapping[str, Mapping[str, Any]] | None = None,
    formula_query: str | None = None,
    theme: str = "light",
    title: str = "Clinical knowledge graph",
) -> GraphSpec:
    """Build the variable/formula knowledge graph.

    Args:
        taxonomy: the (possibly filtered) variable taxonomy ``{original_name: attributes}``.
        formulas: formulas registry ``{formula_key: {name, input_variables, output_variable, ...}}``;
            when empty, the legacy free-text ``related_formulas`` strings are parsed instead.
        full_taxonomy: the complete taxonomy, used to restore variables hidden by a filter when a
            formula references them.
        formula_query: case-insensitive text filter; keeps formulas (and their variables) matching it.
        theme: ``"light"`` or ``"dark"`` palette.
    """
    kind = GraphKind.TAXONOMY
    taxonomy = dict(taxonomy or {})
    formulas = dict(formulas or {})
    full_taxonomy = dict(full_taxonomy or {})
    query = (formula_query or "").strip().lower() or None
    draft = _Draft(kind, theme)

    std_to_orig: dict[str, str] = {}
    for key, attrs in taxonomy.items():
        std = str((attrs or {}).get("standard_name", "")).strip().lower()
        if std:
            std_to_orig.setdefault(std, str(key))

    def resolve_id(raw_name: Any) -> str:
        raw = str(raw_name).strip()
        lowered = raw.lower()
        if raw in draft.nodes or raw in taxonomy or raw in full_taxonomy:
            return raw
        if lowered in std_to_orig:
            return std_to_orig[lowered]
        for key, attrs in full_taxonomy.items():
            if str((attrs or {}).get("standard_name", "")).strip().lower() == lowered:
                return str(key)
        if len(raw) > 4:
            for std, key in std_to_orig.items():
                if lowered in std or std in lowered:
                    return key
        return raw

    def formula_matches(f_id: str, f_data: Mapping[str, Any]) -> bool:
        if query is None:
            return True
        haystack = " ".join(
            str(part)
            for part in (
                f_data.get("name", f_id),
                f_data.get("description", ""),
                f_data.get("input_variables", []),
                f_data.get("output_variable", ""),
            )
        ).lower()
        return query in haystack

    def add_category_link(var_id: str, attrs: Mapping[str, Any]) -> None:
        category = str((attrs or {}).get("category") or UNCATEGORIZED)
        cat_id = make_category_id(category)
        draft.add_node(
            cat_id,
            category,
            VariableNodeType.CATEGORY,
            "Data category",
            {"node_type": VariableNodeType.CATEGORY.value, "name": category},
        )
        draft.add_edge(
            Relation.CATEGORY_MEMBER, cat_id, var_id, directed=False, data={"category": category}
        )

    def add_variable(
        var_id: str, attrs: Mapping[str, Any], forced_type: VariableNodeType | None = None
    ) -> None:
        attrs = attrs or {}
        node_type = forced_type or normalize_node_type(attrs.get("node_type"))
        draft.add_node(
            var_id,
            attrs.get("standard_name", var_id),
            node_type,
            attrs.get("description", ""),
            _variable_payload(var_id, attrs, node_type),
        )
        add_category_link(var_id, attrs)

    allowed: set[str] | None = None
    if query is not None and formulas:
        allowed = set()
        for f_id, f_data in formulas.items():
            if formula_matches(str(f_id), f_data or {}):
                allowed.update(
                    resolve_id(i) for i in (f_data or {}).get("input_variables", []) or []
                )
                out = (f_data or {}).get("output_variable")
                if out:
                    allowed.add(resolve_id(out))

    # 1. variables and their categories
    for var_id, attrs in taxonomy.items():
        var_id = str(var_id)
        if allowed is not None and var_id not in allowed:
            continue
        add_variable(var_id, attrs)

    def ensure_variable(resolved: str, *, as_output: bool) -> None:
        """Make sure a formula endpoint exists: restore hidden variables, else add an external one."""
        if resolved in draft.nodes:
            return
        if resolved in full_taxonomy:
            attrs = full_taxonomy[resolved] or {}
            forced = (
                VariableNodeType.DERIVED_INTERNAL
                if as_output
                and normalize_node_type(attrs.get("node_type")) is not VariableNodeType.OUTCOME
                else None
            )
            add_variable(resolved, attrs, forced_type=forced)
            draft.restored_hidden += 1
        else:
            node_type = (
                VariableNodeType.DERIVED_EXTERNAL if as_output else VariableNodeType.INPUT_EXTERNAL
            )
            draft.add_node(
                resolved,
                resolved,
                node_type,
                "",
                {"node_type": node_type.value, "original_name": resolved},
            )

    # 2. formulas: structured registry
    if formulas:
        for f_key, f_data in formulas.items():
            f_key = str(f_key)
            f_data = f_data or {}
            if not formula_matches(f_key, f_data):
                continue
            inputs = [resolve_id(i) for i in (f_data.get("input_variables") or [])]
            output = (
                resolve_id(f_data["output_variable"]) if f_data.get("output_variable") else None
            )
            relevant = (output in taxonomy) if output else False
            relevant = relevant or any(i in taxonomy for i in inputs)
            if not relevant:
                continue
            formula_id = make_formula_id(f_key)
            registry_key = parse_formula_id(formula_id) or f_key
            draft.add_node(
                formula_id,
                f_data.get("name", f_key),
                VariableNodeType.FORMULA,
                f_data.get("description", ""),
                _json_safe(
                    {
                        "node_type": VariableNodeType.FORMULA.value,
                        "formula_key": f_key,
                        "registry_key": registry_key,
                        "name": f_data.get("name", f_key),
                        "description": f_data.get("description", ""),
                        "expression": f_data.get("expression", f_data.get("formula", "")),
                        "markdown_formula": f_data.get("markdown_formula", ""),
                        "input_variables": list(f_data.get("input_variables") or []),
                        "output_variable": f_data.get("output_variable", ""),
                    }
                ),
            )
            for inp in inputs:
                ensure_variable(inp, as_output=False)
                draft.add_edge(
                    Relation.INPUT,
                    inp,
                    formula_id,
                    directed=True,
                    label="input",
                    data={"formula_key": f_key},
                )
            if output:
                ensure_variable(output, as_output=True)
                current = draft.nodes[output]["type"]
                if current in (VariableNodeType.INPUT_INTERNAL, VariableNodeType.DERIVED_INTERNAL):
                    draft.set_type(output, VariableNodeType.DERIVED_INTERNAL)
                elif current in (
                    VariableNodeType.INPUT_EXTERNAL,
                    VariableNodeType.DERIVED_EXTERNAL,
                ):
                    draft.set_type(output, VariableNodeType.DERIVED_EXTERNAL)
                draft.add_edge(
                    Relation.YIELDS,
                    formula_id,
                    output,
                    directed=True,
                    label="yields",
                    data={"formula_key": f_key},
                )
    else:
        # 2b. legacy free-text formulas ("Parameter in: X" / "Result of: Y")
        for var_id, attrs in taxonomy.items():
            var_id = str(var_id)
            if var_id not in draft.nodes:
                continue
            raw_formulas = (attrs or {}).get("related_formulas", []) or []
            if isinstance(raw_formulas, str):
                raw_formulas = [raw_formulas]
            for text in raw_formulas:
                text = str(text)
                match_input = re.search(r"Parameter in:?\s*(.*?)(?:\.|:|\(|$)", text, re.IGNORECASE)
                match_output = re.search(r"Result of:?\s*(.*?)(?:\.|:|\(|$)", text, re.IGNORECASE)
                match = match_input or match_output
                if not match:
                    continue
                form_name = match.group(1).strip()
                formula_id = make_formula_id(_slug(form_name))
                draft.add_node(
                    formula_id,
                    form_name,
                    VariableNodeType.FORMULA,
                    text,
                    {
                        "node_type": VariableNodeType.FORMULA.value,
                        "name": form_name,
                        "legacy_text": text,
                    },
                )
                if match_input:
                    draft.add_edge(Relation.INPUT, var_id, formula_id, directed=True, label="input")
                else:
                    if draft.nodes[var_id]["type"] is VariableNodeType.INPUT_INTERNAL:
                        draft.set_type(var_id, VariableNodeType.DERIVED_INTERNAL)
                    draft.add_edge(
                        Relation.YIELDS, formula_id, var_id, directed=True, label="yields"
                    )

    # 3. declared relationships between variables
    for var_id, attrs in taxonomy.items():
        var_id = str(var_id)
        if var_id not in draft.nodes:
            continue
        for target in (attrs or {}).get("relationships", []) or []:
            target = str(target)
            if target in draft.nodes and target in taxonomy:
                draft.add_edge(Relation.RELATED, var_id, target, directed=False, label="relates to")

    meta = {"n_variables_in": len(taxonomy), "n_formulas_in": len(formulas), "formula_query": query}
    return draft.finalize(title, meta)


__all__ = [
    "CATEGORY_LABELS",
    "CATEGORY_PREFIX",
    "FORMULA_PREFIX",
    "RELATION_LABELS",
    "ROLE_FOR",
    "Relation",
    "VariableNodeType",
    "build_taxonomy_graph",
    "make_category_id",
    "make_formula_id",
    "normalize_node_type",
    "parse_formula_id",
]
