"""Compatibility shim: the transformation logic lives in :mod:`enrich.transformation_logic` (no UI code)."""

from enrich.transformation_logic import apply_variable_transformation

__all__ = ["apply_variable_transformation"]
