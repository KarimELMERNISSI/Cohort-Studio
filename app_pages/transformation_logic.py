"""Compatibility shim: the transformation logic lives in :mod:`enrich.transformation_logic` (no UI code)."""

from enrich.transformation_logic import *  # noqa: F403
from enrich.transformation_logic import (
    apply_variable_transformation,  # noqa: F401 - explicit for static analysers
)
