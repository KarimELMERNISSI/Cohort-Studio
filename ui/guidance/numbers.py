"""
How a number is written in a plan, in a sentence and in a result of the analysis pages: one rule, so that a number
is not written two ways on one screen. The rule itself is ``utils.numbers``, below the pages, so that the Word
dossier writes by it too; the plans, the readings and the screens read it here.
"""

from utils.numbers import (
    MOST,
    alike,
    decimals,
    fixed,
    interval,
    known,
    number,
    p_text,
    percent,
    prevalence_text,
    ratio_parts,
    share_decimals,
    share_parts,
    span,
    together,
    with_interval,
)

__all__ = [
    "MOST",
    "alike",
    "decimals",
    "fixed",
    "interval",
    "known",
    "number",
    "p_text",
    "percent",
    "prevalence_text",
    "ratio_parts",
    "share_decimals",
    "share_parts",
    "span",
    "together",
    "with_interval",
]
