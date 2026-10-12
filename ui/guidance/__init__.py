"""Guidance: the test chooser, the glossary and CI-first interpretation templates."""

from ui.guidance.glossary import GLOSSARY, define
from ui.guidance.interpretation import how_to_read
from ui.guidance.test_chooser import DESIGNS, DISTRIBUTIONS, GROUPS, OUTCOMES, Recommendation, recommend

__all__ = ["DESIGNS", "DISTRIBUTIONS", "GLOSSARY", "GROUPS", "OUTCOMES", "Recommendation", "define", "how_to_read", "recommend"]
