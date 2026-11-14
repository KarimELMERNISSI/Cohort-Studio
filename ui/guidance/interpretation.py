"""CI-first interpretation text (ASA statement on p-values, 2016); never 'proves', never 'no effect exists'."""

from __future__ import annotations

from biostats._types import TestResult
from biostats.assumptions import read_assumptions
from ui.guidance.numbers import number, p_text, with_interval


def _fmt(value: float | None, digits: int = 3) -> str:
    """A number of a reading, by the rule of the results (no exponent: a mean change of 2,473 read "2.47e+03")."""
    if value is None:
        return "not available"
    return number(value, digits)


def _capital(name: str) -> str:
    """An effect size's name at the start of a sentence ("matched-pairs rank-biserial r = ...")."""
    return name[:1].upper() + name[1:]


def how_to_read(result: TestResult) -> str:
    """A short paragraph for the result card's 'How to read this' popover."""
    parts: list[str] = []
    if result.estimate is not None and result.estimate_ci is not None:
        # the estimate and its interval as the result's headline writes them ("-5.01 (95% CI -5.677 to -4.343)"
        # stood under "-5.01, 95% CI -5.68 to -4.34")
        value, bounds = with_interval(result.estimate, result.estimate_ci)
        parts.append(
            f"The {result.estimate_label or 'estimate'} is {value}"
            f"{f' ({bounds})' if bounds else ''}. The interval is the range of true values compatible with the data; "
            "if it includes 0, no difference remains plausible, but so do the differences at both ends of the interval."
        )
    effect = result.primary_effect
    if effect is not None and effect.ci is not None:
        size, limits = with_interval(effect.estimate, effect.ci)
        parts.append(
            f"{_capital(effect.name)} = {size}{f' ({limits})' if limits else ''}, a "
            f"{effect.magnitude() or 'standardised'} "
            "effect by Cohen's conventions; the interval shows which other magnitudes remain compatible."
        )
    elif effect is not None:
        parts.append(
            f"{_capital(effect.name)} = {_fmt(effect.estimate)} ({effect.magnitude() or 'standardised'} by Cohen's "
            "conventions)."
        )
    p = result.p_adjusted if result.p_adjusted is not None else result.p_value
    if p is not None:
        label = "adjusted p" if result.p_adjusted is not None else "p"
        parts.append(
            f"The {label}-value ({'below 0.001' if p < 0.001 else p_text(p).removeprefix('= ')}) is the probability of a "
            "result at least this "
            "extreme if the true effect were zero; "
            "it is not the probability that there is no effect, and a value above 0.05 is not evidence of absence."
        )
    worries = [r for r in read_assumptions(result) if r.status in ("caution", "violated")]
    if worries:
        parts.append(
            "Diagnostics to keep in mind: "
            + "; ".join(f"{r.check} ({r.finding}): {r.meaning}" for r in worries)
        )
    if result.family == "nonparametric":
        # one group, or pairs: there is no second group to have larger values
        parts.append(
            "Rank-based test: the conclusion is about the typical difference (the pseudo-median), not about the mean."
            if len(result.groups or ()) < 2
            else "Rank-based test: the conclusion is about which group tends to have larger values, not about means."
        )
    parts.append(
        "Report the estimate with its interval, the test, the sample sizes and the reference group."
    )
    return " ".join(parts)


__all__ = ["how_to_read"]
