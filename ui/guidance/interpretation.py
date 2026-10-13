"""CI-first interpretation text (ASA statement on p-values, 2016); never 'proves', never 'no effect exists'."""

from __future__ import annotations

from biostats._types import TestResult


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "not available"
    return f"{value:.{digits}g}"


def how_to_read(result: TestResult) -> str:
    """A short paragraph for the result card's 'How to read this' popover."""
    parts: list[str] = []
    if result.estimate is not None and result.estimate_ci is not None:
        parts.append(
            f"The {result.estimate_label or 'estimate'} is {_fmt(result.estimate)} "
            f"(95% CI {result.estimate_ci.as_text(3)}). The interval is the range of true values compatible with the data; "
            "if it includes 0, no difference remains plausible, but so do the differences at both ends of the interval."
        )
    effect = result.primary_effect
    if effect is not None and effect.ci is not None:
        parts.append(
            f"{effect.name} = {_fmt(effect.estimate)} (95% CI {effect.ci.as_text(2)}), a {effect.magnitude() or 'standardised'} "
            "effect by Cohen's conventions; the interval shows which other magnitudes remain compatible."
        )
    elif effect is not None:
        parts.append(
            f"{effect.name} = {_fmt(effect.estimate)} ({effect.magnitude() or 'standardised'} by Cohen's conventions)."
        )
    p = result.p_adjusted if result.p_adjusted is not None else result.p_value
    if p is not None:
        label = "adjusted p" if result.p_adjusted is not None else "p"
        parts.append(
            f"The {label}-value ({_fmt(p, 2)}) is the probability of a result at least this extreme if the true effect were zero; "
            "it is not the probability that there is no effect, and a value above 0.05 is not evidence of absence."
        )
    worries = [a for a in result.assumptions if a.status in ("caution", "violated")]
    if worries:
        parts.append(
            "Diagnostics to keep in mind: "
            + "; ".join(f"{a.name} ({a.detail})" for a in worries)
            + "."
        )
    if result.family == "nonparametric":
        parts.append(
            "Rank-based test: the conclusion is about which group tends to have larger values, not about means."
        )
    parts.append(
        "Report the estimate with its interval, the test, the sample sizes and the reference group."
    )
    return " ".join(parts)


__all__ = ["how_to_read"]
