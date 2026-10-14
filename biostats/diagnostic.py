"""Diagnostic-accuracy measures with confidence intervals (audit S10)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

from biostats._types import CI, RANDOM_STATE
from biostats.references import cite


def _wilson(count: int, nobs: int, alpha: float) -> CI:
    if nobs == 0:
        return CI(lower=float("nan"), upper=float("nan"), level=1 - alpha, method="Wilson")
    lo, hi = proportion_confint(count, nobs, alpha=alpha, method="wilson")
    return CI(lower=float(lo), upper=float(hi), level=1 - alpha, method="Wilson")


def _log_ci(estimate: float, se_log: float, alpha: float, method: str) -> CI | None:
    if not np.isfinite(estimate) or estimate <= 0 or not np.isfinite(se_log):
        return None
    z = stats.norm.ppf(1 - alpha / 2)
    return CI(
        lower=float(math.exp(math.log(estimate) - z * se_log)),
        upper=float(math.exp(math.log(estimate) + z * se_log)),
        level=1 - alpha,
        method=method,
    )


@dataclass
class DiagnosticAccuracy:
    tp: int
    fp: int
    fn: int
    tn: int
    sensitivity: float
    sensitivity_ci: CI
    specificity: float
    specificity_ci: CI
    ppv: float
    ppv_ci: CI
    npv: float
    npv_ci: CI
    accuracy: float
    prevalence_observed: float
    lr_positive: float | None
    lr_positive_ci: CI | None
    lr_negative: float | None
    lr_negative_ci: CI | None
    dor: float | None
    dor_ci: CI | None
    youden: float
    ppv_at_prevalence: float | None = None
    npv_at_prevalence: float | None = None
    prevalence_assumed: float | None = None
    references: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def diagnostic_accuracy(
    tp: int, fp: int, fn: int, tn: int, *, alpha: float = 0.05, prevalence: float | None = None
) -> DiagnosticAccuracy:
    """Sensitivity, specificity, predictive values (Wilson CIs), likelihood ratios (Simel log CIs), DOR, Youden."""
    tp, fp, fn, tn = int(tp), int(fp), int(fn), int(tn)
    n_pos, n_neg = tp + fn, fp + tn
    n = n_pos + n_neg
    warnings: list[str] = []
    sens = tp / n_pos if n_pos else float("nan")
    spec = tn / n_neg if n_neg else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    accuracy = (tp + tn) / n if n else float("nan")
    prev_obs = n_pos / n if n else float("nan")

    lr_pos = sens / (1 - spec) if n_pos and n_neg and spec < 1 else None
    lr_neg = (1 - sens) / spec if n_pos and n_neg and spec > 0 else None
    lr_pos_ci = (
        _log_ci(lr_pos, math.sqrt((1 - sens) / tp + spec / fp), alpha, "Simel log")
        if lr_pos and tp and fp
        else None
    )
    lr_neg_ci = (
        _log_ci(lr_neg, math.sqrt(sens / fn + (1 - spec) / tn), alpha, "Simel log")
        if lr_neg and fn and tn
        else None
    )
    if lr_pos is None or lr_neg is None:
        warnings.append(
            "a likelihood ratio is undefined (sensitivity or specificity equals 0 or 1)"
        )

    cells = [tp, fp, fn, tn]
    if all(c > 0 for c in cells):
        dor = (tp * tn) / (fp * fn)
        dor_ci = _log_ci(dor, math.sqrt(sum(1 / c for c in cells)), alpha, "Woolf log")
    elif n > 0:
        adj = [c + 0.5 for c in cells]
        dor = (adj[0] * adj[3]) / (adj[1] * adj[2])
        dor_ci = _log_ci(
            dor, math.sqrt(sum(1 / c for c in adj)), alpha, "Woolf log, Haldane-Anscombe 0.5 added"
        )
        warnings.append(
            "a cell is zero: the diagnostic odds ratio uses the Haldane-Anscombe correction"
        )
    else:
        dor, dor_ci = None, None

    ppv_prev = npv_prev = None
    if prevalence is not None and n_pos and n_neg:
        ppv_prev = sens * prevalence / (sens * prevalence + (1 - spec) * (1 - prevalence))
        npv_prev = spec * (1 - prevalence) / (spec * (1 - prevalence) + (1 - sens) * prevalence)

    return DiagnosticAccuracy(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        sensitivity=sens,
        sensitivity_ci=_wilson(tp, n_pos, alpha),
        specificity=spec,
        specificity_ci=_wilson(tn, n_neg, alpha),
        ppv=ppv,
        ppv_ci=_wilson(tp, tp + fp, alpha),
        npv=npv,
        npv_ci=_wilson(tn, tn + fn, alpha),
        accuracy=accuracy,
        prevalence_observed=prev_obs,
        lr_positive=lr_pos,
        lr_positive_ci=lr_pos_ci,
        lr_negative=lr_neg,
        lr_negative_ci=lr_neg_ci,
        dor=dor,
        dor_ci=dor_ci,
        youden=(sens + spec - 1) if n_pos and n_neg else float("nan"),
        ppv_at_prevalence=ppv_prev,
        npv_at_prevalence=npv_prev,
        prevalence_assumed=prevalence,
        references=cite("wilson1927", "simel1991", "youden1950"),
        warnings=warnings,
    )


def _binary(y_true: Sequence, pos_label) -> np.ndarray:
    return np.asarray([v == pos_label for v in y_true], dtype=bool)


def delong_auc(y_true: Sequence, y_score: Sequence, *, pos_label=None, alpha: float = 0.05) -> dict:
    """AUC with the DeLong (1988) variance via placement values (Sun & Xu 2014); normal CI clipped to [0, 1]."""
    scores = np.asarray(pd.to_numeric(pd.Series(list(y_score)), errors="coerce"), dtype=float)
    labels = list(y_true)
    mask = ~np.isnan(scores)
    scores = scores[mask]
    labels = [labels[i] for i in np.flatnonzero(mask)]
    if pos_label is None:
        uniques = sorted(set(labels), key=str)
        if len(uniques) != 2:
            raise ValueError("the outcome must be binary")
        pos_label = uniques[1]
    pos = _binary(labels, pos_label)
    x, y = scores[pos], scores[~pos]
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        raise ValueError("both outcome classes are required")
    # placement values (ties count one half), by sorted search: O((m + n) log(m + n)) instead of m x n
    ys, xs = np.sort(y), np.sort(x)
    below = np.searchsorted(ys, x, side="left")
    v10 = (below + 0.5 * (np.searchsorted(ys, x, side="right") - below)) / n
    not_above = np.searchsorted(xs, y, side="right")
    v01 = (m - not_above + 0.5 * (not_above - np.searchsorted(xs, y, side="left"))) / m
    auc = float(v10.mean())
    var = (np.var(v10, ddof=1) / m if m > 1 else 0.0) + (np.var(v01, ddof=1) / n if n > 1 else 0.0)
    z = stats.norm.ppf(1 - alpha / 2)
    se = math.sqrt(var)
    ci = CI(
        lower=float(max(0.0, auc - z * se)),
        upper=float(min(1.0, auc + z * se)),
        level=1 - alpha,
        method="DeLong",
    )
    return {
        "auc": auc,
        "variance": float(var),
        "se": se,
        "ci": ci,
        "n_positive": m,
        "n_negative": n,
        "pos_label": pos_label,
        "references": cite("delong1988", "sun2014"),
    }


def confusion_at(
    y_true: Sequence, y_score: Sequence, cutoff: float, *, pos_label
) -> tuple[int, int, int, int]:
    """(tp, fp, fn, tn) when ``score >= cutoff`` is called positive."""
    scores = np.asarray(pd.to_numeric(pd.Series(list(y_score)), errors="coerce"), dtype=float)
    pos = _binary(list(y_true), pos_label)
    pred = scores >= cutoff
    tp = int(np.sum(pred & pos))
    fp = int(np.sum(pred & ~pos))
    fn = int(np.sum(~pred & pos))
    tn = int(np.sum(~pred & ~pos))
    return tp, fp, fn, tn


def youden_cutoff(
    y_true: Sequence,
    y_score: Sequence,
    *,
    pos_label=None,
    alpha: float = 0.05,
    n_boot: int = 500,
    seed: int = RANDOM_STATE,
) -> dict:
    """Cut-off maximising Youden's J on the sample (optimistic by construction), with a bootstrap CI for the cut-off."""
    from sklearn.metrics import roc_curve

    scores = np.asarray(pd.to_numeric(pd.Series(list(y_score)), errors="coerce"), dtype=float)
    labels = list(y_true)
    mask = ~np.isnan(scores)
    scores = scores[mask]
    labels = [labels[i] for i in np.flatnonzero(mask)]
    if pos_label is None:
        pos_label = sorted(set(labels), key=str)[1]
    y = _binary(labels, pos_label).astype(int)

    def _best(yy: np.ndarray, ss: np.ndarray) -> tuple[float, float, float]:
        fpr, tpr, thr = roc_curve(yy, ss)
        j = tpr - fpr
        i = int(np.argmax(j))
        return float(thr[i]), float(tpr[i]), float(1 - fpr[i])

    cutoff, sens, spec = _best(y, scores)
    ci = None
    if n_boot > 0 and y.sum() >= 5 and (len(y) - y.sum()) >= 5:
        rng = np.random.default_rng(seed)
        boots = []
        idx = np.arange(len(y))
        for _ in range(n_boot):
            b = rng.choice(idx, len(idx), replace=True)
            if y[b].sum() == 0 or y[b].sum() == len(b):
                continue
            boots.append(_best(y[b], scores[b])[0])
        if len(boots) > 50:
            lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            ci = CI(
                lower=float(lo),
                upper=float(hi),
                level=1 - alpha,
                method=f"bootstrap percentile ({len(boots)})",
            )
    return {
        "cutoff": cutoff,
        "sensitivity": sens,
        "specificity": spec,
        "youden": sens + spec - 1,
        "ci": ci,
        "pos_label": pos_label,
        "note": "Chosen on the same data it is evaluated on: sensitivity and specificity at this cut-off are optimistic; "
        "confirm on independent data.",
        "references": cite("youden1950"),
    }


__all__ = [
    "DiagnosticAccuracy",
    "confusion_at",
    "delong_auc",
    "diagnostic_accuracy",
    "youden_cutoff",
]
