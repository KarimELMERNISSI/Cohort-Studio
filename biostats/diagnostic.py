"""Diagnostic-accuracy measures with confidence intervals (audit S10): the 2x2 table of a test against a reference
standard, read from a score at a cut-off, from a yes/no test's two columns or from four counts; the ROC curve's
area; a cut-off found on the data."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, stats
from statsmodels.stats.proportion import proportion_confint

from biostats._types import CI, RANDOM_STATE, YOUDEN_BOOTSTRAP
from biostats.references import cite


def _wilson(count: int, nobs: int, alpha: float) -> CI:
    """The Wilson interval of ``count`` of ``nobs``. With none or all of them, its bound on that side is 0 or 1
    exactly (as R's ``prop.test`` gives it): the formula leaves 2.8e-17 or 0.9999999999999999 there for many sizes,
    which a table then wrote as a share strictly inside the interval, "1.00000000 (0.75750599 to 1.00000000)"."""
    if nobs == 0:
        return CI(lower=float("nan"), upper=float("nan"), level=1 - alpha, method=WILSON)
    lo, hi = proportion_confint(count, nobs, alpha=alpha, method="wilson")
    lower = 0.0 if count == 0 else float(lo)
    upper = 1.0 if count == nobs else float(hi)
    return CI(lower=lower, upper=upper, level=1 - alpha, method=WILSON)


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


# The ids kept with a result of the accuracy of a test, never renamed: the record, the dossier and the page read
# them from here.
# How the participants were sampled. The stratified analysis asks the same question of another catalogue
# (biostats.study_designs, where "not stated" is the 2x2 panel's older id "unknown"): the constant is named apart, so
# that one catalogue is never keyed with the other's id.
SERIES = "series"  # taken as they came, before the condition was known (consecutive or random)
TWO_GATE = "two_gate"  # those with and without the condition sampled apart
SAMPLING_NOT_STATED = "not_stated"  # asked and not said: only what every sampling supports
SAMPLINGS = (SAMPLING_NOT_STATED, SERIES, TWO_GATE)  # as offered
# the form of the index test, the cut-off of a score and the way a score points
SCORE_FORM, YES_NO_FORM = "number", "yes_no"
INDEX_FORMS = (SCORE_FORM, YES_NO_FORM)
NO_CUTOFF, FIXED_CUTOFF, FOUND_CUTOFF = "No cut-off", "Manual", "Youden-optimal (data-driven)"
CUTOFF_MODES = (NO_CUTOFF, FIXED_CUTOFF, FOUND_CUTOFF)
HIGHER, LOWER = "higher", "lower"
DIRECTIONS = (HIGHER, LOWER)
# the methods of the intervals, as a result's record holds them: a rewording changes what a record says of the same
# numbers (tests/test_accuracy_yes_no.py pins them)
WILSON = "Wilson"
SIMEL = "Simel log"
KOOPMAN = "Koopman score"
DELONG = "DeLong"
WOOLF = "Woolf log"
WOOLF_CORRECTED = "Woolf log, Haldane-Anscombe 0.5 added"
MERCALDO = "logit (Mercaldo)"
SCORE_CARRIED = "from the score bound of the likelihood ratio"


def reads_share(sampling: str) -> bool:
    """Whether the share with the condition in the sample is the setting's prevalence: predictive values are read at
    it, and the share classified correctly weighs sensitivity and specificity by it, for a series only. Sampled
    apart, that share is the sampling's (a test of sensitivity 85% and specificity 90%, read on 100 with and 100
    without, gives a positive predictive value of 89.5% where the setting's, at 2%, is 14.8%)."""
    return sampling == SERIES


def score_ratio_interval(
    x1: int, n1: int, x0: int, n0: int, *, alpha: float = 0.05
) -> tuple[float, float]:
    """
    The score interval of a ratio of two shares, (x1 / n1) / (x0 / n0) (Koopman 1984): the ratios that the score
    test does not reject, their ends found by root-finding. The lower end is 0 when x1 is 0, and there is no upper
    end (infinity) when x0 is 0: the one bound that a zero cell leaves, where the log interval has no value.

    Computed here: the library's own (statsmodels, ``confint_proportions_2indep(method="score", compare="ratio")``)
    is wrong whenever a share is 100%: complex numbers, or an end the score test rejects (the negative likelihood
    ratio of a perfect table, 0 of 20 over 20 of 20, read "0.5 to 0.5", where the interval is 0 to 0.161). Where no
    share is 100% the two agree to 4e-13.
    """
    nan = float("nan")
    if min(n1, n0) <= 0 or (x1 == 0 and x0 == 0):
        return nan, nan
    z = float(stats.norm.ppf(1 - alpha / 2))
    total = n1 + n0

    def statistic(log_ratio: float) -> float:
        ratio = math.exp(log_ratio)
        # the shares that fit best under p1 = ratio x p0 (Miettinen & Nurminen 1985; Farrington & Manning 1990)
        a = total * ratio
        b = -(n1 * ratio + x1 + n0 + x0 * ratio)
        c = x1 + x0
        p0 = (-b - math.sqrt(max(b * b - 4 * a * c, 0.0))) / (2 * a)
        p0 = min(max(p0, 0.0), 1.0)
        p1 = min(max(ratio * p0, 0.0), 1.0)
        variance = p1 * (1 - p1) / n1 + ratio**2 * p0 * (1 - p0) / n0
        gap = x1 / n1 - ratio * x0 / n0
        if variance <= 0:
            return 0.0 if gap == 0 else math.copysign(1e12, gap)
        return max(min(gap / math.sqrt(variance), 1e12), -1e12)

    # the statistic falls as the ratio rises: the lower end solves z = +z, the upper end z = -z
    low_log, high_log = -30.0, 30.0
    lower, upper = 0.0, math.inf
    if x1 > 0 and statistic(low_log) > z > statistic(high_log):
        lower = math.exp(optimize.brentq(lambda t: statistic(t) - z, low_log, high_log, xtol=1e-12))
    if x0 > 0 and statistic(low_log) > -z > statistic(high_log):
        upper = math.exp(optimize.brentq(lambda t: statistic(t) + z, low_log, high_log, xtol=1e-12))
    return float(lower), float(upper)


def _short(value: float) -> str:
    """A bound inside a sentence, as the table of every measure writes it: three significant figures, the whole part
    kept and grouped, no exponent ("as large as 0.13" stood beside "at most 0.126", and "at least 1.3e+03" beside
    "1,303")."""
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return np.format_float_positional(value, precision=3, unique=False, fractional=False, trim="-")


def _likelihood_ratio(
    x1: int, n1: int, x0: int, n0: int, alpha: float, name: str, result: str
) -> tuple[float, CI | None, str]:
    """
    A likelihood ratio (x1 / n1) / (x0 / n0), its interval and, at a zero cell, the sentence that says its one
    bound. ``name`` is "LR+" or "LR-", ``result`` the test result it reads ("positive", "negative"). Simel's log
    interval wherever it has a value: published tables give it, and a table typed from a paper is set beside the
    paper's. At a zero cell the log interval has none, and the score interval gives the bound the data leave.
    """
    level = 1 - alpha
    if x1 > 0 and x0 > 0:
        ratio = (x1 / n1) / (x0 / n0)
        se = math.sqrt((1 - x1 / n1) / x1 + (1 - x0 / n0) / x0)
        return ratio, _log_ci(ratio, se, alpha, SIMEL), ""
    lower, upper = score_ratio_interval(x1, n1, x0, n0, alpha=alpha)
    if x0 == 0:  # no upper bound: nobody without the condition has this result
        return (
            math.inf,
            CI(lower=lower, upper=math.inf, level=level, method=KOOPMAN),
            f"{name} has no upper bound: no participant without the condition tested {result}; it is at least "
            f"{_short(lower)} ({level:.0%} interval)",
        )
    return (
        0.0,
        CI(lower=0.0, upper=upper, level=level, method=KOOPMAN),
        f"{name} is 0: no participant with the condition tested {result}; it may be as large as {_short(upper)} "
        f"({level:.0%} interval)",
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
    ppv: float  # at the share with the condition in the sample: a number for a series only (reads_share)
    ppv_ci: CI
    npv: float
    npv_ci: CI
    accuracy: float  # the share classified correctly: for a series only
    prevalence_observed: float
    lr_positive: (
        float | None
    )  # 0 or infinite at a zero cell, with the one bound of the score interval
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
    # how the participants were sampled (SAMPLINGS): not stated unless said
    sampling: str = SAMPLING_NOT_STATED
    # the likelihood ratio's interval carried to the predictive value at the prevalence given (taken as known)
    ppv_at_prevalence_ci: CI | None = None
    npv_at_prevalence_ci: CI | None = None
    dor_corrected: bool = False  # 0.5 added to every cell (a cell is zero)


def diagnostic_accuracy(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    *,
    alpha: float = 0.05,
    prevalence: float | None = None,
    sampling: str = SAMPLING_NOT_STATED,
) -> DiagnosticAccuracy:
    """
    The measures of a 2x2 table of a test against a reference standard: sensitivity and specificity (Wilson),
    likelihood ratios (Simel's log interval; at a zero cell, the one bound of Koopman's score interval), the
    diagnostic odds ratio (Woolf; 0.5 added to every cell when one is zero), Youden's J.

    ``sampling`` says how the participants were sampled, and what the table then supports: predictive values at the
    share in the sample, and the share classified correctly, for a series only (:func:`reads_share`); without a
    value otherwise, and when it is not stated, so that no reader can show them. ``prevalence``, strictly between 0 and 1, gives predictive
    values at it whatever the sampling, with the likelihood ratio's interval carried through the same formula (the
    logit interval of Mercaldo, Lau & Zhou 2007 where the log interval holds; the prevalence is taken as known).

    A test result that nobody has (no one called positive, or everyone) leaves no likelihood ratio, no diagnostic
    odds ratio and no predictive value of that result; so does a side of the reference standard without
    participants.
    """
    if sampling not in SAMPLINGS:
        raise ValueError(f"unknown sampling; the samplings: {', '.join(SAMPLINGS)}")
    if prevalence is not None and not 0 < float(prevalence) < 1:
        raise ValueError("the prevalence is a share strictly between 0 and 1")
    tp, fp, fn, tn = int(tp), int(fp), int(fn), int(tn)
    n_pos, n_neg = tp + fn, fp + tn
    called, uncalled = tp + fp, fn + tn
    n = n_pos + n_neg
    warnings: list[str] = []
    cited = ["wilson1927", "simel1991", "youden1950"]
    sens = tp / n_pos if n_pos else float("nan")
    spec = tn / n_neg if n_neg else float("nan")
    at_share = reads_share(sampling)
    nan = float("nan")
    ppv = tp / called if at_share and called else nan
    npv = tn / uncalled if at_share and uncalled else nan
    accuracy = (tp + tn) / n if at_share and n else nan
    prev_obs = n_pos / n if n else nan

    both_sides = bool(n_pos and n_neg)
    both_results = bool(called and uncalled)
    lr_pos = lr_neg = None
    lr_pos_ci = lr_neg_ci = None
    dor, dor_ci, corrected = None, None, False
    if both_sides and both_results:
        lr_pos, lr_pos_ci, said = _likelihood_ratio(tp, n_pos, fp, n_neg, alpha, "LR+", "positive")
        if said:
            warnings.append(said)
        lr_neg, lr_neg_ci, said = _likelihood_ratio(fn, n_pos, tn, n_neg, alpha, "LR-", "negative")
        if said:
            warnings.append(said)
        if any(ci is not None and ci.method == KOOPMAN for ci in (lr_pos_ci, lr_neg_ci)):
            cited.append("koopman1984")
        cells = [tp, fp, fn, tn]
        if all(c > 0 for c in cells):
            dor = (tp * tn) / (fp * fn)
            dor_ci = _log_ci(dor, math.sqrt(sum(1 / c for c in cells)), alpha, WOOLF)
        else:
            adj = [c + 0.5 for c in cells]
            corrected = True
            dor = (adj[0] * adj[3]) / (adj[1] * adj[2])
            dor_ci = _log_ci(
                dor,
                math.sqrt(sum(1 / c for c in adj)),
                alpha,
                WOOLF_CORRECTED,
            )
            warnings.append(
                "a cell is zero: the diagnostic odds ratio adds 0.5 to every cell (Haldane-Anscombe)"
            )
            cited.append("glas2003")
    elif both_sides and not called:
        warnings.append(
            "no one is called positive: sensitivity 0% and specificity 100% by construction; no likelihood ratio, "
            "diagnostic odds ratio, positive predictive value or predictive value at a prevalence: a negative "
            "result says nothing beyond the prevalence"
        )
    elif both_sides:
        warnings.append(
            "everyone is called positive: sensitivity 100% and specificity 0% by construction; no likelihood ratio, "
            "diagnostic odds ratio, negative predictive value or predictive value at a prevalence: a positive "
            "result says nothing beyond the prevalence"
        )
    if sampling == TWO_GATE:
        cited.append("lijmer1999")

    ppv_prev = npv_prev = None
    ppv_prev_ci = npv_prev_ci = None
    if prevalence is not None and both_sides and both_results:
        share = float(prevalence)
        odds = share / (1 - share)
        # with a test result that nobody has there is none: the result that everyone has would give the prevalence
        # back (a likelihood ratio of 1 by construction, which is not given either), written as if it were exact
        positive = sens * share + (1 - spec) * (1 - share)
        negative = spec * (1 - share) + (1 - sens) * share
        ppv_prev = sens * share / positive
        npv_prev = spec * (1 - share) / negative

        def after_positive(ratio: float) -> float:
            return 1.0 if math.isinf(ratio) else ratio * odds / (1 + ratio * odds)

        def after_negative(ratio: float) -> float:
            return 0.0 if math.isinf(ratio) else 1 / (1 + ratio * odds)

        level = 1 - alpha
        if lr_pos_ci is not None:
            ppv_prev_ci = CI(
                lower=after_positive(lr_pos_ci.lower),
                upper=after_positive(lr_pos_ci.upper),
                level=level,
                method=_carried(lr_pos_ci),
            )
        if lr_neg_ci is not None:  # the predictive value falls as the likelihood ratio rises
            npv_prev_ci = CI(
                lower=after_negative(lr_neg_ci.upper),
                upper=after_negative(lr_neg_ci.lower),
                level=level,
                method=_carried(lr_neg_ci),
            )
        if ppv_prev_ci is not None or npv_prev_ci is not None:
            cited.append("mercaldo2007")

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
        ppv_ci=_wilson(tp, called if at_share else 0, alpha),
        npv=npv,
        npv_ci=_wilson(tn, uncalled if at_share else 0, alpha),
        accuracy=accuracy,
        prevalence_observed=prev_obs,
        lr_positive=lr_pos,
        lr_positive_ci=lr_pos_ci,
        lr_negative=lr_neg,
        lr_negative_ci=lr_neg_ci,
        dor=dor,
        dor_ci=dor_ci,
        youden=(sens + spec - 1) if both_sides else nan,
        ppv_at_prevalence=ppv_prev,
        npv_at_prevalence=npv_prev,
        prevalence_assumed=None if prevalence is None else float(prevalence),
        references=cite(*cited),
        warnings=warnings,
        sampling=sampling,
        ppv_at_prevalence_ci=ppv_prev_ci,
        npv_at_prevalence_ci=npv_prev_ci,
        dor_corrected=corrected,
    )


def _carried(ci: CI) -> str:
    """The method of a predictive value's interval at a prevalence given, from its likelihood ratio's."""
    return MERCALDO if ci.method == SIMEL else SCORE_CARRIED


def yes_no_table(
    df: pd.DataFrame, reference: str, test: str, *, present: object, positive: object
) -> tuple[int, int, int, int]:
    """
    (tp, fp, fn, tn) of a yes/no test against a reference standard, on the rows where both are recorded:
    ``reference`` = ``present`` is the condition, ``test`` = ``positive`` a positive test. Counted by explicit
    masks, never read by position off a sorted cross-tabulation (the level that sorts last is not always the one
    that means "present"). Each column holds two values in those rows, else there is no 2x2 table to read.
    """
    frame = df[[reference, test]].dropna()
    for column, value, what in ((reference, present, "present"), (test, positive, "positive")):
        held = int(frame[column].nunique())
        if held != 2:
            raise ValueError(
                f"{column} holds {held} value{'s' if held != 1 else ''} in the rows where both are recorded: a 2x2 "
                "table needs two"
            )
        if not bool((frame[column] == value).any()):
            raise ValueError(f"no row of {column} holds the value read as {what}")
    with_it = (frame[reference] == present).to_numpy(dtype=bool)
    called = (frame[test] == positive).to_numpy(dtype=bool)
    return (
        int(np.sum(called & with_it)),
        int(np.sum(called & ~with_it)),
        int(np.sum(~called & with_it)),
        int(np.sum(~called & ~with_it)),
    )


@dataclass(frozen=True)
class Verification:
    """Among the rows where the index test is recorded: those called positive and negative, and how many of each
    hold the reference standard."""

    positive: int
    positive_with_reference: int
    negative: int
    negative_with_reference: int

    @property
    def shares(self) -> tuple[float, float]:
        """The share with the reference standard among those who tested positive, and among those negative."""
        nan = float("nan")
        return (
            self.positive_with_reference / self.positive if self.positive else nan,
            self.negative_with_reference / self.negative if self.negative else nan,
        )

    @property
    def ratio(self) -> float:
        """How many times as often those who tested positive had the reference standard. If it was done because of
        the test result, the odds of the sensitivity read are multiplied by it and those of the specificity divided
        (Begg & Greenes 1983). Infinite when no one who tested negative had it, 0 when no one who tested positive
        did: the check said nothing of a screening programme whose biopsies follow a positive screen only. Without
        a value when a share has none (nobody tested positive, or nobody negative) or when both are 0."""
        positive, negative = self.shares
        if math.isnan(positive) or math.isnan(negative) or (positive == 0 and negative == 0):
            return float("nan")
        return math.inf if negative == 0 else positive / negative

    @property
    def partial(self) -> bool:
        """The reference standard was done unequally by test result: the ratio outside 0.8 to 1.25."""
        ratio = self.ratio
        return bool(not math.isnan(ratio) and not 0.8 <= ratio <= 1.25)

    @property
    def one_sided(self) -> bool:
        """One test result never had the reference standard: sensitivity and specificity cannot be read at all."""
        return self.partial and 0 in (self.positive_with_reference, self.negative_with_reference)

    @property
    def may_be_chance(self) -> bool:
        """The score interval of the ratio of the two shares includes 1: with few participants, shares that differ
        by chance (a reference standard missing at random fired the caution in half the samples of 20 + 80)."""
        if not self.partial:
            return False
        lower, upper = score_ratio_interval(
            self.positive_with_reference,
            self.positive,
            self.negative_with_reference,
            self.negative,
        )
        return bool(lower <= 1 <= upper)


def verification(called: pd.Series, reference: pd.Series) -> Verification:
    """
    The check of partial verification: ``called`` says of each row whether the index test is positive (True),
    negative (False) or not recorded (missing); ``reference`` is the reference standard's column, missing where it
    was not done. The table of the accuracy holds only the rows with both; the dataset knows the others.
    """
    recorded = called.notna()
    positive = recorded & called.where(recorded, False).astype(bool)
    negative = recorded & ~positive
    verified = reference.notna()
    return Verification(
        positive=int(positive.sum()),
        positive_with_reference=int((positive & verified).sum()),
        negative=int(negative.sum()),
        negative_with_reference=int((negative & verified).sum()),
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
        method=DELONG,
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


YOUDEN_MIN_CLASS = 5  # rows of each level below which the cut-off's range is not drawn


def youden_cutoff(
    y_true: Sequence,
    y_score: Sequence,
    *,
    pos_label=None,
    alpha: float = 0.05,
    n_boot: int = YOUDEN_BOOTSTRAP.resamples,
    seed: int = RANDOM_STATE,
) -> dict:
    """
    Cut-off maximising Youden's J on the sample (optimistic by construction), with a percentile bootstrap range for
    the cut-off: the middle of the cut-offs found in the resamples, not a confidence interval (the cut-off is the
    place of a maximum, and the level such a range holds is not known). Where the ROC curve is flat near its top (a
    marker that discriminates little), the resampled cut-offs jump between distant scores and the range is rough
    whatever the number of resamples. The cut-off is always an observed score: the curve's first point, above every
    score, is no cut-off. The rows are put in order (by score, then by level) before they are resampled, so that the
    same data give the same range in any order of the rows. ``caution`` says when the marker points the other way
    (an area under the curve below 0.5), when the cut-off found is the smallest score (everyone is called positive)
    or when it is the largest and next to no one is called positive. It speaks of the marker as given: a caller
    that reverses the marker to read it the other way writes its own words.
    """
    from sklearn.metrics import roc_curve

    scores = np.asarray(pd.to_numeric(pd.Series(list(y_score)), errors="coerce"), dtype=float)
    labels = list(y_true)
    mask = ~np.isnan(scores)
    scores = scores[mask]
    labels = [labels[i] for i in np.flatnonzero(mask)]
    if pos_label is None:
        pos_label = sorted(set(labels), key=str)[1]
    y = _binary(labels, pos_label).astype(int)
    order = np.lexsort((y, scores))
    scores, y = scores[order], y[order]

    def _best(yy: np.ndarray, ss: np.ndarray) -> tuple[float, float, float]:
        fpr, tpr, thr = roc_curve(yy, ss)
        # the first threshold is infinite (nobody is called positive): for a marker that points the other way it won
        # the maximum, with a sensitivity of 0
        j = np.where(np.isfinite(thr), tpr - fpr, -np.inf)
        i = int(np.argmax(j))
        return float(thr[i]), float(tpr[i]), float(1 - fpr[i])

    cutoff, sens, spec = _best(y, scores)
    caution = ""
    if 0 < y.sum() < len(y):
        fpr, tpr, _ = roc_curve(y, scores)
        area = float(np.trapezoid(tpr, fpr))
        if area < 0.5 - 1e-9:  # a half computed as 0.49999999999999994 is a half
            caution = (
                f"The marker points the other way (area under the curve {area:.3f}, below 0.5): higher values go "
                "with the other level, and no cut-off of the form 'score >= cut-off' separates the two. Read the "
                "marker the other way round: lower values go with this level."
            )
        elif cutoff == float(scores.min()):
            caution = (
                "The cut-off found is the smallest observed score: everyone is called positive. The marker does "
                "not separate the two levels."
            )
        elif cutoff == float(scores.max()) and int((scores >= cutoff).sum()) < YOUDEN_MIN_CLASS:
            # the largest score of a grade from 1 to 5 can hold a third of the rows: no caution then
            called = int((scores >= cutoff).sum())
            caution = (
                f"The cut-off found is the largest observed score: {called} of {len(scores):,} "
                f"{'is' if called == 1 else 'are'} called positive."
            )
    ci = None
    drawn = 0
    if n_boot > 0 and min(y.sum(), len(y) - y.sum()) >= YOUDEN_MIN_CLASS:
        rng = np.random.default_rng(seed)
        boots = []
        idx = np.arange(len(y))
        for _ in range(n_boot):
            b = rng.choice(idx, len(idx), replace=True)
            if y[b].sum() == 0 or y[b].sum() == len(b):
                continue
            boots.append(_best(y[b], scores[b])[0])
        if len(boots) > 50:
            drawn = len(boots)
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
        "resamples": drawn,
        "caution": caution,
        "pos_label": pos_label,
        "note": "Chosen on the same data it is evaluated on: sensitivity and specificity at this cut-off are optimistic, "
        "and their intervals do not know that the cut-off was chosen; check them on independent data. Its range is "
        "the middle of the cut-offs found in bootstrap resamples: not a "
        "confidence interval (its level is not known), and rough when the ROC curve is flat near its top, where the "
        "best cut-off jumps between distant scores from one resample to the next.",
        "references": cite("youden1950"),
    }


__all__ = [
    "CUTOFF_MODES",
    "DELONG",
    "DIRECTIONS",
    "FIXED_CUTOFF",
    "FOUND_CUTOFF",
    "HIGHER",
    "INDEX_FORMS",
    "KOOPMAN",
    "LOWER",
    "MERCALDO",
    "NO_CUTOFF",
    "SAMPLINGS",
    "SAMPLING_NOT_STATED",
    "SCORE_CARRIED",
    "SCORE_FORM",
    "SERIES",
    "SIMEL",
    "TWO_GATE",
    "WILSON",
    "WOOLF",
    "WOOLF_CORRECTED",
    "YES_NO_FORM",
    "YOUDEN_MIN_CLASS",
    "DiagnosticAccuracy",
    "Verification",
    "confusion_at",
    "delong_auc",
    "diagnostic_accuracy",
    "reads_share",
    "score_ratio_interval",
    "verification",
    "yes_no_table",
    "youden_cutoff",
]
