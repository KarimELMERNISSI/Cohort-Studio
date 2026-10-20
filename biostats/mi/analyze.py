"""
Run one analysis on every imputed dataset and pool it.

``analyze_mi(fn, mi)`` calls ``fn(data)`` (or ``fn(data, i)``, to skip costly extras beyond the first
imputations) on each completed dataset, picks the adapter for the type of result, pools the estimates
(:mod:`biostats.mi.pooling`) and returns an :class:`MIResult`: the pooled table, D1/D2 tests, what the
adapter adds (ratios, curves, adjusted means, a pooled t-test), the complete-case analysis for comparison
and the warnings counted across imputations. An analysis that fails on one imputation stops the pooling
(dropping imputations would bias it); a time budget can stop early and labels the result as a preview.
"""

from __future__ import annotations

import inspect
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from biostats.mi.adapters import Adapter, NotPoolableError, adapter_for
from biostats.mi.data import MIData
from biostats.mi.pooling import MultiParameterTest, PooledTable, PooledTerm, pool_terms


class MIAnalysisError(RuntimeError):
    """The analysis failed on one of the imputed datasets."""

    def __init__(self, imputation: int, cause: Exception) -> None:
        super().__init__(f"the analysis failed on imputed dataset {imputation}: {cause}")
        self.imputation = imputation
        self.cause = cause


@dataclass
class MIResult:
    kind: str
    m: int  # imputed datasets pooled
    m_total: int  # imputed datasets available
    table: PooledTable | None
    tests: dict[str, MultiParameterTest] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    complete_case: Any | None = None
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    partial: bool = False
    elapsed: float = 0.0

    def term(self, name: str) -> PooledTerm:
        if self.table is None:
            raise KeyError(name)
        return self.table.term(name)

    def to_frame(self) -> pd.DataFrame:
        return self.table.to_frame() if self.table is not None else pd.DataFrame()


def _takes_index(fn: Callable[..., Any]) -> bool:
    try:
        params = [
            p
            for p in inspect.signature(fn).parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty
        ]
    except TypeError, ValueError:
        return False
    return len(params) >= 2


def analyze_mi(
    fn: Callable[..., Any],
    mi: MIData,
    *,
    columns: Sequence[str] | None = None,
    alpha: float = 0.05,
    complete_case: bool = True,
    time_budget: float | None = None,
    min_imputations: int = 5,
    adapter: Adapter | None = None,
) -> MIResult:
    """
    Pool ``fn`` over the imputed datasets of ``mi``.

    ``columns`` restricts the completed datasets to what the analysis reads (faster). With ``time_budget``
    (seconds) the loop stops once at least ``min_imputations`` datasets are analysed and the budget is spent;
    the result is then flagged ``partial`` with a note.
    """
    takes_index = _takes_index(fn)
    started = time.perf_counter()
    results: list[Any] = []
    partial = False
    floor = max(2, min(min_imputations, mi.m))
    for i, data in mi.completed(columns):
        try:
            results.append(fn(data, i) if takes_index else fn(data))
        except NotPoolableError:
            raise
        except Exception as exc:
            raise MIAnalysisError(i, exc) from exc
        if (
            time_budget is not None
            and i < mi.m
            and i >= floor
            and time.perf_counter() - started > time_budget
        ):
            partial = True
            break
    if len(results) < 2:
        raise ValueError("at least two imputed datasets are needed to pool")
    chosen = adapter or adapter_for(results[0])
    extracts, dropped = chosen.prepare([chosen.extract(r) for r in results])
    notes: list[str] = []
    dfcoms = {e.dfcom for e in extracts}
    dfcom = min(dfcoms)
    if len(dfcoms) > 1:
        notes.append("the complete-data df differ between imputed datasets; the smallest is used")
    table: PooledTable | None = None
    if len(extracts[0].params):
        # terms with their own complete-data df (Welch-type) use the smallest over the imputed datasets
        names = set(extracts[0].term_dfcom)
        if any(set(e.term_dfcom) != names for e in extracts):
            raise ValueError("the imputed datasets give their terms different complete-data df")
        term_dfcom = (
            {name: min(e.term_dfcom[name] for e in extracts) for name in names} if names else None
        )
        common = dict(dfcom=term_dfcom or dfcom, alpha=alpha, scales=extracts[0].scales)
        if all(e.vcov is not None for e in extracts):
            table = pool_terms(
                [e.params for e in extracts], covariances=[e.vcov for e in extracts], **common
            )
        else:
            table = pool_terms(
                [e.params for e in extracts], variances=[e.variances for e in extracts], **common
            )
    tests = chosen.tests(extracts, table)
    extra = chosen.finish(extracts, table, alpha=alpha)
    counts = Counter(w for e in extracts for w in dict.fromkeys(e.warnings))
    warnings = [
        f"{w} (in {k} of {len(extracts)} imputed datasets)" for w, k in counts.items()
    ] + dropped
    if partial:
        notes.append(
            f"Preview: pooled over {len(results)} of {mi.m} imputed datasets (time budget of {time_budget:g} s); "
            "run all of them for the final result."
        )
    cc: Any | None = None
    if complete_case:
        try:
            observed = mi.complete(0, columns)
            cc = fn(observed, 0) if takes_index else fn(observed)
        except Exception as exc:  # the comparison is informative only
            notes.append(f"complete-case analysis unavailable: {exc}")
    return MIResult(
        kind=chosen.kind,
        m=len(results),
        m_total=mi.m,
        table=table,
        tests=tests,
        extra=extra,
        complete_case=cc,
        warnings=warnings,
        notes=notes,
        partial=partial,
        elapsed=time.perf_counter() - started,
    )


__all__ = ["MIAnalysisError", "MIResult", "analyze_mi"]
