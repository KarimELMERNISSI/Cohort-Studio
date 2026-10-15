"""
The chained-equations engine (fully conditional specification, van Buuren & Groothuis-Oudshoorn 2011).

Each of the m imputations is an independent chain: the missing cells start as random draws from the
observed values, then, ``maxit`` times, every incomplete variable is imputed from its predictors (the
current values of the others) with its method (:mod:`biostats.mi.methods`). Chain c uses its own random
stream ``SeedSequence(seed, spawn_key=(stratum, c))``: the result is reproducible and does not depend on m
or on the order in which chains run (imputations 1-3 are the same whether m is 3 or 20).

Predictors are numeric blocks (a number, a 0/1 column, or the dummies of a category) kept up to date
variable by variable. Before each fit, predictors that are constant, almost collinear with the variable
or linearly dependent on the others are left out, as mice's ``remove.lindep`` (with scale-free criteria).

Options: imputation within the levels of a complete variable (``by``, e.g. the trial arm); for a Cox
analysis, the event indicator and the Nelson-Aalen cumulative hazard H(T) replace the follow-up time as
predictors (White & Royston 2009).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from biostats.mi import methods as fcs
from biostats.mi.data import MIData
from biostats.mi.spec import (
    CATEGORICAL_METHODS,
    METHODS,
    NUMERIC_METHODS,
    ImputationSpec,
    default_method,
    is_numeric,
)

ENGINE = "cohortstudio-fcs 1"
HAZARD = "Nelson-Aalen H(T)"


@dataclass
class ImputationRun:
    mi: MIData
    spec: ImputationSpec
    methods: dict[str, str]  # resolved method of every imputed variable
    predictors: dict[str, list[str]]  # resolved predictors of every imputed variable
    trace: dict[
        str, dict[str, np.ndarray]
    ]  # variable -> {"mean", "sd"}: (iterations x chains) of the imputed values
    rhat: dict[str, float]  # indicative potential scale reduction of the chain means
    dropped: dict[str, list[str]] = field(default_factory=dict)  # predictors left out at least once
    warnings: list[str] = field(default_factory=list)
    elapsed: float = 0.0


@dataclass
class _Var:
    name: str
    numeric: bool
    levels: list[Any]  # original values, ordered by their text (categories only)
    values: np.ndarray  # float (NaN missing) or int codes (-1 missing)
    missing: np.ndarray

    @property
    def k(self) -> int:
        return len(self.levels)


def _encode(series: pd.Series) -> _Var:
    if is_numeric(series):
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        return _Var(str(series.name), True, [], values, np.isnan(values))
    observed = series.notna().to_numpy()
    text = series.astype(str).to_numpy()
    first: dict[str, Any] = {}
    for raw, label in zip(series.to_numpy()[observed], text[observed], strict=True):
        first.setdefault(label, raw)
    order = sorted(first)
    index = {label: i for i, label in enumerate(order)}
    codes = np.array(
        [index[label] if obs else -1 for label, obs in zip(text, observed, strict=True)], dtype=int
    )
    return _Var(str(series.name), False, [first[label] for label in order], codes, ~observed)


def _block(var: _Var, values: np.ndarray) -> np.ndarray:
    if var.numeric:
        return values.astype(float)[:, None]
    if var.k <= 2:
        return (values == 1).astype(float)[:, None]
    return np.eye(var.k)[values][:, 1:]


def _labels(var: _Var) -> list[str]:
    if var.numeric or var.k <= 2:
        return [var.name]
    return [f"{var.name}={level}" for level in var.levels[1:]]


def nelson_aalen(times: np.ndarray, events: np.ndarray) -> np.ndarray:
    """Nelson-Aalen cumulative hazard at each subject's own follow-up time."""
    t = np.asarray(times, dtype=float)
    e = np.asarray(events, dtype=float)
    unique, inverse = np.unique(t, return_inverse=True)
    deaths = np.bincount(inverse, weights=e, minlength=len(unique))
    at_risk = len(t) - np.concatenate(
        [[0], np.cumsum(np.bincount(inverse, minlength=len(unique)))[:-1]]
    )
    hazard = np.cumsum(deaths / at_risk)
    return hazard[inverse]


def _keep(xo: np.ndarray, yo: np.ndarray, *, maxcor: float = 0.99, eps: float = 1e-4) -> np.ndarray:
    """Predictors kept for one fit: not constant, not almost collinear with the variable, not dependent."""
    p = xo.shape[1]
    if p == 0 or len(yo) < 2 or np.ptp(yo) == 0:
        return np.zeros(p, dtype=bool)
    keep = np.ptp(xo, axis=0) > 0
    if not keep.any():
        return keep
    xc = xo - xo.mean(axis=0)
    yc = yo - yo.mean()
    gram = xc.T @ xc  # one product gives every variance and correlation
    scale = np.sqrt(np.clip(np.diag(gram), 1e-300, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        cor = (xc.T @ yc) / (scale * np.sqrt(yc @ yc))
    keep &= ~(np.abs(np.nan_to_num(cor)) >= maxcor)
    corr = gram / np.outer(scale, scale)
    while keep.sum() > 1:
        idx = np.flatnonzero(keep)
        values, vectors = np.linalg.eigh(corr[np.ix_(idx, idx)])
        if values[0] / values[-1] >= eps:
            break
        keep[idx[int(np.argmax(np.abs(vectors[:, 0])))]] = False
    return keep


def _incomplete(series: pd.Series) -> bool:
    return bool(series.isna().to_numpy().any())


def _check(df: pd.DataFrame, spec: ImputationSpec) -> None:
    missing = [c for c in spec.columns if c not in df.columns]
    if missing:
        raise ValueError(f"unknown column(s): {missing}")
    if not df.index.is_unique:
        raise ValueError("the row ids (index) must be unique")
    for col in spec.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            raise ValueError(
                f"{col} holds dates: convert it to a number (e.g. days) or leave it out of the model"
            )
    for col, method in spec.methods.items():
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r} for {col}; expected one of {METHODS}")
        if col not in spec.columns:
            raise ValueError(f"a method is given for {col}, which is not in the model")
        if is_numeric(df[col]) and method and method not in NUMERIC_METHODS:
            raise ValueError(f"{col} is numeric: use pmm, norm or rf, not {method}")
        if not is_numeric(df[col]) and method and method not in CATEGORICAL_METHODS:
            raise ValueError(f"{col} is categorical: use logreg, polyreg or rf, not {method}")
        if method == "logreg" and df[col].dropna().astype(str).nunique() > 2:
            raise ValueError(f"{col} has more than two categories: use polyreg")
    if spec.by is not None:
        if spec.by not in df.columns:
            raise ValueError(f"unknown stratification column {spec.by}")
        if _incomplete(df[spec.by]):
            raise ValueError(f"the stratification column {spec.by} must be complete")
    if spec.survival is not None:
        for col in spec.survival:
            if col not in df.columns:
                raise ValueError(f"unknown survival column {col}")
            if _incomplete(df[col]):
                raise ValueError(f"the survival column {col} must be complete to compute H(T)")


def _resolve(
    df: pd.DataFrame, spec: ImputationSpec
) -> tuple[dict[str, str], dict[str, list[str]], list[str]]:
    warnings: list[str] = []
    methods: dict[str, str] = {}
    for col in spec.columns:
        if col == spec.by or not _incomplete(df[col]):
            continue
        method = spec.methods.get(col, default_method(df[col]))
        if method:
            methods[col] = method
        else:
            warnings.append(
                f"{col} is incomplete and not imputed, so it is not used as a predictor"
            )
    usable = [c for c in spec.columns if c != spec.by and (c in methods or not _incomplete(df[c]))]
    time_col = spec.survival[0] if spec.survival else None
    predictors: dict[str, list[str]] = {}
    for target in methods:
        chosen = spec.predictors.get(target)
        if chosen is None:
            chosen = [c for c in usable if c != target and c != time_col]
            if spec.survival:
                chosen = (
                    chosen
                    + ([spec.survival[1]] if spec.survival[1] not in chosen else [])
                    + [HAZARD]
                )
        else:
            unknown = [c for c in chosen if c not in usable and c != HAZARD]
            if unknown:
                raise ValueError(
                    f"predictors of {target} that are not usable (unknown or incomplete): {unknown}"
                )
        predictors[target] = [c for c in chosen if c != target]
    return methods, predictors, warnings


def _rhat(trace: np.ndarray) -> float:
    """Gelman-Rubin potential scale reduction of the chain means over the second half of the iterations."""
    half = trace[trace.shape[0] // 2 :]
    n, m = half.shape
    if n < 2 or m < 2:
        return float("nan")
    within = float(half.var(axis=0, ddof=1).mean())
    between = n * float(half.mean(axis=0).var(ddof=1))
    if within == 0:
        return float("nan")
    return float(np.sqrt(((n - 1) / n * within + between / n) / within))


@dataclass
class _ChainOutput:
    draws: dict[str, np.ndarray]
    mean: dict[str, np.ndarray]  # per iteration
    sd: dict[str, np.ndarray]
    dropped: dict[str, set[str]]


def _chain(
    vars_: dict[str, _Var],
    methods: dict[str, str],
    predictors: dict[str, list[str]],
    spec: ImputationSpec,
    stratum: int,
    chain: int,
    tick: Callable[[], None] | None = None,
) -> _ChainOutput:
    """One imputation: an independent chain with its own random stream SeedSequence(seed, (stratum, chain))."""
    rng = np.random.default_rng(np.random.SeedSequence(spec.seed, spawn_key=(stratum, chain)))
    state = {name: var.values.copy() for name, var in vars_.items()}
    for target in methods:
        var = vars_[target]
        state[target][var.missing] = fcs.sample(
            state[target][~var.missing], int(var.missing.sum()), rng
        )
    blocks = {name: _block(var, state[name]) for name, var in vars_.items()}
    labels = {name: _labels(var) for name, var in vars_.items()}
    mean = {t: np.full(spec.maxit, np.nan) for t in methods}
    sd = {t: np.full(spec.maxit, np.nan) for t in methods}
    dropped: dict[str, set[str]] = {t: set() for t in methods}
    for iteration in range(spec.maxit):
        for target, method in methods.items():
            var = vars_[target]
            obs, mis = ~var.missing, var.missing
            names = predictors[target]
            x = np.hstack([blocks[p] for p in names]) if names else np.zeros((len(obs), 0))
            cols = [label for p in names for label in labels[p]]
            yo = state[target][obs]
            keep = _keep(x[obs], yo.astype(float))
            dropped[target].update(c for c, k in zip(cols, keep, strict=True) if not k)
            xo, xm = x[obs][:, keep], x[mis][:, keep]
            if var.numeric:
                if method == "pmm":
                    new = fcs.pmm(yo, xo, xm, rng, donors=spec.donors, ridge=spec.ridge)
                elif method == "norm":
                    new = fcs.norm(yo, xo, xm, rng, ridge=spec.ridge)
                else:
                    new = fcs.rf(yo, xo, xm, rng, categorical=False, ntree=spec.rf_trees)
            elif var.k == 1:
                new = np.zeros(int(mis.sum()), dtype=int)
            elif method == "logreg":
                new = fcs.logreg(yo.astype(int), xo, xm, rng)
            elif method == "polyreg":
                new = fcs.polyreg(yo.astype(int), var.k, xo, xm, rng)
            else:
                new = fcs.rf(yo.astype(int), xo, xm, rng, categorical=True, ntree=spec.rf_trees)
            state[target][mis] = new
            blocks[target] = _block(var, state[target])
            values = np.asarray(new, dtype=float)
            mean[target][iteration] = values.mean() if len(values) else np.nan
            sd[target][iteration] = values.std(ddof=1) if len(values) > 1 else np.nan
        if tick is not None:
            tick()
    draws = {target: state[target][vars_[target].missing] for target in methods}
    return _ChainOutput(draws=draws, mean=mean, sd=sd, dropped=dropped)


def _one_thread() -> None:
    """Worker initialiser: one BLAS thread per process (the processes already use the cores)."""
    from threadpoolctl import threadpool_limits

    threadpool_limits(1)


def _run_chain(job: tuple[Any, ...]) -> _ChainOutput:
    """Process-pool entry point."""
    return _chain(*job)


def impute(
    df: pd.DataFrame,
    spec: ImputationSpec,
    *,
    progress: Callable[[float], None] | None = None,
    workers: int = 1,
) -> ImputationRun:
    """
    Multiple imputation of ``df`` by chained equations; returns the imputed data and the diagnostics.

    ``workers`` > 1 runs the chains in that many processes; the result is identical (one stream per chain).
    """
    started = time.perf_counter()
    _check(df, spec)
    methods, predictors, warnings = _resolve(df, spec)
    strata: list[tuple[int, Any, np.ndarray]] = []
    if spec.by is None:
        strata.append((0, None, np.ones(len(df), dtype=bool)))
    else:
        labels = df[spec.by].astype(str)
        for s, level in enumerate(sorted(labels.unique())):
            strata.append((s, level, (labels == level).to_numpy()))
    total = max(1, len(strata) * spec.m * spec.maxit)
    done = 0

    def tick() -> None:
        nonlocal done
        done += 1
        if progress is not None:
            progress(done / total)

    imputed: dict[str, list[pd.DataFrame]] = {t: [] for t in methods}
    trace: dict[str, dict[str, np.ndarray]] = {}
    dropped: dict[str, set[str]] = {t: set() for t in methods}
    for s, level, rows in strata:
        part = df.loc[rows]
        vars_ = {
            c: _encode(part[c])
            for c in spec.columns
            if c != spec.by and (c in methods or not _incomplete(part[c]))
        }
        for c in spec.columns:
            if c in methods and c not in vars_:
                vars_[c] = _encode(part[c])
        if spec.survival is not None:
            hazard = nelson_aalen(
                part[spec.survival[0]].to_numpy(dtype=float),
                part[spec.survival[1]].to_numpy(dtype=float),
            )
            vars_[HAZARD] = _Var(HAZARD, True, [], hazard, np.zeros(len(part), dtype=bool))
            if spec.survival[1] not in vars_:
                vars_[spec.survival[1]] = _encode(part[spec.survival[1]])
        targets = {t: m for t, m in methods.items() if vars_[t].missing.any()}
        for t in targets:
            if vars_[t].missing.all():
                where = f" in stratum {spec.by} = {level}" if level is not None else ""
                raise ValueError(f"{t} has no observed value{where}: it cannot be imputed")
        suffix = f" [{spec.by} = {level}]" if level is not None else ""
        jobs = [(vars_, targets, predictors, spec, s, chain) for chain in range(spec.m)]
        if workers > 1 and spec.m > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            outputs: list[_ChainOutput | None] = [None] * spec.m
            with ProcessPoolExecutor(
                max_workers=min(workers, spec.m), initializer=_one_thread
            ) as pool:
                futures = {pool.submit(_run_chain, job): k for k, job in enumerate(jobs)}
                for future in as_completed(futures):
                    outputs[futures[future]] = future.result()
                    for _ in range(spec.maxit):
                        tick()
            chains = [o for o in outputs if o is not None]
        else:
            chains = [_chain(*job, tick=tick) for job in jobs]
        draws = {t: [c.draws[t] for c in chains] for t in targets}
        trace_mean = {t: np.column_stack([c.mean[t] for c in chains]) for t in targets}
        trace_sd = {t: np.column_stack([c.sd[t] for c in chains]) for t in targets}
        for c in chains:
            for t, cols in c.dropped.items():
                dropped[t].update(cols)
        for t in targets:
            var = vars_[t]
            index = part.index[var.missing]
            if var.numeric:
                columns = {i + 1: draws[t][i] for i in range(spec.m)}
            else:
                columns = {i + 1: [var.levels[code] for code in draws[t][i]] for i in range(spec.m)}
            imputed[t].append(pd.DataFrame(columns, index=index))
            trace[f"{t}{suffix}"] = {"mean": trace_mean[t], "sd": trace_sd[t]}
    imputations = {t: pd.concat(frames).sort_index() for t, frames in imputed.items() if frames}
    mi = MIData(
        base=df.copy(),
        imputations=imputations,
        m=spec.m,
        methods=dict(methods),
        info={"engine": ENGINE, "spec": spec.model_dump(), "predictors": predictors},
    )
    return ImputationRun(
        mi=mi,
        spec=spec,
        methods=methods,
        predictors=predictors,
        trace=trace,
        rhat={name: _rhat(values["mean"]) for name, values in trace.items()},
        dropped={t: sorted(cols) for t, cols in dropped.items() if cols},
        warnings=warnings,
        elapsed=time.perf_counter() - started,
    )


__all__ = ["ENGINE", "HAZARD", "ImputationRun", "impute", "nelson_aalen"]
