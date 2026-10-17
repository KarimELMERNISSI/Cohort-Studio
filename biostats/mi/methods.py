"""
Imputation methods for one incomplete variable: the conditional models of the chained equations.

Each method receives the observed values of the variable, the predictor matrices of the observed and of
the missing rows (without intercept) and a numpy ``Generator``, and returns values for the missing rows.
They follow R mice (van Buuren & Groothuis-Oudshoorn 2011):

========= ==================================================================================================
pmm       predictive mean matching (mice.impute.pmm): Bayesian linear-regression draw (ridge 1e-5), type-1
          matching, one of the 5 nearest donors; imputes observed values only
norm      Bayesian linear regression (mice.impute.norm): prediction from the drawn coefficients plus noise
logreg    logistic regression with augmentation against perfect prediction (mice.impute.logreg; White,
          Daniel & Royston 2010), coefficients drawn from their approximate posterior
polyreg   multinomial logistic regression with the same augmentation (mice.impute.polyreg); the observed
          rows are bootstrapped so that the draw carries parameter uncertainty (mice draws none here)
rf        random-forest donors (mice.impute.rf; Doove, van Buuren & Dusseldorp 2014): trees grown on
          bootstrap samples, a donor drawn among the observed rows sharing a leaf with the missing row
sample    random draw from the observed values (start of the chains)
========= ==================================================================================================
"""

from __future__ import annotations

import math

import numpy as np

from biostats.references import REFERENCES

REFERENCES.setdefault(
    "white2010",
    "White IR, Daniel R, Royston P (2010). Avoiding bias due to perfect prediction in multiple imputation of "
    "incomplete categorical variables. Comput Stat Data Anal 54:2267-2275.",
)
REFERENCES.setdefault(
    "doove2014",
    "Doove LL, van Buuren S, Dusseldorp E (2014). Recursive partitioning for missing data imputation in the "
    "presence of interaction effects. Comput Stat Data Anal 72:92-104.",
)
REFERENCES.setdefault(
    "little1988_pmm",
    "Little RJA (1988). Missing-data adjustments in large surveys. J Bus Econ Stat 6:287-296.",
)


def _with_intercept(x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x])


def _expit(eta: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(eta, -40, 40)))


def _cholesky(v: np.ndarray) -> np.ndarray:
    """Lower Cholesky factor of a symmetric matrix, with a small jitter when it is not positive definite."""
    sym = (v + v.T) / 2
    try:
        return np.linalg.cholesky(sym)
    except np.linalg.LinAlgError:
        jitter = 1e-10 * max(float(np.mean(np.diag(sym))), 1e-12)
        return np.linalg.cholesky(sym + np.eye(len(sym)) * jitter)


def norm_draw(
    y: np.ndarray, x: np.ndarray, rng: np.random.Generator, *, ridge: float = 1e-5
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    mice's .norm.draw: least-squares coefficients, sigma drawn from its posterior (scaled inverse chi-square)
    and coefficients drawn from N(beta, sigma^2 V), V = (X'X + ridge diag(X'X))^-1. ``x`` includes the intercept.
    """
    n, p = x.shape
    df = max(n - p, 1)
    xtx = x.T @ x
    try:  # normal equations (the predictors are cleared of dependencies beforehand); QR when singular
        beta_hat = np.linalg.solve(xtx, x.T @ y)
    except np.linalg.LinAlgError:
        beta_hat = np.linalg.lstsq(x, y, rcond=None)[0]
    v = np.linalg.inv(xtx + np.diag(np.diag(xtx) * ridge))
    resid = y - x @ beta_hat
    sigma_star = math.sqrt(float(resid @ resid) / rng.chisquare(df))
    beta_star = beta_hat + (_cholesky(v) @ rng.standard_normal(p)) * sigma_star
    return beta_hat, beta_star, sigma_star


def match_donors(
    yhat_obs: np.ndarray, yhat_mis: np.ndarray, donors: int, rng: np.random.Generator
) -> np.ndarray:
    """For each missing row, the index of one of the ``donors`` observed rows with the closest prediction."""
    n = len(yhat_obs)
    k = max(1, min(donors, n))
    order = np.argsort(yhat_obs, kind="stable")
    sorted_obs = yhat_obs[order]
    width = min(2 * k, n)
    start = np.clip(np.searchsorted(sorted_obs, yhat_mis) - k, 0, n - width)
    window = start[:, None] + np.arange(width)[None, :]
    distance = np.abs(sorted_obs[window] - yhat_mis[:, None])
    distance += rng.random(distance.shape) * 1e-12 * (1 + np.abs(distance))  # ties broken at random
    nearest = (
        np.argpartition(distance, k - 1, axis=1)[:, :k]
        if width > k
        else np.tile(np.arange(width), (len(yhat_mis), 1))
    )
    pick = nearest[np.arange(len(yhat_mis)), rng.integers(0, k, len(yhat_mis))]
    return order[window[np.arange(len(yhat_mis)), pick]]


def pmm(
    y_obs: np.ndarray,
    x_obs: np.ndarray,
    x_mis: np.ndarray,
    rng: np.random.Generator,
    *,
    donors: int = 5,
    ridge: float = 1e-5,
) -> np.ndarray:
    """Predictive mean matching, type 1: observed rows predicted with beta-hat, missing rows with beta-star."""
    xo, xm = _with_intercept(x_obs), _with_intercept(x_mis)
    beta_hat, beta_star, _ = norm_draw(y_obs, xo, rng, ridge=ridge)
    return y_obs[match_donors(xo @ beta_hat, xm @ beta_star, donors, rng)]


def norm(
    y_obs: np.ndarray,
    x_obs: np.ndarray,
    x_mis: np.ndarray,
    rng: np.random.Generator,
    *,
    ridge: float = 1e-5,
) -> np.ndarray:
    """Bayesian linear regression: drawn coefficients plus normal noise with the drawn sigma."""
    _, beta_star, sigma_star = norm_draw(y_obs, _with_intercept(x_obs), rng, ridge=ridge)
    xm = _with_intercept(x_mis)
    return xm @ beta_star + rng.standard_normal(len(xm)) * sigma_star


def augment(
    codes_obs: np.ndarray, x_obs: np.ndarray, x_all: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    mice's augment(): 2 p k pseudo-observations (each predictor at its mean +/- half an SD, the others at their
    mean, every outcome category) with total weight p + 1, so that no category is perfectly predicted.
    Returns the augmented predictors, outcome codes and weights of the observed rows.
    """
    p = x_all.shape[1]
    weights = np.ones(len(codes_obs))
    if p == 0:
        return x_obs, codes_obs, weights
    nr = 2 * p * k
    mean, sd = x_all.mean(axis=0), x_all.std(axis=0, ddof=1) if len(x_all) > 1 else np.zeros(p)
    lo, hi = x_all.min(axis=0), x_all.max(axis=0)
    rows = np.tile(mean, (nr, 1))
    for j in range(p):
        block = slice(j * 2 * k, (j + 1) * 2 * k)
        rows[block, j] = mean[j] + np.tile([0.5, -0.5], k) * sd[j]
    rows = np.clip(rows, lo, hi)
    codes = np.tile(np.repeat(np.arange(k), 2), p)
    x_aug = np.vstack([x_obs, rows])
    y_aug = np.concatenate([codes_obs, codes])
    w_aug = np.concatenate([weights, np.full(nr, (p + 1) / nr)])
    return x_aug, y_aug, w_aug


def _logistic_fit(
    x: np.ndarray, y: np.ndarray, w: np.ndarray, *, maxit: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted logistic regression by Newton-Raphson; coefficients and unscaled covariance (X'WX)^-1."""
    beta = np.zeros(x.shape[1])
    for _ in range(maxit):
        mu = _expit(x @ beta)
        grad = x.T @ (w * (y - mu))
        hessian = x.T @ (x * (w * mu * (1 - mu))[:, None]) + np.eye(x.shape[1]) * 1e-10
        step = np.linalg.solve(hessian, grad)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-9:
            break
    mu = _expit(x @ beta)
    hessian = x.T @ (x * (w * mu * (1 - mu))[:, None]) + np.eye(x.shape[1]) * 1e-10
    return beta, np.linalg.inv(hessian)


def logreg(
    y01_obs: np.ndarray, x_obs: np.ndarray, x_mis: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Binary variable (codes 0/1): augmented logistic regression, drawn coefficients, Bernoulli draws."""
    x_all = np.vstack([x_obs, x_mis])
    xa, ya, wa = augment(y01_obs.astype(int), x_obs, x_all, 2)
    beta, cov = _logistic_fit(_with_intercept(xa), ya.astype(float), wa)
    beta_star = beta + _cholesky(cov) @ rng.standard_normal(len(beta))
    prob = _expit(_with_intercept(x_mis) @ beta_star)
    return (rng.random(len(prob)) <= prob).astype(int)


def polyreg(
    codes_obs: np.ndarray, k: int, x_obs: np.ndarray, x_mis: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Unordered variable with k categories (codes 0..k-1): augmented multinomial model on a bootstrap sample."""
    from sklearn.linear_model import LogisticRegression

    x_all = np.vstack([x_obs, x_mis])
    boot = rng.integers(0, len(codes_obs), len(codes_obs))
    xa, ya, wa = augment(codes_obs[boot].astype(int), x_obs[boot], x_all, k)
    if xa.shape[1] == 0:
        counts = np.bincount(ya, weights=wa, minlength=k)
        prob = np.tile(counts / counts.sum(), (len(x_mis), 1))
    else:
        center, scale = x_all.mean(axis=0), x_all.std(axis=0)
        scale[scale == 0] = 1.0
        model = LogisticRegression(C=np.inf, max_iter=300)
        model.fit((xa - center) / scale, ya, sample_weight=wa)
        prob = np.zeros((len(x_mis), k))
        prob[:, model.classes_] = model.predict_proba((x_mis - center) / scale)
    cumulative = np.cumsum(prob, axis=1)
    draws = rng.random(len(x_mis))[:, None]
    return np.minimum((draws > cumulative).sum(axis=1), k - 1)


def rf(
    y_obs: np.ndarray,
    x_obs: np.ndarray,
    x_mis: np.ndarray,
    rng: np.random.Generator,
    *,
    categorical: bool,
    ntree: int = 10,
) -> np.ndarray:
    """Random-forest donors: a missing row takes the value of an observed row in one of its leaves (all trees)."""
    if x_obs.shape[1] == 0 or len(y_obs) < 2:
        return sample(y_obs, len(x_mis), rng)
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    seed = int(rng.integers(0, 2**31 - 1))
    if categorical:
        forest = RandomForestClassifier(
            n_estimators=ntree, max_features="sqrt", min_samples_leaf=1, random_state=seed
        )
    else:
        forest = RandomForestRegressor(
            n_estimators=ntree, max_features=1 / 3, min_samples_leaf=5, random_state=seed
        )
    forest.fit(x_obs, y_obs)
    leaves_obs = forest.apply(x_obs)
    leaves_mis = forest.apply(x_mis)
    out = np.empty(len(x_mis), dtype=y_obs.dtype)
    groups = [
        {leaf: np.flatnonzero(leaves_obs[:, t] == leaf) for leaf in np.unique(leaves_obs[:, t])}
        for t in range(leaves_obs.shape[1])
    ]
    for i in range(len(x_mis)):
        pools = [
            groups[t].get(leaves_mis[i, t], np.empty(0, dtype=int)) for t in range(len(groups))
        ]
        sizes = np.array([len(p) for p in pools], dtype=float)
        if sizes.sum() == 0:
            out[i] = y_obs[rng.integers(0, len(y_obs))]
            continue
        t = rng.choice(len(pools), p=sizes / sizes.sum())
        out[i] = y_obs[pools[t][rng.integers(0, len(pools[t]))]]
    return out


def sample(y_obs: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Random draw (with replacement) from the observed values."""
    return y_obs[rng.integers(0, len(y_obs), n)]


__all__ = [
    "augment",
    "logreg",
    "match_donors",
    "norm",
    "norm_draw",
    "pmm",
    "polyreg",
    "rf",
    "sample",
]
