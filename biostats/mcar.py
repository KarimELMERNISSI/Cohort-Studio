"""
Little's MCAR test with EM estimates of the mean and covariance (audit S15).

Under MCAR the pattern-wise observed means are consistent with one multivariate-normal mean and
covariance; the statistic d^2 = sum_j m_j (ybar_j - mu_j)' Sigma_jj^-1 (ybar_j - mu_j) is chi-square
with df = sum_j p_j - p (Little 1988). mu and Sigma are maximum-likelihood estimates from the EM
algorithm (Dempster 1977), not available-case or pairwise estimates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from biostats.references import cite


def em_mean_cov(
    x: np.ndarray, *, max_iter: int = 200, tol: float = 1e-6
) -> tuple[np.ndarray, np.ndarray, int]:
    """EM for the multivariate normal with values missing (NaN). Returns (mu, Sigma, iterations)."""
    x = np.asarray(x, dtype=float)
    n, p = x.shape
    observed = ~np.isnan(x)
    mu = np.nanmean(x, axis=0)
    mu = np.where(np.isnan(mu), 0.0, mu)
    filled = np.where(observed, x, mu)
    sigma = np.cov(filled, rowvar=False, ddof=0) + 1e-6 * np.eye(p)
    patterns: dict[tuple[bool, ...], np.ndarray] = {}
    for i, row in enumerate(observed):
        patterns.setdefault(tuple(row), []).append(i)  # type: ignore[arg-type]
    pattern_rows = {k: np.array(v) for k, v in patterns.items()}
    iterations = 0
    for iterations in range(1, max_iter + 1):  # noqa: B007 - the final count is reported after the loop
        sum_x = np.zeros(p)
        sum_xx = np.zeros((p, p))
        for pattern, rows in pattern_rows.items():
            obs = np.array(pattern)
            mis = ~obs
            block = x[rows]
            xhat = block.copy()
            cond_cov = np.zeros((p, p))
            if mis.any():
                if obs.any():
                    s_oo = sigma[np.ix_(obs, obs)]
                    s_mo = sigma[np.ix_(mis, obs)]
                    beta = np.linalg.solve(s_oo, s_mo.T).T  # Sigma_mo Sigma_oo^-1
                    xhat[:, mis] = mu[mis] + (block[:, obs] - mu[obs]) @ beta.T
                    cond_cov[np.ix_(mis, mis)] = sigma[np.ix_(mis, mis)] - beta @ s_mo.T
                else:
                    xhat[:, mis] = mu[mis]
                    cond_cov[np.ix_(mis, mis)] = sigma[np.ix_(mis, mis)]
            sum_x += xhat.sum(axis=0)
            sum_xx += xhat.T @ xhat + len(rows) * cond_cov
        new_mu = sum_x / n
        new_sigma = sum_xx / n - np.outer(new_mu, new_mu)
        new_sigma = (new_sigma + new_sigma.T) / 2 + 1e-9 * np.eye(p)
        shift = max(np.max(np.abs(new_mu - mu)), np.max(np.abs(new_sigma - sigma)))
        mu, sigma = new_mu, new_sigma
        if shift < tol:
            break
    return mu, sigma, iterations


def littles_mcar_test(df: pd.DataFrame, *, max_iter: int = 200) -> dict:
    """Little's (1988) chi-square test of MCAR on the numeric columns; EM estimates of mu and Sigma."""
    numeric = df.select_dtypes(include=[np.number]).astype(float)
    numeric = numeric.loc[:, numeric.notna().any()]
    if numeric.shape[1] < 2:
        return {"error": "At least two numeric columns with data are required."}
    if not numeric.isna().any().any():
        return {"error": "No missing values in the numeric columns: nothing to test."}
    x = numeric.to_numpy()
    n, p = x.shape
    observed = ~np.isnan(x)
    patterns = pd.Series([tuple(r) for r in observed])
    mu, sigma, iterations = em_mean_cov(x, max_iter=max_iter)
    d2 = 0.0
    df_total = 0
    n_patterns = 0
    for pattern, idx in patterns.groupby(patterns).groups.items():
        obs = np.array(pattern)
        p_j = int(obs.sum())
        if p_j == 0:
            continue
        rows = x[np.asarray(list(idx))][:, obs]
        m_j = rows.shape[0]
        diff = rows.mean(axis=0) - mu[obs]
        s_oo = sigma[np.ix_(obs, obs)]
        try:
            d2 += float(m_j * diff @ np.linalg.solve(s_oo, diff))
        except np.linalg.LinAlgError:
            d2 += float(m_j * diff @ np.linalg.pinv(s_oo) @ diff)
        df_total += p_j
        n_patterns += 1
    dof = df_total - p
    if dof <= 0:
        return {"error": "Insufficient degrees of freedom (a single missingness pattern)."}
    p_value = float(stats.chi2.sf(d2, dof))
    return {
        "statistic": float(d2),
        "df": int(dof),
        "p_value": p_value,
        "n_patterns": n_patterns,
        "n": int(n),
        "em_iterations": iterations,
        "result": "Evidence against MCAR" if p_value < 0.05 else "No evidence against MCAR",
        "interpretation": (
            "The observed means differ between missingness patterns more than expected under MCAR; the mechanism is "
            "MAR or MNAR (this test cannot tell them apart)."
            if p_value < 0.05
            else "No evidence against MCAR. A non-significant result does not prove MCAR, and the test cannot "
            "distinguish MAR from MNAR; inspect the dependency scan as well."
        ),
        "references": cite("little1988", "dempster1977"),
    }


__all__ = ["em_mean_cov", "littles_mcar_test"]
