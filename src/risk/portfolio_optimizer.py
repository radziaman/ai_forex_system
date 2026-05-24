"""
Portfolio Optimization Engine.

Implements three allocation methods:
  - Mean-Variance (Markowitz)          — analytical solution
  - Risk Parity (equal risk contribution)  — cyclic coordinate descent
  - Hierarchical Risk Parity (HRP)     — Lopez de Prado (2016)

Also provides efficient frontier computation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PortfolioWeights:
    """Result of a portfolio optimisation."""

    weights: Dict[str, float] = field(default_factory=dict)
    expected_return: float = 0.0
    expected_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    method: str = ""


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _to_mu_cov(
    returns: Dict[str, np.ndarray],
) -> tuple:
    """Convert a dict-of-arrays to mean vector and covariance matrix.

    Returns
    -------
    mu : ndarray, shape (n,)
    cov : ndarray, shape (n, n)
    columns : list[str]
    """
    df = pd.DataFrame(returns)
    if df.empty:
        return np.array([]), np.empty((0, 0)), []
    return df.mean().values, df.cov().values, list(df.columns)


# ---------------------------------------------------------------------------
# Mean-Variance (Markowitz) — analytical
# ---------------------------------------------------------------------------


def mean_variance_optimize(
    returns: Dict[str, np.ndarray],
    risk_aversion: float = 1.0,
) -> Dict[str, float]:
    """Analytical mean-variance optimisation.

    Solves  w = (1 / λ) Σ⁻¹ μ   and rescales so that weights sum to 1.

    Parameters
    ----------
    returns : dict[str, ndarray]
    risk_aversion : float
        λ — higher values produce more conservative portfolios.

    Returns
    -------
    dict[str, float]
        Asset weights that sum to (approximately) 1.
    """
    mu, cov, cols = _to_mu_cov(returns)
    n = len(mu)
    if n == 0:
        return {}

    # Invert covariance (pseudo-inverse for singular matrices)
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov)

    w = inv_cov @ mu / risk_aversion

    # Rescale to sum to 1
    w_sum = w.sum()
    if abs(w_sum) > 1e-12:
        w = w / w_sum
    else:
        # Degenerate case — fall back to equal weights
        w = np.ones(n) / n

    return dict(zip(cols, w.tolist()))


# ---------------------------------------------------------------------------
# Risk Parity — equal risk contribution
# ---------------------------------------------------------------------------


def risk_parity_optimize(
    returns: Dict[str, np.ndarray],
    max_iter: int = 100,
    tol: float = 1e-8,
) -> Dict[str, float]:
    """Risk parity via cyclic coordinate descent (Spinu-style fixed-point).

    Finds weights such that every asset has the same risk contribution:
        RC_i = w_i · (Σ w)_i / √(wᵀ Σ w) = 1 / n · √(wᵀ Σ w)

    Parameters
    ----------
    returns : dict[str, ndarray]
    max_iter : int
    tol : float

    Returns
    -------
    dict[str, float]
        Long-only weights summing to 1.
    """
    _, cov, cols = _to_mu_cov(returns)
    n = len(cols)
    if n == 0:
        return {}

    w = np.ones(n) / n  # equal start

    for _ in range(max_iter):
        w_old = w.copy()

        sigma_sq = max(w @ cov @ w, tol)
        mrc = cov @ w  # marginal risk contribution

        # Fixed-point update:  w_i = σ² / (n · MRC_i)
        for i in range(n):
            if abs(mrc[i]) > tol:
                w[i] = max(0.0, sigma_sq / (n * mrc[i]))

        # Renormalise
        s = w.sum()
        if s > 0:
            w = w / s
        else:
            w = np.ones(n) / n

        if np.max(np.abs(w - w_old)) < tol:
            break

    return dict(zip(cols, w.tolist()))


# ---------------------------------------------------------------------------
# Hierarchical Risk Parity (Lopez de Prado 2016)
# ---------------------------------------------------------------------------


def hrp_optimize(returns: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Hierarchical Risk Parity using single-linkage clustering.

    1.  Build distance matrix from correlations.
    2.  Single-linkage clustering (scipy).
    3.  Quasi-diagonalisation (dendrogram leaf order).
    4.  Recursive bisection to allocate weight inversely to variance.

    Parameters
    ----------
    returns : dict[str, ndarray]

    Returns
    -------
    dict[str, float]
        Long-only weights summing to 1.
    """
    _, cov, cols = _to_mu_cov(returns)
    n = len(cols)
    if n == 0:
        return {}
    if n == 1:
        return {cols[0]: 1.0}

    # 1. Correlation → distance
    corr = _cov_to_corr(cov)
    dist = np.sqrt(2.0 * (1.0 - np.clip(corr, -1.0, 1.0)))
    # Ensure symmetry and zero diagonal
    np.fill_diagonal(dist, 0.0)

    # 2. Clustering (single linkage)
    linkage_matrix = linkage(squareform(dist), method="single")

    # 3. Quasi-diagonalisation — leaf order from dendrogram
    ordered_idx = _get_leaf_order(linkage_matrix, n)

    # 4. Recursive bisection
    weights = _recursive_bisection(ordered_idx, cov)

    # Map back to original column order
    result = {cols[i]: float(weights[idx]) for idx, i in enumerate(ordered_idx)}
    return result


# -- HRP helpers ------------------------------------------------------------


def _cov_to_corr(cov: np.ndarray) -> np.ndarray:
    """Convert covariance to correlation matrix."""
    d = np.sqrt(np.diag(cov))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov / np.outer(d, d)
    corr[np.isnan(corr)] = 0.0
    corr = np.clip(corr, -1.0, 1.0)
    return corr


def _get_leaf_order(linkage_matrix: np.ndarray, n: int) -> List[int]:
    """Return leaf indices in left-to-right dendrogram order.

    Parameters
    ----------
    linkage_matrix : ndarray
        Output of ``scipy.cluster.hierarchy.linkage``.
    n : int
        Number of original assets.

    Returns
    -------
    list[int]
        Asset indices in dendrogram order.
    """

    def _traverse(node: int) -> List[int]:
        if node < n:
            return [node]
        # The linkage matrix row for this cluster
        row = linkage_matrix[node - n]
        left = int(row[0])
        right = int(row[1])
        return _traverse(left) + _traverse(right)

    root = 2 * n - 2  # last cluster index created by linkage
    return _traverse(root)


def _recursive_bisection(
    assets: List[int],
    cov: np.ndarray,
) -> np.ndarray:
    """Recursive bisection step of HRP.

    Parameters
    ----------
    assets : list[int]
        Asset indices in current cluster (in dendrogram order).
    cov : ndarray
        Full covariance matrix.

    Returns
    -------
    ndarray
        Weight for each asset in ``assets`` (same order).
    """
    n = len(assets)
    if n == 1:
        return np.array([1.0])

    # Split ordered list in half
    mid = n // 2
    left_assets = assets[:mid]
    right_assets = assets[mid:]

    # Inverse-variance weights within each subset
    left_w = _compute_ivp(left_assets, cov)
    right_w = _compute_ivp(right_assets, cov)

    # Cluster variance
    left_var = _compute_cluster_var(left_assets, cov, left_w)
    right_var = _compute_cluster_var(right_assets, cov, right_w)

    # Allocate — lower variance gets higher weight
    total_var = left_var + right_var
    if total_var < 1e-12:
        alpha = 0.5
    else:
        alpha = 1.0 - left_var / total_var

    # Recurse
    left_result = _recursive_bisection(left_assets, cov)
    right_result = _recursive_bisection(right_assets, cov)

    return np.concatenate([alpha * left_result, (1.0 - alpha) * right_result])


def _compute_ivp(assets: List[int], cov: np.ndarray) -> np.ndarray:
    """Inverse-variance portfolio for a subset of assets."""
    sub_cov = cov[np.ix_(assets, assets)]
    inv_diag = 1.0 / np.diag(sub_cov)
    return inv_diag / inv_diag.sum()


def _compute_cluster_var(
    assets: List[int],
    cov: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Variance of a cluster given its inverse-variance weights."""
    sub_cov = cov[np.ix_(assets, assets)]
    return float(weights @ sub_cov @ weights)


# ---------------------------------------------------------------------------
# Efficient frontier
# ---------------------------------------------------------------------------


def compute_efficient_frontier(
    returns: Dict[str, np.ndarray],
    n_points: int = 20,
) -> List[Dict]:
    """Compute the mean-variance efficient frontier.

    Varies the risk-aversion parameter over a wide range to trace out the
    frontier from low-risk/low-return to high-risk/high-return portfolios.

    Parameters
    ----------
    returns : dict[str, ndarray]
    n_points : int
        Number of points along the frontier.

    Returns
    -------
    list[dict]
        Each entry has keys ``return``, ``volatility``, ``weights``.
    """
    mu, cov, cols = _to_mu_cov(returns)
    n = len(mu)
    if n == 0:
        return []

    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov)

    # Vary risk aversion from very low (aggressive) to high (conservative)
    risk_aversions = np.logspace(-1.5, 2.5, n_points)

    frontier: List[Dict] = []
    for lam in risk_aversions:
        w = inv_cov @ mu / lam
        s = w.sum()
        if abs(s) > 1e-12:
            w = w / s
        else:
            w = np.ones(n) / n

        port_ret = float(w @ mu)
        port_vol = float(np.sqrt(max(w @ cov @ w, 1e-16)))

        frontier.append(
            {
                "return": port_ret,
                "volatility": port_vol,
                "weights": dict(zip(cols, w.tolist())),
            }
        )

    return frontier


# ---------------------------------------------------------------------------
# PortfolioOptimizer — unified interface
# ---------------------------------------------------------------------------


class PortfolioOptimizer:
    """Unified interface to all three optimisation methods.

    Parameters
    ----------
    target_volatility : float or None
        If set, portfolio weights are scaled (levered / delevered) so that
        the expected volatility equals this value.
    """

    def __init__(self, target_volatility: Optional[float] = 0.15):
        self.target_volatility = target_volatility

    # ------------------------------------------------------------------
    def optimize(
        self,
        returns: Dict[str, np.ndarray],
        method: str = "hrp",
        risk_aversion: float = 1.0,
    ) -> PortfolioWeights:
        """Run portfolio optimisation and return a ``PortfolioWeights``.

        Parameters
        ----------
        returns : dict[str, ndarray]
            Asset return series.
        method : str
            One of ``"mean_variance"``, ``"risk_parity"``, ``"hrp"``.
        risk_aversion : float
            Only used for ``"mean_variance"``.

        Returns
        -------
        PortfolioWeights
        """
        methods = {
            "mean_variance": mean_variance_optimize,
            "risk_parity": risk_parity_optimize,
            "hrp": hrp_optimize,
        }

        if method not in methods:
            raise ValueError(
                f"Unknown method '{method}'. " f"Choose from {list(methods.keys())}"
            )

        # Dispatch
        kwargs = {}
        if method == "mean_variance":
            kwargs["risk_aversion"] = risk_aversion
        elif method == "risk_parity":
            kwargs["max_iter"] = 100
            kwargs["tol"] = 1e-8

        raw_weights = methods[method](returns, **kwargs)

        # Compute portfolio statistics
        mu, cov, cols = _to_mu_cov(returns)
        if not raw_weights:
            return PortfolioWeights(
                weights={},
                expected_return=0.0,
                expected_volatility=0.0,
                sharpe_ratio=0.0,
                method=method,
            )

        w = np.array([raw_weights[c] for c in cols], dtype=float)
        expected_return = float(w @ mu)
        expected_vol = float(np.sqrt(max(w @ cov @ w, 1e-16)))

        # Scale to target volatility
        if self.target_volatility is not None and expected_vol > 1e-12:
            scale = self.target_volatility / expected_vol
            w = w * scale
            expected_vol = self.target_volatility
            expected_return = float(w @ mu)

        # Rebuild weight dict
        scaled_weights = dict(zip(cols, w.tolist()))
        sharpe = expected_return / expected_vol if expected_vol > 1e-12 else 0.0

        return PortfolioWeights(
            weights=scaled_weights,
            expected_return=expected_return,
            expected_volatility=expected_vol,
            sharpe_ratio=sharpe,
            method=method,
        )
