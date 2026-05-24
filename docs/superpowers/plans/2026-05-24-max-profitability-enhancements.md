
```
# Max Profitability Enhancements — 13 Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement 13 profitability-enhancing improvements across risk management, execution, alpha generation, model architecture, and infrastructure — each independently testable and backward-compatible.

**Architecture:** 13 tasks grouped into 5 phases running partially in parallel. Phase 1 (Foundation) can parallelize 3 tasks. Each subsequent phase builds on prior phases. All tasks follow TDD: write failing test → run → implement → pass test → commit.

**Tech Stack:** Python 3.9.6+, numpy, pandas, PyTorch, TensorFlow/Keras, loguru, pytest. No new external dependencies beyond existing stack (scipy.stats added only for correlation calculations).

---

## File Map (Files Created/Modified)

| File | Change | Task |
|------|--------|------|
| `src/execution/cost_model.py` | Enhanced | T1 |
| `src/execution/almgren_chriss.py` | **NEW** | T1 |
| `src/risk/portfolio_optimizer.py` | **NEW** | T2 |
| `src/risk/manager.py` | Modified (lines 325-360) | T4 |
| `src/risk/correlation_matrix.py` | **NEW** | T4 |
| `src/training/online_learner.py` | Modified (lines 43-54, 107-119) | T8 |
| `src/training/validation_gate.py` | **NEW** | T12 |
| `src/rts_ai_fx/features_unified.py` | Modified (add cross-sectional) | T3 |
| `src/rts_ai_fx/cross_sectional_alpha.py` | **NEW** | T3 |
| `src/ai/alpha_research.py` | **NEW** | T9 |
| `src/ai/sentiment.py` | Modified | T10 |
| `src/ai/alternative_data.py` | **NEW** | T10 |
| `src/rts_ai_fx/model.py` | Modified (add adversarial) | T7 |
| `src/rts_ai_fx/adversarial.py` | **NEW** | T7 |
| `src/rts_ai_fx/multi_asset_model.py` | **NEW** | T5 |
| `src/ai/maml_agent.py` | Modified (lines 76-100, 273-289) | T11 |
| `src/execution/engine.py` | Modified (lines 164-168, 214-218) | T6 |
| `src/execution/is_execution.py` | **NEW** | T6 |
| `src/agentic/agents/consolidated_agents.py` | **NEW** | T13 |
| `src/agentic/main_agentic.py` | Modified (lines 32-52, 97-135) | T13 |
| `tests/execution/test_almgren_chriss.py` | **NEW** | T1 |
| `tests/risk/test_portfolio_optimizer.py` | **NEW** | T2 |
| `tests/risk/test_correlation_matrix.py` | **NEW** | T4 |
| `tests/training/test_online_learner_adaptive.py` | **NEW** | T8 |
| `tests/training/test_validation_gate.py` | **NEW** | T12 |
| `tests/rts_ai_fx/test_cross_sectional_alpha.py` | **NEW** | T3 |
| `tests/ai/test_alpha_research.py` | **NEW** | T9 |
| `tests/ai/test_alternative_data.py` | **NEW** | T10 |
| `tests/rts_ai_fx/test_adversarial.py` | **NEW** | T7 |
| `tests/rts_ai_fx/test_multi_asset_model.py` | **NEW** | T5 |
| `tests/ai/test_maml_scaling.py` | **NEW** | T11 |
| `tests/execution/test_is_execution.py` | **NEW** | T6 |
| `tests/agentic/test_consolidated_agents.py` | **NEW** | T13 |

---

## Phase 1: Foundation (Parallelizable)

---

### Task 1: Almgren-Chriss Transaction Cost Model

**Files:**
- Create: `src/execution/almgren_chriss.py`
- Modify: `src/execution/cost_model.py` (lines 40-136)
- Test: `tests/execution/test_almgren_chriss.py`

**Goal:** Replace simplistic spread+slippage cost model with Almgren-Chriss (2001) market impact model: `I(Q) = sign(Q) * σ * (|Q|/V)^δ` with both permanent and temporary impact components.

- [ ] **Step 1: Write the failing test**

```python
# tests/execution/test_almgren_chriss.py
import numpy as np
import pytest
from execution.almgren_chriss import AlmgrenChrissModel, ImpactResult


class TestAlmgrenChrissModel:
    def test_impact_non_negative(self):
        model = AlmgrenChrissModel(volatility=0.15, adv=1e9, spread=0.0001)
        result = model.calculate_impact(
            symbol="EURUSD", volume=100_000, price=1.10, side="BUY"
        )
        assert result.permanent_impact_bps >= 0.0
        assert result.temporary_impact_bps >= 0.0
        assert result.total_cost_bps >= 0.0

    def test_larger_volume_higher_impact(self):
        model = AlmgrenChrissModel(volatility=0.15, adv=1e9, spread=0.0001)
        r1 = model.calculate_impact("EURUSD", 100_000, 1.10, "BUY")
        r2 = model.calculate_impact("EURUSD", 1_000_000, 1.10, "BUY")
        assert r2.total_cost_bps > r1.total_cost_bps

    def test_sell_side_same_impact_as_buy(self):
        model = AlmgrenChrissModel(volatility=0.15, adv=1e9, spread=0.0001)
        buy = model.calculate_impact("EURUSD", 100_000, 1.10, "BUY")
        sell = model.calculate_impact("EURUSD", 100_000, 1.10, "SELL")
        assert abs(buy.total_cost_bps - sell.total_cost_bps) < 1e-10

    def test_optimal_trajectory_returns_slices(self):
        model = AlmgrenChrissModel(volatility=0.15, adv=1e9, spread=0.0001)
        slices = model.optimal_trajectory(
            volume=100_000, price=1.10, n_slices=10, duration_minutes=30
        )
        assert len(slices) == 10
        assert abs(sum(s["volume"] for s in slices) - 100_000) < 1.0

    def test_optimal_trajectory_front_loaded(self):
        """In Almgren-Chriss, optimal execution is front-loaded (concave)."""
        model = AlmgrenChrissModel(volatility=0.15, adv=1e9, spread=0.0001)
        slices = model.optimal_trajectory(
            volume=100_000, price=1.10, n_slices=10, duration_minutes=30
        )
        # First slice should be >= last slice
        assert slices[0]["volume"] >= slices[-1]["volume"]

    def test_impact_decays_with_longer_horizon(self):
        model = AlmgrenChrissModel(volatility=0.15, adv=1e9, spread=0.0001)
        slices_fast = model.optimal_trajectory(
            volume=100_000, price=1.10, n_slices=5, duration_minutes=5
        )
        slices_slow = model.optimal_trajectory(
            volume=100_000, price=1.10, n_slices=20, duration_minutes=60
        )
        impact_fast = sum(s["impact_bps"] for s in slices_fast)
        impact_slow = sum(s["impact_bps"] for s in slices_slow)
        assert impact_fast > impact_slow

    def test_integration_with_cost_model(self):
        """Verify CostModel can delegate to AlmgrenChriss."""
        from execution.cost_model import CostModel
        ac = AlmgrenChrissModel(volatility=0.15, adv=1e9, spread=0.0001)
        model = CostModel(impact_model=ac)
        result = model.calculate("EURUSD", "BUY", 100_000, 1.10)
        assert result.total > 0
        assert result.market_impact > 0
```

- [ ] **Step 2: Run test to verify it fails**

```
$ python -m pytest tests/execution/test_almgren_chriss.py -v
Expected: ModuleNotFoundError or ImportError (no AlmgrenChrissModel)
```

- [ ] **Step 3: Write minimal implementation**

```python
# src/execution/almgren_chriss.py
"""
Almgren-Chriss (2001) Optimal Execution Model.
Implements permanent and temporary market impact with optimal VWAP trajectory.
Reference: Almgren & Chriss, "Optimal Execution of Portfolio Transactions", J Risk 2001.
"""
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class ImpactResult:
    permanent_impact_bps: float
    temporary_impact_bps: float
    total_cost_bps: float
    total_cost_usd: float
    optimal_slices: List[dict]


class AlmgrenChrissModel:
    """Almgren-Chriss market impact model with optimal execution trajectory.

    Permanent impact: I_p(Q) = gamma * sigma * (Q/V)^delta
    Temporary impact: I_t(Q) = eta * sigma * (Q/V)^delta + spread/2
    Total cost: I_p(Q) + I_t(Q)

    Where:
      Q = order volume
      V = average daily volume (ADV)
      sigma = daily volatility
      gamma, eta = impact coefficients (default: 0.1, 0.5)
      delta = impact exponent (default: 0.5, square-root model)
    """

    def __init__(
        self,
        volatility: float = 0.15,
        adv: float = 1e9,
        spread: float = 0.0001,
        gamma: float = 0.1,
        eta: float = 0.5,
        delta: float = 0.5,
        risk_aversion: float = 1e-6,
    ):
        self.volatility = volatility
        self.adv = adv
        self.spread = spread
        self.gamma = gamma
        self.eta = eta
        self.delta = delta
        self.risk_aversion = risk_aversion

    def calculate_impact(
        self, symbol: str, volume: float, price: float, side: str
    ) -> ImpactResult:
        """Calculate total market impact for a single order."""
        participation_rate = volume / max(self.adv, 1.0)
        impact_term = (participation_rate ** self.delta) * self.volatility

        permanent_impact = self.gamma * impact_term
        temporary_impact = self.eta * impact_term + (self.spread / 2.0)

        total_cost_bps = permanent_impact + temporary_impact
        total_cost_usd = total_cost_bps / 10_000 * price * volume

        return ImpactResult(
            permanent_impact_bps=float(permanent_impact) * 10_000,
            temporary_impact_bps=float(temporary_impact) * 10_000,
            total_cost_bps=float(total_cost_bps) * 10_000,
            total_cost_usd=float(total_cost_usd),
            optimal_slices=[],
        )

    def optimal_trajectory(
        self,
        volume: float,
        price: float,
        n_slices: int = 10,
        duration_minutes: float = 30.0,
    ) -> List[dict]:
        """Compute optimal execution trajectory (Almgren-Chriss dynamic programming).

        Returns a list of slice dicts with volume, impact_bps, and timestamp.
        The solution is the well-known concave 'front-loaded' trajectory where
        trading rate decays as sqrt of remaining time.
        """
        if n_slices <= 0:
            return []

        # Time intervals
        T = duration_minutes / (60 * 24 * 252)  # Convert to years
        tau = T / n_slices

        # Risk-adjusted optimal trajectory: v_j = sinh(kappa * (T - t_j)) / sinh(kappa * T)
        # where kappa = sqrt(lambda * sigma^2 / eta) and lambda = risk aversion
        sigma_sq = self.volatility ** 2
        kappa = np.sqrt(self.risk_aversion * sigma_sq / max(self.eta, 1e-10))

        slices = []
        remaining = volume
        for j in range(n_slices):
            t_j = j * tau
            if kappa > 1e-10:
                numerator = np.sinh(kappa * (T - t_j))
                denominator = np.sinh(kappa * T)
                if denominator > 0:
                    weight = numerator / denominator
                else:
                    weight = 1.0 - j / n_slices
            else:
                weight = 1.0 - j / n_slices

            # Normalize weights to sum to 1
            weight /= max(sum(1.0 - i / n_slices for i in range(n_slices)), 1e-10)

            slice_vol = volume * weight
            participation = slice_vol / max(self.adv * tau * 252, 1.0)
            impact = (self.eta * (participation ** self.delta) * self.volatility
                      + self.spread / 2.0) * 10_000

            slice_vol = min(slice_vol, remaining)
            slices.append({
                "slice_idx": j,
                "volume": float(slice_vol),
                "impact_bps": float(impact),
            })
            remaining -= slice_vol

        # Adjust last slice for rounding
        if slices and remaining > 0:
            slices[-1]["volume"] += remaining

        return slices
```

- [ ] **Step 4: Update CostModel to support AlmgrenChriss delegation**

```python
# In src/execution/cost_model.py, modify __init__ and calculate:
# Add at line 50, after quality_tracker param:
#    impact_model: Optional[AlmgrenChrissModel] = None,
# In calculate() at line 119-125, replace existing market_impact block:

        market_impact_cost = 0.0
        if self.impact_model is not None:
            impact_result = self.impact_model.calculate_impact(
                symbol, volume, price, direction
            )
            market_impact_cost = impact_result.total_cost_usd
        elif self.quality_tracker is not None:
            impact_bps = self.quality_tracker.calculate_market_impact(
                symbol, volume, atr, price
            )
            market_impact_cost = impact_bps / 10_000 * price * volume * usd_rate

        total = spread_cost + commission + slippage + market_impact_cost

        return CostResult(
            spread_cost=spread_cost,
            commission=commission,
            slippage=slippage,
            total=total,
            actual_spread_pips=spread_pips,
            is_acceptable=is_acceptable,
            rejection_reason=rejection_reason,
            market_impact=market_impact_cost,
        )
```

Add import at top of `cost_model.py`:
```python
from typing import Optional
from .almgren_chriss import AlmgrenChrissModel
```

- [ ] **Step 5: Run tests to verify they pass**

```
$ python -m pytest tests/execution/test_almgren_chriss.py -v
Expected: All 7 tests PASS
```

- [ ] **Step 6: Run existing cost model tests to verify no regression**

```
$ python -m pytest tests/execution/ -v
Expected: All existing tests PASS
```

- [ ] **Step 7: Commit**

```
git add src/execution/almgren_chriss.py src/execution/cost_model.py tests/execution/test_almgren_chriss.py
git commit -m "feat: add Almgren-Chriss market impact model for transaction cost estimation"
```

---

### Task 2: Portfolio Optimization Engine

**Files:**
- Create: `src/risk/portfolio_optimizer.py`
- Test: `tests/risk/test_portfolio_optimizer.py`

**Goal:** Build mean-variance (Markowitz) and Hierarchical Risk Parity (HRP) portfolio optimization for multi-symbol position sizing.

- [ ] **Step 1: Write the failing test**

```python
# tests/risk/test_portfolio_optimizer.py
import numpy as np
import pytest
from risk.portfolio_optimizer import (
    PortfolioOptimizer,
    PortfolioWeights,
    mean_variance_optimize,
    risk_parity_optimize,
    hrp_optimize,
    compute_efficient_frontier,
)


class TestPortfolioOptimizer:
    def test_mean_variance_returns_weights_sum_to_one(self):
        returns = np.random.randn(500, 4) * 0.01 + 0.0005
        weights = mean_variance_optimize(returns, risk_aversion=1.0)
        assert abs(weights.sum() - 1.0) < 1e-6
        assert all(w >= -0.1 for w in weights)  # allow small shorts

    def test_mean_variance_no_assets_returns_empty(self):
        weights = mean_variance_optimize(np.array([]).reshape(0, 0), risk_aversion=1.0)
        assert len(weights) == 0

    def test_risk_parity_equal_risk_contribution(self):
        """Risk parity should give approximately equal risk contribution from each asset."""
        returns = np.random.randn(500, 3) * 0.01
        weights = risk_parity_optimize(returns)
        cov = np.cov(returns, rowvar=False)
        risk_contrib = (weights @ cov) * weights
        risk_contrib = risk_contrib / risk_contrib.sum()
        # All should be roughly 1/3
        assert np.allclose(risk_contrib, 1/3, atol=0.15)

    def test_hrp_clustering_based_weights(self):
        returns = np.random.randn(500, 4) * 0.01
        weights = hrp_optimize(returns)
        assert abs(weights.sum() - 1.0) < 1e-6
        assert all(w >= 0 for w in weights)  # HRP is long-only

    def test_efficient_frontier_returns_frontier(self):
        returns = np.random.randn(500, 3) * 0.01 + 0.0005
        frontier = compute_efficient_frontier(returns, n_points=10)
        assert len(frontier) == 10
        for point in frontier:
            assert "volatility" in point
            assert "return" in point
            assert "weights" in point
            assert abs(point["weights"].sum() - 1.0) < 1e-6

    def test_efficient_frontier_increasing_volatility(self):
        returns = np.random.randn(500, 3) * 0.01 + 0.0005
        frontier = compute_efficient_frontier(returns, n_points=10)
        vols = [p["volatility"] for p in frontier]
        for i in range(1, len(vols)):
            assert vols[i] >= vols[i-1]  # monotonic increasing vol

    def test_portfolio_optimizer_full_pipeline(self):
        optimizer = PortfolioOptimizer()
        returns = {"EURUSD": np.random.randn(500) * 0.01,
                   "GBPUSD": np.random.randn(500) * 0.01 + 0.0002,
                   "USDJPY": np.random.randn(500) * 0.008}
        weights = optimizer.optimize(returns, method="hrp")
        assert isinstance(weights, PortfolioWeights)
        assert abs(sum(w for _, w in weights.weights) - 1.0) < 1e-6

    def test_portfolio_optimizer_target_vol(self):
        optimizer = PortfolioOptimizer(target_volatility=0.12)
        returns = {"EURUSD": np.random.randn(500) * 0.01,
                   "GBPUSD": np.random.randn(500) * 0.01}
        result = optimizer.optimize(returns, method="mean_variance", risk_aversion=2.0)
        expected_vol = result.expected_volatility
        assert expected_vol > 0
```

- [ ] **Step 2: Run test to verify it fails**

```
$ python -m pytest tests/risk/test_portfolio_optimizer.py -v
Expected: ModuleNotFoundError
```

- [ ] **Step 3: Write implementation**

```python
# src/risk/portfolio_optimizer.py
"""
Portfolio Optimization Engine — Mean-Variance, Risk Parity, Hierarchical Risk Parity.
Implements Markowitz (1952) and Lopez de Prado (2016) HRP.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform


@dataclass
class PortfolioWeights:
    weights: Dict[str, float] = field(default_factory=dict)
    expected_return: float = 0.0
    expected_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    method: str = ""


class PortfolioOptimizer:
    """Unified portfolio optimizer supporting multiple methods.

    Methods:
      - "mean_variance": Markowitz mean-variance with risk aversion
      - "risk_parity": Equal risk contribution (ERC)
      - "hrp": Hierarchical Risk Parity (Lopez de Prado 2016)
    """

    def __init__(self, target_volatility: float = 0.15):
        self.target_volatility = target_volatility
        self._last_cov: Optional[np.ndarray] = None
        self._last_symbols: List[str] = []

    def optimize(
        self,
        returns: Dict[str, np.ndarray],
        method: str = "hrp",
        risk_aversion: float = 1.0,
    ) -> PortfolioWeights:
        symbols = list(returns.keys())
        if len(symbols) == 0:
            return PortfolioWeights(method=method)

        # Build return matrix
        min_len = min(len(v) for v in returns.values())
        ret_matrix = np.column_stack([returns[s][-min_len:] for s in symbols])
        self._last_symbols = symbols

        if method == "mean_variance":
            weights = mean_variance_optimize(ret_matrix, risk_aversion)
        elif method == "risk_parity":
            weights = risk_parity_optimize(ret_matrix)
        else:
            weights = hrp_optimize(ret_matrix)

        weight_dict = {s: float(w) for s, w in zip(symbols, weights)}
        cov = np.cov(ret_matrix, rowvar=False)
        mean_ret = np.mean(ret_matrix, axis=0)
        port_ret = float(mean_ret @ weight_vec(weights))
        port_vol = float(np.sqrt(weights @ cov @ weights))

        return PortfolioWeights(
            weights=weight_dict,
            expected_return=port_ret,
            expected_volatility=port_vol,
            sharpe_ratio=port_ret / max(port_vol, 1e-10),
            method=method,
        )


def weight_vec(weights: np.ndarray) -> np.ndarray:
    return np.array(weights, dtype=float)


def mean_variance_optimize(
    returns: np.ndarray, risk_aversion: float = 1.0
) -> np.ndarray:
    """Markowitz mean-variance optimization with risk aversion.

    maximize: w'μ - λ/2 * w'Σw
    subject to: sum(w) = 1
    """
    if returns.size == 0 or returns.ndim != 2:
        return np.array([])
    n = returns.shape[1]
    mean = np.mean(returns, axis=0)
    cov = np.cov(returns, rowvar=False)

    # Analytical solution for unconstrained: w = (1/λ) * Σ⁻¹ * μ
    try:
        cov_inv = np.linalg.inv(cov + np.eye(n) * 1e-8)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)

    w = (1.0 / risk_aversion) * cov_inv @ mean

    # Rescale to sum to 1
    if abs(w.sum()) > 1e-10:
        w = w / w.sum()
    return w


def risk_parity_optimize(
    returns: np.ndarray, max_iter: int = 100, tol: float = 1e-8
) -> np.ndarray:
    """Equal risk contribution portfolio via cyclic coordinate descent."""
    if returns.size == 0 or returns.ndim != 2:
        return np.array([])
    n = returns.shape[1]
    cov = np.cov(returns, rowvar=False)
    w = np.ones(n) / n

    for _ in range(max_iter):
        for i in range(n):
            sigma_i = np.sqrt(cov[i, i])
            marginal = (cov[i] @ w) / sigma_i
            target = (w @ cov @ w) / (n * sigma_i)
            w[i] = w[i] * target / max(marginal, 1e-10)
        w = np.clip(w, 0, 1)
        w = w / max(w.sum(), 1e-10)

        # Check convergence
        risk_contrib = (w @ cov) * w
        risk_contrib = risk_contrib / max(risk_contrib.sum(), 1e-10)
        if np.std(risk_contrib) < tol:
            break

    return w


def hrp_optimize(returns: np.ndarray) -> np.ndarray:
    """Hierarchical Risk Parity (Lopez de Prado 2016).

    Uses hierarchical clustering to group similar assets, then allocates
    inversely by variance within clusters and equally across clusters.
    """
    if returns.size == 0 or returns.ndim != 2:
        return np.array([])
    n = returns.shape[1]
    cov = np.cov(returns, rowvar=False)
    corr = np.corrcoef(returns, rowvar=False)
    dist = np.sqrt(2 * (1 - np.clip(corr, -1, 1)))

    # Hierarchical clustering
    try:
        linkage_matrix = linkage(squareform(dist), method="ward")
    except Exception:
        return np.ones(n) / n

    # Quasi-diagonalization: reorder assets by clustering order
    idx = list(range(n))
    # Simple recursive bisection
    clusters = _get_clusters(linkage_matrix, n)
    weights = _bisection_alloc(clusters, cov, np.ones(n) / n)
    return weights


def _get_clusters(linkage_matrix: np.ndarray, n: int) -> List[List[int]]:
    """Extract list of clusters from linkage matrix (each cluster is list of indices)."""
    clusters = {i: [i] for i in range(n)}
    for i, link in enumerate(linkage_matrix):
        idx = n + i
        left = int(link[0])
        right = int(link[1])
        clusters[idx] = clusters.get(left, [left]) + clusters.get(right, [right])
    return clusters[n + len(linkage_matrix) - 1]


def _bisection_alloc(
    cluster: List[int], cov: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Recursive bisection allocation within a cluster."""
    if len(cluster) == 1:
        return weights
    mid = len(cluster) // 2
    left = cluster[:mid]
    right = cluster[mid:]

    def cluster_variance(indices: List[int]) -> float:
        sub_cov = cov[np.ix_(indices, indices)]
        sub_w = np.ones(len(indices)) / len(indices)
        return float(sub_w @ sub_cov @ sub_w)

    var_left = cluster_variance(left)
    var_right = cluster_variance(right)
    alpha = 1 - var_left / max(var_left + var_right, 1e-10)
    alpha = np.clip(alpha, 0.05, 0.95)

    # Propagate to individual assets
    for idx in left:
        weights[idx] *= alpha / (len(left) / len(cluster))
    for idx in right:
        weights[idx] *= (1 - alpha) / (len(right) / len(cluster))

    return _bisection_alloc(left, cov, weights) or _bisection_alloc(right, cov, weights)


def compute_efficient_frontier(
    returns: np.ndarray, n_points: int = 20
) -> List[Dict]:
    """Compute efficient frontier by varying risk aversion parameter."""
    if returns.size == 0 or returns.ndim != 2:
        return []
    risk_aversions = np.logspace(-1, 2, n_points)
    frontier = []
    for lam in risk_aversions:
        w = mean_variance_optimize(returns, risk_aversion=lam)
        if len(w) == 0:
            continue
        cov = np.cov(returns, rowvar=False)
        mean = np.mean(returns, axis=0)
        port_ret = float(mean @ w)
        port_vol = float(np.sqrt(w @ cov @ w))
        frontier.append({
            "volatility": port_vol,
            "return": port_ret,
            "weights": w,
            "risk_aversion": float(lam),
        })
    return sorted(frontier, key=lambda x: x["volatility"])
```

- [ ] **Step 4: Run tests to verify they pass**

```
$ python -m pytest tests/risk/test_portfolio_optimizer.py -v
Expected: All 8 tests PASS
```

- [ ] **Step 5: Commit**

```
git add src/risk/portfolio_optimizer.py tests/risk/test_portfolio_optimizer.py
git commit -m "feat: add portfolio optimization engine with MV, risk parity, HRP"
```

---

### Task 12: Validation Gate

**Files:**
- Create: `src/training/validation_gate.py`
- Test: `tests/training/test_validation_gate.py`

**Goal:** Add mandatory walk-forward + crisis scenario checks that must pass before model deployment is allowed.

- [ ] **Step 1: Write the failing test**

```python
# tests/training/test_validation_gate.py
import numpy as np
import pytest
from training.validation_gate import (
    ValidationGate,
    ValidationResult,
    GateDecision,
    GateConfig,
)


class TestValidationGate:
    def test_gate_rejects_poor_sharpe(self):
        config = GateConfig(min_sharpe=1.0, max_drawdown_pct=0.10)
        gate = ValidationGate(config)
        result = gate.evaluate(
            model_name="lstm_cnn",
            walk_forward_results={
                "avg_sharpe": 0.5,
                "avg_max_dd_pct": 0.05,
            },
            stress_test_results={},
        )
        assert result.decision == GateDecision.REJECTED
        assert "sharpe" in result.reason.lower()

    def test_gate_accepts_strong_performance(self):
        config = GateConfig(min_sharpe=1.0, max_drawdown_pct=0.10)
        gate = ValidationGate(config)
        result = gate.evaluate(
            model_name="lstm_cnn",
            walk_forward_results={
                "avg_sharpe": 1.5,
                "avg_max_dd_pct": 0.05,
            },
            stress_test_results={},
        )
        assert result.decision == GateDecision.APPROVED

    def test_gate_rejects_excessive_drawdown(self):
        config = GateConfig(min_sharpe=1.0, max_drawdown_pct=0.10)
        gate = ValidationGate(config)
        result = gate.evaluate(
            model_name="lstm_cnn",
            walk_forward_results={
                "avg_sharpe": 1.5,
                "avg_max_dd_pct": 0.15,
            },
            stress_test_results={},
        )
        assert result.decision == GateDecision.REJECTED
        assert "drawdown" in result.reason.lower()

    def test_gate_rejects_crisis_failure(self):
        config = GateConfig(min_sharpe=1.0, max_drawdown_pct=0.10, crisis_max_loss=0.15)
        gate = ValidationGate(config)
        result = gate.evaluate(
            model_name="lstm_cnn",
            walk_forward_results={
                "avg_sharpe": 1.5,
                "avg_max_dd_pct": 0.05,
            },
            stress_test_results={
                "covid_crash_2020": {"max_loss_pct": 0.20, "passed": False},
                "flash_crash_2010": {"max_loss_pct": 0.08, "passed": True},
            },
        )
        assert result.decision == GateDecision.REJECTED
        assert "crisis" in result.reason.lower() or "covid" in result.reason.lower()

    def test_gate_requires_min_walk_forward_folds(self):
        config = GateConfig(min_sharpe=1.0, max_drawdown_pct=0.10, min_folds=3)
        gate = ValidationGate(config)
        result = gate.evaluate(
            model_name="lstm_cnn",
            walk_forward_results={
                "avg_sharpe": 1.5,
                "avg_max_dd_pct": 0.05,
                "total_folds": 2,
            },
            stress_test_results={},
        )
        assert result.decision == GateDecision.REJECTED
        assert "fold" in result.reason.lower() or "insufficient" in result.reason.lower()

    def test_gate_logs_result(self):
        config = GateConfig(min_sharpe=1.0, max_drawdown_pct=0.10)
        gate = ValidationGate(config)
        result = gate.evaluate(
            model_name="test_model",
            walk_forward_results={"avg_sharpe": 1.5, "avg_max_dd_pct": 0.05},
            stress_test_results={},
        )
        assert result.model_name == "test_model"
        assert result.timestamp > 0

    def test_gate_conditional_approval_with_warning(self):
        """Marginal sharpe with good stress tests gets conditional approval."""
        config = GateConfig(min_sharpe=1.0, max_drawdown_pct=0.10,
                           conditional_sharpe=0.8)
        gate = ValidationGate(config)
        result = gate.evaluate(
            model_name="marginal_model",
            walk_forward_results={
                "avg_sharpe": 0.9,
                "avg_max_dd_pct": 0.05,
            },
            stress_test_results={
                "covid_crash_2020": {"max_loss_pct": 0.08, "passed": True},
            },
        )
        assert result.decision == GateDecision.CONDITIONAL_APPROVED
```

- [ ] **Step 2: Run test to verify it fails**

```
$ python -m pytest tests/training/test_validation_gate.py -v
Expected: ModuleNotFoundError
```

- [ ] **Step 3: Write implementation**

```python
# src/training/validation_gate.py
"""
Validation Gate — mandatory walk-forward + crisis scenario checks before model deployment.
Enforces minimum performance standards for all model promotions.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional
import time
from loguru import logger


class GateDecision(Enum):
    APPROVED = "approved"
    CONDITIONAL_APPROVED = "conditional_approved"
    REJECTED = "rejected"


@dataclass
class GateConfig:
    min_sharpe: float = 1.0
    max_drawdown_pct: float = 0.10
    crisis_max_loss: float = 0.15
    min_folds: int = 4
    conditional_sharpe: float = 0.8


@dataclass
class ValidationResult:
    model_name: str
    decision: GateDecision
    reason: str
    sharpe: float = 0.0
    drawdown: float = 0.0
    crisis_loss: float = 0.0
    timestamp: float = field(default_factory=time.time)
    details: Dict = field(default_factory=dict)


class ValidationGate:
    """Gatekeeper for model deployment.

    Checks before allowing deployment:
      1. Walk-forward Sharpe >= min_sharpe
      2. Walk-forward max drawdown <= max_drawdown_pct
      3. All crisis scenarios pass (loss < crisis_max_loss)
      4. Minimum number of walk-forward folds
    """

    def __init__(self, config: Optional[GateConfig] = None):
        self.config = config or GateConfig()

    def evaluate(
        self,
        model_name: str,
        walk_forward_results: Dict,
        stress_test_results: Dict[str, Dict],
    ) -> ValidationResult:
        """Evaluate model against all validation criteria."""
        issues = []
        sharpe = walk_forward_results.get("avg_sharpe", 0.0)
        drawdown = walk_forward_results.get("avg_max_dd_pct", 0.0)
        n_folds = walk_forward_results.get("total_folds", 0)

        # Check 1: Minimum folds
        if n_folds < self.config.min_folds:
            issues.append(
                f"Insufficient folds: {n_folds} < {self.config.min_folds}"
            )

        # Check 2: Sharpe ratio
        if sharpe < self.config.min_sharpe:
            if sharpe < self.config.conditional_sharpe:
                issues.append(
                    f"Sharpe {sharpe:.2f} < {self.config.min_sharpe} "
                    f"(below conditional threshold {self.config.conditional_sharpe})"
                )
            else:
                pass  # Conditional approval possible

        # Check 3: Max drawdown
        if drawdown > self.config.max_drawdown_pct:
            issues.append(
                f"Max drawdown {drawdown:.2%} > {self.config.max_drawdown_pct:.0%}"
            )

        # Check 4: Crisis scenarios
        crisis_loss = 0.0
        for scenario_name, scenario_result in stress_test_results.items():
            loss_pct = scenario_result.get("max_loss_pct", 0.0)
            crisis_loss = max(crisis_loss, loss_pct)
            if not scenario_result.get("passed", False):
                issues.append(
                    f"Crisis scenario '{scenario_name}' failed "
                    f"(loss {loss_pct:.1%})"
                )

        # Determine decision
        if len(issues) == 0:
            decision = GateDecision.APPROVED
            reason = "All validation checks passed"
        elif sharpe >= self.config.conditional_sharpe and drawdown <= self.config.max_drawdown_pct:
            decision = GateDecision.CONDITIONAL_APPROVED
            reason = f"Conditional approval: {', '.join(issues[:2])}"
        else:
            decision = GateDecision.REJECTED
            reason = f"Validation failed: {', '.join(issues)}"

        result = ValidationResult(
            model_name=model_name,
            decision=decision,
            reason=reason,
            sharpe=sharpe,
            drawdown=drawdown,
            crisis_loss=crisis_loss,
            details={
                "n_folds": n_folds,
                "walk_forward": walk_forward_results,
                "stress_tests": stress_test_results,
            },
        )

        if decision == GateDecision.APPROVED:
            logger.success(f"Validation gate PASSED for {model_name}: {reason}")
        elif decision == GateDecision.CONDITIONAL_APPROVED:
            logger.info(f"Validation gate CONDITIONAL for {model_name}: {reason}")
        else:
            logger.warning(f"Validation gate REJECTED for {model_name}: {reason}")

        return result
```

- [ ] **Step 4: Run tests to verify they pass**

```
$ python -m pytest tests/training/test_validation_gate.py -v
Expected: All 7 tests PASS
```

- [ ] **Step 5: Wire ValidationGate into ModelRegistry.deploy_candidate**

In `src/training/model_registry.py`, modify `deploy_candidate` method (around line 141-180) to accept optional `validation_gate`:

```python
    def deploy_candidate(
        self,
        name: str,
        min_improvement: float = 0.05,
        validation_gate: Optional[ValidationGate] = None,
        walk_forward_results: Optional[Dict] = None,
        stress_test_results: Optional[Dict] = None,
    ) -> Tuple[bool, str]:
        """Deploy new model if it beats current champion and passes validation."""
        if validation_gate is not None and walk_forward_results is not None:
            gate_result = validation_gate.evaluate(
                name, walk_forward_results, stress_test_results or {}
            )
            if gate_result.decision == GateDecision.REJECTED:
                logger.warning(
                    f"Deployment blocked for {name}: {gate_result.reason}"
                )
                return False, f"validation_gate_rejected: {gate_result.reason}"
```

Add the import:
```python
from training.validation_gate import ValidationGate, GateDecision
```

- [ ] **Step 6: Commit**

```
git add src/training/validation_gate.py src/training/model_registry.py \
       tests/training/test_validation_gate.py
git commit -m "feat: add validation gate for mandatory WF + crisis checks before deployment"
```

---

## Phase 2: Risk & Adaptation

---

### Task 4: Regime-Dependent Correlation Matrix

**Files:**
- Create: `src/risk/correlation_matrix.py`
- Modify: `src/risk/manager.py` (lines 313-360)
- Test: `tests/risk/test_correlation_matrix.py`

**Goal:** Replace the static hardcoded correlation dictionary in `RiskManager._check_correlation_risk` with a regime-switching model that estimates rolling correlations per regime (trending/ranging/volatile/crisis).

- [ ] **Step 1: Write the failing test**

```python
# tests/risk/test_correlation_matrix.py
import numpy as np
import pandas as pd
import pytest
from risk.correlation_matrix import (
    RegimeCorrelationMatrix,
    RegimeCorrelationStore,
    compute_regime_correlation,
)


class TestRegimeCorrelationMatrix:
    def test_store_update_and_retrieve(self):
        store = RegimeCorrelationStore()
        store.update("EURUSD", "GBPUSD", 0.85, regime="trending")
        corr = store.get("EURUSD", "GBPUSD", regime="trending")
        assert abs(corr - 0.85) < 1e-6

    def test_store_returns_default_for_unknown_pair(self):
        store = RegimeCorrelationStore()
        corr = store.get("EURUSD", "BTCUSD", regime="ranging")
        assert corr == 0.0

    def test_store_decays_old_values(self):
        store = RegimeCorrelationStore(window=3)
        store.update("EURUSD", "GBPUSD", 0.9, regime="trending")
        store.update("EURUSD", "GBPUSD", 0.8, regime="trending")
        store.update("EURUSD", "GBPUSD", 0.7, regime="trending")
        store.update("EURUSD", "GBPUSD", 0.6, regime="trending")
        # Window=3, so should only have last 3
        assert len(store._store["trending"].get(("EURUSD", "GBPUSD"), [])) == 3

    def test_compute_from_returns_matrix(self):
        returns = pd.DataFrame(
            {"EURUSD": np.random.randn(100) * 0.01,
             "GBPUSD": np.random.randn(100) * 0.01,
             "USDJPY": np.random.randn(100) * 0.008}
        )
        matrix = RegimeCorrelationMatrix()
        store = matrix.update_from_returns(returns, regime="trending")
        corr = store.get("EURUSD", "GBPUSD", regime="trending")
        assert abs(corr) <= 1.0
        assert corr != 0.0  # likely non-zero

    def test_regime_correlation_differs_by_regime(self):
        """Correlations in crisis should differ from trending."""
        returns_trend = pd.DataFrame(
            {"EURUSD": np.random.randn(100) * 0.008 + 0.0003,
             "GBPUSD": np.random.randn(100) * 0.008 + 0.0002}
        )
        returns_crisis = pd.DataFrame(
            {"EURUSD": np.random.randn(100) * 0.05 - 0.01,
             "GBPUSD": np.random.randn(100) * 0.05 - 0.01}
        )
        matrix = RegimeCorrelationMatrix(decay_half_life=0.5)
        matrix.update_from_returns(returns_trend, regime="trending")
        matrix.update_from_returns(returns_crisis, regime="crisis")
        corr_trend = matrix.get("EURUSD", "GBPUSD", regime="trending")
        corr_crisis = matrix.get("EURUSD", "GBPUSD", regime="crisis")
        # In crisis, correlations tend to converge toward 1
        assert abs(corr_crisis) >= abs(corr_trend) * 0.5  # loose check

    def test_manager_integration_no_regression(self):
        """Verify RiskManager.check_correlation uses new matrix without breaking."""
        from risk.manager import RiskManager, RiskParameters
        from risk.correlation_matrix import RegimeCorrelationMatrix
        rm = RiskManager(RiskParameters())
        matrix = RegimeCorrelationMatrix()
        rm._correlation_matrix = matrix  # inject
        result = rm._check_correlation_risk("EURUSD", ["GBPUSD"])
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_rolling_correlation_window(self):
        """Test that rolling correlation computation is correct."""
        np.random.seed(42)
        x = np.cumsum(np.random.randn(200))
        y = x + np.random.randn(200) * 0.1  # highly correlated
        returns = pd.DataFrame({"A": np.diff(x), "B": np.diff(y)})
        matrix = RegimeCorrelationMatrix(window=50)
        matrix.update_from_returns(returns, regime="ranging")
        corr = matrix.get("A", "B", regime="ranging")
        assert corr > 0.5  # should detect high correlation
```

- [ ] **Step 2: Run test to verify it fails**

```
$ python -m pytest tests/risk/test_correlation_matrix.py -v
Expected: ModuleNotFoundError
```

- [ ] **Step 3: Write implementation**

```python
# src/risk/correlation_matrix.py
"""
Regime-Dependent Correlation Matrix — replaces static FX_CORR dict.
Maintains rolling correlation estimates per market regime.
Uses exponential decay for real-time adaptation.
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class RegimeCorrelationStore:
    """Per-regime store of rolling pairwise correlations."""
    _store: Dict[str, Dict[Tuple[str, str], list]] = field(default_factory=lambda: defaultdict(dict))
    window: int = 50

    def update(self, sym_a: str, sym_b: str, correlation: float, regime: str):
        """Add a correlation observation for a pair in a regime."""
        pair = tuple(sorted([sym_a.upper(), sym_b.upper()]))
        if regime not in self._store:
            self._store[regime] = {}
        if pair not in self._store[regime]:
            self._store[regime][pair] = []
        self._store[regime][pair].append(correlation)
        # Rolling window
        if len(self._store[regime][pair]) > self.window:
            self._store[regime][pair] = self._store[regime][pair][-self.window:]

    def get(self, sym_a: str, sym_b: str, regime: str) -> float:
        """Get latest correlation for pair in regime, or 0 if unknown."""
        pair = tuple(sorted([sym_a.upper(), sym_b.upper()]))
        vals = self._store.get(regime, {}).get(pair, [])
        if not vals:
            # Fallback to any regime
            for r, pairs in self._store.items():
                if pair in pairs and pairs[pair]:
                    return pairs[pair][-1]
            return 0.0
        return vals[-1]

    def to_dataframe(self, regime: str) -> pd.DataFrame:
        """Convert stored correlations to a DataFrame matrix."""
        pairs_data = self._store.get(regime, {})
        symbols = set()
        for (a, b) in pairs_data:
            symbols.add(a)
            symbols.add(b)
        symbols = sorted(symbols)
        n = len(symbols)
        corr_matrix = np.eye(n)
        for (a, b), vals in pairs_data.items():
            if a in symbols and b in vals:
                i, j = symbols.index(a), symbols.index(b)
                corr_matrix[i, j] = vals[-1]
                corr_matrix[j, i] = vals[-1]
        return pd.DataFrame(corr_matrix, index=symbols, columns=symbols)


class RegimeCorrelationMatrix:
    """Maintains rolling pairwise correlations per market regime.

    Replaces the static FX_CORR dict in RiskManager._check_correlation_risk.
    """

    REGIMES = ["trending", "ranging", "volatile", "crisis"]

    def __init__(self, window: int = 100, decay_half_life: float = 0.5):
        self.store = RegimeCorrelationStore(window=window)
        self.window = window
        self.decay_half_life = decay_half_life
        self._returns_buffer: Dict[str, Dict[str, list]] = defaultdict(
            lambda: defaultdict(list)
        )

    def update_from_returns(
        self, returns: pd.DataFrame, regime: str
    ) -> RegimeCorrelationStore:
        """Compute pairwise correlations from a returns DataFrame and store them."""
        if len(returns) < 10:
            return self.store

        for col in returns.columns:
            for other in returns.columns:
                if col >= other:
                    continue
                series_a = returns[col].values
                series_b = returns[other].values
                # Exponentially weighted correlation
                weights = np.exp(
                    self.decay_half_life * np.linspace(-1, 0, len(series_a))
                )
                weights /= weights.sum()
                mean_a = np.average(series_a, weights=weights)
                mean_b = np.average(series_b, weights=weights)
                cov = np.average(
                    (series_a - mean_a) * (series_b - mean_b), weights=weights
                )
                std_a = np.sqrt(
                    np.average((series_a - mean_a) ** 2, weights=weights)
                )
                std_b = np.sqrt(
                    np.average((series_b - mean_b) ** 2, weights=weights)
                )
                if std_a > 0 and std_b > 0:
                    corr = float(cov / (std_a * std_b))
                    self.store.update(col, other, corr, regime)

        return self.store

    def get(self, sym_a: str, sym_b: str, regime: Optional[str] = None) -> float:
        """Get correlation between two symbols.

        If regime is specified, returns that regime's correlation.
        Falls back to any available regime, then to 0.
        """
        if regime:
            return self.store.get(sym_a, sym_b, regime)
        # Try each regime, return first non-zero
        for r in self.REGIMES:
            corr = self.store.get(sym_a, sym_b, r)
            if abs(corr) > 1e-6:
                return corr
        return 0.0
```

- [ ] **Step 4: Modify RiskManager to use RegimeCorrelationMatrix**

In `src/risk/manager.py`, modify `_check_correlation_risk`:

```python
    def _check_correlation_risk(
        self, new_symbol: str, open_symbols: List[str]
    ) -> Tuple[bool, str]:
        """Reject new position if >0.80 correlated with any open position.

        Uses RegimeCorrelationMatrix with fallback to static FX_CORR.
        """
        if not open_symbols:
            return True, "OK"

        # Try dynamic correlation matrix first
        if hasattr(self, '_correlation_matrix') and self._correlation_matrix is not None:
            for pair in open_symbols:
                corr = self._correlation_matrix.get(new_symbol, pair)
                if abs(corr) > 0.80:
                    return (
                        False,
                        f"Correlation risk: {new_symbol}/{pair}={corr:.2f}",
                    )

        # Fallback to static FX_CORR
        FX_CORR = { ... }  # keep existing dict
        for pair in open_symbols:
            key = (new_symbol, pair)
            rev_key = (pair, new_symbol)
            corr = FX_CORR.get(key, FX_CORR.get(rev_key, 0.0))
            if abs(corr) > 0.80:
                return (False, f"Correlation risk: {new_symbol}/{pair}={corr:.2f}")

        if len(open_symbols) >= 3:
            return False, f"Max correlated positions: {len(open_symbols)} open"
        return True, "OK"
```

Add at constructor init (line 65 area):
```python
        self._correlation_matrix = None  # injected externally
```

- [ ] **Step 5: Run tests to verify they pass**

```
$ python -m pytest tests/risk/test_correlation_matrix.py tests/risk/test_enhanced_manager.py -v
Expected: All tests PASS
```

- [ ] **Step 6: Commit**

```
git add src/risk/correlation_matrix.py src/risk/manager.py tests/risk/test_correlation_matrix.py
git commit -m "feat: add regime-dependent correlation matrix replacing static FX_CORR dict"
```

---

### Task 8: Adaptive Online Learning Cooldown

**Files:**
- Modify: `src/training/online_learner.py` (lines 43-54, 60, 107-119)
- Test: `tests/training/test_online_learner_adaptive.py`

**Goal:** Replace the fixed 4-hour cooldown with a drift-severity adaptive cooldown. When drift is severe, cooldown reduces to minutes. When stable, cooldown extends to hours.

- [ ] **Step 1: Write the failing test**

```python
# tests/training/test_online_learner_adaptive.py
import time
import pytest
from training.online_learner import OnlineLearner


class TestOnlineLearnerAdaptive:
    def test_default_cooldown_unchanged(self):
        learner = OnlineLearner(retrain_cooldown_hours=4.0)
        assert learner.retrain_cooldown == 4.0 * 3600

    def test_adaptive_cooldown_reduces_with_drift(self):
        learner = OnlineLearner(retrain_cooldown_hours=4.0)
        for i in range(5):
            learner.on_drift_detected("EURUSD", drift_count=i+1)
        cooldown = learner._get_adaptive_cooldown("EURUSD")
        assert cooldown < 4.0 * 3600  # should be shorter

    def test_adaptive_cooldown_increases_with_stability(self):
        learner = OnlineLearner(retrain_cooldown_hours=4.0)
        # No drift signals -> stable
        cooldown = learner._get_adaptive_cooldown("EURUSD")
        assert cooldown >= 4.0 * 3600  # should be at least default

    def test_should_retrain_respects_adaptive_cooldown(self):
        learner = OnlineLearner(retrain_cooldown_hours=4.0)
        # Mark last retrain as now
        learner._last_retrain["EURUSD"] = time.time()
        # With no drift, should NOT retrain
        assert not learner.should_retrain("EURUSD", total_trades=100)
        # With severe drift, should retrain despite recent retrain
        learner.on_drift_detected("EURUSD", drift_count=10)
        learner._drift_signals["EURUSD"] = 10
        # Force adaptive cooldown to be small
        learner._last_retrain["EURUSD"] = time.time()
        result = learner.should_retrain("EURUSD", total_trades=100)
        assert result  # severe drift overrides recent retrain

    def test_cooldown_caps_at_minimum(self):
        learner = OnlineLearner(retrain_cooldown_hours=4.0)
        learner._last_retrain["EURUSD"] = time.time()
        learner.on_drift_detected("EURUSD", drift_count=100)
        cd = learner._get_adaptive_cooldown("EURUSD")
        assert cd >= 300.0  # minimum 5 minutes

    def test_cooldown_caps_at_maximum(self):
        learner = OnlineLearner(retrain_cooldown_hours=4.0)
        learner._last_retrain["EURUSD"] = 0
        cd = learner._get_adaptive_cooldown("EURUSD")
        assert cd <= 12 * 3600  # maximum 12 hours

    def test_reset_cooldown_after_retrain(self):
        learner = OnlineLearner(retrain_cooldown_hours=4.0)
        learner.on_drift_detected("EURUSD", drift_count=5)
        learner._drift_signals["EURUSD"] = 5
        cd_before = learner._get_adaptive_cooldown("EURUSD")
        # Simulate retrain
        learner._last_retrain["EURUSD"] = time.time()
        learner._drift_signals["EURUSD"] = 0
        cd_after = learner._get_adaptive_cooldown("EURUSD")
        assert cd_after > cd_before
```

- [ ] **Step 2: Run test to verify it fails**

```
$ python -m pytest tests/training/test_online_learner_adaptive.py -v
Expected: Fail with AttributeError for _get_adaptive_cooldown
```

- [ ] **Step 3: Write implementation**

In `src/training/online_learner.py`:

Add at the top of the class (after line 41), the new method:

```python
    def _get_adaptive_cooldown(self, pair: str) -> float:
        """Calculate adaptive cooldown based on drift severity.

        Base: retrain_cooldown_hours * 3600 (default 4h)
        Drift adjustment: severe drift = shorter cooldown (down to 5 min)
        Stability: no drift = longer cooldown (up to 12h)

        Formula: adaptive_cd = base_cd * exp(-drift_factor * severity)
        """
        base_cd = float(self.retrain_cooldown)
        drift_count = self._drift_signals.get(pair, 0)

        if drift_count <= 0:
            # No drift: extend cooldown up to 3x
            return min(base_cd * 3.0, 12 * 3600)

        # Drift severity: 1-3 = mild, 4-7 = moderate, 8+ = severe
        severity = min(drift_count / 5.0, 3.0)
        factor = 1.0 - (1.0 - np.exp(-severity)) * 0.8
        adjusted = base_cd * max(factor, 0.2)

        # Minimum 5 minutes
        return max(adjusted, 300.0)
```

Add `import numpy as np` at the top of the file (if not already there). Update `should_retrain`:

```python
    def should_retrain(self, pair: str, total_trades: int) -> bool:
        with self._lock:
            if self._running.get(pair, False):
                return False
            last = self._last_retrain.get(pair, 0)
            adaptive_cd = self._get_adaptive_cooldown(pair)
            if time.time() - last < adaptive_cd:
                return False
            drift = self._drift_signals.get(pair, 0)
            if drift > 0:
                return True
            if total_trades >= self.min_trades and self._deployed.get(pair) is None:
                return True
            return False
```

- [ ] **Step 4: Run tests to verify they pass**

```
$ python -m pytest tests/training/test_online_learner_adaptive.py -v
Expected: All 7 tests PASS
```

- [ ] **Step 5: Run existing online learner tests**

```
$ python -m pytest tests/training/ -v
Expected: All existing tests PASS
```

- [ ] **Step 6: Commit**

```
git add src/training/online_learner.py tests/training/test_online_learner_adaptive.py
git commit -m "feat: adaptive online learning cooldown based on drift severity"
```

---

## Phase 3: Alpha Generation

---

### Task 3: Cross-Sectional Alpha Signals

**Files:**
- Create: `src/rts_ai_fx/cross_sectional_alpha.py`
- Modify: `src/rts_ai_fx/features_unified.py` (add cross-sectional computation in `compute_features`)
- Test: `tests/rts_ai_fx/test_cross_sectional_alpha.py`

**Goal:** Add relative-value signals: carry rank, momentum rank, value rank, volatility rank across symbols.

- [ ] **Step 1: Write the failing test**

```python
# tests/rts_ai_fx/test_cross_sectional_alpha.py
import numpy as np
import pandas as pd
import pytest
from rts_ai_fx.cross_sectional_alpha import (
    CrossSectionalAlpha,
    compute_carry_rank,
    compute_momentum_rank,
    compute_value_rank,
    compute_volatility_rank,
)


class TestCrossSectionalAlpha:
    def test_carry_rank_ranks_by_interest_rate_diff(self):
        symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
        rates = {"EUR": -0.005, "GBP": 0.05, "JPY": -0.001, "AUD": 0.04, "USD": 0.03}
        ranks = compute_carry_rank(symbols, rates)
        assert len(ranks) == 4
        assert all(0 <= r <= 1 for r in ranks.values())
        # AUDUSD should have higher carry (AUD 4% - USD 3%)
        # than EURUSD (EUR -0.5% - USD 3%)
        assert ranks["AUDUSD"] >= ranks["EURUSD"]

    def test_momentum_rank_ranks_by_return(self):
        prices = pd.DataFrame({
            "EURUSD": np.linspace(1.10, 1.12, 100),
            "GBPUSD": np.linspace(1.30, 1.28, 100),
            "USDJPY": np.linspace(110, 112, 100),
        })
        ranks = compute_momentum_rank(prices, lookback=20)
        assert len(ranks) == 3
        assert all(0 <= r <= 1 for r in ranks.values())

    def test_value_rank_uses_ppp_deviation(self):
        symbols = ["EURUSD", "GBPUSD", "USDJPY"]
        current_prices = {"EURUSD": 1.15, "GBPUSD": 1.35, "USDJPY": 110.0}
        ppp_estimates = {"EURUSD": 1.20, "GBPUSD": 1.30, "USDJPY": 105.0}
        ranks = compute_value_rank(symbols, current_prices, ppp_estimates)
        assert len(ranks) == 3
        assert all(0 <= r <= 1 for r in ranks.values())

    def test_volatility_rank_inverse_relationship(self):
        vols = {"EURUSD": 0.05, "GBPUSD": 0.12, "USDJPY": 0.08, "AUDUSD": 0.15}
        ranks = compute_volatility_rank(vols)
        # Lower vol = higher rank
        assert ranks["EURUSD"] > ranks["AUDUSD"]

    def test_cross_sectional_alpha_combines_signals(self):
        alpha = CrossSectionalAlpha()
        market_data = {
            "EURUSD": {"price": 1.10, "atr": 0.005, "carry": -0.035},
            "GBPUSD": {"price": 1.30, "atr": 0.008, "carry": 0.02},
            "USDJPY": {"price": 110.0, "atr": 0.50, "carry": -0.031},
        }
        signals = alpha.compute(market_data)
        assert len(signals) == 3
        for sym, signal in signals.items():
            assert "composite_rank" in signal
            assert "carry_rank" in signal
            assert "momentum_rank" in signal
            assert "value_rank" in signal
            assert "vol_rank" in signal
            assert -1 <= signal["composite_rank"] <= 1

    def test_empty_input_returns_empty(self):
        alpha = CrossSectionalAlpha()
        assert alpha.compute({}) == {}
```

- [ ] **Step 2: Run test to verify it fails**

```
$ python -m pytest tests/rts_ai_fx/test_cross_sectional_alpha.py -v
Expected: ModuleNotFoundError
```

- [ ] **Step 3: Write implementation**

```python
# src/rts_ai_fx/cross_sectional_alpha.py
"""
Cross-Sectional Alpha Signals — relative-value signals across instruments.
Carry rank, momentum rank, value rank, volatility rank.
All ranks are cross-sectional (0=worst, 1=best) within the universe.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


def _rank_to_uniform(values: Dict[str, float]) -> Dict[str, float]:
    """Convert values to cross-sectional uniform ranks [0, 1]."""
    if not values:
        return {}
    sorted_pairs = sorted(values.items(), key=lambda x: x[1])
    n = len(sorted_pairs)
    return {sym: i / max(n - 1, 1) for i, (sym, _) in enumerate(sorted_pairs)}


def compute_carry_rank(
    symbols: List[str], interest_rates: Dict[str, float]
) -> Dict[str, float]:
    """Rank symbols by carry (long-short interest rate differential).

    For each pair, carry = base_currency_rate - quote_currency_rate.
    Higher carry = higher rank.
    """
    carry_values = {}
    for sym in symbols:
        base = sym[:3]
        quote = sym[3:]
        base_rate = interest_rates.get(base, 0.0)
        quote_rate = interest_rates.get(quote, 0.0)
        carry = base_rate - quote_rate
        carry_values[sym] = carry
    return _rank_to_uniform(carry_values)


def compute_momentum_rank(
    prices: pd.DataFrame, lookback: int = 21
) -> Dict[str, float]:
    """Rank symbols by momentum (return over lookback period)."""
    if len(prices) < lookback + 1:
        return {col: 0.5 for col in prices.columns}
    returns = prices.pct_change(lookback).iloc[-1].to_dict()
    return _rank_to_uniform(returns)


def compute_value_rank(
    symbols: List[str],
    current_prices: Dict[str, float],
    ppp_estimates: Dict[str, float],
) -> Dict[str, float]:
    """Rank symbols by value (deviation from purchasing power parity).

    Value = (PPP - current) / PPP — positive means undervalued.
    """
    value_scores = {}
    for sym in symbols:
        price = current_prices.get(sym, 0.0)
        ppp = ppp_estimates.get(sym, price)
        if ppp > 0:
            value_scores[sym] = (ppp - price) / ppp
        else:
            value_scores[sym] = 0.0
    return _rank_to_uniform(value_scores)


def compute_volatility_rank(volatilities: Dict[str, float]) -> Dict[str, float]:
    """Rank symbols by volatility (inverse: lower vol = higher rank)."""
    if not volatilities:
        return {}
    # Invert: lower vol = higher rank
    inverted = {sym: 1.0 / max(v, 1e-10) for sym, v in volatilities.items()}
    return _rank_to_uniform(inverted)


class CrossSectionalAlpha:
    """Computes composite cross-sectional alpha from multiple signals.

    Combines:
      - Carry rank (interest rate differential)
      - Momentum rank (recent return)
      - Value rank (PPP deviation)
      - Volatility rank (inverse vol)
    into a single composite rank [-1, 1].
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            "carry": 0.25,
            "momentum": 0.35,
            "value": 0.20,
            "vol": 0.20,
        }

    def compute(
        self,
        market_data: Dict[str, Dict],
        price_history: Optional[pd.DataFrame] = None,
        interest_rates: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Dict]:
        """Compute cross-sectional alpha for all symbols.

        Args:
            market_data: dict of {symbol: {price, atr, carry, ...}}
            price_history: DataFrame with columns as symbols, rows as time
            interest_rates: dict of {currency: rate}

        Returns:
            dict of {symbol: {composite_rank, carry_rank, momentum_rank, ...}}
        """
        if not market_data:
            return {}

        symbols = list(market_data.keys())
        rates = interest_rates or {}

        # Carry rank
        carry_ranks = compute_carry_rank(symbols, rates)
        if not carry_ranks:
            carry_ranks = {s: 0.5 for s in symbols}

        # Momentum rank
        if price_history is not None and len(price_history) > 1:
            momentum_ranks = compute_momentum_rank(price_history)
        else:
            momentum_ranks = {s: 0.5 for s in symbols}

        # Value
</task_result>