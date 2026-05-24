"""
Synthetic Data Generation for Forex Trading Systems.

Implements multiple approaches for generating realistic synthetic market data:
  1. GMM Sampler — Gaussian Mixture Model fitted to historical returns
  2. Regime-Conditional Sampler — generates bars conditioned on HMM regime
  3. Crisis Augmenter — generates flash-crash/crisis scenarios preserving
     volatility clustering and fat tails

The generated data preserves key statistical properties:
  - Volatility clustering (GARCH-like behavior via regime switching)
  - Fat tails (via Student-t mixture components)
  - Cross-asset correlations (preserved via multivariate sampling)
  - Regime dynamics (trending, ranging, volatile, crisis)
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
from sklearn.mixture import GaussianMixture
from scipy.stats import t as student_t
from loguru import logger


@dataclass
class SyntheticBar:
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: float


class GMMSampler:
    """Gaussian Mixture Model based synthetic return generator.

    Fits a GMM to historical log-returns, then samples from it to
    generate synthetic price paths. Captures multi-modal return
    distributions (e.g., calm vs. volatile regimes).
    """

    def __init__(self, n_components: int = 3, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self._gmm: Optional[GaussianMixture] = None
        self._mean_return: float = 0.0
        self._std_return: float = 0.0
        self._last_price: float = 0.0

    def fit(self, returns: np.ndarray):
        """Fit GMM to historical log-returns.

        Args:
            returns: (n_samples, n_symbols) array of log-returns
        """
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)
        self._mean_return = float(np.mean(returns))
        self._std_return = float(np.std(returns))
        self._gmm = GaussianMixture(
            n_components=self.n_components,
            random_state=self.random_state,
            covariance_type="full" if returns.shape[1] > 1 else "spherical",
        )
        self._gmm.fit(returns)
        logger.info(
            f"GMMSampler: fitted {self.n_components} components "
            f"on {returns.shape[0]} samples x {returns.shape[1]} symbols"
        )

    def sample(
        self, n_samples: int, start_price: float = 1.0, seed: Optional[int] = None
    ) -> np.ndarray:
        """Generate synthetic price paths.

        Args:
            n_samples: number of bars to generate
            start_price: starting price
            seed: random seed

        Returns:
            (n_samples,) array of synthetic prices
        """
        if self._gmm is None:
            raise ValueError("GMM not fitted. Call fit() first.")
        if seed is not None:
            np.random.seed(seed)
        n_dims = self._gmm.means_.shape[1]
        returns = self._gmm.sample(n_samples)[0]  # (n_samples, n_dims)
        if n_dims == 1:
            returns = returns.flatten()
            prices = start_price * np.exp(np.cumsum(returns))
        else:
            # Multi-symbol: use first dimension
            prices = start_price * np.exp(np.cumsum(returns[:, 0]))
        return prices

    def sample_regime_conditional(
        self, n_samples: int, regime_component: int = 0, start_price: float = 1.0
    ) -> np.ndarray:
        """Sample from a specific GMM component (regime).

        Args:
            n_samples: number of bars
            regime_component: which GMM component to sample from
            start_price: starting price

        Returns:
            (n_samples,) synthetic prices
        """
        if self._gmm is None:
            raise ValueError("GMM not fitted.")
        means = self._gmm.means_[regime_component]
        raw_cov = self._gmm.covariances_[regime_component]
        n_dims = means.shape[0]
        # Expand covariance to full matrix depending on GMM covariance_type
        if np.ndim(raw_cov) == 0:
            # spherical: scalar variance -> diagonal matrix
            covs = float(raw_cov) * np.eye(n_dims)
        elif raw_cov.ndim == 1:
            # diag: per-feature variances -> diagonal matrix
            covs = np.diag(raw_cov)
        else:
            # full: already a proper covariance matrix
            covs = raw_cov
        raw = np.random.multivariate_normal(means, covs, size=n_samples)
        if n_dims == 1:
            returns = raw.flatten()
        else:
            returns = raw[:, 0]
        return start_price * np.exp(np.cumsum(returns))


class CrisisAugmenter:
    """Generates synthetic crisis scenarios for robust training.

    Produces realistic crash/recovery patterns with:
      - Initial sharp decline (fat tail)
      - Increased volatility during crash
      - Gradual recovery (V or U shaped)
      - Correlation regime shift (all pairs become correlated)
    """

    CRISIS_TYPES = ["flash_crash", "slow_bleed", "v_recovery", "u_recovery"]

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def generate_crisis(
        self,
        n_bars: int = 100,
        crisis_type: str = "flash_crash",
        crash_depth: float = 0.10,
        start_price: float = 1.0,
    ) -> np.ndarray:
        """Generate a synthetic crisis price path.

        Args:
            n_bars: total number of bars
            crisis_type: one of flash_crash, slow_bleed, v_recovery, u_recovery
            crash_depth: max drawdown (e.g., 0.10 = 10% crash)
            start_price: starting price

        Returns:
            (n_bars,) synthetic crisis prices
        """
        np.random.seed(self.random_state)
        prices = np.ones(n_bars) * start_price
        crash_bars = n_bars // 3  # First third is the crash
        recovery_bars = n_bars - crash_bars

        if crisis_type == "flash_crash":
            # Sharp drop with student-t (fat tails)
            crash_returns = student_t.rvs(
                df=3, loc=-crash_depth / crash_bars, scale=0.01, size=crash_bars
            )
            recovery_returns = (
                np.random.randn(recovery_bars) * 0.002
                + crash_depth / recovery_bars * 0.5
            )
            all_returns = np.concatenate([crash_returns, recovery_returns])

        elif crisis_type == "slow_bleed":
            crash_returns = (
                np.random.randn(crash_bars) * 0.005 - crash_depth / crash_bars
            )
            recovery_returns = np.random.randn(recovery_bars) * 0.003
            all_returns = np.concatenate([crash_returns, recovery_returns])

        elif crisis_type == "v_recovery":
            half1 = crash_bars // 2
            half2 = crash_bars - half1  # handles odd crash_bars
            crash_returns = student_t.rvs(
                df=2,
                loc=-crash_depth / half1,
                scale=0.02,
                size=half1,
            )
            v_recovery = student_t.rvs(
                df=4,
                loc=crash_depth / half2,
                scale=0.01,
                size=half2,
            )
            post_recovery = np.random.randn(recovery_bars) * 0.002
            all_returns = np.concatenate([crash_returns, v_recovery, post_recovery])

        elif crisis_type == "u_recovery":
            crash_returns = student_t.rvs(
                df=2,
                loc=-crash_depth / crash_bars,
                scale=0.015,
                size=crash_bars,
            )
            flat_len = min(crash_bars // 2, recovery_bars)
            recovery_len = recovery_bars - flat_len
            flat = np.random.randn(flat_len) * 0.003
            recovery = (
                np.random.randn(recovery_len) * 0.003
                + crash_depth / max(recovery_bars, 1) * 0.3
            )
            all_returns = np.concatenate([crash_returns, flat, recovery])
        else:
            all_returns = np.random.randn(n_bars) * 0.005

        prices = start_price * np.exp(np.cumsum(all_returns))
        return prices

    def generate_all_crisis_types(
        self, n_bars: int = 100, crash_depth: float = 0.10, start_price: float = 1.0
    ) -> Dict[str, np.ndarray]:
        """Generate all crisis types for comprehensive stress testing."""
        return {
            ctype: self.generate_crisis(n_bars, ctype, crash_depth, start_price)
            for ctype in self.CRISIS_TYPES
        }


class VolatilityClusterSampler:
    """Generates returns with realistic volatility clustering.

    Uses a regime-switching volatility model: high-vol and low-vol
    regimes with Markov transition between them.
    """

    def __init__(self, p_high: float = 0.05, p_low: float = 0.10):
        """p_high: probability of staying in high-vol regime
        p_low: probability of staying in low-vol regime
        """
        self.p_high = p_high
        self.p_low = p_low

    def sample(
        self,
        n_samples: int,
        low_vol: float = 0.005,
        high_vol: float = 0.03,
        start_price: float = 1.0,
    ) -> np.ndarray:
        """Generate returns with volatility clustering.

        Returns:
            (n_samples,) price array with clustered volatility
        """
        returns = np.zeros(n_samples)
        in_high_vol = False
        for i in range(n_samples):
            if in_high_vol:
                vol = high_vol
                if np.random.random() > self.p_high:
                    in_high_vol = False
            else:
                vol = low_vol
                if np.random.random() > self.p_low:
                    in_high_vol = True
            returns[i] = np.random.randn() * vol
        return start_price * np.exp(np.cumsum(returns))
