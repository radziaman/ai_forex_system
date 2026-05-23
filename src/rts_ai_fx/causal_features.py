"""
Causal Feature Selection — Remove spurious correlations.
Uses PC-MCI (Peter-Clark Momentary Conditional Independence) for time series causality.
Only keeps features that CAUSE the target (not just correlate with it).
"""

import pandas as pd
from typing import List, Dict
from loguru import logger

try:
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr
    import tigramite.data_processing as pp

    CAUSAL_AVAILABLE = True
except ImportError:
    CAUSAL_AVAILABLE = False
    logger.warning("tigramite not installed. Using correlation fallback.")


class CausalFeatureSelector:
    """
    Use causal discovery to find TRUE predictive features.
    Removes spurious correlations that break in live trading.
    """

    def __init__(self, max_lag: int = 5, alpha: float = 0.01):
        self.max_lag = max_lag
        self.alpha = alpha
        self.causal_features_: List[str] = []
        self.causal_graph_ = None
        self.p_matrix_ = None
        self.val_matrix_ = None

    def fit(self, features: pd.DataFrame, target: pd.Series) -> List[str]:
        """
        Discover causal relationships, not just correlations.
        Returns list of features that directly cause the target.
        """
        if not CAUSAL_AVAILABLE:
            logger.info(
                "Causal discovery not available, using correlation-based selection."
            )
            return self._correlation_fallback(features, target)

        if features.empty or len(features) < 50:
            logger.warning("Insufficient data for causal discovery.")
            return list(features.columns)

        # Prepare data: add target as a column
        data = features.copy()
        data["_target_"] = target.values

        try:
            # Run PCMCI algorithm
            dataframe = self._to_causal_format(data)
            pcmci = PCMCI(
                dataframe=dataframe,
                cond_ind_test=ParCorr(significance="analytic"),
                verbosity=0,
            )
            results = pcmci.run_pcmci(
                tau_min=0,
                tau_max=self.max_lag,
                pc_alpha=self.alpha,
            )

            self.p_matrix_ = results["p_matrix"]
            self.val_matrix_ = results["val_matrix"]
            self.causal_graph_ = results

            # Extract features that CAUSE the target (not just correlate)
            causal_features = self._extract_causal_features(features, results, data)

            if causal_features:
                self.causal_features_ = causal_features
                logger.info(
                    f"Causal selection: {len(causal_features)}/{len(features.columns)} "
                    f"features kept (causal only)"
                )
            else:
                logger.warning("No causal features found, keeping all features.")
                return list(features.columns)

            return self.causal_features_

        except Exception as e:
            logger.error(f"Causal discovery failed: {e}")
            return self._correlation_fallback(features, target)

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Keep only causally-validated features."""
        if not self.causal_features_:
            return features
        available = [f for f in self.causal_features_ if f in features.columns]
        return features[available]

    def fit_transform(self, features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        """Fit and transform in one call."""
        self.fit(features, target)
        return self.transform(features)

    def _extract_causal_features(
        self, features: pd.DataFrame, results: dict, data: pd.DataFrame
    ) -> List[str]:
        """Extract features that have causal link to target."""
        causal = []
        target_idx = len(data.columns) - 1  # Last column is target

        for i, col in enumerate(features.columns):
            # Check if feature has ANY causal link to target
            has_causal_link = False
            for tau in range(self.max_lag + 1):
                p_val = results["p_matrix"][i, target_idx, tau]
                if p_val < self.alpha:  # Significant causal link
                    has_causal_link = True
                    break

            if has_causal_link:
                causal.append(col)

        return causal

    def _correlation_fallback(
        self, features: pd.DataFrame, target: pd.Series
    ) -> List[str]:
        """Fallback to correlation when causal discovery unavailable."""
        if features.empty:
            return []
        corrs = features.apply(
            lambda col: abs(col.corr(target)) if col.std() > 0 else 0.0
        )
        # Keep features with correlation > threshold
        threshold = 0.05
        selected = corrs[corrs > threshold].index.tolist()
        logger.info(
            f"Correlation fallback: {len(selected)}/{len(features.columns)} features kept"  # noqa: E501
        )
        return selected

    def _to_causal_format(self, df: pd.DataFrame):
        """Convert pandas DataFrame to tigramite format."""
        return pp.DataFrame(df.values, var_names=list(df.columns))

    def get_causal_summary(self) -> Dict:
        """Return summary of causal analysis."""
        return {
            "n_causal_features": len(self.causal_features_),
            "causal_features": self.causal_features_[:10],  # Top 10
            "algorithm": "PCMCI" if CAUSAL_AVAILABLE else "correlation_fallback",
            "max_lag": self.max_lag,
            "alpha": self.alpha,
        }
