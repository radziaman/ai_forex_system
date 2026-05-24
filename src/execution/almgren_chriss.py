"""
Almgren-Chriss (2001) Optimal Execution Model.
Implements permanent and temporary market impact with optimal VWAP trajectory.
"""

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class ImpactResult:
    permanent_impact_bps: float
    temporary_impact_bps: float
    total_cost_bps: float
    total_cost_usd: float
    optimal_slices: List[dict]


class AlmgrenChrissModel:
    """
    Almgren-Chriss market impact model with optimal execution trajectory.

    Permanent impact: I_p(Q) = gamma * sigma * (Q/V)^delta
    Temporary impact: I_t(Q) = eta * sigma * (Q/V)^delta + spread/2
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
        participation_rate = volume / max(self.adv, 1.0)
        impact_term = (participation_rate**self.delta) * self.volatility

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
        if n_slices <= 0:
            return []

        T = duration_minutes / (60 * 24 * 252)
        tau = T / n_slices
        sigma_sq = self.volatility**2
        kappa = np.sqrt(self.risk_aversion * sigma_sq / max(self.eta, 1e-10))

        slices = []
        remaining = volume
        for j in range(n_slices):
            t_j = j * tau
            if kappa > 1e-10:
                numerator = np.sinh(kappa * (T - t_j))
                denominator = np.sinh(kappa * T)
                weight = (
                    numerator / denominator if denominator > 0 else (1.0 - j / n_slices)
                )
            else:
                weight = 1.0 - j / n_slices

            total_weight = sum(max(1.0 - i / n_slices, 0.01) for i in range(n_slices))
            weight /= max(total_weight, 1e-10)

            slice_vol = volume * weight
            participation = slice_vol / max(self.adv * tau * 252, 1.0)
            impact = (
                self.eta * (participation**self.delta) * self.volatility
                + self.spread / 2.0
            ) * 10_000

            slice_vol = min(slice_vol, remaining)
            slices.append(
                {
                    "slice_idx": j,
                    "volume": float(slice_vol),
                    "impact_bps": float(impact),
                }
            )
            remaining -= slice_vol

        if slices and remaining > 0:
            slices[-1]["volume"] += remaining

        return slices
