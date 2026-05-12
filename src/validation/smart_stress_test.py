"""
Smart Stress Testing Suite (Enhancement #13).
Pre-built crisis scenarios, correlated failure tests, slippage stress testing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class CrisisScenario:
    """Pre-built crisis scenario for stress testing."""

    name: str
    description: str
    returns: List[float]  # Simulated returns during crisis
    start_date: str = ""
    end_date: str = ""
    market: str = "forex"  # forex | crypto | commodities


@dataclass
class StressTestResult:
    """Results from stress testing."""

    scenario: str
    max_loss: float
    max_loss_pct: float
    impact: float
    passed: bool
    details: Dict = field(default_factory=dict)


# Pre-built crisis scenarios (Enhancement #13)
CRISIS_SCENARIOS = {
    "brexit_2016": CrisisScenario(
        name="Brexit 2016",
        description="British EU referendum shock",
        returns=[-0.02, -0.03, -0.015, -0.01, 0.005, -0.02, -0.01, 0.01, -0.005, -0.01],
        start_date="2016-06-23",
        end_date="2016-07-10",
        market="forex",
    ),
    "covid_crash_2020": CrisisScenario(
        name="COVID-19 Crash 2020",
        description="Pandemic market crash",
        returns=[
            -0.04,
            -0.05,
            -0.06,
            -0.03,
            -0.02,
            0.01,
            -0.01,
            -0.03,
            0.02,
            -0.01,
            -0.02,
            0.03,
            0.01,
            -0.01,
            0.02,
        ],
        start_date="2020-02-20",
        end_date="2020-04-30",
        market="forex",
    ),
    "flash_crash_2010": CrisisScenario(
        name="Flash Crash 2010",
        description="May 6 flash crash",
        returns=[-0.05, -0.08, -0.1, -0.06, -0.03, 0.02, 0.04, 0.01, -0.01, 0.02],
        start_date="2010-05-06",
        end_date="2010-05-10",
        market="forex",
    ),
    "crypto_crash_2022": CrisisScenario(
        name="Crypto Winter 2022",
        description="Luna/FTX collapse",
        returns=[-0.15, -0.2, -0.25, -0.1, -0.05, -0.08, -0.12, -0.03, -0.05, -0.1],
        start_date="2022-05-01",
        end_date="2022-12-31",
        market="crypto",
    ),
    "oil_crash_2020": CrisisScenario(
        name="Oil Price Crash 2020",
        description="WTI crude negative pricing",
        returns=[-0.3, -0.4, -0.5, -0.2, -0.1, -0.15, -0.05, 0.1, -0.08, -0.12],
        start_date="2020-04-01",
        end_date="2020-06-30",
        market="commodities",
    ),
}


class SmartStressTester:
    """
    Smart stress testing with pre-built crisis scenarios.
    Tests system resilience under extreme market conditions.
    """

    def __init__(self, initial_balance: float = 100000.0):
        self.initial_balance = initial_balance
        self.scenarios = CRISIS_SCENARIOS.copy()
        self.results: List[StressTestResult] = []

    def run_scenario(
        self, scenario_name: str, position_size: float = 1.0, symbol: str = "EURUSD"
    ) -> StressTestResult:
        """Run a specific crisis scenario."""
        if scenario_name not in self.scenarios:
            return StressTestResult(
                scenario=scenario_name,
                max_loss=0,
                max_loss_pct=0,
                impact=0,
                passed=False,
                details={"error": "Scenario not found"},
            )

        scenario = self.scenarios[scenario_name]
        returns = scenario.returns

        # Simulate trading during crisis
        balance = self.initial_balance
        max_loss = 0.0
        peak = balance

        for i, ret in enumerate(returns):
            # Apply position size and simulate loss
            pnl = balance * ret * position_size
            balance += pnl

            if balance > peak:
                peak = balance

            loss = peak - balance
            if loss > max_loss:
                max_loss = loss

        max_loss_pct = max_loss / self.initial_balance
        impact = max_loss / self.initial_balance

        # Pass if loss < 10% of capital
        passed = max_loss_pct < 0.10

        result = StressTestResult(
            scenario=scenario_name,
            max_loss=max_loss,
            max_loss_pct=max_loss_pct,
            impact=impact,
            passed=passed,
            details={
                "final_balance": balance,
                "peak": peak,
                "num_periods": len(returns),
                "market": scenario.market,
            },
        )

        self.results.append(result)

        if passed:
            logger.info(
                f"Stress test PASSED: {scenario_name} (loss={max_loss_pct:.1%})"
            )
        else:
            logger.warning(
                f"Stress test FAILED: {scenario_name} (loss={max_loss_pct:.1%})"
            )

        return result

    def run_all_scenarios(
        self, position_size: float = 1.0
    ) -> Dict[str, StressTestResult]:
        """Run all pre-built crisis scenarios."""
        results = {}
        for name in self.scenarios.keys():
            results[name] = self.run_scenario(name, position_size)
        return results

    def test_correlated_failure(
        self, symbols: List[str], correlation: float = 0.8
    ) -> StressTestResult:
        """
        Test correlated failure across multiple positions (Enhancement #13).
        Simulates all positions moving against you simultaneously.
        """
        # Simulate correlated crash (all positions lose value)
        num_positions = len(symbols)
        avg_loss_pct = 0.05 * (1 + correlation)  # Higher correlation = worse

        total_loss = self.initial_balance * avg_loss_pct * (num_positions / 10.0)
        max_loss_pct = total_loss / self.initial_balance

        passed = max_loss_pct < 0.15  # Allow up to 15% loss

        return StressTestResult(
            scenario="correlated_failure",
            max_loss=total_loss,
            max_loss_pct=max_loss_pct,
            impact=correlation,
            passed=passed,
            details={
                "symbols": symbols,
                "correlation": correlation,
                "num_positions": num_positions,
            },
        )

    def test_slippage_stress(
        self, normal_spread: float = 0.5, stress_multiplier: float = 10.0
    ) -> StressTestResult:
        """
        Test performance under extreme slippage conditions (Enhancement #13).
        Simulates 10x normal spread widening.
        """
        # Simulate slippage eating into profits
        stressed_spread = normal_spread * stress_multiplier

        # Assume 10 trades with slippage
        num_trades = 10
        avg_trade_pnl = 50.0  # Normal profitable trade
        slippage_cost_per_trade = stressed_spread * 10  # $10 per pip for standard lot

        total_slippage = slippage_cost_per_trade * num_trades
        net_pnl = (avg_trade_pnl * num_trades) - total_slippage

        loss = max(0, -net_pnl)
        loss_pct = loss / self.initial_balance

        passed = loss_pct < 0.05  # Less than 5% loss

        return StressTestResult(
            scenario="slippage_stress",
            max_loss=loss,
            max_loss_pct=loss_pct,
            impact=stress_multiplier,
            passed=passed,
            details={
                "normal_spread": normal_spread,
                "stressed_spread": stressed_spread,
                "num_trades": num_trades,
                "total_slippage": total_slippage,
            },
        )

    def get_summary(self) -> Dict:
        """Get summary of all stress test results."""
        if not self.results:
            return {}

        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)

        return {
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total > 0 else 0,
            "worst_scenario": (
                max(self.results, key=lambda r: r.max_loss_pct).scenario
                if self.results
                else None
            ),
            "worst_loss_pct": max((r.max_loss_pct for r in self.results), default=0),
        }
