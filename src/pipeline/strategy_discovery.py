"""StrategyDiscovery — auto-discovers profitable trading strategies.

Runs candidate strategy configurations in a sandboxed paper-trading
environment, evaluates their performance, and auto-registers promising
strategies as experts in the MoE ensemble.

Events consumed:
    tick -> {symbol, bid, ask, volume, timestamp}
    position_closed -> {position: dict}

Events emitted:
    strategy_discovered -> {strategy: str, sharpe: float, win_rate: float}
    strategy_registered -> {strategy: str, expert_name: str}
"""

import asyncio
import time
import random
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from loguru import logger


@dataclass
class StrategyCandidate:
    """A candidate strategy configuration being evaluated."""

    name: str
    strategy_type: str  # "breakout", "mean_reversion", "momentum", "volatility"
    params: Dict[str, Any]
    predict_fn: Optional[Callable] = None
    confidence_fn: Optional[Callable] = None

    # Performance tracking
    trades: int = 0
    wins: int = 0
    total_pnl: float = 0.0
    sharpe: float = 0.0
    win_rate: float = 0.0
    returns: List[float] = field(default_factory=list)
    is_active: bool = False


class StrategyDiscovery:
    """Auto-discovers profitable trading strategies by running experiments.

    Maintains a pool of parameterized strategy templates, runs them on live
    tick data in evaluation mode, and promotes high-performers to the expert
    ensemble. Low-performers are retired after a minimum evaluation period.
    """

    # Strategy templates with parameter ranges to sweep
    STRATEGY_TEMPLATES: List[Dict[str, Any]] = [
        {
            "name": "breakout",
            "params": [
                {"lookback": 20, "multiplier": 1.5},
                {"lookback": 20, "multiplier": 2.0},
                {"lookback": 50, "multiplier": 1.5},
                {"lookback": 50, "multiplier": 2.0},
            ],
        },
        {
            "name": "mean_reversion",
            "params": [
                {"lookback": 10, "entry_std": 1.5},
                {"lookback": 10, "entry_std": 2.0},
                {"lookback": 20, "entry_std": 1.5},
                {"lookback": 20, "entry_std": 2.0},
            ],
        },
        {
            "name": "momentum",
            "params": [
                {"lookback": 5, "threshold": 0.001},
                {"lookback": 10, "threshold": 0.001},
                {"lookback": 5, "threshold": 0.002},
                {"lookback": 10, "threshold": 0.002},
            ],
        },
        {
            "name": "volatility",
            "params": [
                {"lookback": 10, "vol_threshold": 1.5},
                {"lookback": 20, "vol_threshold": 1.5},
                {"lookback": 10, "vol_threshold": 2.0},
                {"lookback": 20, "vol_threshold": 2.0},
            ],
        },
    ]

    def __init__(
        self,
        event_bus,
        expert_registry=None,
        min_eval_trades: int = 20,
        max_candidates: int = 20,
        eval_interval: float = 86400.0,  # Daily
        sharpe_threshold: float = 1.0,
        win_rate_threshold: float = 0.55,
    ):
        self._bus = event_bus
        self._registry = expert_registry
        self._min_eval_trades = min_eval_trades
        self._max_candidates = max_candidates
        self._eval_interval = eval_interval
        self._sharpe_threshold = sharpe_threshold
        self._win_rate_threshold = win_rate_threshold

        # Price history for strategy computation
        self._price_history: Dict[str, List[float]] = defaultdict(list)
        self._candidates: Dict[str, StrategyCandidate] = {}
        self._evaluated: List[str] = []  # Names of evaluated candidates

        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        """Subscribe to ticks and start evaluation loop."""
        self._bus.on("tick", self._on_tick)
        self._bus.on("position_closed", self._on_position_closed)
        self._bus.on("strategy_discovery_scan", self._on_manual_scan)
        self._running = True

        # Initialize candidates from templates
        self._init_candidates()
        self._task = asyncio.create_task(self._eval_loop())
        logger.info(
            f"StrategyDiscovery started: {len(self._candidates)} candidates, "
            f"eval every {self._eval_interval}s"
        )

    async def stop(self):
        """Stop strategy discovery."""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        self._bus.off("tick", self._on_tick)
        self._bus.off("position_closed", self._on_position_closed)
        self._bus.off("strategy_discovery_scan", self._on_manual_scan)
        logger.info("StrategyDiscovery stopped")

    def _init_candidates(self):
        """Create candidate strategies from templates."""
        count = 0
        for template in self.STRATEGY_TEMPLATES:
            base_name = template["name"]
            for params in template["params"]:
                name = f"candidate_{base_name}_{count}"
                candidate = StrategyCandidate(
                    name=name,
                    strategy_type=base_name,
                    params=params,
                )
                self._candidates[name] = candidate
                count += 1
                if count >= self._max_candidates:
                    break
            if count >= self._max_candidates:
                break

    async def _on_manual_scan(self, **data):
        """Handle manual evaluation trigger."""
        logger.info("StrategyDiscovery: manual evaluation triggered")
        await self._evaluate_candidates()

    async def _on_tick(self, **data):
        """Collect tick data for strategy evaluation."""
        symbol = data.get("symbol")
        bid = data.get("bid", 0)
        ask = data.get("ask", 0)
        mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else 0.0
        if mid > 0:
            self._price_history[symbol].append(mid)
            if len(self._price_history[symbol]) > 500:
                self._price_history[symbol] = self._price_history[symbol][-500:]

    async def _on_position_closed(self, **data):
        """Feed closed trade data into candidate evaluation."""
        position = data.get("position", data)
        if isinstance(position, dict):
            strategy = position.get("strategy", "")
            if strategy in self._candidates:
                pnl = float(position.get("pnl", position.get("profit", 0.0)))
                candidate = self._candidates[strategy]
                candidate.trades += 1
                candidate.total_pnl += pnl
                candidate.returns.append(pnl)
                if pnl > 0:
                    candidate.wins += 1
                if len(candidate.returns) > 100:
                    candidate.returns = candidate.returns[-100:]

    async def _eval_loop(self):
        """Periodic evaluation loop."""
        await asyncio.sleep(60)
        await self._evaluate_candidates()
        while self._running:
            await asyncio.sleep(self._eval_interval)
            await self._evaluate_candidates()

    async def _evaluate_candidates(self):
        """Evaluate all candidates and promote/retire."""
        for name, candidate in list(self._candidates.items()):
            if candidate.trades < self._min_eval_trades:
                continue

            # Compute performance metrics
            if candidate.returns:
                mean_r = np.mean(candidate.returns)
                std_r = np.std(candidate.returns)
                candidate.sharpe = (mean_r / std_r * np.sqrt(252)) if std_r > 1e-12 else 0.0
            candidate.win_rate = candidate.wins / candidate.trades if candidate.trades > 0 else 0.0

            # Evaluate against thresholds
            if (candidate.sharpe >= self._sharpe_threshold
                    and candidate.win_rate >= self._win_rate_threshold
                    and not candidate.is_active):
                await self._promote_candidate(candidate)
            elif (candidate.sharpe < -0.5 and candidate.trades >= self._min_eval_trades * 2):
                # Retire poor performers
                logger.info(f"StrategyDiscovery: retiring {name} "
                           f"(sharpe={candidate.sharpe:.2f}, trades={candidate.trades})")
                del self._candidates[name]

    async def _promote_candidate(self, candidate: StrategyCandidate):
        """Promote a candidate to active expert."""
        candidate.is_active = True

        await self._bus.emit(
            "strategy_discovered",
            strategy=candidate.name,
            strategy_type=candidate.strategy_type,
            sharpe=round(candidate.sharpe, 4),
            win_rate=round(candidate.win_rate, 4),
            params=candidate.params,
        )

        # Register with expert registry
        expert_name = f"discovered_{candidate.strategy_type}_{len(self._evaluated)}"
        if self._registry and hasattr(self._registry, 'register_strategy'):
            try:
                self._registry.register_strategy(
                    name=expert_name,
                    strategy_type=candidate.strategy_type,
                    params=candidate.params,
                )
                await self._bus.emit(
                    "strategy_registered",
                    strategy=candidate.name,
                    expert_name=expert_name,
                )
                logger.info(f"StrategyDiscovery: promoted {candidate.name} "
                           f"as '{expert_name}' (sharpe={candidate.sharpe:.2f})")
            except Exception as e:
                logger.warning(f"StrategyDiscovery: failed to register {expert_name}: {e}")

        self._evaluated.append(candidate.name)

    def get_candidate_report(self) -> Dict:
        """Return evaluation status of all candidates."""
        report = {}
        for name, c in self._candidates.items():
            report[name] = {
                "type": c.strategy_type,
                "params": c.params,
                "trades": c.trades,
                "wins": c.wins,
                "total_pnl": round(c.total_pnl, 2),
                "sharpe": round(c.sharpe, 4),
                "win_rate": round(c.win_rate, 4),
                "active": c.is_active,
            }
        return report
