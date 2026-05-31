"""
train_and_backtest_ml.py — End-to-end PPO agent training & ML-driven backtest.

Pipeline:
  1. Load EURUSD 1h data, compute unified feature set (47+ technical indicators)
  2. Detect market regimes via HMM (trending / ranging / volatile / crisis)
  3. Split data: train on first 70 %, test on last 30 %
  4. Train 3 regime-specific PPO agents (trending, ranging, volatile) on their
     respective regime bars using a TradingEnvironment for reward computation
  5. Build an ML signal function from the trained RegimeSpecialistSystem
  6. Run a vectorized backtest on the test set — with sensitivity analysis,
     regime breakdown, and Monte Carlo significance
  7. Compare trained vs untrained (random-weight) agent results in a clear
     side-by-side report

Usage:
    python -m src.scripts.train_and_backtest_ml
    python -m src.scripts.train_and_backtest_ml --data <path> --epochs 5
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Run with: pip install -e . && python -m src.scripts.train_and_backtest_ml

from loguru import logger  # noqa: E402
from rts_ai_fx.features_unified import compute_features  # noqa: E402
from rts_ai_fx.regime_detector import HMMRegimeDetector  # noqa: E402
from backtest.vectorized_backtester import VectorizedBacktester  # noqa: E402
from ai.rl_agent import PPOAgent, TradingEnvironment  # noqa: E402
from ai.regime_agents import RegimeSpecialistSystem  # noqa: E402

# ─── Constants ────────────────────────────────────────────────────────────────
DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "historical", "EURUSD_1h.csv"
)

REGIME_NAMES = ["ranging", "trending", "volatile", "crisis"]

# Backtest defaults
SPREAD_PIPS = 0.5
COMMISSION_PER_LOT = 7.0
SL_ATR = 2.0
TP_ATR = 4.0
N_REGIMES = 4

# Training defaults
TRAIN_RATIO = 0.7
TRAIN_INTERVAL = 128
N_EPOCHS = 3
BURN_IN = 100

# Monte Carlo defaults
MC_N_PERMUTATIONS = 10_000
MC_ALPHA = 0.05

# Columns that are NOT features — keep out of the state vector
EXCLUDE_COLS = {"timestamp", "open", "high", "low", "close", "volume"}


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: Data Loading & Feature Engineering
# ═══════════════════════════════════════════════════════════════════════════════


def load_and_prepare_data(
    path: str,
) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame, np.ndarray, pd.DataFrame, List[str]]:
    """Load OHLCV CSV, parse timestamps, sort, compute features.

    Returns:
        prices:         close-price array, shape (n,)
        df_raw:         raw OHLCV DataFrame (timestamp sorted)
        df_features:    full feature DataFrame (indicators + OHLCV + timestamp)
        features_np:    numpy feature matrix, shape (n, n_features)
        feature_cols:   list of feature column names (excluding EXCLUDE_COLS)
    """
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df_raw = df.copy()
    prices = df["close"].values.astype(np.float64)

    logger.info(f"Loaded {len(prices)} bars from {path}")
    logger.info(
        f"  Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}"
    )

    df_features = compute_features(df)
    feature_cols = [c for c in df_features.columns if c not in EXCLUDE_COLS]
    features_np = df_features[feature_cols].values.astype(np.float64)

    logger.info(
        f"Features computed: {df_features.shape[1]} columns, "
        f"{len(feature_cols)} feature columns"
    )

    return prices, df_raw, df_features, features_np, feature_cols


def normalize_features_expanding(
    features_np: np.ndarray,
) -> np.ndarray:
    """Standard-scale features using an expanding window to prevent look-ahead.

    For each bar *i*, the mean and std are computed from bars 0..i so that
    future information never leaks into the current state.
    """
    n, nf = features_np.shape
    normed = np.empty_like(features_np, dtype=np.float64)

    for i in range(n):
        window = features_np[: i + 1]
        mu = np.mean(window, axis=0)
        sigma = np.std(window, axis=0)
        sigma = np.where(sigma < 1e-12, 1.0, sigma)
        normed[i] = (features_np[i] - mu) / sigma

    return normed


def prepare_state_vectors(
    prices: np.ndarray,
    features_normed: np.ndarray,
) -> np.ndarray:
    """Build state vectors by prepending the close price to normalized features.

    The TradingEnvironment expects ``state[0]`` to be the current price for
    position PnL calculations, so we prepend the unnormalised close price.
    Shape returned: (n, 1 + n_features).
    """
    prices_2d = prices.reshape(-1, 1)
    states = np.column_stack([prices_2d, features_normed])
    return np.asarray(states, dtype=np.float32)


def load_atr(features_np: np.ndarray, feature_cols: List[str]) -> np.ndarray:
    """Extract ATR-14 from the feature matrix (or return fallback)."""
    try:
        idx = feature_cols.index("atr_14")
        atr = features_np[:, idx].copy()
        atr = np.nan_to_num(atr, nan=0.001, posinf=0.01, neginf=0.001)
        return atr
    except ValueError:
        return np.full(len(features_np), 0.001)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Regime Detection
# ═══════════════════════════════════════════════════════════════════════════════


def detect_regimes(
    prices: np.ndarray,
    df_raw: pd.DataFrame,
    n_regimes: int = N_REGIMES,
) -> Tuple[np.ndarray, np.ndarray, HMMRegimeDetector]:
    """Fit HMM regime detector and return per-bar regime labels / names.

    Returns:
        regime_labels:  int array, shape (n,), values 0..n_regimes-1
        regime_names:   str array,  shape (n,)
        detector:       fitted HMMRegimeDetector
    """
    try:
        detector = HMMRegimeDetector(n_regimes=n_regimes)
        detector.fit(df_raw)

        # Build HMM features (same logic as HMMRegimeDetector._extract_features)
        returns = np.diff(prices) / (prices[:-1] + 1e-10)
        returns = np.clip(returns, -0.1, 0.1)
        vol = pd.Series(prices).rolling(14).std().values
        vol_ratio = np.ones_like(returns)
        valid_vol = vol[1:] > 0
        vol_ratio[valid_vol] = vol[1:][valid_vol] / (
            np.mean(vol[-min(60, len(vol)) :]) + 1e-10
        )
        vol_ratio = np.clip(vol_ratio, 0.1, 10.0)
        hmm_feat = np.column_stack([returns, vol_ratio, np.abs(returns)])
        hmm_feat = np.nan_to_num(hmm_feat, nan=0.0, posinf=5.0, neginf=-5.0)
        hmm_feat = np.clip(hmm_feat, -5, 5)
        hmm_feat = (hmm_feat - hmm_feat.mean(axis=0)) / (hmm_feat.std(axis=0) + 1e-8)

        hidden_states = detector.model.predict(hmm_feat)

        # Map hidden state index → regime name by sorting mean returns
        mean_ret = detector.model.means_[:, 0]
        order = np.argsort(mean_ret)
        label_to_name = {i: REGIME_NAMES[order[i]] for i in range(n_regimes)}

        regime_labels = hidden_states
        regime_names = np.array([label_to_name[s] for s in hidden_states])

        # Align length with prices (n vs n-1)
        regime_labels = np.concatenate([[int(regime_labels[0])], regime_labels])
        regime_names = np.concatenate([[regime_names[0]], regime_names])

        dist = pd.Series(regime_names).value_counts()
        logger.info(f"Regime distribution: {dist.to_dict()}")

        return regime_labels, regime_names, detector

    except Exception as e:
        logger.warning(f"HMM regime detection failed ({e}), using fallback")
        # Volatility-based fallback
        returns = np.abs(np.diff(prices) / (prices[:-1] + 1e-10))
        vol = pd.Series(prices).rolling(20).std().values[1:]
        vol_percentile = np.zeros(len(returns))
        if len(vol) > 0:
            vol_sorted = np.sort(vol)
            for i in range(len(vol)):
                pct = np.searchsorted(vol_sorted, vol[i]) / len(vol_sorted)
                vol_percentile[i] = pct

        fallback_labels = np.zeros(len(returns), dtype=int)
        fallback_labels[vol_percentile > 0.95] = 3
        fallback_labels[(vol_percentile > 0.70) & (vol_percentile <= 0.95)] = 2
        fallback_labels[(vol_percentile > 0.30) & (vol_percentile <= 0.70)] = 1

        regime_labels = np.concatenate([[0], fallback_labels])
        regime_names = np.array([REGIME_NAMES[int(rl)] for rl in regime_labels])

        dummy_detector = HMMRegimeDetector(n_regimes=n_regimes)
        dummy_detector.fit(df_raw)
        return regime_labels, regime_names, dummy_detector


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: Agent Training
# ═══════════════════════════════════════════════════════════════════════════════


def _get_regime_indices(
    regime_names: np.ndarray,
    start: int,
    end: int,
    regime: str,
    burn_in: int = BURN_IN,
) -> np.ndarray:
    """Return sorted indices in [start, end) where regime matches, >= burn_in."""
    indices = (
        np.where(
            (regime_names[start:end] == regime) & (np.arange(start, end) >= burn_in)
        )[0]
        + start
    )
    return indices


def train_regime_agent(
    agent: PPOAgent,
    env: TradingEnvironment,
    states: np.ndarray,
    train_indices: np.ndarray,
    regime: str,
    train_interval: int = TRAIN_INTERVAL,
    n_epochs: int = N_EPOCHS,
) -> Dict:
    """Train a single PPO agent on a chronological subset of bars.

    Walks through *train_indices* in order, calling ``select_action`` and
    ``store_transition`` on each bar, then periodically calling ``agent.train()``.

    Returns a dict of training metrics.
    """
    n_steps = 0
    total_reward = 0.0
    episode_rewards: List[float] = []
    policy_losses: List[float] = []
    value_losses: List[float] = []
    entropies: List[float] = []

    n_bars = len(train_indices)
    logger.info(f"  Training {regime} agent on {n_bars} bars")

    for idx in train_indices:
        state = states[idx]
        env.update_state(state)

        action, sl_raw, tp_raw, size_raw, info = agent.select_action(state)
        next_state, reward, done, _ = env.step(action, sl_raw, tp_raw, size_raw)
        agent.store_transition(reward, done)

        total_reward += reward
        n_steps += 1

        if n_steps % train_interval == 0:
            metrics = agent.train(next_value=0.0, n_epochs=n_epochs)
            if metrics:
                policy_losses.append(metrics.get("policy_loss", 0.0))
                value_losses.append(metrics.get("value_loss", 0.0))
                entropies.append(metrics.get("entropy", 0.0))
                episode_rewards.append(total_reward)
                total_reward = 0.0

    # Final training pass on remaining data
    if agent.states and len(agent.states) >= 2:
        metrics = agent.train(next_value=0.0, n_epochs=n_epochs)
        if metrics:
            policy_losses.append(metrics.get("policy_loss", 0.0))
            value_losses.append(metrics.get("value_loss", 0.0))
            entropies.append(metrics.get("entropy", 0.0))
            episode_rewards.append(total_reward)

    result: Dict = {
        "n_steps": n_steps,
        "n_train_calls": len(policy_losses),
        "avg_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "total_trades": env.total_trades,
        "win_rate": float(env.winning_trades / max(env.total_trades, 1)),
        "avg_policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
        "avg_value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
        "avg_entropy": float(np.mean(entropies)) if entropies else 0.0,
    }
    return result


def train_all_agents(
    system: RegimeSpecialistSystem,
    states: np.ndarray,
    regime_names: np.ndarray,
    train_end: int,
    train_interval: int = TRAIN_INTERVAL,
    n_epochs: int = N_EPOCHS,
) -> Dict[str, Dict]:
    """Train each non-crisis regime agent on its own bars.

    ``train_end`` is the exclusive index that separates train from test data.

    Returns a dict mapping regime names to their training metrics.
    """
    training_metrics: Dict[str, Dict] = {}
    active_regimes = ["trending", "ranging", "volatile"]

    state_dim = states.shape[1]

    for regime in active_regimes:
        agent = system.agents.get(regime)
        if agent is None or not system.should_trade_in_regime(regime):
            logger.info(f"  Skipping {regime} agent (no agent or trading disabled)")
            continue

        indices = _get_regime_indices(regime_names, 0, train_end, regime)
        if len(indices) < 50:
            logger.warning(
                f"  Too few training bars ({len(indices)}) for {regime}, skipping"
            )
            continue

        env = TradingEnvironment(
            state_dim=state_dim,
            spread_pips=SPREAD_PIPS,
            commission_per_lot=COMMISSION_PER_LOT,
        )
        env.reset()

        metrics = train_regime_agent(
            agent, env, states, indices, regime, train_interval, n_epochs
        )
        training_metrics[regime] = metrics

        logger.info(
            f"  {regime}: {metrics['n_steps']} steps, "
            f"{metrics['n_train_calls']} train calls, "
            f"avg_reward={metrics['avg_reward']:.2f}, "
            f"win_rate={metrics['win_rate']:.1%}"
        )

    return training_metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4: ML Signal Function Factory
# ═══════════════════════════════════════════════════════════════════════════════


def make_ml_signal_function(
    regime_system: RegimeSpecialistSystem,
    regime_names: np.ndarray,
    states: np.ndarray,
    burn_in: int = BURN_IN,
) -> Callable:
    """Build a signal function that uses the RegimeSpecialistSystem for decisions.

    The returned callable has the signature ``(prices, features) -> np.ndarray``
    of {-1, 0, 1} signals, as required by ``VectorizedBacktester.run()``.

    Mapping from agent actions:
        HOLD (0)     →  0
        BUY  (1)     →  1
        SELL (2)     → -1
        CLOSE (3)    →  0
        MODIFY (4)   →  0

    The crisis regime is always mapped to 0 (no trades).
    """

    def signal_fn(
        prices_arg: np.ndarray,
        features_arg: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        n = len(regime_names)
        signals = np.zeros(n, dtype=int)

        for i in range(burn_in, n):
            regime = regime_names[i]
            if regime == "crisis":
                signals[i] = 0
                continue

            state = states[i]
            action, sl_raw, tp_raw, size_raw, info = regime_system.select_action(
                state, regime
            )

            if action == 1:  # BUY
                signals[i] = 1
            elif action == 2:  # SELL
                signals[i] = -1
            else:
                signals[i] = 0

        return signals

    return signal_fn


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5: Backtest Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def compute_trade_level_metrics(trade_pnls: np.ndarray) -> Dict:
    """Compute proper trade-level metrics (avoids equity-curve artifacts)."""
    if len(trade_pnls) == 0:
        return {
            "total_return": 0.0,
            "sharpe": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "total_trades": 0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
        }

    total_return = float(np.sum(trade_pnls))
    n = len(trade_pnls)
    wins = trade_pnls[trade_pnls > 0]
    losses = trade_pnls[trade_pnls < 0]
    win_rate = len(wins) / n

    sharpe = (
        float(np.mean(trade_pnls) / max(np.std(trade_pnls), 1e-10) * np.sqrt(252))
        if n > 1
        else 0.0
    )

    gross_win = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_loss = float(abs(np.sum(losses))) if len(losses) > 0 else 1.0
    profit_factor = gross_win / max(gross_loss, 1e-8)

    return {
        "total_return": total_return,
        "sharpe": round(sharpe, 3),
        "win_rate": round(win_rate, 3),
        "profit_factor": round(profit_factor, 3),
        "total_trades": n,
        "avg_win": float(np.mean(wins)) if len(wins) > 0 else 0.0,
        "avg_loss": float(np.mean(losses)) if len(losses) > 0 else 0.0,
    }


def run_single_backtest(
    prices: np.ndarray,
    signal_fn: Callable,
    features_np: np.ndarray,
    atr_np: np.ndarray,
    regime_names_subset: np.ndarray,
    sl_atr: float = SL_ATR,
    tp_atr: float = TP_ATR,
) -> Dict:
    """Run base backtest with the ML signal function.

    Returns ``dict`` with keys ``base`` (BacktestResult), ``regime_breakdown``,
    and ``trade_metrics``.
    """
    bt = VectorizedBacktester(
        spread_pips=SPREAD_PIPS,
        commission_per_lot=COMMISSION_PER_LOT,
        slippage_model="moderate",
    )
    base_result = bt.run(
        prices,
        signal_fn,
        features=features_np,
        atr=atr_np,
        regimes=regime_names_subset,
        sl_atr=sl_atr,
        tp_atr=tp_atr,
    )

    trade_metrics = compute_trade_level_metrics(base_result.trade_pnls)

    # Regime breakdown
    regime_breakdown = {}
    if base_result.trade_regimes:
        unique_r = sorted(set(base_result.trade_regimes))
        for reg in unique_r:
            mask = [t == reg for t in base_result.trade_regimes]
            reg_pnls = base_result.trade_pnls[mask]
            n_trades = len(reg_pnls)
            wins = int(np.sum(reg_pnls > 0))
            total_pnl = float(np.sum(reg_pnls))
            regime_breakdown[reg] = {
                "trades": n_trades,
                "win_rate": wins / n_trades if n_trades > 0 else 0.0,
                "total_return": total_pnl,
            }

    return {
        "base": base_result,
        "regime_breakdown": regime_breakdown,
        "trade_metrics": trade_metrics,
    }


def run_monte_carlo(
    trade_pnls: np.ndarray,
    n_permutations: int = MC_N_PERMUTATIONS,
    alpha: float = MC_ALPHA,
) -> Dict:
    """Run Monte Carlo permutation test on trade PnLs."""
    from validation.monte_carlo import MonteCarloSigTest

    if len(trade_pnls) < 10:
        return {"p_value_sharpe": 1.0, "p_value_return": 1.0, "is_significant": False}

    trades = [{"pnl": float(p)} for p in trade_pnls]
    mc = MonteCarloSigTest(n_permutations=n_permutations, alpha=alpha)
    result = mc.test(trades)

    return {
        "actual_sharpe": result.actual_sharpe,
        "p_value_sharpe": result.p_value_sharpe,
        "p_value_return": result.p_value_return,
        "is_significant": result.is_significant_sharpe,
        "sharpe_percentile": result.sharpe_percentile,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6: Comparison & Report
# ═══════════════════════════════════════════════════════════════════════════════


def print_comparison_report(  # noqa: C901
    untrained: Dict,
    trained: Dict,
    training_metrics: Dict[str, Dict],
    elapsed: float,
) -> None:
    """Print a side-by-side comparison of untrained vs trained agent results."""
    u_metrics = untrained["trade_metrics"]
    t_metrics = trained["trade_metrics"]
    u_base = untrained["base"]
    t_base = trained["base"]

    print()
    print("=" * 78)
    print("    RTS AI FOREX — ML TRAINING & BACKTEST COMPARISON REPORT")
    print("=" * 78)
    print()

    # ── Section 1: Training Summary ──
    print("─" * 78)
    print("  [1] TRAINING SUMMARY")
    print("─" * 78)
    if training_metrics:
        print(
            f"  {'Regime':<16s} {'Steps':>8s} {'TrainCalls':>12s} "
            f"{'AvgReward':>10s} {'WinRate':>10s}"
        )
        print(f"  {'-'*16} {'-'*8} {'-'*12} {'-'*10} {'-'*10}")
        for regime, metrics in sorted(training_metrics.items()):
            print(
                f"  {regime:<16s} {metrics['n_steps']:>8d} "
                f"{metrics['n_train_calls']:>12d} "
                f"{metrics['avg_reward']:>+10.2f} "
                f"{metrics['win_rate']:>9.1%}"
            )
    else:
        print("  (no training performed)")
    print()

    # ── Section 2: Signal Stats ──
    print("─" * 78)
    print("  [2] SIGNAL STATISTICS (test set)")
    print("─" * 78)
    for label, bt_data in [("Untrained", untrained), ("Trained  ", trained)]:
        base = bt_data["base"]
        print(f"  {label}: {base.total_trades} trades")
    print()

    # ── Section 3: Side-by-Side Comparison ──
    print("─" * 78)
    print("  [3] KEY METRICS — UNTRAINED vs TRAINED")
    print("─" * 78)
    print(f"  {'Metric':<28s} {'Untrained':>14s} {'Trained':>14s}  {'Delta':>10s}")
    print(f"  {'-'*28} {'-'*14} {'-'*14} {'-'*10}")

    metrics_to_show = [
        ("Sharpe", "sharpe", "{:.3f}"),
        ("Win Rate", "win_rate", "{:.1%}"),
        ("Profit Factor", "profit_factor", "{:.3f}"),
        ("Net PnL (pips)", "total_return", "{:+.2f}"),
        ("Total Trades", "total_trades", "{:d}"),
        ("Avg Win", "avg_win", "{:.2f}"),
        ("Avg Loss", "avg_loss", "{:.2f}"),
    ]

    for label, key, fmt in metrics_to_show:
        u_val = u_metrics.get(key, 0)
        t_val = t_metrics.get(key, 0)
        if key == "total_trades":
            u_val = (
                len(u_base.trade_pnls)
                if hasattr(u_base, "trade_pnls")
                else u_metrics.get(key, 0)
            )
            t_val = (
                len(t_base.trade_pnls)
                if hasattr(t_base, "trade_pnls")
                else t_metrics.get(key, 0)
            )
        if isinstance(u_val, float) and isinstance(t_val, float):
            delta = t_val - u_val
            delta_str = f"{delta:+.3f}" if isinstance(delta, float) else str(delta)
        else:
            u_val_int = int(u_val) if not isinstance(u_val, int) else u_val
            t_val_int = int(t_val) if not isinstance(t_val, int) else t_val
            delta = t_val_int - u_val_int
            delta_str = f"{delta:+d}"

        u_str = fmt.format(u_val) if isinstance(u_val, (int, float)) else str(u_val)
        t_str = fmt.format(t_val) if isinstance(t_val, (int, float)) else str(t_val)
        print(f"  {label:<28s} {u_str:>14s} {t_str:>14s}  {delta_str:>10s}")

    print()

    # ── Section 4: Regime Breakdown ──
    print("─" * 78)
    print("  [4] REGIME BREAKDOWN (trained agent)")
    print("─" * 78)
    rb = trained.get("regime_breakdown", {})
    if rb:
        print(f"  {'Regime':<16s} {'Trades':>8s} {'WinRate':>10s} {'Return':>12s}")
        print(f"  {'-'*16} {'-'*8} {'-'*10} {'-'*12}")
        for reg in sorted(rb.keys()):
            info = rb[reg]
            print(
                f"  {reg:<16s} {info['trades']:>8d} {info['win_rate']:>9.1%} "
                f"{info['total_return']:>+11.2f}"
            )
    else:
        print("  (no regime breakdown — no trades)")
    print()

    # ── Section 5: Monte Carlo ──
    mc = trained.get("monte_carlo", {})
    if mc and mc.get("p_value_sharpe", 1.0) < 1.0:
        print("─" * 78)
        print("  [5] MONTE CARLO SIGNIFICANCE (trained agent)")
        print("─" * 78)
        sig = "SIGNIFICANT" if mc.get("is_significant") else "NOT significant"
        print(f"  Sharpe p-value:  {mc['p_value_sharpe']:.4f}  [{sig}]")
        print(f"  Return p-value:  {mc['p_value_return']:.4f}")
        print(f"  Sharpe pctl:     {mc['sharpe_percentile']:.1%}")
        print()

    # ── Section 6: Summary ──
    print("─" * 78)
    if t_metrics["sharpe"] > u_metrics["sharpe"]:
        print("  ✅ Training improved Sharpe ratio")
    else:
        print("  ⚠  Training did NOT improve Sharpe ratio")
    if t_metrics["profit_factor"] > u_metrics["profit_factor"]:
        print("  ✅ Training improved Profit Factor")
    else:
        print("  ⚠  Training did NOT improve Profit Factor")
    if t_metrics["total_return"] > u_metrics["total_return"]:
        print("  ✅ Training improved Net PnL")
    else:
        print("  ⚠  Training did NOT improve Net PnL")
    print()
    print(f"  Total runtime: {elapsed:.1f}s")
    print()
    print("=" * 78)
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Section 7: Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════


def run_pipeline(
    data_path: str,
    n_epochs: int = N_EPOCHS,
    train_interval: int = TRAIN_INTERVAL,
    train_ratio: float = TRAIN_RATIO,
) -> Dict:
    """Execute the full training-and-backtest pipeline.

    Returns a dict with keys:
      - training_metrics
      - untrained_results
      - trained_results
    """
    # ── 1. Data Loading & Features ──
    logger.info("─" * 60)
    logger.info("STEP 1/6: Loading data & computing features")
    prices, df_raw, df_features, features_np, feature_cols = load_and_prepare_data(
        data_path
    )

    atr_np = load_atr(features_np, feature_cols)

    # ── 2. Regime Detection ──
    logger.info("─" * 60)
    logger.info("STEP 2/6: Detecting market regimes (HMM)")
    regime_labels, regime_names, detector = detect_regimes(prices, df_raw, N_REGIMES)

    # ── 3. Normalize features & build state vectors ──
    logger.info("─" * 60)
    logger.info("STEP 3/6: Building state vectors (expanding-window normalization)")

    features_normed = normalize_features_expanding(features_np)
    states = prepare_state_vectors(prices, features_normed)
    state_dim = states.shape[1]
    n_total = len(prices)
    train_end = int(n_total * train_ratio)

    # Align regime_names length with states
    if len(regime_names) > n_total:
        regime_names = regime_names[:n_total]
    elif len(regime_names) < n_total:
        regime_names = np.concatenate(
            [regime_names, np.full(n_total - len(regime_names), "ranging")]
        )

    logger.info(
        f"  State dimension: {state_dim} (1 price + {features_np.shape[1]} features)"
    )
    logger.info(f"  Training bars: 0:{train_end}  |  Test bars: {train_end}:{n_total}")

    # ── 4. Create untrained (random) system as baseline ──
    logger.info("─" * 60)
    logger.info("STEP 4/6: Creating systems & running untrained baseline")

    untrained_system = RegimeSpecialistSystem(state_dim=state_dim, n_actions=5)

    # Subset to test range
    test_start = train_end
    test_end = n_total
    prices_test = prices[test_start:test_end]
    states_test = states[test_start:test_end]
    atr_test = atr_np[test_start:test_end]
    regime_names_test = regime_names[test_start:test_end]

    # Align lengths
    min_len_test = min(len(prices_test), len(states_test), len(regime_names_test))
    prices_test = prices_test[:min_len_test]
    states_test = states_test[:min_len_test]
    regime_names_test = regime_names_test[:min_len_test]
    atr_test = atr_test[:min_len_test]

    # Untrained signal function & backtest
    untrained_sig_fn = make_ml_signal_function(
        untrained_system,
        regime_names_test,
        states_test,
        burn_in=max(0, BURN_IN - test_start),
    )
    untrained_results = run_single_backtest(
        prices_test,
        untrained_sig_fn,
        states_test,
        atr_test,
        regime_names_test,
    )

    u_tm = untrained_results["trade_metrics"]
    logger.info(
        f"  Untrained: Sharpe={u_tm['sharpe']:.3f} | "
        f"Return={u_tm['total_return']:+.2f} | "
        f"Trades={u_tm['total_trades']} | "
        f"WinRate={u_tm['win_rate']:.1%}"
    )

    # ── 5. Train regime agents ──
    logger.info("─" * 60)
    logger.info("STEP 5/6: Training regime-specific PPO agents")

    # Create a new system for training (agents start random, then get trained)
    trained_system = RegimeSpecialistSystem(state_dim=state_dim, n_actions=5)

    training_metrics = train_all_agents(
        trained_system,
        states,
        regime_names,
        train_end,
        train_interval=train_interval,
        n_epochs=n_epochs,
    )

    # ── 6. Backtest with trained agents ──
    logger.info("─" * 60)
    logger.info("STEP 6/6: Backtesting with trained agents")

    trained_sig_fn = make_ml_signal_function(
        trained_system,
        regime_names_test,
        states_test,
        burn_in=max(0, BURN_IN - test_start),
    )
    trained_results = run_single_backtest(
        prices_test,
        trained_sig_fn,
        states_test,
        atr_test,
        regime_names_test,
    )

    t_tm = trained_results["trade_metrics"]
    logger.info(
        f"  Trained:   Sharpe={t_tm['sharpe']:.3f} | "
        f"Return={t_tm['total_return']:+.2f} | "
        f"Trades={t_tm['total_trades']} | "
        f"WinRate={t_tm['win_rate']:.1%}"
    )

    # Monte Carlo on trained result
    mc_result = run_monte_carlo(trained_results["base"].trade_pnls)
    trained_results["monte_carlo"] = mc_result

    return {
        "training_metrics": training_metrics,
        "untrained_results": untrained_results,
        "trained_results": trained_results,
        "n_total": n_total,
        "train_end": train_end,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RTS AI Forex — ML Training & Backtest"
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA_PATH,
        help=f"Path to OHLCV CSV (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=N_EPOCHS,
        help=f"PPO training epochs per train call (default: {N_EPOCHS})",
    )
    parser.add_argument(
        "--train-interval",
        type=int,
        default=TRAIN_INTERVAL,
        help=f"Steps between PPO updates (default: {TRAIN_INTERVAL})",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=TRAIN_RATIO,
        help=f"Fraction of data for training (default: {TRAIN_RATIO})",
    )
    args = parser.parse_args()

    t_start = time.time()

    results = run_pipeline(
        data_path=args.data,
        n_epochs=args.epochs,
        train_interval=args.train_interval,
        train_ratio=args.train_ratio,
    )

    elapsed = time.time() - t_start

    print_comparison_report(
        untrained=results["untrained_results"],
        trained=results["trained_results"],
        training_metrics=results["training_metrics"],
        elapsed=elapsed,
    )


if __name__ == "__main__":
    main()
