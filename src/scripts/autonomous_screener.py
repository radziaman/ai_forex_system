"""
Autonomous Instrument Screener — continuously scans markets to find tradeable edges.

The system uses this module to autonomously:
  1. Download fresh data for all available instruments
  2. Test multiple strategies on each instrument
  3. Score and rank by statistical edge
  4. Return ONLY instruments that pass the threshold

Usage:
    python -m src.scripts.autonomous_screener              # Run full scan
    python -m src.scripts.autonomous_screener --update      # Download fresh data
    python -m src.scripts.autonomous_screener --threshold 0.5  # Custom Sharpe min
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

_src = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _src not in sys.path:
    sys.path.insert(0, _src)

from loguru import logger  # noqa: E402

# ─── Instrument Universe ──────────────────────────────────────────────────────
# The system maintains a master list. The screener tests ALL of them and returns
# only those that pass the statistical edge threshold.

INSTRUMENT_UNIVERSE = {
    # Metals
    "GC=F": {"name": "Gold Futures", "asset_class": "metal", "pip": 0.1},
    "SI=F": {"name": "Silver Futures", "asset_class": "metal", "pip": 0.001},
    "HG=F": {"name": "Copper Futures", "asset_class": "metal", "pip": 0.0005},
    "PA=F": {"name": "Palladium Futures", "asset_class": "metal", "pip": 1.0},
    "PL=F": {"name": "Platinum Futures", "asset_class": "metal", "pip": 0.1},
    # Energy
    "CL=F": {"name": "Crude Oil Futures", "asset_class": "energy", "pip": 0.01},
    "HO=F": {"name": "Heating Oil Futures", "asset_class": "energy", "pip": 0.0001},
    "RB=F": {"name": "Gasoline Futures", "asset_class": "energy", "pip": 0.0001},
    "NG=F": {"name": "Natural Gas Futures", "asset_class": "energy", "pip": 0.001},
    # Bonds
    "ZB=F": {"name": "30yr T-Bond Futures", "asset_class": "bond", "pip": 0.01},
    "ZN=F": {"name": "10yr T-Note Futures", "asset_class": "bond", "pip": 0.005},
    "ZF=F": {"name": "5yr T-Note Futures", "asset_class": "bond", "pip": 0.0025},
    "ZT=F": {"name": "2yr T-Note Futures", "asset_class": "bond", "pip": 0.00125},
    # Equity Indices
    "ES=F": {"name": "S&P 500 Futures", "asset_class": "equity", "pip": 0.1},
    "NQ=F": {"name": "Nasdaq Futures", "asset_class": "equity", "pip": 0.25},
    "YM=F": {"name": "Dow Futures", "asset_class": "equity", "pip": 1.0},
    # FX
    "DX-Y.NYB": {"name": "US Dollar Index", "asset_class": "fx", "pip": 0.01},
    "EURUSD=X": {"name": "EUR/USD", "asset_class": "fx", "pip": 0.0001},
    "GBPUSD=X": {"name": "GBP/USD", "asset_class": "fx", "pip": 0.0001},
    # ETFs
    "GLD": {"name": "Gold ETF", "asset_class": "metal", "pip": 0.01},
    "SLV": {"name": "Silver ETF", "asset_class": "metal", "pip": 0.01},
    "USO": {"name": "Oil ETF", "asset_class": "energy", "pip": 0.01},
    "TLT": {"name": "20yr+ Treasury ETF", "asset_class": "bond", "pip": 0.01},
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "historical")
SCREEN_RESULTS_FILE = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "screen_results.json"
)


@dataclass
class InstrumentScore:
    """Score for a single instrument after screening."""

    ticker: str
    name: str
    asset_class: str

    # Momentum strategy scores (5-day, 10-day, 20-day)
    mom5_sharpe: float = 0.0
    mom5_pf: float = 0.0
    mom5_win_rate: float = 0.0
    mom5_trades: int = 0
    mom5_pnl: float = 0.0

    mom10_sharpe: float = 0.0
    mom10_pf: float = 0.0
    mom20_sharpe: float = 0.0
    mom20_pf: float = 0.0

    # Market characteristics
    annual_return: float = 0.0
    annual_volatility: float = 0.0
    lag1_autocorr: float = 0.0
    n_bars: int = 0
    current_price: float = 0.0

    # Composite score
    composite_score: float = 0.0
    edge_detected: bool = False
    recommendation: str = "HOLD"

    # Strategy parameters (set by optimizer)
    optimal_lookback: int = 5
    optimal_sl_atr: float = 3.0
    optimal_tp_atr: float = 6.0


def load_prices(ticker: str) -> Optional[np.ndarray]:  # noqa: C901
    """Load or download daily price data for a ticker."""
    safe = ticker.replace("=", "_").replace("-", "_")
    fpath = os.path.join(DATA_DIR, f"{safe}_daily.csv")

    if os.path.exists(fpath):
        try:
            df = pd.read_csv(fpath, index_col=0, parse_dates=True)
            for col in ["Close", "close", "Adj Close"]:
                if col in df.columns:
                    val = df[col].values
                    if val.dtype == "O":
                        val = pd.to_numeric(val, errors="coerce")
                    return np.nan_to_num(val, nan=0.0).astype(float)
            # Fallback
            for col in df.columns:
                if df[col].dtype in (np.float64, np.float32, np.int64):
                    return df[col].values.astype(float)
        except Exception:
            pass

    # Download
    try:
        data = yf.download(
            ticker, period="10y", interval="1d", progress=False, auto_adjust=True
        )
        if data is None or len(data) < 100:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0] for c in data.columns]
        data.to_csv(fpath)
        for col in ["Close", "close", "Adj Close"]:
            if col in data.columns:
                return data[col].values.astype(float)
        return data.iloc[:, 0].values.astype(float)
    except Exception:
        return None


def test_momentum_strategy(  # noqa: C901
    prices: np.ndarray,
    lookback: int = 5,
    sl_atr: float = 3.0,
    tp_atr: float = 6.0,
    spread_cost: float = 0.0,
) -> Dict:
    """Test a momentum strategy and return performance metrics.

    Args:
        prices: OHLC price array
        lookback: Days for momentum calculation
        sl_atr: Stop loss in ATR multiples (0 = no stop)
        tp_atr: Take profit in ATR multiples (0 = no limit)
        spread_cost: Estimated spread cost in price units

    Returns:
        Dict with sharpe, pf, win_rate, trades, pnl
    """
    p = prices.flatten()
    if len(p) < 100:
        return {"sharpe": 0, "pf": 0, "wr": 0, "trades": 0, "pnl": 0}

    # Compute ATR
    if sl_atr > 0 or tp_atr > 0:
        tr = np.zeros(len(p))
        for i in range(1, len(p)):
            tr[i] = abs(p[i] - p[i - 1])
        atr = np.full(len(p), np.mean(tr[1:14]))
    else:
        atr = np.zeros(len(p))

    signals = np.zeros(len(p), dtype=int)
    for i in range(lookback, len(p)):
        signals[i] = 1 if p[i] > p[i - lookback] else -1

    tp = []
    pos = 0
    ep = 0
    ei = 0

    for i in range(lookback, len(p)):
        sig = signals[i]

        if pos == 0 and sig != 0:
            pos = sig
            ep = p[i]
            ei = i
            continue

        if pos != 0:
            # Check SL/TP
            sl_hit = tp_hit = False
            if sl_atr > 0 and i > ei:
                sl = ep - pos * atr[i] * sl_atr
                sl_hit = (pos == 1 and p[i] <= sl) or (pos == -1 and p[i] >= sl)
            if tp_atr > 0 and i > ei:
                tpl = ep + pos * atr[i] * tp_atr
                tp_hit = (pos == 1 and p[i] >= tpl) or (pos == -1 and p[i] <= tpl)

            exit_sig = sig != 0 and sig != pos

            if sl_hit or tp_hit or exit_sig or i == len(p) - 1:
                pnl = (p[i] - ep) * pos - spread_cost * np.sign(pos)
                tp.append(pnl)
                pos = 0

    tpa = np.array(tp)
    if len(tpa) < 3:
        return {"sharpe": 0, "pf": 0, "wr": 0, "trades": 0, "pnl": 0}

    w = tpa[tpa > 0]
    losses = tpa[tpa < 0]
    sharpe = np.mean(tpa) / np.std(tpa) * np.sqrt(252) if np.std(tpa) > 1e-10 else 0
    pf = np.sum(w) / abs(np.sum(losses)) if np.sum(losses) != 0 else float("inf")
    returns = np.diff(p) / p[:-1]

    return {
        "sharpe": float(sharpe),
        "pf": float(pf),
        "wr": float(np.mean(tpa > 0)),
        "trades": int(len(tpa)),
        "pnl": float(np.sum(tpa)),
        "ann_ret": float(np.mean(returns) * 252 * 100),
        "ann_vol": float(np.std(returns) * np.sqrt(252) * 100),
    }


def screen_instrument(ticker: str, info: Dict) -> Optional[InstrumentScore]:
    """Run full screening on a single instrument."""
    prices = load_prices(ticker)
    if prices is None or len(prices) < 100:
        return None

    p = prices.flatten()
    n = len(p)

    # Market characteristics
    returns = np.diff(p) / (p[:-1] + 1e-10)
    ann_ret = float(np.mean(returns) * 252 * 100)
    ann_vol = float(np.std(returns) * np.sqrt(252) * 100)
    lag1 = (
        float(np.corrcoef(returns[1:], returns[:-1])[0, 1]) if len(returns) > 2 else 0.0
    )

    # Spread cost
    spread_cost = info["pip"] * 0.5 if info.get("pip") else 0.0

    # Test multiple lookback periods
    mom5 = test_momentum_strategy(p, lookback=5, spread_cost=spread_cost)
    mom10 = test_momentum_strategy(p, lookback=10, spread_cost=spread_cost)
    mom20 = test_momentum_strategy(p, lookback=20, spread_cost=spread_cost)

    # Composite score: weighted average of Sharpe ratios
    scores = [
        (mom5["sharpe"], 0.5),
        (mom10["sharpe"], 0.3),
        (mom20["sharpe"], 0.2),
    ]
    composite = sum(s * w for s, w in scores)

    # Find best lookback
    lookbacks = {5: mom5["sharpe"], 10: mom10["sharpe"], 20: mom20["sharpe"]}
    best_lookback = max(lookbacks, key=lookbacks.get)

    # Edge detection
    edge_detected = composite > 0.3 and mom5["pf"] > 1.05
    if composite > 0.5 and mom5["pf"] > 1.15:
        recommendation = "STRONG_BUY"
    elif composite > 0.3 and mom5["pf"] > 1.05:
        recommendation = "BUY"
    elif composite > 0:
        recommendation = "WATCH"
    else:
        recommendation = "HOLD"

    return InstrumentScore(
        ticker=ticker,
        name=info["name"],
        asset_class=info["asset_class"],
        mom5_sharpe=mom5["sharpe"],
        mom5_pf=mom5["pf"],
        mom5_win_rate=mom5["wr"],
        mom5_trades=mom5["trades"],
        mom5_pnl=mom5["pnl"],
        mom10_sharpe=mom10["sharpe"],
        mom10_pf=mom10["pf"],
        mom20_sharpe=mom20["sharpe"],
        mom20_pf=mom20["pf"],
        annual_return=ann_ret,
        annual_volatility=ann_vol,
        lag1_autocorr=lag1,
        n_bars=n,
        current_price=float(p[-1]),
        composite_score=composite,
        edge_detected=edge_detected,
        recommendation=recommendation,
        optimal_lookback=best_lookback,
    )


def screen_all() -> List[InstrumentScore]:
    """Screen ALL instruments and return scores sorted by edge."""
    results = []
    for ticker, info in INSTRUMENT_UNIVERSE.items():
        logger.info(f"Screening {ticker} ({info['name']})...")
        score = screen_instrument(ticker, info)
        if score:
            results.append(score)
            flag = (
                "✅"
                if score.edge_detected
                else "⚠️" if score.composite_score > 0 else "❌"
            )
            logger.info(
                f"  {flag} {ticker}: Sharpe={score.mom5_sharpe:.3f} PF={score.mom5_pf:.2f} "  # noqa: E501
                f"Score={score.composite_score:.3f} -> {score.recommendation}"
            )
        else:
            logger.warning(f"  ❌ {ticker}: Insufficient data")

    results.sort(key=lambda r: r.composite_score, reverse=True)
    return results


def print_report(results: List[InstrumentScore]):
    """Print formatted screening report."""
    print()
    print("=" * 110)
    print("  AUTONOMOUS INSTRUMENT SCREENER — Live Market Scan")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 110)
    print()

    # Header
    print(
        f"  {'Tkr':<8s} {'Name':<22s} {'Class':<8s} {'Mom5S':>7s} {'Mom5PF':>7s} {'WR':>5s} "  # noqa: E501
        f"{'Trds':>5s} {'Score':>7s} {'AnnRet':>7s} {'AnnVol':>7s} {'Lag1':>7s} {'Rec':<12s}"  # noqa: E501
    )
    print(
        f"  {'-'*8} {'-'*22} {'-'*8} {'-'*7} {'-'*7} {'-'*5} "
        f"{'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*12}"
    )

    for r in results:
        flag = "✅" if r.edge_detected else "  "
        print(
            f"  {flag} {r.ticker:<6s} {r.name:<22s} {r.asset_class:<8s} "
            f"{r.mom5_sharpe:>+7.3f} {r.mom5_pf:>7.2f} {r.mom5_win_rate:>4.0%} "
            f"{r.mom5_trades:>5d} {r.composite_score:>+7.3f} {r.annual_return:>+6.2f}% "
            f"{r.annual_volatility:>6.2f}% {r.lag1_autocorr:>+7.4f} {r.recommendation:<12s}"  # noqa: E501
        )

    print()

    # Summary by asset class
    for ac in ["energy", "metal", "bond", "equity", "fx"]:
        ac_results = [r for r in results if r.asset_class == ac]
        if ac_results:
            best = max(ac_results, key=lambda r: r.composite_score)
            print(
                f"  {ac.upper():<10s}: Best={best.ticker} ({best.name}) "
                f"Score={best.composite_score:.2f} -> {best.recommendation}"
            )

    print()

    # Tradeable instruments
    tradeable = [r for r in results if r.edge_detected]
    if tradeable:
        print(f"  ✅ TRADEABLE INSTRUMENTS ({len(tradeable)}):")
        for r in tradeable:
            print(
                f"     {r.ticker:<8s} {r.name:<22s} "
                f"Strategy: {r.optimal_lookback}d momentum | "
                f"Sharpe={r.mom5_sharpe:.2f} PF={r.mom5_pf:.2f}"
            )
    else:
        print("  ❌ NO TRADEABLE INSTRUMENTS currently pass the edge threshold")

    # Saving results
    results_dict = [asdict(r) for r in results]
    with open(SCREEN_RESULTS_FILE, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "n_instruments": len(results),
                "tradeable": len(tradeable),
                "results": results_dict,
            },
            f,
            indent=2,
            default=str,
        )
    logger.info(f"Results saved to {SCREEN_RESULTS_FILE}")

    return tradeable


# ─── API for the trading system ───────────────────────────────────────────────


def get_tradeable_instruments(
    min_sharpe: float = 0.3, min_pf: float = 1.05
) -> List[Dict]:
    """API callable by the trading system to get currently tradeable instruments.

    The trading system calls this to determine which instruments to trade.
    Returns only instruments that pass the edge threshold.
    """
    if os.path.exists(SCREEN_RESULTS_FILE):
        try:
            with open(SCREEN_RESULTS_FILE) as f:
                data = json.load(f)
            results = data.get("results", [])
            tradeable = [
                r
                for r in results
                if r.get("mom5_sharpe", 0) >= min_sharpe
                and r.get("mom5_pf", 0) >= min_pf
            ]
            return tradeable
        except Exception:
            pass

    # No cached results — run full scan
    results = screen_all()
    tradeable = [asdict(r) for r in results if r.edge_detected]
    return tradeable


def update_data():
    """Download fresh data for all instruments."""
    logger.info("Updating data for all instruments...")
    count = 0
    for ticker, info in INSTRUMENT_UNIVERSE.items():
        prices = load_prices(ticker)
        if prices is not None:
            count += 1
    logger.info(f"Data updated for {count}/{len(INSTRUMENT_UNIVERSE)} instruments")
    return count


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Autonomous Instrument Screener")
    parser.add_argument("--update", action="store_true", help="Download fresh data")
    parser.add_argument("--threshold", type=float, default=0.3, help="Minimum Sharpe")
    parser.add_argument("--json", action="store_true", help="Output JSON only")
    args = parser.parse_args()

    if args.update:
        update_data()
        return

    results = screen_all()
    print_report(results)

    # Summary
    tradeable = [r for r in results if r.edge_detected]
    if not tradeable and not args.json:
        print(
            "\n  📡 System is in WATCH mode — scanning continuously for edge to appear"
        )
        print(
            f"  Currently monitoring {len(results)} instruments across "
            f"{len(set(r.asset_class for r in results))} asset classes"
        )


if __name__ == "__main__":
    main()
