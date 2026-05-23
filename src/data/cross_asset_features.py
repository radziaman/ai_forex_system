"""Cross-asset lead-lag features for RTS AI Forex Trading System.

Implements:
- Synthetic DXY correlation
- EURUSD/GBPUSD lagged cross-correlation
- Gold/Oil ratio deviation
- Commodity-FX lead (AUDUSD vs XAUUSD)
- US500 risk-on momentum proxy
"""

from collections import defaultdict, deque
from typing import Dict

import numpy as np


class CrossAssetEngine:
    """Stateful cross-asset feature calculator with rolling price buffers."""

    def __init__(self, maxlen: int = 1000):
        self.maxlen = maxlen
        self._price_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=maxlen)
        )

    def compute_lead_lag_features(
        self, symbol: str, prices_dict: Dict[str, float]
    ) -> Dict[str, float]:
        """Return cross-asset lead-lag features.

        Args:
            symbol: Primary symbol (e.g. ``EURUSD``).
            prices_dict: Mapping ``symbol -> latest_price`` for all 11 symbols.

        Returns:
            Dict of float-valued cross-asset features.
        """
        # Buffer incoming prices
        for sym, price in prices_dict.items():
            if price > 0:
                self._price_history[sym].append(price)

        result = {
            "dxy_impact": 0.0,
            "eurusd_gbpusd_lag30": 0.0,
            "gold_oil_ratio": 0.0,
            "commodity_fx_lead": 0.0,
            "us500_risk_on": 0.0,
        }

        # -- DXY impact: correlation with synthetic DXY --------------------
        dxy = self._compute_synthetic_dxy()
        sym_hist = list(self._price_history.get(symbol, []))
        if len(dxy) >= 30 and len(sym_hist) >= 30:
            dxy_arr = np.array(dxy[-30:])
            sym_arr = np.array(sym_hist[-30:])
            dxy_ret = np.diff(dxy_arr) / dxy_arr[:-1]
            sym_ret = np.diff(sym_arr) / sym_arr[:-1]
            if len(dxy_ret) > 1 and np.std(dxy_ret) > 1e-12 and np.std(sym_ret) > 1e-12:
                result["dxy_impact"] = float(np.corrcoef(dxy_ret, sym_ret)[0, 1])

        # -- EURUSD / GBPUSD 30-period cross-correlation -------------------
        eur_hist = list(self._price_history.get("EURUSD", []))
        gbp_hist = list(self._price_history.get("GBPUSD", []))
        if len(eur_hist) >= 30 and len(gbp_hist) >= 30:
            eur_arr = np.array(eur_hist[-30:])
            gbp_arr = np.array(gbp_hist[-30:])
            eur_ret = np.diff(eur_arr) / eur_arr[:-1]
            gbp_ret = np.diff(gbp_arr) / gbp_arr[:-1]
            if len(eur_ret) > 1 and np.std(eur_ret) > 1e-12 and np.std(gbp_ret) > 1e-12:
                result["eurusd_gbpusd_lag30"] = float(
                    np.corrcoef(eur_ret, gbp_ret)[0, 1]
                )

        # -- Gold / Oil ratio deviation from moving average ----------------
        gold_hist = list(self._price_history.get("XAUUSD", []))
        oil_hist = list(self._price_history.get("XTIUSD", []))
        if len(gold_hist) >= 10 and len(oil_hist) >= 10:
            gold = gold_hist[-1]
            oil = oil_hist[-1]
            if oil > 0:
                ratio = gold / oil
                if len(gold_hist) >= 200 and len(oil_hist) >= 200:
                    gold_ma = np.mean(gold_hist[-200:])
                    oil_ma = np.mean(oil_hist[-200:])
                else:
                    gold_ma = np.mean(gold_hist)
                    oil_ma = np.mean(oil_hist)
                ma_ratio = gold_ma / oil_ma if oil_ma > 0 else ratio
                result["gold_oil_ratio"] = (
                    (ratio - ma_ratio) / ma_ratio if ma_ratio > 0 else 0.0
                )

        # -- Commodity FX lead: AUDUSD correlation with lagged XAUUSD -----
        aud_hist = list(self._price_history.get("AUDUSD", []))
        gold_hist2 = list(self._price_history.get("XAUUSD", []))
        if len(aud_hist) >= 30 and len(gold_hist2) >= 31:
            aud_arr = np.array(aud_hist[-30:])
            gold_arr = np.array(gold_hist2[-31:-1])  # lagged by 1 step
            aud_ret = np.diff(aud_arr) / aud_arr[:-1]
            gold_ret = np.diff(gold_arr) / gold_arr[:-1]
            if (
                len(aud_ret) > 1
                and np.std(aud_ret) > 1e-12
                and np.std(gold_ret) > 1e-12
            ):
                result["commodity_fx_lead"] = float(
                    np.corrcoef(aud_ret, gold_ret)[0, 1]
                )

        # -- US500 risk-on momentum ----------------------------------------
        us500_hist = list(self._price_history.get("US500", []))
        if len(us500_hist) >= 20:
            us500_arr = np.array(us500_hist[-20:])
            result["us500_risk_on"] = float(
                (us500_arr[-1] - us500_arr[0]) / us500_arr[0]
                if us500_arr[0] > 0
                else 0.0
            )

        return result

    def _compute_synthetic_dxy(self) -> list:
        """Compute a simplified synthetic DXY from available USD pairs.

        Weights are approximate DXY basket weights adapted to the
        11-symbol universe.
        """
        eur = self._price_history.get("EURUSD")
        gbp = self._price_history.get("GBPUSD")
        jpy = self._price_history.get("USDJPY")
        cad = self._price_history.get("USDCAD")
        chf = self._price_history.get("USDCHF")

        hist_lens = [len(h) for h in [eur, gbp, jpy, cad, chf] if h is not None]
        if not hist_lens:
            return []
        min_len = min(hist_lens)

        dxy_vals = []
        for i in range(-min_len, 0):
            vals = []
            if eur is not None and len(eur) >= min_len:
                vals.append(0.50 * (1.0 / eur[i]))
            if gbp is not None and len(gbp) >= min_len:
                vals.append(0.12 * (1.0 / gbp[i]))
            if jpy is not None and len(jpy) >= min_len:
                vals.append(0.15 * (jpy[i] / 100.0))
            if cad is not None and len(cad) >= min_len:
                vals.append(0.13 * cad[i])
            if chf is not None and len(chf) >= min_len:
                vals.append(0.10 * chf[i])
            if vals:
                dxy_vals.append(sum(vals))
        return dxy_vals
