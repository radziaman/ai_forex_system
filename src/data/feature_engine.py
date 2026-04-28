import numpy as np
try:
    import talib
except ImportError:
    talib = None
from loguru import logger
from typing import Dict, List, Optional, Tuple

class FeatureEngine:
    """222+ features across 5 timeframes (TICQ AI research)."""
    TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h"]
    TF_MULT = {"1m":1, "5m":5, "15m":15, "1h":60, "4h":240}

    def __init__(self):
        self.feature_names: List[str] = []
        self._build_names()
        logger.info(f"FeatureEngine | {len(self.feature_names)} features")

    def _build_names(self):
        names = []
        for tf in self.TIMEFRAMES:
            p = f"{tf}_"
            for c in ["open","high","low","close"]:
                names.append(f"{p}{c}")
            for r in ["r1","r5","r20"]:
                names.append(f"{p}ret_{r}")
            names += [f"{p}vol", f"{p}vol_rel", f"{p}atr", f"{p}atr_rel"]
            for per in [14,21,50]:
                names.append(f"{p}rsi_{per}")
            names += [f"{p}macd", f"{p}macd_sig", f"{p}macd_hist"]
            names += [f"{p}bb_up", f"{p}bb_mid", f"{p}bb_lo", f"{p}bb_w", f"{p}bb_pos"]
            names += [f"{p}stoch_k", f"{p}stoch_d", f"{p}adx", f"{p}di_p", f"{p}di_m"]
            names += [f"{p}cci", f"{p}mfi", f"{p}obv", f"{p}obv_rel"]
            for s,f in [(10,20),(20,50),(50,100),(100,200)]:
                names.append(f"{p}sma_{s}_{f}_diff")
            names += [f"{p}vol20", f"{p}vol50", f"{p}hurst"]
            names += [f"{p}z_p20", f"{p}z_p50", f"{p}z_v20"]
            names += [f"{p}ob_bull", f"{p}ob_bear", f"{p}fvg_bull", f"{p}fvg_bear"]
            names += [f"{p}bos_bull", f"{p}bos_bear", f"{p}choch_bull", f"{p}choch_bear"]
            names += [f"{p}cvd", f"{p}cvd_sl", f"{p}imbalance", f"{p}large_z"]
        names += ["mtf_trend","mtf_vol_avg","mtf_rsi_avg","mtf_vol_avg2"]
        names += ["acc_bal","acc_eq","acc_margin","acc_pos_cnt","acc_pnl_norm"]
        self.feature_names = names

    def compute_features(self, ohlcv: Dict, order_flow: Optional[Dict]=None,
                          acc: Optional[Dict]=None, positions: Optional[List]=None) -> np.ndarray:
        feats = []
        for tf in self.TIMEFRAMES:
            df = ohlcv.get(tf, None)
            if df is None or len(df) < 50:
                cnt = len([n for n in self.feature_names if n.startswith(f"{tf}_")])
                feats.extend([0.0] * cnt)
                continue
            c = df['close'].values
            h = df['high'].values
            l = df['low'].values
            o = df['open'].values
            v = df['volume'].values if 'volume' in df.columns else np.zeros(len(c))
            lc = c[-1]
            for arr in [o,h,l,c]:
                feats.append((arr[-1]/lc - 1.0) if lc > 0 else 0.0)
            for n in [1,5,20]:
                feats.append((c[-1]/c[-n]-1.0) if len(c)>n else 0.0)
            feats.append(np.log1p(v[-1]) if len(v)>0 else 0.0)
            avg = np.mean(v[-20:]) if len(v)>=20 else (v[-1] if len(v)>0 else 1.0)
            feats.append((v[-1]/avg - 1.0) if avg > 0 else 0.0)
            atr = self._atr(h,l,c,14)
            feats.append(atr/lc if lc>0 else 0.0)
            feats.append(0.0)
            for p in [14,21,50]:
                feats.append((self._rsi(c,p) or 50.0)/100.0)
            m,s,h2 = self._macd(c)
            for val in [m,s,h2]:
                feats.append(val/lc if lc>0 else 0.0)
            bu,bm,bl = self._bb(c)
            for val in [bu,bm,bl]:
                feats.append((val/lc - 1.0) if lc>0 else 0.0)
            bw = (bu-bl)/bm if bm>0 else 0.0
            feats.append(bw)
            feats.append((lc-bl)/(bu-bl) if (bu-bl)>0 else 0.5)
            k,d = self._stoch(h,l,c)
            feats += [k/100.0, d/100.0]
            a,dp,dm = self._adx(h,l,c)
            feats += [a/100.0, dp/100.0, dm/100.0]
            feats.append(np.clip((self._cci(h,l,c) or 0.0)/200.0, -1, 1))
            feats.append((self._mfi(h,l,c,v) or 50.0)/100.0)
            obv = self._obv(c,v)
            feats += [obv[-1]/1e9 if len(obv)>0 else 0.0]
            feats.append(np.clip(obv[-1]/max(abs(np.max(obv[-20:])) if len(obv)>=20 else 1.0, 1e-8), -1, 1))
            for s,f2 in [(10,20),(20,50),(50,100),(100,200)]:
                sf = np.mean(c[-s:]) if len(c)>=s else c[-1]
                sl2 = np.mean(c[-f2:]) if len(c)>=f2 else c[-1]
                feats.append((sf-sl2)/lc if lc>0 else 0.0)
            for n in [20,50]:
                vol = np.std(c[-n:]/c[-n]-1.0) if len(c)>=n else 0.0
                feats.append(vol)
            feats.append(self._hurst(c))
            for n in [20,50]:
                z = (c[-1]-np.mean(c[-n:]))/np.std(c[-n:]) if len(c)>=n else 0.0
                feats.append(np.clip(z,-3,3))
            vz = (v[-1]-np.mean(v[-20:]))/np.std(v[-20:]) if len(v)>=20 else 0.0
            feats.append(np.clip(vz,-3,3))
            ob_b,ob_be = self._ob(h,l,c)
            fvg_b,fvg_be = self._fvg(h,l,c)
            bos_b,bos_be = self._bos(h,l,c)
            cho_b,cho_be = self._choch(c)
            feats += [1.0 if ob_b else 0.0, 1.0 if ob_be else 0.0]
            feats += [1.0 if fvg_b else 0.0, 1.0 if fvg_be else 0.0]
            feats += [1.0 if bos_b else 0.0, 1.0 if bos_be else 0.0]
            feats += [1.0 if cho_b else 0.0, 1.0 if cho_be else 0.0]
            cvd_val = (order_flow or {}).get(f"{tf}_cvd",0.0)
            feats += [np.clip(cvd_val/1e9,-1,1), np.clip((order_flow or {}).get(f"{tf}_cvd_slope",0.0),-1,1)]
            feats += [np.clip((order_flow or {}).get(f"{tf}_imbalance",0.0),-1,1)]
            feats.append(np.clip((order_flow or {}).get(f"{tf}_large_z",0.0) if order_flow else 0.0, -3, 3))
        feats.append(self._mtf_trend(ohlcv))
        feats.append(np.mean([self._atr(ohlcv[t]['high'].values,ohlcv[t]['low'].values,ohlcv[t]['close'].values,14) for t in self.TIMEFRAMES if t in ohlcv and len(ohlcv[t])>14] or [0.0]))
        feats.append(np.mean([self._rsi(ohlcv[t]['close'].values,14) for t in self.TIMEFRAMES if t in ohlcv and len(ohlcv[t])>14] or [50.0])/100.0)
        feats.append(np.mean([np.log1p(ohlcv[t]['volume'].values[-1]) for t in self.TIMEFRAMES if t in ohlcv and len(ohlcv[t])>0] or [0.0]))
        if acc:
            feats += [min(acc.get('balance',0)/100000.0,2.0), min(acc.get('equity',0)/100000.0,2.0), acc.get('margin',0)/max(acc.get('balance',1),1.0)]
        else:
            feats += [0.5,0.5,0.0]
        feats.append(min(len(positions or []),5)/5.0)
        feats.append(np.clip(sum(p.get('unrealized_pnl',0) for p in (positions or []))/1000.0,-1,1))
        return np.array(feats, dtype=np.float32)

    def _atr(self,h,l,c,p=14):
        if len(c)<p+1: return 0.0
        try:
            if talib: return talib.ATR(h,l,c,timeperiod=p)[-1]
        except: pass
        tr = np.maximum(h[-p:]-l[-p:], np.maximum(np.abs(h[-p:]-c[-p-1:-1]), np.abs(l[-p:]-c[-p-1:-1])))
        return np.mean(tr)

    def _rsi(self,c,p=14):
        if len(c)<p+1: return 50.0
        try:
            if talib: return talib.RSI(c,timeperiod=p)[-1]
        except: pass
        return 50.0

    def _macd(self,c):
        if len(c)<26: return 0.0,0.0,0.0
        try:
            if talib:
                m,s,h2 = talib.MACD(c,12,26,9)
                return (m[-1] or 0, s[-1] or 0, h2[-1] or 0)
        except: pass
        return 0.0,0.0,0.0

    def _bb(self,c,p=20):
        if len(c)<p: return c[-1],c[-1],c[-1]
        try:
            if talib:
                u,m,l = talib.BBANDS(c,timeperiod=p)
                return (u[-1] or c[-1], m[-1] or c[-1], l[-1] or c[-1])
        except: pass
        return c[-1],c[-1],c[-1]

    def _stoch(self,h,l,c):
        if len(c)<14: return 50.0,50.0
        try:
            if talib:
                k,d = talib.STOCH(h,l,c,14,3,3)
                return (k[-1] or 50.0, d[-1] or 50.0)
        except: pass
        return 50.0,50.0

    def _adx(self,h,l,c):
        if len(c)<15: return 25.0,25.0,25.0
        try:
            if talib:
                a = talib.ADX(h,l,c,14)[-1] or 25.0
                dp = talib.PLUS_DI(h,l,c,14)[-1] or 25.0
                dm = talib.MINUS_DI(h,l,c,14)[-1] or 25.0
                return a,dp,dm
        except: pass
        return 25.0,25.0,25.0

    def _cci(self,h,l,c,p=20):
        if len(c)<p: return 0.0
        try:
            if talib: return talib.CCI(h,l,c,timeperiod=p)[-1] or 0.0
        except: pass
        return 0.0

    def _mfi(self,h,l,c,v,p=14):
        if len(c)<p or np.all(v==0): return 50.0
        try:
            if talib: return talib.MFI(h,l,c,v,timeperiod=p)[-1] or 50.0
        except: pass
        return 50.0

    def _obv(self,c,v):
        obv=[0]
        for i in range(1,len(c)):
            if c[i]>c[i-1]: obv.append(obv[-1]+v[i])
            elif c[i]<c[i-1]: obv.append(obv[-1]-v[i])
            else: obv.append(obv[-1])
        return np.array(obv)

    def _hurst(self,c,max_lag=20):
        if len(c)<max_lag*2: return 0.5
        lags=list(range(2,min(max_lag,len(c)//2)))
        if not lags: return 0.5
        tau=[np.std(c[lag:]-c[:-lag]) for lag in lags]
        if not tau or any(t==0 for t in tau): return 0.5
        poly=np.polyfit(np.log(lags),np.log(tau),1)
        return poly[0]

    def _ob(self,h,l,c):
        if len(c)<10: return False,False
        bull=(c[-3]<c[-2]<c[-1]) and (l[-2]<l[-1])
        bear=(c[-3]>c[-2]>c[-1]) and (h[-2]>h[-1])
        return bull,bear

    def _fvg(self,h,l,c):
        if len(c)<3: return False,False
        return h[-3]<l[-1], l[-3]>h[-1]

    def _bos(self,h,l,c):
        if len(c)<20: return False,False
        rh=np.max(h[-10:]); rl=np.min(l[-10:])
        ph=np.max(h[-20:-10]); pl=np.min(l[-20:-10])
        return rh>ph, rl<pl

    def _choch(self,c):
        if len(c)<30: return False,False
        sma20=np.mean(c[-20:]); sma30=np.mean(c[-30:])
        return (sma20>sma30 and c[-1]>sma20), (sma20<sma30 and c[-1]<sma20)

    def _mtf_trend(self,ohlcv):
        signals=[]
        for tf in self.TIMEFRAMES:
            df=ohlcv.get(tf)
            if df is None or len(df)<20: continue
            c=df['close'].values
            signals.append(1.0 if np.mean(c[-10:])>np.mean(c[-20:]) else -1.0)
        return np.mean(signals) if signals else 0.0

    def get_feature_dim(self):
        return len(self.feature_names)
