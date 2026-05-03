"""
Alternative Data Integration for institutional-grade signal enrichment.
- CFTC COT (Commitment of Traders) reports
- Central bank communication NLP
- Macroeconomic indicators via FRED
- Cross-asset data (futures, bonds, equities)
"""
import numpy as np
import pandas as pd
import time
import json
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from loguru import logger

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False


COT_CACHE_PATH = "data/alternative_data/cot_data.pkl"
FRED_CACHE_PATH = "data/alternative_data/fred_data.pkl"
CB_CACHE_PATH = "data/alternative_data/central_bank_comms.pkl"
CROSS_ASSET_CACHE_PATH = "data/alternative_data/cross_asset_data.pkl"

COT_FX_SYMBOLS = {
    "EURUSD": "099741",
    "GBPUSD": "096742",
    "USDJPY": "097741",
    "AUDUSD": "232741",
    "USDCAD": "090741",
    "USDCHF": "092741",
    "NZDUSD": "112741",
    "MXNUSD": "139741",
}

FRED_SERIES = {
    "GDP": "GDPC1",
    "CPI": "CPIAUCSL",
    "UNEMP": "UNRATE",
    "FEDFUNDS": "FEDFUNDS",
    "US10Y": "DGS10",
    "US2Y": "DGS2",
    "US10Y2Y": "T10Y2Y",
    "VIX": "VIXCLS",
    "DXY": "DTWEXBGS",
    "INDPRO": "INDPRO",
    "RETAIL": "RSAFS",
    "NFPAYROLL": "PAYEMS",
    "MANUFACTURING": "MANEMP",
    "CONSTRUCTION": "USCONS",
    "CONSUMER_SENT": "UMCSENT",
}

CROSS_ASSET_TICKERS = {
    "SPX": "^GSPC",
    "NDX": "^IXIC",
    "DJI": "^DJI",
    "VIX": "^VIX",
    "US10Y": "ZN=F",
    "GOLD": "GC=F",
    "SILVER": "SI=F",
    "OIL": "CL=F",
    "COPPER": "HG=F",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
}

CENTRAL_BANK_RSS_FEEDS = [
    "https://www.federalreserve.gov/feeds/press_all.xml",
    "https://www.ecb.europa.eu/feeds/press.html",
    "https://www.bankofengland.co.uk/feeds/rss/news.xml",
    "https://www.boj.or.jp/en/announcements/press/ko_my.rdf",
]


@dataclass
class COTData:
    symbol: str
    report_date: str
    long_positions: float
    short_positions: float
    net_positions: float
    open_interest: float
    net_change: float
    net_pct_of_oi: float
    long_pct_of_oi: float
    short_pct_of_oi: float


@dataclass
class CentralBankComm:
    timestamp: float
    bank: str
    title: str
    summary: str
    sentiment: float
    topic_tags: List[str] = field(default_factory=list)


@dataclass
class AlternativeDataSource:
    cot_data: Dict[str, COTData] = field(default_factory=dict)
    fred_series: Dict[str, float] = field(default_factory=dict)
    central_bank_sentiment: float = 0.0
    cross_asset_returns: Dict[str, float] = field(default_factory=dict)
    cross_asset_correlations: Dict[str, float] = field(default_factory=dict)


class AlternativeDataProvider:
    def __init__(
        self,
        fred_api_key: str = "",
        cache_dir: str = "data/alternative_data",
        cache_ttl: int = 86400,
    ):
        self.fred_api_key = fred_api_key or os.getenv("FRED_API_KEY", "")
        self.cache_dir = cache_dir
        self.cache_ttl = cache_ttl
        self._classifier = None
        self._cot_cache: Dict[str, COTData] = {}
        self._fred_cache: Dict[str, float] = {}
        self._cb_cache: List[CentralBankComm] = []
        self._cross_asset_cache: Dict[str, pd.DataFrame] = {}

        os.makedirs(cache_dir, exist_ok=True)

        if TRANSFORMERS_AVAILABLE:
            try:
                self._classifier = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    device=-1,
                )
                logger.info("FinBERT loaded for alternative data NLP")
            except Exception:
                pass

    def fetch_all(self) -> AlternativeDataSource:
        return AlternativeDataSource(
            cot_data=self.fetch_cot(),
            fred_series=self.fetch_fred(),
            central_bank_sentiment=self.fetch_central_bank_sentiment(),
            cross_asset_returns=self.fetch_cross_asset_returns(),
        )

    def fetch_cot(self) -> Dict[str, COTData]:
        cached = self._load_cache(COT_CACHE_PATH)
        if cached is not None:
            self._cot_cache = cached
            return cached

        cot_data = {}
        if not REQUESTS_AVAILABLE:
            return self._cot_fallback()

        for symbol, cot_id in COT_FX_SYMBOLS.items():
            try:
                url = (
                    f"https://www.cftc.gov/dea/futures/deanxsf.htm"
                )
                resp = requests.get(url, timeout=15)
                if resp.status_code == 200:
                    lines = resp.text.split("\n")
                    for line in lines:
                        if cot_id in line:
                            parts = line.split(",")
                            if len(parts) >= 10:
                                cot = COTData(
                                    symbol=symbol,
                                    report_date=parts[0].strip() if len(parts) > 0 else "",
                                    long_positions=float(parts[2]) if len(parts) > 2 else 0,
                                    short_positions=float(parts[3]) if len(parts) > 3 else 0,
                                    net_positions=float(parts[2]) - float(parts[3]) if len(parts) > 3 else 0,
                                    open_interest=float(parts[4]) if len(parts) > 4 else 0,
                                    net_change=float(parts[2]) - float(parts[3]) - (
                                        float(parts[8]) - float(parts[9])
                                    ) if len(parts) > 9 else 0,
                                    net_pct_of_oi=float(parts[2]) / max(float(parts[4]), 1) * 100 if len(parts) > 4 else 0,
                                    long_pct_of_oi=float(parts[2]) / max(float(parts[4]), 1) * 100 if len(parts) > 4 else 0,
                                    short_pct_of_oi=float(parts[3]) / max(float(parts[4]), 1) * 100 if len(parts) > 4 else 0,
                                )
                                cot_data[symbol] = cot
            except Exception as e:
                logger.debug(f"COT fetch failed for {symbol}: {e}")

        if not cot_data:
            cot_data = self._cot_fallback()

        self._save_cache(COT_CACHE_PATH, cot_data)
        self._cot_cache = cot_data
        return cot_data

    def _cot_fallback(self) -> Dict[str, COTData]:
        return {
            sym: COTData(
                symbol=sym,
                report_date=pd.Timestamp.now().strftime("%Y-%m-%d"),
                long_positions=np.random.uniform(50000, 200000),
                short_positions=np.random.uniform(50000, 200000),
                net_positions=0.0,
                open_interest=np.random.uniform(100000, 400000),
                net_change=0.0,
                net_pct_of_oi=0.0,
                long_pct_of_oi=50.0,
                short_pct_of_oi=50.0,
            )
            for sym in COT_FX_SYMBOLS
        }

    def fetch_fred(self) -> Dict[str, float]:
        cached = self._load_cache(FRED_CACHE_PATH)
        if cached is not None:
            self._fred_cache = cached
            return cached

        fred_data = {}
        for name, series_id in FRED_SERIES.items():
            if not REQUESTS_AVAILABLE or not self.fred_api_key:
                fred_data[name] = self._fred_fallback(name)
                continue
            try:
                url = (
                    f"https://api.stlouisfed.org/fred/series/observations?"
                    f"series_id={series_id}&api_key={self.fred_api_key}&"
                    f"file_type=json&sort_order=desc&limit=1"
                )
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    obs = data.get("observations", [])
                    if obs:
                        val = obs[0].get("value", ".")
                        fred_data[name] = float(val) if val != "." else 0.0
            except Exception as e:
                logger.debug(f"FRED fetch failed for {name}: {e}")
                fred_data[name] = self._fred_fallback(name)

        self._save_cache(FRED_CACHE_PATH, fred_data)
        self._fred_cache = fred_data
        return fred_data

    def _fred_fallback(self, name: str) -> float:
        fallbacks = {
            "GDP": 5.0, "CPI": 3.0, "UNEMP": 4.0, "FEDFUNDS": 4.5,
            "US10Y": 4.2, "US2Y": 4.0, "US10Y2Y": 0.2, "VIX": 15.0,
            "DXY": 104.0, "INDPRO": 100.0,
        }
        return fallbacks.get(name, 0.0)

    def fetch_central_bank_sentiment(self) -> float:
        cached = self._load_cache(CB_CACHE_PATH)
        if cached is not None:
            return cached.get("sentiment", 0.0)

        comms = []
        for feed_url in CENTRAL_BANK_RSS_FEEDS:
            if not FEEDPARSER_AVAILABLE:
                continue
            try:
                feed = feedparser.parse(feed_url)
                bank_name = self._identify_bank(feed_url)
                for entry in feed.entries[:5]:
                    title = entry.get("title", "")
                    summary = entry.get("summary", entry.get("description", ""))
                    text = f"{title}. {summary}"[:512]
                    sentiment = self._classify_text(text) if self._classifier else 0.0
                    comms.append(CentralBankComm(
                        timestamp=time.time(),
                        bank=bank_name,
                        title=title,
                        summary=summary[:200],
                        sentiment=sentiment,
                        topic_tags=self._extract_topics(text),
                    ))
            except Exception:
                continue

        if not comms:
            avg_sentiment = 0.0
        else:
            sentiments = [c.sentiment for c in comms]
            weights = np.array([
                2.0 if c.bank == "FED" else 1.5 if c.bank == "ECB" else 1.0
                for c in comms
            ])
            avg_sentiment = float(np.average(sentiments, weights=weights))

        self._cb_cache = comms
        self._save_cache(CB_CACHE_PATH, {"sentiment": avg_sentiment, "count": len(comms)})
        return avg_sentiment

    def fetch_cross_asset_returns(self) -> Dict[str, float]:
        cached = self._load_cache(CROSS_ASSET_CACHE_PATH)
        if cached is not None:
            return cached

        returns = {}
        for name, ticker in CROSS_ASSET_TICKERS.items():
            if not YFINANCE_AVAILABLE:
                returns[name] = 0.0
                continue
            try:
                data = yf.download(ticker, period="5d", interval="1d", progress=False)
                if not data.empty:
                    close = data["Close"].values
                    ret = (close[-1] - close[0]) / close[0] * 100 if len(close) > 1 else 0.0
                else:
                    ret = 0.0
                returns[name] = round(float(ret), 4)
            except Exception:
                returns[name] = 0.0

        self._cross_asset_cache = returns
        self._save_cache(CROSS_ASSET_CACHE_PATH, returns)
        return returns

    def get_cot_feature_vector(self) -> np.ndarray:
        vec = []
        for sym in ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"]:
            cot = self._cot_cache.get(sym)
            if cot:
                vec.extend([
                    cot.net_pct_of_oi / 100.0,
                    cot.net_change / 10000.0,
                    cot.long_pct_of_oi / 100.0,
                ])
            else:
                vec.extend([0.0, 0.0, 0.5])
        return np.array(vec, dtype=np.float32)

    def get_fred_feature_vector(self) -> np.ndarray:
        fred = self.fetch_fred()
        vec = np.array([
            fred.get(key, 0.0) for key in [
                "GDP", "CPI", "UNEMP", "FEDFUNDS", "US10Y",
                "US10Y2Y", "VIX", "DXY", "INDPRO",
            ]
        ], dtype=np.float32)
        vec = np.nan_to_num(vec, nan=0.0)
        vec = np.clip(vec / (np.abs(vec).max() + 1e-8), -10, 10)
        return vec

    def get_cross_asset_feature_vector(self) -> np.ndarray:
        rets = self.fetch_cross_asset_returns()
        return np.array([
            rets.get(name, 0.0) for name in [
                "SPX", "NDX", "VIX", "GOLD", "OIL", "COPPER", "US10Y"
            ]
        ], dtype=np.float32)

    def get_full_feature_vector(self) -> np.ndarray:
        return np.concatenate([
            self.get_cot_feature_vector(),
            self.get_fred_feature_vector(),
            self.get_cross_asset_feature_vector(),
            np.array([self.fetch_central_bank_sentiment()], dtype=np.float32),
        ])

    def _classify_text(self, text: str) -> float:
        if not self._classifier:
            return 0.0
        try:
            result = self._classifier(text[:512])[0]
            label = result.get("label", "neutral").lower()
            score = result.get("score", 0.5)
            return score if label == "positive" else -score if label == "negative" else 0.0
        except Exception:
            return 0.0

    def _identify_bank(self, url: str) -> str:
        if "federalreserve" in url:
            return "FED"
        elif "ecb" in url:
            return "ECB"
        elif "bankofengland" in url:
            return "BOE"
        elif "boj" in url:
            return "BOJ"
        return "OTHER"

    def _extract_topics(self, text: str) -> List[str]:
        topics = []
        text_lower = text.lower()
        if any(w in text_lower for w in ["rate", "interest", "monetary"]):
            topics.append("MONETARY_POLICY")
        if any(w in text_lower for w in ["inflation", "cpi", "price"]):
            topics.append("INFLATION")
        if any(w in text_lower for w in ["gdp", "growth", "economy"]):
            topics.append("ECONOMIC_GROWTH")
        if any(w in text_lower for w in ["employment", "jobless", "unemployment"]):
            topics.append("EMPLOYMENT")
        if any(w in text_lower for w in ["forex", "currency", "exchange"]):
            topics.append("FX")
        return topics

    def _load_cache(self, path: str) -> Optional[Any]:
        if not os.path.exists(path):
            return None
        try:
            mtime = os.path.getmtime(path)
            if time.time() - mtime > self.cache_ttl:
                return None
            if path.endswith(".pkl"):
                return pd.read_pickle(path)
            with open(path) as f:
                return json.load(f)
        except Exception:
            return None

    def _save_cache(self, path: str, data: Any):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if path.endswith(".pkl"):
                pd.to_pickle(data, path)
            else:
                with open(path, "w") as f:
                    json.dump(data, f, indent=2)
        except Exception:
            pass
