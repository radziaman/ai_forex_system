"""
Market Sentiment Analysis via FinBERT and news feeds.
Provides orthogonal signal dimension beyond price-based features.
"""
import numpy as np
import time
import json
import os
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger

try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False

FINBERT_AVAILABLE = False
try:
    from transformers import pipeline
    FINBERT_AVAILABLE = True
except ImportError:
    pass


FINANCE_NEWS_FEEDS = [
    "https://feeds.content.dowjones.io/public/rss/markets",
    "https://www.investopedia.com/feedbuilder/feed/getfeed?feedName=rss_categories_market_analysis",
    "https://www.forexfactory.com/news.xml",
    "https://feeds.reuters.com/reuters/businessNews",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://feeds.bloomberg.com/markets/news.rss",
    "https://www.economist.com/finance-and-economics/rss.xml",
]

CURRENCY_MAP = {
    "EUR": ["euro", "european", "ecb", "eurozone", "germany", "france"],
    "USD": ["dollar", "fed", "federal reserve", "us economy", "treasury"],
    "GBP": ["pound", "sterling", "boe", "uk economy", "british"],
    "JPY": ["yen", "japan", "boj", "japanese"],
    "AUD": ["aussie", "australian", "rba"],
    "CAD": ["loonie", "canadian", "boc", "canada"],
    "NZD": ["kiwi", "new zealand", "rbnz"],
    "CHF": ["swiss", "franc", "snb"],
}

SENTIMENT_CACHE_PATH = "data/alternative_data/sentiment_cache.json"


@dataclass
class NewsItem:
    title: str
    summary: str
    published: float
    source: str
    url: str = ""
    sentiment: float = 0.0
    relevance: float = 0.0
    currencies: List[str] = field(default_factory=list)


@dataclass
class SentimentSnapshot:
    timestamp: float
    overall_score: float
    currency_scores: Dict[str, float]
    volatility_signal: float
    news_count: int
    recent_headlines: List[str]


class SentimentAnalyzer:
    def __init__(
        self,
        use_finbert: bool = True,
        cache_ttl: int = 300,
        max_articles: int = 100,
    ):
        self.use_finbert = use_finbert and FINBERT_AVAILABLE
        self.cache_ttl = cache_ttl
        self.max_articles = max_articles
        self._classifier = None
        self._cache: Dict[str, SentimentSnapshot] = {}
        self._last_fetch: Dict[str, float] = {}
        self._lock = threading.Lock()

        if self.use_finbert:
            self._init_finbert()

        os.makedirs(os.path.dirname(SENTIMENT_CACHE_PATH), exist_ok=True)

    def _init_finbert(self):
        try:
            self._classifier = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                device=-1,
            )
            logger.info("FinBERT model loaded for sentiment analysis")
        except Exception as e:
            logger.warning(f"FinBERT init failed, using fallback: {e}")
            self.use_finbert = False

    def fetch_news(
        self, currencies: Optional[List[str]] = None
    ) -> List[NewsItem]:
        items = []
        for feed_url in FINANCE_NEWS_FEEDS:
            if not FEEDPARSER_AVAILABLE:
                continue
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:20]:
                    title = entry.get("title", "")
                    summary = entry.get("summary", entry.get("description", ""))
                    published = self._parse_date(entry)
                    item = NewsItem(
                        title=title,
                        summary=summary,
                        published=published,
                        source=feed_url.split("/")[2],
                        url=entry.get("link", ""),
                    )
                    item.currencies = self._detect_currencies(title + " " + summary)
                    if currencies:
                        item.relevance = len(
                            [c for c in item.currencies if c in currencies]
                        ) / max(len(currencies), 1)
                    else:
                        item.relevance = len(item.currencies) / max(
                            len(CURRENCY_MAP), 1
                        )
                    items.append(item)
            except Exception:
                continue
        items.sort(key=lambda x: x.published, reverse=True)
        return items[: self.max_articles]

    def analyze_sentiment(
        self, items: List[NewsItem]
    ) -> SentimentSnapshot:
        if not items:
            return SentimentSnapshot(
                timestamp=time.time(),
                overall_score=0.0,
                currency_scores={c: 0.0 for c in CURRENCY_MAP},
                volatility_signal=0.0,
                news_count=0,
                recent_headlines=[],
            )

        texts = [f"{item.title}. {item.summary}"[:512] for item in items]

        if self.use_finbert and self._classifier:
            scores = self._batch_classify(texts)
        else:
            scores = self._lexicon_score(texts)

        currency_scores: Dict[str, List[float]] = {c: [] for c in CURRENCY_MAP}
        for item, score in zip(items, scores):
            for c in item.currencies:
                if c in currency_scores:
                    currency_scores[c].append(score * (0.5 + 0.5 * item.relevance))

        agg_currency = {
            c: float(np.mean(s)) if s else 0.0 for c, s in currency_scores.items()
        }

        overall = float(np.mean(scores)) if scores else 0.0

        vol_signal = float(np.std(scores)) if len(scores) > 1 else 0.0

        snapshot = SentimentSnapshot(
            timestamp=time.time(),
            overall_score=overall,
            currency_scores=agg_currency,
            volatility_signal=vol_signal,
            news_count=len(items),
            recent_headlines=[item.title[:120] for item in items[:5]],
        )

        with self._lock:
            self._cache["latest"] = snapshot
            self._last_fetch["latest"] = time.time()

        self._save_cache(snapshot)
        return snapshot

    def _batch_classify(self, texts: List[str]) -> List[float]:
        scores = []
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                results = self._classifier(batch)
                for r in results:
                    label = r.get("label", "neutral").lower()
                    score = r.get("score", 0.5)
                    if label == "positive":
                        scores.append(score)
                    elif label == "negative":
                        scores.append(-score)
                    else:
                        scores.append(0.0)
            except Exception:
                scores.extend([0.0] * len(batch))
        return scores

    def _lexicon_score(self, texts: List[str]) -> List[float]:
        positive_words = {
            "grow", "growth", "surge", "rally", "bull", "bullish", "gain", "profit",
            "strong", "upgrade", "outperform", "beat", "positive", "recovery", "boom",
            "expansion", "momentum", "uptrend", "breakout", "higher", "improve",
            "improvement", "increase", "rising", "boost", "opportunity", "optimistic",
        }
        negative_words = {
            "drop", "crash", "decline", "bear", "bearish", "loss", "weak", "downgrade",
            "underperform", "miss", "negative", "recession", "slowdown", "crisis",
            "selloff", "plunge", "lower", "decrease", "falling", "risk", "uncertainty",
            "volatile", "downturn", "fear", "panic", "correction", "tumble", "slump",
        }
        scores = []
        for text in texts:
            words = set(text.lower().split())
            pos = len(words & positive_words)
            neg = len(words & negative_words)
            total = pos + neg
            scores.append((pos - neg) / max(total, 1))
        return scores if scores else [0.0]

    def _detect_currencies(self, text: str) -> List[str]:
        text_lower = text.lower()
        detected = []
        for currency, keywords in CURRENCY_MAP.items():
            for kw in keywords:
                if kw in text_lower:
                    detected.append(currency)
                    break
        return detected or ["USD"]

    def _parse_date(self, entry) -> float:
        for attr in ("published_parsed", "updated_parsed"):
            parsed = entry.get(attr)
            if parsed:
                try:
                    return time.mktime(parsed)
                except Exception:
                    pass
        return time.time() - 3600

    def get_latest(
        self, force_refresh: bool = False
    ) -> SentimentSnapshot:
        with self._lock:
            cached = self._cache.get("latest")
            last_fetch = self._last_fetch.get("latest", 0)
            if cached and not force_refresh and time.time() - last_fetch < self.cache_ttl:
                return cached

        cached_file = self._load_cache()
        if cached_file and not force_refresh:
            with self._lock:
                self._cache["latest"] = cached_file
            return cached_file

        items = self.fetch_news()
        return self.analyze_sentiment(items)

    def get_currency_score(self, currency: str) -> float:
        snapshot = self.get_latest()
        return snapshot.currency_scores.get(currency.upper(), 0.0)

    def get_feature_vector(self, currencies: Optional[List[str]] = None) -> np.ndarray:
        snapshot = self.get_latest()
        targets = currencies or ["EUR", "USD", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]
        vec = [
            snapshot.overall_score,
            snapshot.volatility_signal,
            snapshot.news_count / 100.0,
        ]
        for c in targets:
            vec.append(snapshot.currency_scores.get(c, 0.0))
        return np.array(vec, dtype=np.float32)

    def _save_cache(self, snapshot: SentimentSnapshot):
        try:
            data = {
                "timestamp": snapshot.timestamp,
                "overall_score": snapshot.overall_score,
                "currency_scores": snapshot.currency_scores,
                "volatility_signal": snapshot.volatility_signal,
                "news_count": snapshot.news_count,
                "recent_headlines": snapshot.recent_headlines,
            }
            with open(SENTIMENT_CACHE_PATH, "w") as f:
                json.dump(data, f)
        except Exception:
            pass

    def _load_cache(self) -> Optional[SentimentSnapshot]:
        try:
            if not os.path.exists(SENTIMENT_CACHE_PATH):
                return None
            if time.time() - os.path.getmtime(SENTIMENT_CACHE_PATH) > self.cache_ttl:
                return None
            with open(SENTIMENT_CACHE_PATH) as f:
                data = json.load(f)
            return SentimentSnapshot(
                timestamp=data.get("timestamp", 0),
                overall_score=data.get("overall_score", 0),
                currency_scores=data.get("currency_scores", {}),
                volatility_signal=data.get("volatility_signal", 0),
                news_count=data.get("news_count", 0),
                recent_headlines=data.get("recent_headlines", []),
            )
        except Exception:
            return None
