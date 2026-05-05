"""
Social Media / Satellite / News / Behavioral Sentiment AI Logic.
Multi-source sentiment: Twitter/X, Reddit, news, satellite imagery, on-chain metrics.
Uses transformer models for deep sentiment analysis.
"""
import numpy as np
import time
import json
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger
import threading


class SentimentSource(str, Enum):
    """Sentiment data sources."""
    TWITTER = "twitter"
    REDDIT = "reddit"
    NEWS = "news"
    SATELLITE = "satellite"
    ONCHAIN = "onchain"
    ECONOMIC = "economic"


@dataclass
class SocialMediaPost:
    """A social media post with metadata."""
    id: str
    source: SentimentSource
    text: str
    author: str = ""
    timestamp: float = field(default_factory=time.time)
    likes: int = 0
    retweets: int = 0
    comments: int = 0
    url: str = ""
    sentiment_score: float = 0.0
    confidence: float = 0.0
    tickers: List[str] = field(default_factory=list)
    viral_score: float = 0.0


@dataclass
class SatelliteSignal:
    """Satellite imagery analysis for commodities/energy."""
    symbol: str
    region: str
    activity_score: float = 0.0  # 0-1, higher = more active
    storage_level: float = 0.0  # % of capacity
    shipping_traffic: int = 0
    night_lights: float = 0.0  # Economic activity proxy
    timestamp: float = field(default_factory=time.time)


@dataclass
class OnChainMetrics:
    """On-chain metrics for crypto sentiment."""
    symbol: str
    network: str = "bitcoin"  # bitcoin | ethereum | etc.
    active_addresses: int = 0
    transaction_volume: float = 0.0
    exchange_inflows: float = 0.0
    exchange_outflows: float = 0.0
    hodl_waves: Dict[str, float] = field(default_factory=dict)  # Age bands
    nvt_ratio: float = 0.0  # Network Value to Transactions
    sentiment_score: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class BehavioralSentimentSnapshot:
    """Comprehensive sentiment snapshot across all sources."""
    timestamp: float = field(default_factory=time.time)
    overall_score: float = 0.0
    confidence: float = 0.0
    
    # Per-source scores
    twitter_score: float = 0.0
    reddit_score: float = 0.0
    news_score: float = 0.0
    satellite_score: float = 0.0
    onchain_score: float = 0.0
    
    # Behavioral metrics
    fear_greed_index: float = 50.0  # 0-100
    social_volume: int = 0
    viral_posts: int = 0
    trending_tickers: List[str] = field(default_factory=list)
    
    # Cross-asset spillover
    spillover_matrix: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    source_counts: Dict[str, int] = field(default_factory=dict)
    recent_headlines: List[str] = field(default_factory=list)


class BehavioralSentimentAI:
    """
    Multi-source sentiment analysis with behavioral AI.
    Combines social media, news, satellite, and on-chain data.
    """

    def __init__(
        self,
        use_transformers: bool = True,
        cache_ttl: int = 300,
        max_posts: int = 200,
        satellite_enabled: bool = True,
        onchain_enabled: bool = True,
    ):
        self.use_transformers = use_transformers
        self.cache_ttl = cache_ttl
        self.max_posts = max_posts
        self.satellite_enabled = satellite_enabled
        self.onchain_enabled = onchain_enabled
        
        self._classifier = None
        self._cache: Dict[str, BehavioralSentimentSnapshot] = {}
        self._last_fetch: Dict[str, float] = {}
        self._social_history: List[Dict] = []
        self._lock = threading.Lock()
        
        # Initialize models
        self._init_models()
        
        # Cache path
        self.cache_path = "data/alternative_data/behavioral_sentiment.json"
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        
        logger.info("Behavioral Sentiment AI initialized")

    def _init_models(self):
        """Initialize sentiment analysis models."""
        # FinBERT for financial news
        if self.use_transformers:
            try:
                from transformers import pipeline
                self._classifier = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    device=-1,  # CPU
                )
                logger.info("FinBERT model loaded for sentiment analysis")
            except Exception as e:
                logger.warning(f"FinBERT init failed, using lexicon fallback: {e}")
                self.use_transformers = False
        
        # Initialize satellite analyzer (placeholder)
        if self.satellite_enabled:
            self._satellite_cache: Dict[str, SatelliteSignal] = {}
            
        # Initialize on-chain analyzer (placeholder)
        if self.onchain_enabled:
            self._onchain_cache: Dict[str, OnChainMetrics] = {}

    def analyze_social_media(
        self,
        symbols: Optional[List[str]] = None,
        sources: Optional[List[SentimentSource]] = None,
    ) -> BehavioralSentimentSnapshot:
        """
        Analyze sentiment from social media and news.
        Returns comprehensive sentiment snapshot.
        """
        sources = sources or [SentimentSource.TWITTER, SentimentSource.REDDIT, SentimentSource.NEWS]
        symbols = symbols or ["EURUSD", "GBPUSD", "BTCUSD"]
        
        cache_key = f"social_{'_'.join(sorted(symbols))}"
        
        # Check cache
        with self._lock:
            cached = self._cache.get(cache_key)
            last_fetch = self._last_fetch.get(cache_key, 0)
            if cached and time.time() - last_fetch < self.cache_ttl:
                return cached
        
        snapshot = BehavioralSentimentSnapshot()
        
        # Collect posts from each source
        all_posts: List[SocialMediaPost] = []
        
        if SentimentSource.TWITTER in sources:
            all_posts.extend(self._fetch_twitter_posts(symbols))
        
        if SentimentSource.REDDIT in sources:
            all_posts.extend(self._fetch_reddit_posts(symbols))
            
        if SentimentSource.NEWS in sources:
            all_posts.extend(self._fetch_news_posts(symbols))
        
        # Analyze sentiment
        if all_posts:
            snapshot = self._analyze_posts(all_posts, snapshot)
        
        # Add satellite signals for commodities
        if self.satellite_enabled:
            for sym in symbols:
                if sym in ["XAUUSD", "XAGUSD", "XTIUSD", "XBRUSD"]:
                    sat = self._get_satellite_signal(sym)
                    if sat:
                        snapshot.satellite_score = (snapshot.satellite_score + sat.activity_score) / 2
        
        # Add on-chain metrics for crypto
        if self.onchain_enabled:
            for sym in symbols:
                if sym in ["BTCUSD", "ETHUSD", "LTCUSD", "XRPUSD"]:
                    chain = self._get_onchain_metrics(sym)
                    if chain:
                        snapshot.onchain_score = chain.sentiment_score
        
        # Calculate fear & greed index
        snapshot.fear_greed_index = self._calculate_fear_greed(snapshot)
        
        # Calculate spillover effects
        snapshot.spillover_matrix = self._calculate_spillover(snapshot, symbols)
        
        # Update cache
        with self._lock:
            self._cache[cache_key] = snapshot
            self._last_fetch[cache_key] = time.time()
        
        self._save_cache(snapshot)
        
        return snapshot

    def _fetch_twitter_posts(self, symbols: List[str]) -> List[SocialMediaPost]:
        """Fetch Twitter/X posts (placeholder - needs API)."""
        # In production: use Twitter API v2
        # For now, generate synthetic posts based on recent price action
        posts = []
        for sym in symbols[:3]:  # Limit for demo
            post = SocialMediaPost(
                id=f"tw_{int(time.time())}_{sym}",
                source=SentimentSource.TWITTER,
                text=f"${sym} looking bullish! Great momentum building up. #trading",
                author="crypto_analyst",
                timestamp=time.time(),
                likes=np.random.randint(10, 1000),
                retweets=np.random.randint(5, 500),
                tickers=[sym],
            )
            posts.append(post)
        return posts

    def _fetch_reddit_posts(self, symbols: List[str]) -> List[SocialMediaPost]:
        """Fetch Reddit posts (placeholder - needs PRAW)."""
        # In production: use PRAW (Python Reddit API Wrapper)
        posts = []
        for sym in symbols[:3]:
            post = SocialMediaPost(
                id=f"rd_{int(time.time())}_{sym}",
                source=SentimentSource.REDDIT,
                text=f"DD: Why {sym} is set for a major breakout. Technicals align.",
                author="wallstreetbets_user",
                timestamp=time.time(),
                likes=np.random.randint(50, 5000),
                comments=np.random.randint(10, 1000),
                tickers=[sym],
            )
            posts.append(post)
        return posts

    def _fetch_news_posts(self, symbols: List[str]) -> List[SocialMediaPost]:
        """Fetch news articles."""
        posts = []
        try:
            import feedparser
            feed_urls = [
                "https://feeds.reuters.com/reuters/businessNews",
                "https://www.forexfactory.com/news.xml",
            ]
            for url in feed_urls:
                feed = feedparser.parse(url)
                for entry in feed.entries[:10]:
                    post = SocialMediaPost(
                        id=f"news_{hash(entry.get('link', ''))}",
                        source=SentimentSource.NEWS,
                        text=f"{entry.get('title', '')}. {entry.get('summary', '')[:200]}",
                        timestamp=time.time(),
                        url=entry.get('link', ''),
                        tickers=self._extract_tickers(entry.get('title', '') + ' ' + entry.get('summary', '')),
                    )
                    posts.append(post)
        except Exception as e:
            logger.debug(f"News fetch error: {e}")
        return posts

    def _analyze_posts(
        self, posts: List[SocialMediaPost], snapshot: BehavioralSentimentSnapshot,
    ) -> BehavioralSentimentSnapshot:
        """Analyze sentiment of posts using transformer or lexicon."""
        if not posts:
            return snapshot
        
        texts = [f"{p.text}"[:512] for p in posts]
        scores = []
        
        if self.use_transformers and self._classifier:
            # Batch classify
            try:
                results = self._batch_classify(texts)
                for post, (score, label) in zip(posts, results):
                    post.sentiment_score = score
                    post.confidence = abs(score)
                    scores.append(score)
            except Exception as e:
                logger.debug(f"Transformer classification failed: {e}")
                scores = self._lexicon_score(texts)
        else:
            scores = self._lexicon_score(texts)
        
        # Weight by engagement (viral posts matter more)
        weights = []
        for p in posts:
            engagement = np.log1p(p.likes + p.retweets + p.comments)
            weights.append(engagement)
        
        if scores and weights:
            weighted_scores = [s * w for s, w in zip(scores, weights)]
            snapshot.overall_score = float(np.mean(weighted_scores))
            snapshot.confidence = float(np.std(weighted_scores))
        
        # Per-source breakdown
        twitter_scores = [p.sentiment_score for p in posts if p.source == SentimentSource.TWITTER]
        reddit_scores = [p.sentiment_score for p in posts if p.source == SentimentSource.REDDIT]
        news_scores = [p.sentiment_score for p in posts if p.source == SentimentSource.NEWS]
        
        snapshot.twitter_score = float(np.mean(twitter_scores)) if twitter_scores else 0.0
        snapshot.reddit_score = float(np.mean(reddit_scores)) if reddit_scores else 0.0
        snapshot.news_score = float(np.mean(news_scores)) if news_scores else 0.0
        
        snapshot.social_volume = len(posts)
        snapshot.viral_posts = sum(1 for p in posts if p.likes + p.retweets > 1000)
        snapshot.source_counts = {
            "twitter": len([p for p in posts if p.source == SentimentSource.TWITTER]),
            "reddit": len([p for p in posts if p.source == SentimentSource.REDDIT]),
            "news": len([p for p in posts if p.source == SentimentSource.NEWS]),
        }
        snapshot.recent_headlines = [p.text[:100] for p in posts[:5]]
        
        return snapshot

    def _batch_classify(self, texts: List[str]) -> List[Tuple[float, str]]:
        """Batch classify texts using transformer."""
        results = self._classifier(texts[:self.max_posts])
        scores = []
        for r in results:
            label = r.get("label", "neutral").lower()
            score = r.get("score", 0.5)
            if "positive" in label:
                scores.append(score)
            elif "negative" in label:
                scores.append(-score)
            else:
                scores.append(0.0)
        return [(s, "positive" if s > 0 else "negative") for s in scores]

    def _lexicon_score(self, texts: List[str]) -> List[float]:
        """Fallback lexicon-based sentiment."""
        positive = {"bull", "bullish", "buy", "long", "gain", "profit", "surge", "rally", "growth"}
        negative = {"bear", "bearish", "sell", "short", "loss", "crash", "drop", "decline"}
        
        scores = []
        for text in texts:
            words = set(text.lower().split())
            pos = len(words & positive)
            neg = len(words & negative)
            total = pos + neg
            scores.append((pos - neg) / max(total, 1))
        return scores

    def _get_satellite_signal(self, symbol: str) -> Optional[SatelliteSignal]:
        """Get satellite imagery analysis (placeholder)."""
        # In production: use Sentinel Hub API or similar
        # For now, return synthetic signal
        if symbol == "XTIUSD":
            signal = SatelliteSignal(
                symbol=symbol,
                region="Cushing_OK",  # Oil storage hub
                activity_score=np.random.uniform(0.3, 0.8),
                storage_level=np.random.uniform(0.4, 0.9),
                shipping_traffic=np.random.randint(20, 100),
                night_lights=np.random.uniform(10, 50),
            )
            self._satellite_cache[symbol] = signal
            return signal
        return None

    def _get_onchain_metrics(self, symbol: str) -> Optional[OnChainMetrics]:
        """Get on-chain metrics for crypto (placeholder)."""
        # In production: use Glassnode or similar API
        if symbol == "BTCUSD":
            metrics = OnChainMetrics(
                symbol=symbol,
                network="bitcoin",
                active_addresses=np.random.randint(800000, 1200000),
                transaction_volume=np.random.uniform(50000, 200000),
                exchange_inflows=np.random.uniform(10000, 50000),
                exchange_outflows=np.random.uniform(10000, 50000),
                nvt_ratio=np.random.uniform(20, 80),
            )
            # Simple sentiment: high NVT = overvalued (bearish)
            metrics.sentiment_score = (50 - metrics.nvt_ratio) / 50.0
            self._onchain_cache[symbol] = metrics
            return metrics
        return None

    def _calculate_fear_greed(self, snapshot: BehavioralSentimentSnapshot) -> float:
        """Calculate Fear & Greed Index from multiple sources."""
        # Weighted combination
        weights = {
            "sentiment": 0.4,
            "momentum": 0.3,
            "social": 0.3,
        }
        
        sentiment_component = (snapshot.overall_score + 1) / 2 * 100  # -1,1 -> 0,100
        social_component = (snapshot.twitter_score + snapshot.reddit_score + 1) / 2 * 100
        
        fg = (
            sentiment_component * weights["sentiment"] +
            social_component * weights["social"] +
            50 * weights["momentum"]  # Placeholder
        )
        
        return float(np.clip(fg, 0, 100))

    def _calculate_spillover(self, snapshot: BehavioralSentimentSnapshot, symbols: List[str]) -> Dict[str, float]:
        """Calculate cross-asset sentiment spillover."""
        spillover = {}
        base_score = snapshot.overall_score
        
        # Simple spillover model (in production: use Graph Neural Network)
        for sym in symbols:
            if "JPY" in sym:
                spillover[sym] = base_score * 0.8  # JPY less sensitive
            elif "USD" in sym:
                spillover[sym] = base_score * 1.2  # USD more sensitive
            else:
                spillover[sym] = base_score
        
        return spillover

    def _extract_tickers(self, text: str) -> List[str]:
        """Extract ticker symbols from text."""
        import re
        tickers = re.findall(r'\$([A-Z]{3,6})', text)
        return tickers

    def get_feature_vector(self) -> np.ndarray:
        """Get sentiment features as vector for ML models."""
        snapshot = self.analyze_social_media()
        
        vec = [
            snapshot.overall_score,
            snapshot.confidence,
            snapshot.twitter_score,
            snapshot.reddit_score,
            snapshot.news_score,
            snapshot.satellite_score,
            snapshot.onchain_score,
            snapshot.fear_greed_index / 100.0,
            np.log1p(snapshot.social_volume) / 10.0,
            snapshot.viral_posts / 100.0,
        ]
        
        return np.array(vec, dtype=np.float32)

    def _save_cache(self, snapshot: BehavioralSentimentSnapshot):
        """Save snapshot to cache file."""
        try:
            data = {
                "timestamp": snapshot.timestamp,
                "overall_score": snapshot.overall_score,
                "confidence": snapshot.confidence,
                "twitter_score": snapshot.twitter_score,
                "reddit_score": snapshot.reddit_score,
                "news_score": snapshot.news_score,
                "satellite_score": snapshot.satellite_score,
                "onchain_score": snapshot.onchain_score,
                "fear_greed_index": snapshot.fear_greed_index,
                "social_volume": snapshot.social_volume,
                "viral_posts": snapshot.viral_posts,
                "source_counts": snapshot.source_counts,
                "recent_headlines": snapshot.recent_headlines,
            }
            with open(self.cache_path, "w") as f:
                json.dump(data, f)
        except Exception:
            pass

    def _load_cache(self) -> Optional[BehavioralSentimentSnapshot]:
        """Load snapshot from cache."""
        try:
            if not os.path.exists(self.cache_path):
                return None
            if time.time() - os.path.getmtime(self.cache_path) > self.cache_ttl:
                return None
            with open(self.cache_path) as f:
                data = json.load(f)
            return BehavioralSentimentSnapshot(**data)
        except Exception:
            return None
