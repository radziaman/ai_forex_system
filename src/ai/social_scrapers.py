"""
Social media and sentiment scrapers for alternative data signals.

Provides:
  - FearGreedScraper: Fear & Greed Index via alternative.me (free, no auth).
  - RedditScraper: Forex sentiment from Reddit via PRAW or requests + OAuth.
  - TwitterScraper: Forex sentiment from Twitter/X via tweepy or requests.

All scrapers cache results internally and degrade gracefully on failure.
"""

import os
import time
import math
import base64
import threading
from typing import Any, Dict, List, Optional, Set, Tuple
from loguru import logger

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Optional PRAW for Reddit
PRAW_AVAILABLE = False
try:
    import praw

    PRAW_AVAILABLE = True
except ImportError:
    pass

# Optional tweepy for Twitter
TWEEPY_AVAILABLE = False
try:
    import tweepy

    TWEEPY_AVAILABLE = True
except ImportError:
    pass

# Default forex-related subreddits
FOREX_SUBREDDITS = ["Forex", "wallstreetbets", "trading", "investing"]

# Lexicon for Reddit text scoring (mirrors sentiment.py lexicon)
_POSITIVE_WORDS: Set[str] = {
    "bullish",
    "moon",
    "pump",
    "buy",
    "long",
    "breakout",
    "rally",
    "surge",
    "gain",
    "profit",
    "strong",
    "positive",
    "growth",
    "uptrend",
    "momentum",
    "higher",
    "improve",
    "improvement",
    "increase",
    "rising",
    "boost",
    "opportunity",
    "optimistic",
    "rocket",
    "green",
    "calls",
    "beat",
    "recovery",
    "boom",
    "expansion",
    "outperform",
    "upgrade",
}

_NEGATIVE_WORDS: Set[str] = {
    "bearish",
    "dump",
    "sell",
    "short",
    "crash",
    "decline",
    "drop",
    "loss",
    "weak",
    "negative",
    "recession",
    "slowdown",
    "crisis",
    "selloff",
    "plunge",
    "lower",
    "decrease",
    "falling",
    "risk",
    "uncertainty",
    "volatile",
    "downturn",
    "fear",
    "panic",
    "correction",
    "tumble",
    "slump",
    "red",
    "puts",
    "rekt",
    "bag",
    "overvalued",
    "downgrade",
    "underperform",
    "miss",
}

RATE_LIMIT_SLEEP = 1.1  # ~55 req/min, safely under Reddit's 60/min limit

# ──────────────────────────────────────────────────────────────────────
# NASA satellite data provider constants
# ──────────────────────────────────────────────────────────────────────

# Financial centres for mapping natural events to currencies
FINANCIAL_CENTRES: List[Dict[str, Any]] = [
    {
        "name": "New York",
        "lat": 40.7,
        "lon": -74.0,
        "currencies": ["USD"],
        "radius_km": 600,
    },
    {
        "name": "London",
        "lat": 51.5,
        "lon": -0.1,
        "currencies": ["EUR", "GBP"],
        "radius_km": 600,
    },
    {
        "name": "Tokyo",
        "lat": 35.7,
        "lon": 139.7,
        "currencies": ["JPY"],
        "radius_km": 600,
    },
    {
        "name": "Sydney",
        "lat": -33.9,
        "lon": 151.2,
        "currencies": ["AUD", "NZD"],
        "radius_km": 600,
    },
    {
        "name": "Zurich",
        "lat": 47.4,
        "lon": 8.5,
        "currencies": ["CHF"],
        "radius_km": 600,
    },
]

# Agricultural regions whose weather impacts commodity / commodity-currency prices
AG_REGIONS: List[Dict[str, Any]] = [
    {"name": "US Midwest", "lat": 41.0, "lon": -93.0, "currencies": ["USD"]},
    {"name": "Brazil Cerrado", "lat": -15.0, "lon": -55.0, "currencies": ["USD"]},
    {"name": "Australia Wheat", "lat": -30.0, "lon": 145.0, "currencies": ["AUD"]},
    {"name": "New Zealand Dairy", "lat": -40.0, "lon": 175.0, "currencies": ["NZD"]},
    {"name": "Europe Wheat", "lat": 48.0, "lon": 5.0, "currencies": ["EUR"]},
    {"name": "Canada Prairies", "lat": 50.0, "lon": -105.0, "currencies": ["CAD"]},
]

# EONET category IDs → impact weight
EONET_WEIGHTS: Dict[str, float] = {
    "earthquakes": 0.30,
    "severeStorms": 0.25,
    "floods": 0.20,
    "volcanoes": 0.15,
    "wildfires": 0.10,
    "drought": 0.15,
    "landslides": 0.10,
    "seaLakeIce": 0.05,
    "temperatureExtremes": 0.15,
    "snow": 0.10,
    "waterColor": 0.05,
    "dustHaze": 0.08,
}

SATELLITE_CACHE_TTL = 21_600  # 6 hours

FOREX_SEARCH_QUERIES = [
    "(EURUSD OR forex OR dollar OR trading) -is:retweet lang:en",
    "(GBP OR sterling OR pound OR GBPUSD) -is:retweet lang:en",
    "(JPY OR yen OR USDJPY) -is:retweet lang:en",
    "(AUD OR aussie OR EURUSD) -is:retweet lang:en",
]

DEFAULT_CURRENCY_KEYWORDS = {
    "EUR": ["euro", "european", "ecb", "eurozone"],
    "USD": ["dollar", "fed", "federal reserve", "treasury"],
    "GBP": ["pound", "sterling", "boe", "uk economy"],
    "JPY": ["yen", "japan", "boj", "japanese"],
    "AUD": ["aussie", "australian", "rba"],
    "CAD": ["loonie", "canadian", "boc", "canada"],
    "NZD": ["kiwi", "new zealand", "rbnz"],
    "CHF": ["swiss", "franc", "snb"],
}

_TWITTER_POSITIVE_WORDS = {
    "bullish",
    "rally",
    "surge",
    "gain",
    "profit",
    "strong",
    "outperform",
    "beat",
    "positive",
    "recovery",
    "boom",
    "momentum",
    "uptrend",
    "breakout",
    "higher",
    "improve",
    "increase",
    "rising",
    "boost",
    "opportunity",
    "optimistic",
    "buy",
    "long",
    "support",
    "growth",
    "upgrade",
}

_TWITTER_NEGATIVE_WORDS = {
    "bearish",
    "crash",
    "decline",
    "loss",
    "weak",
    "downgrade",
    "negative",
    "recession",
    "crisis",
    "selloff",
    "plunge",
    "lower",
    "decrease",
    "falling",
    "risk",
    "uncertainty",
    "volatile",
    "downturn",
    "fear",
    "panic",
    "correction",
    "tumble",
    "slump",
    "sell",
    "short",
    "resistance",
    "drop",
}


class FearGreedScraper:
    """
    Scrapes the Fear & Greed Index via alternative.me (free, no auth required).

    Caches the result for 1 hour (3600 seconds) to avoid redundant API calls.
    Never crashes — returns None on failure.
    """

    API_URL = "https://api.alternative.me/fng/?limit=1"
    CACHE_TTL = 3600  # 1 hour

    def __init__(self):
        self._cached_value: Optional[int] = None
        self._cached_classification: Optional[str] = None
        self._cached_at: float = 0.0

    def fetch_index(self) -> Optional[int]:
        """
        Fetch Fear & Greed Index value (1–100).

        Returns the cached value if TTL hasn't expired, or the stale cached
        value if the API call fails. Returns None only if both cache AND
        API fail.
        """
        now = time.time()
        if self._cached_value is not None and (now - self._cached_at) < self.CACHE_TTL:
            return self._cached_value

        if not REQUESTS_AVAILABLE:
            return self._cached_value

        try:
            resp = requests.get(self.API_URL, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            value = int(data["data"][0]["value"])
            classification = data["data"][0]["value_classification"]

            self._cached_value = value
            self._cached_classification = classification
            self._cached_at = now

            logger.debug(f"Fear & Greed Index: {value} ({classification})")
            return value

        except (
            requests.RequestException,
            KeyError,
            IndexError,
            ValueError,
            TypeError,
        ) as e:
            logger.warning(f"Fear & Greed fetch failed: {e}")
            return self._cached_value

    def fetch_classification(self) -> Optional[str]:
        """Fetch Fear & Greed classification label (e.g. 'Fear', 'Greed')."""
        self.fetch_index()  # ensure cache is warm
        return self._cached_classification


class RedditScraper:
    """Scrapes forex-related sentiment from Reddit.

    Uses PRAW if available, otherwise falls back to requests + OAuth.
    Gracefully degrades if credentials are missing or network fails.
    """

    def __init__(self):
        self._client_id: str = ""
        self._client_secret: str = ""
        self._praw_reddit = None
        self._user_agent: str = "RTS-AI-Forex-Trading-System/1.0 (by /u/rts_ai)"
        self._oauth_token: Optional[str] = None
        self._token_expires_at: float = 0
        self._available: bool = False

        # Load credentials from environment
        self._client_id = os.getenv("REDDIT_CLIENT_ID", "")
        self._client_secret = os.getenv("REDDIT_CLIENT_SECRET", "")

        if not self._client_id or not self._client_secret:
            logger.warning(
                "Reddit credentials not found (REDDIT_CLIENT_ID/SECRET) "
                "— RedditScraper disabled"
            )
            return

        # Try PRAW first
        if PRAW_AVAILABLE:
            try:
                self._praw_reddit = praw.Reddit(
                    client_id=self._client_id,
                    client_secret=self._client_secret,
                    user_agent=self._user_agent,
                )
                self._available = True
                logger.info("RedditScraper initialized with PRAW")
                return
            except Exception as e:
                logger.warning(f"PRAW init failed, falling back to requests: {e}")
                self._praw_reddit = None

        # Requests fallback — will get token on demand
        if not REQUESTS_AVAILABLE:
            logger.warning("requests not available — RedditScraper disabled")
            return

        self._available = True
        logger.info("RedditScraper initialized with requests fallback")

    def _reddit_oauth_token(self) -> Optional[str]:
        """Get an OAuth2 access token from Reddit API.

        Returns None if authentication fails.
        """
        now = time.time()
        if self._oauth_token and now < self._token_expires_at - 60:
            return self._oauth_token

        try:
            auth = base64.b64encode(
                f"{self._client_id}:{self._client_secret}".encode()
            ).decode()
            headers = {
                "Authorization": f"Basic {auth}",
                "User-Agent": self._user_agent,
            }
            data = {"grant_type": "client_credentials"}
            resp = requests.post(
                "https://www.reddit.com/api/v1/access_token",
                headers=headers,
                data=data,
                timeout=10,
            )
            resp.raise_for_status()
            token_data = resp.json()
            self._oauth_token = token_data["access_token"]
            self._token_expires_at = now + token_data.get("expires_in", 3600)
            logger.debug("Reddit OAuth token acquired")
            return self._oauth_token
        except Exception as e:
            logger.warning(f"Reddit OAuth token acquisition failed: {e}")
            return None

    def search_forex_posts(
        self,
        subreddits: Optional[List[str]] = None,
        limit: int = 50,
    ) -> List[Dict]:
        """Search forex-related subreddits for recent posts.

        Args:
            subreddits: List of subreddit names to search.
                        Defaults to FOREX_SUBREDDITS.
            limit: Max posts per subreddit.

        Returns:
            List of dicts with keys:
                text, subreddit, title, score, num_comments, created_utc
        """
        if not self._available:
            logger.debug("RedditScraper not available — skipping search")
            return []

        subreddits = subreddits or FOREX_SUBREDDITS
        all_posts: List[Dict] = []

        for subreddit in subreddits:
            posts = self._fetch_subreddit_posts(subreddit, limit)
            all_posts.extend(posts)

        # Deduplicate by title
        seen_titles: Set[str] = set()
        unique_posts = []
        for post in all_posts:
            title_lower = post.get("title", "").lower().strip()
            if title_lower and title_lower not in seen_titles:
                seen_titles.add(title_lower)
                unique_posts.append(post)

        logger.debug(
            f"Reddit: fetched {len(unique_posts)} unique posts "
            f"from {len(subreddits)} subreddits"
        )
        return unique_posts

    def _fetch_subreddit_posts(self, subreddit: str, limit: int) -> List[Dict]:
        """Fetch recent posts from a single subreddit."""
        if self._praw_reddit is not None:
            return self._fetch_via_praw(subreddit, limit)
        return self._fetch_via_requests(subreddit, limit)

    def _fetch_via_praw(self, subreddit: str, limit: int) -> List[Dict]:
        """Fetch posts using PRAW."""
        posts = []
        try:
            sub = self._praw_reddit.subreddit(subreddit)
            for post in sub.new(limit=limit):
                text = f"{post.title} {post.selftext}"
                posts.append(
                    {
                        "text": text,
                        "subreddit": subreddit,
                        "title": post.title,
                        "score": post.score,
                        "num_comments": post.num_comments,
                        "created_utc": post.created_utc,
                    }
                )
            time.sleep(RATE_LIMIT_SLEEP)
        except Exception as e:
            logger.warning(f"Reddit PRAW fetch failed for r/{subreddit}: {e}")
        return posts

    def _fetch_via_requests(self, subreddit: str, limit: int) -> List[Dict]:
        """Fetch posts using requests + OAuth."""
        token = self._reddit_oauth_token()
        if not token:
            return []

        posts = []
        try:
            headers = {
                "Authorization": f"Bearer {token}",
                "User-Agent": self._user_agent,
            }
            url = f"https://oauth.reddit.com/r/{subreddit}/new" f"?limit={limit}"
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            for child in data.get("data", {}).get("children", []):
                p = child.get("data", {})
                text = f"{p.get('title', '')} {p.get('selftext', '')}"
                posts.append(
                    {
                        "text": text,
                        "subreddit": subreddit,
                        "title": p.get("title", ""),
                        "score": p.get("score", 0),
                        "num_comments": p.get("num_comments", 0),
                        "created_utc": p.get("created_utc", 0),
                    }
                )
            time.sleep(RATE_LIMIT_SLEEP)
        except Exception as e:
            logger.warning(f"Reddit requests fetch failed for r/{subreddit}: {e}")
        return posts

    def get_currency_sentiment(
        self, keywords_by_currency: Dict[str, List[str]]
    ) -> Dict[str, float]:
        """Score posts by currency sentiment.

        Args:
            keywords_by_currency: Dict mapping currency code to list of
                detection keywords, e.g.
                {"EUR": ["euro", "ecb"], "USD": ["dollar", "fed"]}

        Returns:
            Dict mapping currency code to sentiment score in [-1, 1].
        """
        if not self._available:
            return {c: 0.0 for c in keywords_by_currency}

        posts = self.search_forex_posts(limit=50)
        if not posts:
            return {c: 0.0 for c in keywords_by_currency}

        # Collect weighted scores per currency
        currency_scores: Dict[str, List[float]] = {c: [] for c in keywords_by_currency}
        currency_weights: Dict[str, List[float]] = {c: [] for c in keywords_by_currency}

        for post in posts:
            text = post.get("text", "")
            if not text.strip():
                continue

            score = self._score_text(text)
            detected = self._detect_currencies(text, keywords_by_currency)

            if not detected:
                continue

            # Weight by post quality signals
            weight = 1.0
            # Upvote score influence (saturate at 100)
            upvotes = post.get("score", 0)
            weight *= 0.5 + 0.5 * min(max(upvotes, 0), 100) / 100
            # Comment count influence (saturate at 50)
            comments = post.get("num_comments", 0)
            weight *= 0.5 + 0.5 * min(comments, 50) / 50

            for c in detected:
                currency_scores[c].append(score * weight)
                currency_weights[c].append(weight)

        # Weighted average per currency
        result: Dict[str, float] = {}
        for c in keywords_by_currency:
            if currency_scores[c] and currency_weights[c]:
                total_weight = sum(currency_weights[c])
                if total_weight > 0:
                    result[c] = sum(currency_scores[c]) / total_weight
                else:
                    result[c] = 0.0
            else:
                result[c] = 0.0

        # Clamp to [-1, 1]
        for c in result:
            result[c] = float(max(-1.0, min(1.0, result[c])))

        logger.debug(f"Reddit currency sentiment: {result}")
        return result

    def _score_text(self, text: str) -> float:
        """Simple positive/negative lexicon scoring for Reddit text.

        Returns a score in [-1, 1].
        """
        words = set(text.lower().split())
        pos_count = len(words & _POSITIVE_WORDS)
        neg_count = len(words & _NEGATIVE_WORDS)
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        return (pos_count - neg_count) / max(total, 1)

    @staticmethod
    def _detect_currencies(
        text: str,
        keywords_by_currency: Dict[str, List[str]],
    ) -> List[str]:
        """Detect which currencies are mentioned in text."""
        text_lower = text.lower()
        detected = []
        for currency, keywords in keywords_by_currency.items():
            for kw in keywords:
                if kw in text_lower:
                    detected.append(currency)
                    break
        return detected

    @property
    def is_available(self) -> bool:
        return self._available


class TwitterScraper:
    """Scraper for Twitter/X sentiment data with rate limiting and graceful degradation.

    Uses tweepy if available, otherwise falls back to requests with bearer token
    for the Twitter API v2. Responds gracefully to missing credentials, network
    errors, and rate limits — never raises, always returns empty dicts on failure.
    """

    def __init__(self, bearer_token: Optional[str] = None):
        self._lock = threading.Lock()
        self._last_request_time: float = 0.0
        self._min_request_interval: float = 15.0
        self._bearer_token: Optional[str] = None
        self._tweepy_client = None
        self._available = False

        secrets = self._load_secrets(bearer_token)
        if not secrets:
            logger.debug(
                "TwitterScraper: No credentials available, running in degraded mode"
            )
            return

        bearer_token_val, api_key, api_secret = secrets

        if TWEEPY_AVAILABLE and api_key and api_secret:
            try:
                auth = tweepy.AppAuthHandler(api_key, api_secret)
                self._tweepy_client = tweepy.API(auth, wait_on_rate_limit=True)
                self._available = True
                logger.debug("TwitterScraper initialized with tweepy")
                return
            except Exception as e:
                logger.debug(f"TwitterScraper: tweepy init failed: {e}")

        if bearer_token_val:
            self._bearer_token = bearer_token_val
            self._available = True
            logger.debug("TwitterScraper initialized with requests (bearer token)")
        else:
            logger.debug("TwitterScraper: No valid credentials found")

    def _load_secrets(
        self, bearer_token: Optional[str]
    ) -> Optional[Tuple[str, str, str]]:
        """Load Twitter credentials from Secrets or env vars."""
        try:
            from infrastructure.secrets import Secrets

            s = Secrets()
            bt = bearer_token or s.twitter_bearer_token
            ak = s.twitter_api_key
            as_ = s.twitter_api_secret
            if bt or (ak and as_):
                return (bt or "", ak or "", as_ or "")
        except Exception as e:
            logger.debug(f"TwitterScraper: Secrets load failed: {e}")

        bt = bearer_token or os.getenv("TWITTER_BEARER_TOKEN", "")
        ak = os.getenv("TWITTER_API_KEY", "")
        as_ = os.getenv("TWITTER_API_SECRET", "")
        if bt or (ak and as_):
            return (bt or "", ak or "", as_ or "")
        return None

    @property
    def available(self) -> bool:
        """Whether the scraper has valid credentials and can make requests."""
        return self._available

    def _rate_limit(self):
        """Enforce minimum interval between requests (thread-safe)."""
        with self._lock:
            elapsed = time.time() - self._last_request_time
            if elapsed < self._min_request_interval:
                sleep_time = self._min_request_interval - elapsed
                time.sleep(sleep_time)
            self._last_request_time = time.time()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search_forex_tweets(self, count: int = 50) -> List[Dict]:
        """Search for forex-related tweets.

        Returns list of dicts with keys:
            text, created_at, user_followers, retweet_count, like_count.
        Filters out retweets. Returns empty list on any failure.
        """
        if not self._available:
            return []

        all_tweets: List[Dict] = []
        per_query = max(1, count // len(FOREX_SEARCH_QUERIES))

        for query in FOREX_SEARCH_QUERIES:
            try:
                self._rate_limit()
                tweets = self._search_tweets(query, max_results=min(per_query, 100))
                all_tweets.extend(tweets)
                if len(all_tweets) >= count:
                    break
            except Exception as e:
                logger.warning(f"Twitter search failed for '{query}': {e}")
                continue

        all_tweets.sort(key=lambda t: t.get("retweet_count", 0), reverse=True)
        return all_tweets[:count]

    def get_currency_sentiment(
        self, keywords_by_currency: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, float]:
        """Get sentiment scores per currency from Twitter.

        Args:
            keywords_by_currency: Dict mapping currency code to list of keywords.
                If None, uses DEFAULT_CURRENCY_KEYWORDS.

        Returns:
            Dict mapping currency code to sentiment score in [-1, 1].
            Returns empty dict on any failure or if scraper is unavailable.
        """
        if not self._available:
            return {}

        kw_map = keywords_by_currency or DEFAULT_CURRENCY_KEYWORDS
        results: Dict[str, float] = {}

        for currency, keywords in kw_map.items():
            try:
                query_parts = [f"({kw})" for kw in keywords[:3]]
                query = f"({' OR '.join(query_parts)}) -is:retweet lang:en"

                self._rate_limit()
                tweets = self._search_tweets(query, max_results=30)

                if not tweets:
                    results[currency] = 0.0
                    continue

                score = self._compute_lexicon_sentiment([t["text"] for t in tweets])
                results[currency] = score

            except Exception as e:
                logger.warning(f"Twitter sentiment failed for {currency}: {e}")
                results[currency] = 0.0
                continue

        return results

    def get_health(self) -> Dict[str, bool]:
        """Simple health check."""
        return {
            "available": self._available,
            "tweepy": self._tweepy_client is not None,
            "bearer_token": self._bearer_token is not None,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _search_tweets(self, query: str, max_results: int = 50) -> List[Dict]:
        """Execute a single tweet search via available backend."""
        if self._tweepy_client:
            return self._search_tweepy(query, max_results)
        return self._search_requests(query, max_results)

    def _search_tweepy(self, query: str, max_results: int) -> List[Dict]:
        """Search using tweepy (v1.1 API)."""
        assert self._tweepy_client is not None
        try:
            results = self._tweepy_client.search_tweets(
                q=query,
                count=max_results,
                tweet_mode="extended",
                result_type="mixed",
            )
            tweets = []
            for status in results:
                if hasattr(status, "retweeted_status") and status.retweeted_status:
                    continue
                text = getattr(status, "full_text", getattr(status, "text", ""))
                tweets.append(
                    {
                        "text": text,
                        "created_at": str(getattr(status, "created_at", "")),
                        "user_followers": getattr(
                            getattr(status, "user", None), "followers_count", 0
                        ),
                        "retweet_count": getattr(status, "retweet_count", 0),
                        "like_count": getattr(status, "favorite_count", 0),
                    }
                )
            return tweets
        except Exception as e:
            logger.warning(f"tweepy search failed: {e}")
            return []

    def _search_requests(self, query: str, max_results: int) -> List[Dict]:
        """Search using requests with Twitter API v2 recent search endpoint."""
        if not self._bearer_token:
            return []

        url = "https://api.twitter.com/2/tweets/search/recent"
        headers = {"Authorization": f"Bearer {self._bearer_token}"}
        params: Dict[str, Any] = {
            "query": query,
            "max_results": min(max_results, 100),
            "tweet.fields": "created_at,public_metrics,author_id",
            "expansions": "author_id",
            "user.fields": "public_metrics",
        }

        try:
            resp = requests.get(url, headers=headers, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            users = {}
            if "includes" in data and "users" in data["includes"]:
                for u in data["includes"]["users"]:
                    users[u["id"]] = u.get("public_metrics", {}).get(
                        "followers_count", 0
                    )

            tweets = []
            for tweet in data.get("data", []):
                text = tweet.get("text", "")
                if text.startswith("RT @"):
                    continue
                metrics = tweet.get("public_metrics", {})
                author_id = tweet.get("author_id", "")
                tweets.append(
                    {
                        "text": text,
                        "created_at": tweet.get("created_at", ""),
                        "user_followers": users.get(author_id, 0),
                        "retweet_count": metrics.get("retweet_count", 0),
                        "like_count": metrics.get("like_count", 0),
                    }
                )
            return tweets

        except requests.exceptions.HTTPError as e:
            status_code = getattr(e.response, "status_code", 0)
            if status_code == 429:
                logger.warning("Twitter API rate limited (429), backing off")
            elif status_code == 401:
                logger.warning("Twitter API auth failed (401), disabling scraper")
                self._available = False
            else:
                logger.warning(f"Twitter API HTTP error: {e}")
            return []
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Twitter API connection error: {e}")
            return []
        except requests.exceptions.Timeout:
            logger.warning("Twitter API request timed out")
            return []
        except Exception as e:
            logger.warning(f"Twitter API request failed: {e}")
            return []

    @staticmethod
    def _compute_lexicon_sentiment(texts: List[str]) -> float:
        """Simple lexicon-based sentiment scoring for tweet texts."""
        if not texts:
            return 0.0

        scores = []
        for text in texts:
            words = set(text.lower().split())
            pos = len(words & _TWITTER_POSITIVE_WORDS)
            neg = len(words & _TWITTER_NEGATIVE_WORDS)
            total = pos + neg
            scores.append((pos - neg) / max(total, 1))

        return sum(scores) / len(scores)


# ══════════════════════════════════════════════════════════════════════
# NASA Satellite Alternative Data Provider
# ══════════════════════════════════════════════════════════════════════


class NASASatelliteDataProvider:
    """
    Fetches alternative data from NASA APIs that may correlate with forex markets.

    Data sources used (both free / no API key required for basic access):

      1. **EONET** (Earth Observatory Natural Event Tracker)
         https://eonet.gsfc.nasa.gov/api/v3/events
         → Natural disasters near financial centres → short-term currency impacts.

      2. **NASA POWER** (Prediction of Worldwide Energy Resources)
         https://power.larc.nasa.gov/api/temporal/monthly/point
         → Temperature / precipitation for key agricultural regions
         → Impacts commodity-currency pairs (AUD, NZD, CAD, USD).

    API-key rate limits are handled gracefully (uses DEMO_KEY by default).
    All results are cached for 6 hours because NASA data changes very slowly.

    Usage::

        provider = NASASatelliteDataProvider()
        score = provider.compute_satellite_score()   # 0.0 – 100.0
        impacts = provider.get_currency_impact()     # per-currency dict

    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or "DEMO_KEY"
        # Internal LRU-like cache: key -> (value, timestamp)
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._session: Any = None
        if REQUESTS_AVAILABLE:
            self._session = requests.Session()
            self._session.headers.update(
                {"User-Agent": "RTS-AI-Forex-System/1.0 (nasa-satellite-provider)"}
            )

    # ── Public API ──────────────────────────────────────────────────

    def compute_satellite_score(self) -> float:
        """Combined satellite risk score: **0.0** (benign) to **100.0** (catastrophic).

        Components:

        * **EONET natural disasters** (0–50) — events near financial centres.
        * **Agricultural anomalies** (0–50) — temperature / precipitation deviation
          in key growing regions.

        Returns ``0.0`` if all APIs are unreachable.
        """
        events = self._fetch_eonet_events()
        eonet_score = self._score_eonet_events(events)  # 0–50
        ag_score = self._score_agricultural_anomalies()  # 0–50
        total = eonet_score + ag_score
        return min(100.0, max(0.0, total))

    def get_currency_impact(self) -> Dict[str, float]:
        """Per-currency disruption scores **0.0–1.0** (higher = worse).

        Based on detected natural events geographically mapped to financial
        centres that trade each currency pair.
        """
        impact: Dict[str, float] = {
            c: 0.0 for c in ["USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF"]
        }
        events = self._fetch_eonet_events()
        if not events:
            return impact

        for event in events:
            weight = self._event_weight(event)
            if weight <= 0:
                continue
            geometries = event.get("geometry", [])
            if not geometries:
                continue
            last_geo = geometries[-1]
            coords = last_geo.get("coordinates", [])
            if len(coords) < 2:
                continue
            event_lon, event_lat = coords[:2]

            for centre in FINANCIAL_CENTRES:
                d = self._haversine(event_lat, event_lon, centre["lat"], centre["lon"])
                if d < centre["radius_km"]:
                    proximity = 1.0 - (d / centre["radius_km"])
                    for currency in centre["currencies"]:
                        impact[currency] = max(impact[currency], proximity * weight)

        return impact

    # ── EONET ────────────────────────────────────────────────────────

    def _fetch_eonet_events(self) -> List[Dict]:
        """Return cached or freshly-fetched list of current natural events."""
        cached = self._cache_get("eonet_events")
        if cached is not None:
            return cached  # type: ignore[no-any-return]

        if not REQUESTS_AVAILABLE or self._session is None:
            return []

        try:
            url = "https://eonet.gsfc.nasa.gov/api/v3/events"
            params: Dict[str, Any] = {
                "status": "open",
                "limit": 100,
            }
            resp = self._session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            events: List[Dict] = data.get("events", [])
            self._cache_set("eonet_events", events)
            logger.info(f"NASA EONET: fetched {len(events)} open events")
            return events
        except Exception as exc:
            logger.warning(f"NASA EONET fetch failed: {exc}")
            return []

    @staticmethod
    def _event_weight(event: Dict) -> float:
        """Return impact weight (0–1) for an EONET event based on its categories."""
        categories = event.get("categories", [])
        if not categories:
            return 0.0
        best = 0.0
        for cat in categories:
            cat_id: str = cat.get("id", "")
            for known_id, w in EONET_WEIGHTS.items():
                if known_id.lower() in cat_id.lower():
                    best = max(best, w)
        return best if best > 0 else 0.08

    def _score_eonet_events(self, events: List[Dict]) -> float:
        """Score 0–50 based on events close to financial centres."""
        if not events:
            return 0.0

        total_impact = 0.0
        for event in events:
            weight = self._event_weight(event)
            if weight <= 0:
                continue
            geometries = event.get("geometry", [])
            if not geometries:
                continue
            last_geo = geometries[-1]
            coords = last_geo.get("coordinates", [])
            if len(coords) < 2:
                continue
            event_lon, event_lat = coords[:2]

            for centre in FINANCIAL_CENTRES:
                d = self._haversine(event_lat, event_lon, centre["lat"], centre["lon"])
                if d < centre["radius_km"]:
                    proximity = 1.0 - (d / centre["radius_km"])
                    total_impact += proximity * weight

        # Scale: each moderate event near a centre gives ~10 pts, cap at 50
        return min(50.0, total_impact * 10.0)

    # ── NASA POWER (agricultural) ───────────────────────────────────

    def _fetch_power(self, lat: float, lon: float) -> Optional[Dict]:
        """Fetch POWER monthly data for a single point.  Cached per lat/lon."""
        key = f"power_{lat:.1f}_{lon:.1f}"
        cached = self._cache_get(key)
        if cached is not None:
            return cached  # type: ignore[no-any-return]

        if not REQUESTS_AVAILABLE or self._session is None:
            return None

        try:
            url = "https://power.larc.nasa.gov/api/temporal/monthly/point"
            params: Dict[str, Any] = {
                "parameters": "T2M,PRECTOTCORR",
                "community": "AG",
                "longitude": lon,
                "latitude": lat,
                "start": 2024,
                "end": 2025,
                "format": "JSON",
            }
            resp = self._session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data: Dict = resp.json()
            self._cache_set(key, data)
            return data
        except Exception as exc:
            logger.debug(f"NASA POWER fetch failed for ({lat:.1f}, {lon:.1f}): {exc}")
            return None

    def _score_agricultural_anomalies(self) -> float:
        """Score 0–50: average temperature/precipitation anomaly across ag regions."""
        scores: List[float] = []
        for region in AG_REGIONS:
            data = self._fetch_power(region["lat"], region["lon"])
            if data is None:
                continue
            anomaly = self._region_anomaly(data)
            scores.append(anomaly)

        if not scores:
            return 0.0
        avg = sum(scores) / len(scores)
        # anomaly ~2.0 → 50 pts, scale linearly
        return min(50.0, avg * 25.0)

    @staticmethod
    def _region_anomaly(data: Dict) -> float:
        """Return 0–2 anomaly for a single POWER response."""
        try:
            props = data.get("properties", {})
            params = props.get("parameter", {})
            if not params:
                return 0.0

            temp_vals: Dict[str, float] = params.get("T2M", {})
            precip_vals: Dict[str, float] = params.get("PRECTOTCORR", {})

            # Average of last 3 monthly values
            temp_avgs = list(temp_vals.values())[-3:] if temp_vals else []
            precip_avgs = list(precip_vals.values())[-3:] if precip_vals else []

            if not temp_avgs and not precip_avgs:
                return 0.0

            anomaly = 0.0
            count = 0

            if temp_avgs:
                avg_t = sum(temp_avgs) / len(temp_avgs)
                # "normal" ag temp ~25 °C; scale 15 °C deviation → 1.0
                anomaly += abs(avg_t - 25.0) / 15.0
                count += 1

            if precip_avgs:
                avg_p = sum(precip_avgs) / len(precip_avgs)
                # "normal" monthly precip ~100 mm; scale 100 mm deviation → 1.0
                anomaly += abs(avg_p - 100.0) / 100.0
                count += 1

            return anomaly / max(count, 1)
        except Exception as exc:
            logger.debug(f"Region anomaly calc failed: {exc}")
            return 0.0

    # ── Caching ──────────────────────────────────────────────────────

    def _cache_get(self, key: str) -> Any:
        entry = self._cache.get(key)
        if entry is None:
            return None
        value, ts = entry
        if time.time() - ts > SATELLITE_CACHE_TTL:
            del self._cache[key]
            return None
        return value

    def _cache_set(self, key: str, value: Any) -> None:
        self._cache[key] = (value, time.time())

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Great-circle distance in kilometres between two lat/lon points."""
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return 6371.0 * c
