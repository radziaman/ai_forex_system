"""Azure AI-powered sentiment analysis for forex news."""
import os
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class AzureSentimentConfig:
    """Configuration for Azure AI sentiment analysis."""
    endpoint: str = ""
    api_key: str = ""
    use_text_analytics: bool = True
    cache_ttl: int = 300
    max_articles: int = 50


class AzureSentimentAnalyzer:
    """Enhanced sentiment analysis using Azure AI Text Analytics."""
    
    def __init__(self, config: Optional[AzureSentimentConfig] = None):
        self.config = config or AzureSentimentConfig()
        self._client = None
        self._cache: Dict[str, Tuple[float, float]] = {}  # text -> (timestamp, score)
        
        if self.config.use_text_analytics:
            self._init_client()
    
    def _init_client(self):
        """Initialize Azure Text Analytics client."""
        try:
            from azure.ai.textanalytics import TextAnalyticsClient
            from azure.core.credentials import AzureKeyCredential
            
            endpoint = self.config.endpoint or os.getenv("AZURE_TEXT_ANALYTICS_ENDPOINT", "")
            api_key = self.config.api_key or os.getenv("AZURE_TEXT_ANALYTICS_KEY", "")
            
            if endpoint and api_key:
                self._client = TextAnalyticsClient(
                    endpoint=endpoint,
                    credential=AzureKeyCredential(api_key)
                )
        except ImportError:
            pass
    
    def analyze_texts(self, texts: List[str]) -> List[float]:
        """Analyze sentiment of multiple texts using Azure AI."""
        if not self._client or not texts:
            return [0.0] * len(texts)
        
        # Check cache first
        scores = []
        to_analyze = []
        indices = []
        
        for i, text in enumerate(texts):
            cached = self._get_cached(text)
            if cached is not None:
                scores.append(cached)
            else:
                scores.append(None)
                to_analyze.append(text)
                indices.append(i)
        
        if to_analyze:
            try:
                response = self._client.analyze_sentiment(to_analyze)
                for idx, result in zip(indices, response):
                    if not result.is_error:
                        score = self._convert_sentiment(result)
                        scores[idx] = score
                        self._cache[to_analyze[indices.index(idx)]] = (time.time(), score)
                    else:
                        scores[idx] = 0.0
            except Exception:
                for idx in indices:
                    scores[idx] = 0.0
        
        return scores
    
    def _convert_sentiment(self, result) -> float:
        """Convert Azure sentiment result to -1 to 1 score."""
        label = result.sentiment
        confidence = result.confidence_scores
        
        if label == "positive":
            return confidence.positive
        elif label == "negative":
            return -confidence.negative
        return 0.0
    
    def _get_cached(self, text: str) -> Optional[float]:
        """Get cached sentiment score if not expired."""
        if text in self._cache:
            timestamp, score = self._cache[text]
            if time.time() - timestamp < self.config.cache_ttl:
                return score
        return None
    
    def get_currency_sentiment_vector(self, currency_scores: Dict[str, float]) -> np.ndarray:
        """Convert currency sentiment scores to feature vector."""
        currencies = ["EUR", "USD", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]
        vec = []
        for c in currencies:
            vec.append(currency_scores.get(c, 0.0))
        return np.array(vec, dtype=np.float32)
