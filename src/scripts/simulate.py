"""Unified simulation engine with sentiment, ensemble, and risk management."""
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rts_ai_fx.data import DataFetcher
from rts_ai_fx.model import LSTMCNNHybrid, ProfitabilityClassifier
from rts_ai_fx.ensemble import MoEEnsemble, Expert
from rts_ai_fx.regime_detector import HMMRegimeDetector
from rts_ai_fx.features_unified import FeaturePipeline
from risk.manager import RiskManager, RiskParameters
from ai.sentiment import SentimentAnalyzer
import pickle
import numpy as np
import pandas as pd


@dataclass
class SimulationResult:
    """Results from a simulation run."""
    pair: str
    initial_balance: float
    final_balance: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    profit_loss: float
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    trades: List[dict] = field(default_factory=list)


class SimulationEngine:
    """Consolidated simulation engine with all improvements."""
    
    def __init__(
        self,
        pairs: List[str],
        lookback: int = 30,
        use_sentiment: bool = True,
        use_regime_detection: bool = True,
    ):
        self.pairs = pairs
        self.lookback = lookback
        self.use_sentiment = use_sentiment
        self.use_regime_detection = use_regime_detection
        
        self.ensembles: Dict[str, MoEEnsemble] = {}
        self.preprocessors: Dict[str, FeaturePipeline] = {}
        self.risk_managers: Dict[str, RiskManager] = {}
        self.regime_detector: Optional[HMMRegimeDetector] = None
        self.sentiment_analyzer: Optional[SentimentAnalyzer] = None
        
        if use_sentiment:
            self.sentiment_analyzer = SentimentAnalyzer()
        
        if use_regime_detection:
            self.regime_detector = HMMRegimeDetector()
    
    def load_models(self, model_dir: str = "models"):
        """Load all models, preprocessors, and setup ensembles."""
        for pair in self.pairs:
            symbol_safe = pair.replace("=", "_")
            
            # Load models
            model_path = f"{model_dir}/{symbol_safe}_v2_lstm_transformer.keras"
            classifier_path = f"{model_dir}/{symbol_safe}_v2_classifier.keras"
            preprocessor_path = f"{model_dir}/{symbol_safe}_v2_preprocessor.pkl"
            
            if not (Path(model_path).exists() and Path(classifier_path).exists()):
                print(f"Missing models for {pair}, skipping...")
                continue
            
            try:
                model = LSTMCNNHybrid.load(model_path)
                classifier = ProfitabilityClassifier.load(classifier_path)
                
                # Setup ensemble with multiple experts
                ensemble = MoEEnsemble()
                ensemble.add_expert(
                    name=f"LSTM-CNN-{pair}",
                    predict_fn=model.predict,
                    confidence_fn=lambda X, c=classifier: float(c.predict_proba(X)[0][0]),
                    regime="ranging",
                )
                
                # Add more experts for different regimes
                ensemble.add_expert(
                    name=f"Trend-Expert-{pair}",
                    predict_fn=model.predict,
                    confidence_fn=lambda X, c=classifier: float(c.predict_proba(X)[0][0]) * 1.2,
                    regime="trending",
                )
                
                self.ensembles[pair] = ensemble
                
                # Load preprocessor
                with open(preprocessor_path, "rb") as f:
                    preprocessor = pickle.load(f)
                self.preprocessors[pair] = preprocessor
                
                # Setup risk manager
                risk_params = RiskParameters(
                    max_risk_per_trade=0.02,
                    max_drawdown=0.10,
                    max_positions=5,
                )
                self.risk_managers[pair] = RiskManager(risk_params, initial_balance=10000)
                
                print(f"Loaded {pair} with ensemble of {len(ensemble.experts)} experts")
                
            except Exception as e:
                print(f"Failed to load {pair}: {e}")
    
    def run_simulation(
        self,
        start_date: str,
        end_date: Optional[str] = None,
        initial_balance: float = 10000,
    ) -> Dict[str, SimulationResult]:
        """Run simulation with all enhancements."""
        print(f"\n{'='*60}")
        print(f"ENHANCED SIMULATION: {start_date} to {end_date or 'present'}")
        print(f"Features: Sentiment={self.use_sentiment}, Regime={self.use_regime_detection}")
        print(f"{'='*60}\n")
        
        fetcher = DataFetcher()
        results = {}
        
        for pair in self.ensembles.keys():
            print(f"\nProcessing {pair}...")
            
            # Fetch data
            df = fetcher.fetch_ohlcv(pair, "1d", start_date)
            if end_date:
                df = df[df.index <= end_date].copy()
            
            if len(df) < self.lookback + 10:
                print(f"Insufficient data for {pair}")
                continue
            
            # Get sentiment scores
            sentiment_scores = {}
            if self.sentiment_analyzer:
                snapshot = self.sentiment_analyzer.get_latest()
                sentiment_scores = snapshot.currency_scores
            
            # Detect regime
            regime = "ranging"
            if self.regime_detector:
                self.regime_detector.fit(df)
                regime = self.regime_detector.detect_regime(df)
                print(f"Current regime: {regime}")
            
            # Run ensemble prediction
            ensemble = self.ensembles[pair]
            preprocessor = self.preprocessors[pair]
            
            # Create feature sequences
            features, _ = preprocessor.create_sequences(
                {"1d": df}, pair, flatten=False
            )
            
            if features is None or len(features) == 0:
                print(f"No features generated for {pair}")
                continue
            
            # Generate signals
            signals = []
            for i in range(len(features)):
                X = features[i:i+1]
                pred = ensemble.predict(X, regime)
                signals.append(pred)
            
            # Calculate results
            result = self._calculate_results(
                pair, df, signals, initial_balance
            )
            results[pair] = result
            
            print(f"Completed {pair}: PnL={result.profit_loss:.2f}, Trades={result.total_trades}")
        
        return results
    
    def _calculate_results(
        self,
        pair: str,
        df: pd.DataFrame,
        signals: List,
        initial_balance: float,
    ) -> SimulationResult:
        """Calculate simulation results from signals."""
        balance = initial_balance
        trades = []
        wins = 0
        losses = 0
        
        for i, pred in enumerate(signals):
            if i >= len(df) - 1:
                break
            
            current_price = df["close"].iloc[i]
            next_price = df["close"].iloc[i + 1]
            
            # Simple trading logic based on ensemble direction
            if pred.direction == "BUY":
                pnl = (next_price - current_price) / current_price * balance * 0.01
                balance += pnl
                trades.append({"type": "BUY", "pnl": pnl, "price": current_price})
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1
            elif pred.direction == "SELL":
                pnl = (current_price - next_price) / current_price * balance * 0.01
                balance += pnl
                trades.append({"type": "SELL", "pnl": pnl, "price": current_price})
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1
        
        total_trades = wins + losses
        profit_loss = balance - initial_balance
        
        return SimulationResult(
            pair=pair,
            initial_balance=initial_balance,
            final_balance=balance,
            total_trades=total_trades,
            winning_trades=wins,
            losing_trades=losses,
            profit_loss=profit_loss,
            trades=trades,
        )
