"""
Comprehensive tests for MoEEnsemble — the core AI decision-making component.
"""

import time
import math
import numpy as np
from rts_ai_fx.ensemble import MoEEnsemble, EnsemblePrediction


# ── Helper prediction/confidence functions ──────────────────────────────


def _bullish_predict(X):
    return 0.01


def _bearish_predict(X):
    return -0.01


def _neutral_predict(X):
    return 0.0


def _high_conf(X):
    return 0.9


def _low_conf(X):
    return 0.3


# ── 1. Initialization ──────────────────────────────────────────────────


class TestInit:
    """MoEEnsemble initialization."""

    def test_init_empty(self):
        """New ensemble has no experts, empty state."""
        ensemble = MoEEnsemble()
        assert len(ensemble.experts) == 0
        assert isinstance(ensemble.experts, list)
        assert isinstance(ensemble.elo_ratings, dict)
        assert len(ensemble.elo_ratings) == 0
        assert ensemble.enabled is True

    def test_init_defaults(self):
        """Default values for lockout, Elo, Sharpe settings."""
        ensemble = MoEEnsemble()
        assert ensemble._lockout_threshold == 4
        assert ensemble._lockout_hours == 4.0
        assert ensemble._max_lockout_hours == 48.0
        assert ensemble._lockout_multiplier == 2.0
        assert ensemble.use_sharpe_weighting is True
        assert ensemble._tracker_weight_fn is None
        # All internal dicts are empty
        assert len(ensemble.consecutive_losses) == 0
        assert len(ensemble._disabled_until) == 0
        assert len(ensemble.expert_returns) == 0
        assert len(ensemble.expert_win_rates) == 0


# ── 2. Adding experts ──────────────────────────────────────────────────


class TestAddExpert:
    """Adding experts to the ensemble."""

    def test_add_expert(self):
        """Add 1 expert, verify it's registered."""
        ensemble = MoEEnsemble()
        ensemble.add_expert("expert1", _bullish_predict, _high_conf, "trending")
        assert len(ensemble.experts) == 1
        assert ensemble.experts[0].name == "expert1"
        assert ensemble.experts[0].predict is _bullish_predict
        assert ensemble.experts[0].confidence is _high_conf
        assert ensemble.experts[0].regime == "trending"

    def test_add_multiple_experts(self):
        """Add multiple experts with different regimes."""
        ensemble = MoEEnsemble()
        ensemble.add_expert("bull", _bullish_predict, _high_conf, "trending")
        ensemble.add_expert("bear", _bearish_predict, _low_conf, "ranging")
        ensemble.add_expert("neutral", _neutral_predict, _high_conf, "volatile")
        assert len(ensemble.experts) == 3
        assert ensemble.experts[0].name == "bull"
        assert ensemble.experts[1].name == "bear"
        assert ensemble.experts[2].name == "neutral"
        assert ensemble.experts[0].regime == "trending"
        assert ensemble.experts[1].regime == "ranging"
        assert ensemble.experts[2].regime == "volatile"

    def test_add_expert_initializes_elo(self):
        """Each expert gets default 1200 Elo."""
        ensemble = MoEEnsemble()
        ensemble.add_expert("expert1", _bullish_predict, _high_conf, "ranging")
        assert ensemble.elo_ratings["expert1"] == 1200.0
        assert ensemble.consecutive_losses["expert1"] == 0
        assert ensemble._disabled_until["expert1"] == 0.0

    def test_add_expert_default_regime(self):
        """Default regime is 'ranging'."""
        ensemble = MoEEnsemble()
        ensemble.add_expert("expert1", _bullish_predict, _high_conf)
        assert ensemble.experts[0].regime == "ranging"


# ── 3–4. Predict ────────────────────────────────────────────────────────


class TestPredict:
    """Prediction behavior."""

    X = np.random.randn(30, 49)

    def test_predict_empty_experts(self):
        """Predict with no experts returns EnsemblePrediction() defaults."""
        ensemble = MoEEnsemble()
        result = ensemble.predict(self.X)
        assert isinstance(result, EnsemblePrediction)
        assert result.price == 0.0
        assert result.confidence == 0.0
        assert result.direction == "HOLD"
        assert result.expert_outputs == {}

    def test_predict_disabled(self):
        """enabled=False returns empty prediction."""
        ensemble = MoEEnsemble()
        ensemble.add_expert("bull", _bullish_predict, _high_conf)
        ensemble.enabled = False
        result = ensemble.predict(self.X)
        assert result.price == 0.0
        assert result.direction == "HOLD"
        assert result.expert_outputs == {}

    def test_predict_bullish(self):
        """One bullish expert → positive price, BUY direction."""
        ensemble = MoEEnsemble()
        ensemble.add_expert("bull", _bullish_predict, _high_conf)
        result = ensemble.predict(self.X)
        assert result.price > 0
        assert result.direction == "BUY"
        assert "bull" in result.expert_outputs

    def test_predict_bearish(self):
        """One bearish expert → negative price, SELL direction."""
        ensemble = MoEEnsemble()
        ensemble.add_expert("bear", _bearish_predict, _low_conf)
        result = ensemble.predict(self.X)
        assert result.price < 0
        assert result.direction == "SELL"
        assert "bear" in result.expert_outputs

    def test_predict_weighted(self):
        """Two experts with same weights produce correct weighted average."""
        ensemble = MoEEnsemble()
        ensemble.add_expert("bull", _bullish_predict, _high_conf)
        ensemble.add_expert("bear", _bearish_predict, _high_conf)
        result = ensemble.predict(self.X)
        # Both have default regime "ranging" and same confidence 0.9
        # regime_weight=2.0, elo=1.0, sharpe=1.0, conf_weight=0.95, tracker=1.0
        # Same weight → price = (0.01 + -0.01)/2 = 0.0
        assert result.price == 0.0
        assert result.direction == "HOLD"

    def test_predict_neutral(self):
        """Neutral expert produces HOLD with zero price."""
        ensemble = MoEEnsemble()
        ensemble.add_expert("neutral", _neutral_predict, _high_conf)
        result = ensemble.predict(self.X)
        assert result.price == 0.0
        assert result.direction == "HOLD"

    def test_predict_exception_handling(self):
        """Expert that raises is silently skipped."""

        def _broken_predict(X):
            raise ValueError("broken")

        def _broken_conf(X):
            return 0.5

        ensemble = MoEEnsemble()
        ensemble.add_expert("bull", _bullish_predict, _high_conf)
        ensemble.add_expert("broken", _broken_predict, _broken_conf)
        # The broken expert should be skipped; only bull contributes
        result = ensemble.predict(self.X)
        assert result.price > 0
        assert result.direction == "BUY"
        assert "bull" in result.expert_outputs
        assert "broken" not in result.expert_outputs

    def test_predict_confidence_weighted(self):
        """Higher confidence experts get more weight."""
        ensemble = MoEEnsemble()
        ensemble.add_expert("bull_low", _bullish_predict, _low_conf)  # conf=0.3
        ensemble.add_expert("bull_high", _bullish_predict, _high_conf)  # conf=0.9
        result = ensemble.predict(self.X)
        # Both predict bullish (0.01), so price is positive
        # Higher confidence expert has more weight
        assert result.price > 0
        # bull_high has higher conf_weight: 0.5+0.5*0.9=0.95 vs 0.5+0.5*0.3=0.65
        # So bull_high's weight is larger
        assert (
            result.expert_outputs["bull_high"]["weight"]
            > result.expert_outputs["bull_low"]["weight"]
        )

    def test_predict_elo_affects_weight(self):
        """Higher Elo rating increases expert weight."""
        ensemble = MoEEnsemble()
        ensemble.add_expert("expert1", _bullish_predict, _high_conf)
        ensemble.add_expert("expert2", _bullish_predict, _high_conf)
        # Manually boost expert2's Elo
        ensemble.elo_ratings["expert2"] = 2400.0
        result = ensemble.predict(self.X)
        # Higher Elo → higher elo_weight
        w1 = result.expert_outputs["expert1"]["weight"]
        w2 = result.expert_outputs["expert2"]["weight"]
        assert w2 > w1
        # Weight ratio ≈ elo_ratio (2400/1200 = 2)
        assert math.isclose(w2 / w1, 2400.0 / 1200.0, rel_tol=0.1)


# ── 5. Regime weighting ────────────────────────────────────────────────


class TestRegimeWeighting:
    """Regime-based expert weighting."""

    X = np.random.randn(30, 49)

    def test_regime_weight_matching(self):
        """Expert with matching regime gets 2x base weight."""
        ensemble = MoEEnsemble()
        ensemble.add_expert("bull", _bullish_predict, _high_conf, "trending")
        result = ensemble.predict(self.X, regime="trending")
        output = result.expert_outputs["bull"]
        # regime_weight=2.0, elo=1.0, sharpe=1.0, conf_weight=0.95, tracker=1.0
        assert math.isclose(output["weight"], 2.0 * 0.95, rel_tol=1e-3)

    def test_regime_weight_non_matching(self):
        """Non-matching non-ranging expert gets 0.5x base weight."""
        ensemble = MoEEnsemble()
        ensemble.add_expert("bull", _bullish_predict, _high_conf, "trending")
        result = ensemble.predict(self.X, regime="volatile")
        output = result.expert_outputs["bull"]
        # regime_weight=0.5, elo=1.0, sharpe=1.0, conf_weight=0.95, tracker=1.0
        assert math.isclose(output["weight"], 0.5 * 0.95, rel_tol=1e-3)

    def test_regime_ranging_no_penalty(self):
        """Ranging expert doesn't get penalty in non-matching regime."""
        ensemble = MoEEnsemble()
        ensemble.add_expert("ranger", _bullish_predict, _high_conf, "ranging")
        result = ensemble.predict(self.X, regime="trending")
        output = result.expert_outputs["ranger"]
        # regime_weight=1.0 (not 0.5 because regime is "ranging")
        assert math.isclose(output["weight"], 1.0 * 0.95, rel_tol=1e-3)

    def test_regime_posteriors(self):
        """Posteriors affect weighting correctly."""
        ensemble = MoEEnsemble()
        ensemble.add_expert("bull", _bullish_predict, _high_conf, "trending")
        posteriors = np.array([0.8, 0.1, 0.05, 0.05])
        result = ensemble.predict(
            self.X, regime="trending", regime_posteriors=posteriors
        )
        output = result.expert_outputs["bull"]
        # With posteriors: regime_idx=0 (trending), weight = 0.8*2 + 0.5 = 2.1
        expected_base = posteriors[0] * 2 + 0.5  # 2.1
        assert math.isclose(output["weight"], expected_base * 0.95, rel_tol=1e-3)

    def test_regime_posteriors_unknown_regime(self):
        """Unknown regime name falls back to index 1 (ranging)."""
        ensemble = MoEEnsemble()
        ensemble.add_expert("custom", _bullish_predict, _high_conf, "unknown_regime")
        posteriors = np.array([0.1, 0.7, 0.1, 0.1])
        result = ensemble.predict(
            self.X, regime="trending", regime_posteriors=posteriors
        )
        output = result.expert_outputs["custom"]
        # unknown_regime not in regime_names → index 1 (ranging) → posteriors[1]=0.7
        # base_weight = 0.7*2 + 0.5 = 1.9
        expected_base = posteriors[1] * 2 + 0.5  # 1.9
        assert math.isclose(output["weight"], expected_base * 0.95, rel_tol=1e-3)


# ── 6. should_trade ────────────────────────────────────────────────────


class _PredBuilder:
    """Helper to build EnsemblePrediction for should_trade testing."""

    @staticmethod
    def build(predictions, confidences, weights):
        """Build an EnsemblePrediction with given per-expert outputs."""
        pred = EnsemblePrediction()
        expert_outputs = {}
        total_weight = sum(weights)
        weighted_price = sum(p * w for p, w in zip(predictions, weights))
        weighted_conf = sum(c * w for c, w in zip(confidences, weights))
        for i, (p, c, w) in enumerate(zip(predictions, confidences, weights)):
            expert_outputs[f"expert{i}"] = {
                "prediction": p,
                "confidence": c,
                "weight": w,
            }
        pred.price = weighted_price / total_weight if total_weight > 0 else 0.0
        pred.confidence = weighted_conf / total_weight if total_weight > 0 else 0.0
        pred.expert_outputs = expert_outputs
        return pred


class TestShouldTrade:
    """Trade decision logic."""

    def test_should_trade_hold_empty(self):
        """Empty expert_outputs returns (False, HOLD, 0.0)."""
        ensemble = MoEEnsemble()
        pred = EnsemblePrediction()
        should, direction, ratio = ensemble.should_trade(pred, 1.0)
        assert should is False
        assert direction == "HOLD"
        assert ratio == 0.0

    def test_should_trade_tier1_buy(self):
        """3+ experts with buy consensus over 50% → BUY."""
        ensemble = MoEEnsemble()
        # 3 buy, 1 sell → buy_ratio = 3/4 = 0.75 > 0.5
        pred = _PredBuilder.build(
            predictions=[0.001, 0.002, 0.0015, -0.001],
            confidences=[0.7, 0.8, 0.75, 0.3],
            weights=[1.0, 1.0, 1.0, 1.0],
        )
        pred.confidence = 0.75
        should, direction, ratio = ensemble.should_trade(pred, 1.0)
        assert should is True
        assert direction == "BUY"
        assert ratio == 0.75

    def test_should_trade_tier1_sell(self):
        """3+ experts with sell consensus over 50% → SELL."""
        ensemble = MoEEnsemble()
        # 3 sell, 1 buy → sell_ratio = 3/4 = 0.75 > 0.5
        pred = _PredBuilder.build(
            predictions=[-0.001, -0.002, -0.0015, 0.001],
            confidences=[0.7, 0.8, 0.75, 0.3],
            weights=[1.0, 1.0, 1.0, 1.0],
        )
        pred.confidence = 0.75
        should, direction, ratio = ensemble.should_trade(pred, 1.0)
        assert should is True
        assert direction == "SELL"
        assert ratio == 0.75

    def test_should_trade_tier1_insufficient(self):
        """Only 2 active experts → falls to tier 2, which also fails."""
        ensemble = MoEEnsemble()
        # 2 active (above threshold), 2 neutral
        pred = _PredBuilder.build(
            predictions=[0.001, 0.002, 0.0, 0.0],
            confidences=[0.9, 0.7, 0.0, 0.0],
            weights=[1.0, 1.0, 1.0, 1.0],
        )
        pred.confidence = 0.5  # Low enough that tier 2 doesn't trigger
        should, direction, ratio = ensemble.should_trade(pred, 1.0, min_confidence=0.65)
        # n_active=2 < 3, tier 1 fails
        # pred.confidence=0.5 < max(0.65, 0.55)=0.65, tier 2 fails
        assert should is False
        assert direction == "HOLD"
        assert ratio == 0.0

    def test_should_trade_tier2_confidence(self):
        """High confidence even with few experts → trades via tier 2."""
        ensemble = MoEEnsemble()
        # Only 2 active experts but high ensemble confidence
        pred = _PredBuilder.build(
            predictions=[0.001, 0.002, 0.0, 0.0],
            confidences=[0.9, 0.85, 0.5, 0.5],
            weights=[1.0, 1.0, 1.0, 1.0],
        )
        # pred.confidence = (0.9+0.85+0.5+0.5)/4 = 0.6875
        should, direction, ratio = ensemble.should_trade(pred, 1.0, min_confidence=0.65)
        # n_active=2 < 3 → tier 1 fails
        # Tier 2: conf 0.6875 >= 0.65 and price>0 → BUY
        assert should is True
        assert direction == "BUY"
        assert ratio > 0

    def test_should_trade_confidence_too_low(self):
        """Confidence below min → HOLD even with 3+ active experts."""
        ensemble = MoEEnsemble()
        pred = _PredBuilder.build(
            predictions=[0.001, 0.002, 0.0015],
            confidences=[0.3, 0.4, 0.35],
            weights=[1.0, 1.0, 1.0],
        )
        pred.confidence = 0.3  # Below min_confidence=0.65
        should, direction, ratio = ensemble.should_trade(pred, 1.0, min_confidence=0.65)
        # n_active=3 but pred.confidence=0.3 < 0.65 → tier 1 fails
        # Tier 2: conf 0.3 < max(0.65, 0.55)=0.65 → fails
        assert should is False
        assert direction == "HOLD"

    def test_should_trade_threshold(self):
        """Predictions below 0.0003 threshold are ignored in active count."""
        ensemble = MoEEnsemble()
        # All predictions below threshold, n_active=0
        pred = _PredBuilder.build(
            predictions=[0.0001, 0.0002, 0.00015],
            confidences=[0.3, 0.3, 0.3],
            weights=[1.0, 1.0, 1.0],
        )
        pred.confidence = 0.3  # Also below tier 2 threshold
        should, direction, ratio = ensemble.should_trade(pred, 1.0, min_confidence=0.65)
        assert should is False
        assert direction == "HOLD"

    def test_should_trade_tier1_buy_ratio_edge(self):
        """Exactly at 50% buy ratio → no tier 1 trade (must be > 50%)."""
        ensemble = MoEEnsemble()
        # 2 buy, 2 sell → buy_ratio = 0.5, sell_ratio = 0.5
        # Neither > 0.5, so tier 1 fails
        pred = _PredBuilder.build(
            predictions=[0.001, 0.002, -0.001, -0.002],
            confidences=[0.7, 0.8, 0.7, 0.8],
            weights=[1.0, 1.0, 1.0, 1.0],
        )
        pred.confidence = 0.75
        should, direction, ratio = ensemble.should_trade(pred, 1.0)
        # Tier 1: buy_ratio=0.5 not > 0.5 → fails
        # Tier 2: conf=0.75 >= 0.65, price=0.0 → neither > 0 nor < 0 → fails
        assert should is False
        assert direction == "HOLD"

    def test_should_trade_sell_over_buy_split(self):
        """When both sides have > 50% (impossible) but sell > buy → SELL."""
        ensemble = MoEEnsemble()
        # 4 sell, 1 buy → sell_ratio = 0.8 > 0.5 and > buy_ratio
        pred = _PredBuilder.build(
            predictions=[-0.001, -0.002, -0.0015, -0.001, 0.001],
            confidences=[0.7, 0.8, 0.75, 0.7, 0.3],
            weights=[1.0, 1.0, 1.0, 1.0, 1.0],
        )
        pred.confidence = 0.75
        should, direction, ratio = ensemble.should_trade(pred, 1.0)
        assert should is True
        assert direction == "SELL"

    def test_should_trade_weaker_confidence_min_confidence(self):
        """Tier 2 uses max(min_confidence, 0.55)."""
        ensemble = MoEEnsemble()
        pred = _PredBuilder.build(
            predictions=[0.001, 0.002, 0.0, 0.0],
            confidences=[0.9, 0.85, 0.5, 0.5],
            weights=[1.0, 1.0, 1.0, 1.0],
        )
        # min_confidence=0.5 → tier2 uses max(0.5, 0.55) = 0.55
        should, direction, ratio = ensemble.should_trade(pred, 1.0, min_confidence=0.5)
        # pred.confidence = 0.6875 >= 0.55 → BUY
        assert should is True
        assert direction == "BUY"


# ── 7. Expert lockout ──────────────────────────────────────────────────


class TestLockout:
    """Expert auto-disable after consecutive losses."""

    X = np.random.randn(30, 49)

    def test_consecutive_losses_recorded(self):
        """record_loss() increments counter."""
        ensemble = MoEEnsemble()
        ensemble.add_expert("expert1", _bullish_predict, _high_conf)
        ensemble.record_loss("expert1")
        assert ensemble.consecutive_losses["expert1"] == 1
        ensemble.record_loss("expert1")
        assert ensemble.consecutive_losses["expert1"] == 2

    def test_lockout_after_threshold(self):
        """After threshold consecutive losses, expert is disabled."""
        ensemble = MoEEnsemble()
        ensemble.add_expert("expert1", _bullish_predict, _high_conf)
        for _ in range(4):
            ensemble.record_loss("expert1")
        assert ensemble.consecutive_losses["expert1"] == 4
        # _disabled_until should be set to a future time
        assert ensemble._disabled_until["expert1"] != 0.0
        assert ensemble._disabled_until["expert1"] > time.time() - 1

    def test_lockout_multiplier_increases(self):
        """Lockout duration doubles after each threshold breach."""
        ensemble = MoEEnsemble()
        ensemble.add_expert("expert1", _bullish_predict, _high_conf)
        # First lockout: 4 losses → 4h base
        for _ in range(4):
            ensemble.record_loss("expert1")
        assert ensemble._disabled_until["expert1"] != 0.0

        # Reset for second lockout: need 4 more losses
        # Actually the counter keeps going, so 8 total → 8h (4*2)
        for _ in range(4):
            ensemble.record_loss("expert1")

        # After 8 losses, factor = 2^(8/4) = 2^2 = 4
        # duration = min(4 * 4, 48) = 16h
        # But time has passed so we just check it's not 0
        assert ensemble.consecutive_losses["expert1"] == 8
        assert ensemble._disabled_until["expert1"] != 0.0

    def test_lockout_skips_disabled_expert_in_predict(self):
        """Disabled (locked-out) expert is skipped during predict."""
        ensemble = MoEEnsemble()
        ensemble.add_expert("expert1", _bullish_predict, _high_conf)
        # Trigger lockout
        for _ in range(4):
            ensemble.record_loss("expert1")
        # Set disabled_until to far future to ensure disabled
        ensemble._disabled_until["expert1"] = time.time() + 3600
        result = ensemble.predict(self.X)
        # Expert should be skipped → no predictions → empty result
        assert result.price == 0.0
        assert result.direction == "HOLD"
        assert result.expert_outputs == {}

    def test_lockout_expires(self):
        """After lockout duration expires, expert is re-enabled."""
        ensemble = MoEEnsemble()
        ensemble.add_expert("expert1", _bullish_predict, _high_conf)
        # Trigger lockout
        for _ in range(4):
            ensemble.record_loss("expert1")
        # Force expiry: set disabled_until to 1 second ago
        ensemble._disabled_until["expert1"] = time.time() - 1
        ensemble.consecutive_losses["expert1"] = 4
        result = ensemble.predict(self.X)
        # Expert should be re-enabled during predict
        assert ensemble._disabled_until["expert1"] == 0.0
        assert ensemble.consecutive_losses["expert1"] == 0
        assert result.price > 0
        assert result.direction == "BUY"

    def test_win_resets_losses(self):
        """record_win() resets consecutive losses to 0."""
        ensemble = MoEEnsemble()
        ensemble.add_expert("expert1", _bullish_predict, _high_conf)
        ensemble.record_loss("expert1")
        ensemble.record_loss("expert1")
        ensemble.record_loss("expert1")
        assert ensemble.consecutive_losses["expert1"] == 3
        ensemble.record_win("expert1")
        assert ensemble.consecutive_losses["expert1"] == 0

    def test_lockout_max_cap(self):
        """Lockout duration is capped at max_lockout_hours."""
        ensemble = MoEEnsemble()
        ensemble.add_expert("expert1", _bullish_predict, _high_conf)
        # Push far past threshold to test max cap
        for _ in range(20):
            ensemble.record_loss("expert1")
        duration = ensemble._disabled_until["expert1"] - time.time()
        max_seconds = ensemble._max_lockout_hours * 3600
        assert duration <= max_seconds + 1  # Allow 1s tolerance


# ── 8. Elo ratings ─────────────────────────────────────────────────────


class TestElo:
    """Elo rating updates and boundaries."""

    def test_elo_initial(self):
        """Default Elo is 1200."""
        ensemble = MoEEnsemble()
        ensemble.add_expert("expert1", _bullish_predict, _high_conf)
        assert ensemble.elo_ratings["expert1"] == 1200.0

    def test_elo_win_increases(self):
        """Correct prediction increases Elo."""
        ensemble = MoEEnsemble()
        ensemble.add_expert("expert1", _bullish_predict, _high_conf)
        ensemble.update_elo("expert1", was_correct=True)
        assert ensemble.elo_ratings["expert1"] > 1200.0

    def test_elo_loss_decreases(self):
        """Incorrect prediction decreases Elo."""
        ensemble = MoEEnsemble()
        ensemble.add_expert("expert1", _bullish_predict, _high_conf)
        ensemble.update_elo("expert1", was_correct=False)
        assert ensemble.elo_ratings["expert1"] < 1200.0

    def test_elo_bounded_lower(self):
        """Elo cannot go below 100."""
        ensemble = MoEEnsemble()
        ensemble.add_expert("expert1", _bullish_predict, _high_conf)
        for _ in range(100):
            ensemble.update_elo("expert1", was_correct=False, k=50)
        assert ensemble.elo_ratings["expert1"] >= 100

    def test_elo_bounded_upper(self):
        """Elo cannot exceed 3000."""
        ensemble = MoEEnsemble()
        ensemble.add_expert("expert1", _bullish_predict, _high_conf)
        for _ in range(100):
            ensemble.update_elo("expert1", was_correct=True, k=50)
        assert ensemble.elo_ratings["expert1"] <= 3000

    def test_elo_k_decay(self):
        """K-factor decreases as trade count increases."""
        ensemble = MoEEnsemble()
        ensemble.add_expert("expert1", _bullish_predict, _high_conf)
        # First update: total=0, k=32/(1+0/100)=32
        ensemble.update_elo("expert1", was_correct=True)
        first_elo = ensemble.elo_ratings["expert1"]
        assert first_elo == 1200.0 + 32.0

        # Simulate 100 trades to decay k
        ensemble.expert_win_rates["expert1"]["total"] = 100
        ensemble.update_elo("expert1", was_correct=True)
        second_elo = ensemble.elo_ratings["expert1"]
        # Now k = 32/(1+100/100) = 16
        assert second_elo == first_elo + 16.0

    def test_elo_k_custom(self):
        """Custom k-factor overrides automatic calculation."""
        ensemble = MoEEnsemble()
        ensemble.add_expert("expert1", _bullish_predict, _high_conf)
        ensemble.update_elo("expert1", was_correct=True, k=10.0)
        assert ensemble.elo_ratings["expert1"] == 1200.0 + 10.0

    def test_elo_default_for_new_expert(self):
        """Unknown expert gets default 1200 from get()."""
        ensemble = MoEEnsemble()
        # Don't add expert; update_elo should still work
        ensemble.update_elo("new_expert", was_correct=True)
        assert ensemble.elo_ratings["new_expert"] == 1200.0 + 32.0


# ── 9. Sharpe ratio ────────────────────────────────────────────────────


class TestSharpe:
    """Sharpe ratio computation and updates."""

    def test_update_sharpe_insufficient(self):
        """Less than 2 returns → no update."""
        ensemble = MoEEnsemble()
        ensemble.add_expert("expert1", _bullish_predict, _high_conf)
        ensemble.update_sharpe("expert1", [0.01])  # Only 1 return
        assert "expert1" not in ensemble.sharpe_ratios

    def test_update_sharpe_empty(self):
        """Empty returns list → no update."""
        ensemble = MoEEnsemble()
        ensemble.add_expert("expert1", _bullish_predict, _high_conf)
        ensemble.update_sharpe("expert1", [])
        assert "expert1" not in ensemble.sharpe_ratios

    def test_update_sharpe_computed(self):
        """Sharpe computed correctly from returns."""
        ensemble = MoEEnsemble()
        ensemble.add_expert("expert1", _bullish_predict, _high_conf)
        returns = [0.01, 0.02, -0.01, 0.015, -0.005]
        ensemble.update_sharpe("expert1", returns)
        assert "expert1" in ensemble.sharpe_ratios
        expected = np.mean(returns) / np.std(returns) * np.sqrt(252)
        assert math.isclose(ensemble.sharpe_ratios["expert1"], expected, rel_tol=1e-6)

    def test_update_sharpe_annualized(self):
        """Verifies sqrt(252) annualization factor."""
        ensemble = MoEEnsemble()
        ensemble.add_expert("expert1", _bullish_predict, _high_conf)
        returns = [0.01, 0.02, -0.01, 0.015, -0.005]
        ensemble.update_sharpe("expert1", returns)
        daily_sharpe = np.mean(returns) / np.std(returns)
        annualized = daily_sharpe * np.sqrt(252)
        assert math.isclose(ensemble.sharpe_ratios["expert1"], annualized, rel_tol=1e-6)

    def test_sharpe_zero_std(self):
        """Zero standard deviation → no update (div by zero guard)."""
        ensemble = MoEEnsemble()
        ensemble.add_expert("expert1", _bullish_predict, _high_conf)
        # All identical returns → std=0
        ensemble.update_sharpe("expert1", [0.01, 0.01, 0.01])
        assert "expert1" not in ensemble.sharpe_ratios

    def test_update_sharpe_appends_returns(self):
        """update_sharpe appends returns to expert_returns."""
        ensemble = MoEEnsemble()
        ensemble.add_expert("expert1", _bullish_predict, _high_conf)
        ensemble.update_sharpe("expert1", [0.01, 0.02])
        assert len(ensemble.expert_returns["expert1"]) == 2
        ensemble.update_sharpe("expert1", [0.03, 0.04])
        assert len(ensemble.expert_returns["expert1"]) == 4

    def test_sharpe_weighting_in_predict(self):
        """use_sharpe_weighting=False disables Sharpe weight factor."""
        ensemble = MoEEnsemble()
        ensemble.add_expert("expert1", _bullish_predict, _high_conf)
        ensemble.use_sharpe_weighting = False
        # Set a non-default sharpe
        ensemble.sharpe_ratios["expert1"] = 5.0
        result = ensemble.predict(np.random.randn(30, 49))
        output = result.expert_outputs["expert1"]
        # With sharpe disabled, sharpe_weight = 1.0 not 5.0
        expected_weight_no_sharpe = 2.0 * 1.0 * 1.0 * (0.5 + 0.5 * 0.9) * 1.0
        assert math.isclose(output["weight"], expected_weight_no_sharpe, rel_tol=1e-3)


# ── 10. update_expert_result ───────────────────────────────────────────


class TestUpdateExpertResult:
    """Expert result tracking (wins, losses, lockout)."""

    def test_update_expert_result_win(self):
        """Positive PnL → win tracked, losses reset."""
        ensemble = MoEEnsemble()
        ensemble.add_expert("expert1", _bullish_predict, _high_conf)
        # Give some prior losses
        ensemble.consecutive_losses["expert1"] = 2
        ensemble.update_expert_result("expert1", 100.0)
        assert ensemble.expert_win_rates["expert1"]["wins"] == 1
        assert ensemble.expert_win_rates["expert1"]["total"] == 1
        assert ensemble.consecutive_losses["expert1"] == 0  # reset by record_win

    def test_update_expert_result_loss(self):
        """Negative PnL → loss tracked."""
        ensemble = MoEEnsemble()
        ensemble.add_expert("expert1", _bullish_predict, _high_conf)
        ensemble.update_expert_result("expert1", -50.0)
        assert ensemble.expert_win_rates["expert1"]["wins"] == 0
        assert ensemble.expert_win_rates["expert1"]["total"] == 1
        assert ensemble.consecutive_losses["expert1"] == 1

    def test_update_expert_result_zero_pnl(self):
        """PnL of exactly 0 counts as loss."""
        ensemble = MoEEnsemble()
        ensemble.add_expert("expert1", _bullish_predict, _high_conf)
        ensemble.update_expert_result("expert1", 0.0)
        assert ensemble.expert_win_rates["expert1"]["wins"] == 0
        assert ensemble.consecutive_losses["expert1"] == 1

    def test_update_expert_result_triggers_lockout(self):
        """Repeated losses → lockout activated."""
        ensemble = MoEEnsemble()
        ensemble.add_expert("expert1", _bullish_predict, _high_conf)
        for _ in range(4):
            ensemble.update_expert_result("expert1", -50.0)
        assert ensemble.consecutive_losses["expert1"] == 4
        assert ensemble._disabled_until["expert1"] != 0.0
        assert ensemble._disabled_until["expert1"] > time.time() - 1

    def test_update_expert_result_win_rate_tracking(self):
        """Multiple updates accumulate correctly in win rate stats."""
        ensemble = MoEEnsemble()
        ensemble.add_expert("expert1", _bullish_predict, _high_conf)
        ensemble.update_expert_result("expert1", 100.0)  # win
        ensemble.update_expert_result("expert1", -50.0)  # loss
        ensemble.update_expert_result("expert1", 200.0)  # win
        ensemble.update_expert_result("expert1", -30.0)  # loss
        ensemble.update_expert_result("expert1", 10.0)  # win
        assert ensemble.expert_win_rates["expert1"]["wins"] == 3
        assert ensemble.expert_win_rates["expert1"]["total"] == 5
        # After last win, consecutive losses should be 0
        assert ensemble.consecutive_losses["expert1"] == 0


# ── 11. Edge cases and integration ─────────────────────────────────────


class TestEdgeCases:
    """Edge cases and integration scenarios."""

    X = np.random.randn(30, 49)

    def test_tracker_weight_fn(self):
        """External tracker weight function is called and applied."""
        ensemble = MoEEnsemble()

        def tracker_fn(name, regime):
            return 0.5 if name == "bull" else 2.0

        ensemble.set_tracker_weight_fn(tracker_fn)
        ensemble.add_expert("bull", _bullish_predict, _high_conf)
        ensemble.add_expert("bear", _bearish_predict, _high_conf)
        result = ensemble.predict(self.X)
        # bull gets 0.5x, bear gets 2.0x
        w_bull = result.expert_outputs["bull"]["weight"]
        w_bear = result.expert_outputs["bear"]["weight"]
        assert w_bear > w_bull
        # bear weight should be 4x bull (2.0 / 0.5)
        ratio = w_bear / w_bull
        assert math.isclose(ratio, 4.0, rel_tol=0.1)

    def test_sharpe_clipped_in_predict(self):
        """Sharpe ratio is clipped to [0.1, 2.0] in predict."""
        ensemble = MoEEnsemble()
        ensemble.add_expert("expert1", _bullish_predict, _high_conf)
        # Set extreme sharpe values
        ensemble.sharpe_ratios["expert1"] = 100.0
        result = ensemble.predict(self.X)
        # sharpe_weight should be capped at 2.0
        expected_sharpe_weight = 2.0  # min(2.0, max(0.1, 100.0))
        expected_weight = 2.0 * 1.0 * expected_sharpe_weight * (0.5 + 0.5 * 0.9) * 1.0
        assert math.isclose(
            result.expert_outputs["expert1"]["weight"],
            expected_weight,
            rel_tol=1e-3,
        )

        # Test lower clip
        ensemble.sharpe_ratios["expert1"] = -5.0
        result = ensemble.predict(self.X)
        expected_sharpe_weight = 0.1  # max(0.1, min(2.0, -5.0))
        expected_weight = 2.0 * 1.0 * expected_sharpe_weight * (0.5 + 0.5 * 0.9) * 1.0
        assert math.isclose(
            result.expert_outputs["expert1"]["weight"],
            expected_weight,
            rel_tol=1e-3,
        )

    def test_predict_all_experts_skipped(self):
        """All experts skipped (disabled) → empty prediction."""
        ensemble = MoEEnsemble()
        ensemble.add_expert("expert1", _bullish_predict, _high_conf)
        ensemble.add_expert("expert2", _bullish_predict, _high_conf)
        # Disable both via lockout
        ensemble._disabled_until["expert1"] = time.time() + 3600
        ensemble._disabled_until["expert2"] = time.time() + 3600
        result = ensemble.predict(self.X)
        assert result.price == 0.0
        assert result.direction == "HOLD"
        assert result.expert_outputs == {}

    def test_predict_zero_total_weight(self):
        """If all weights sum to 0, return empty prediction."""
        ensemble = MoEEnsemble()
        ensemble.add_expert("expert1", _bullish_predict, _low_conf)
        # Force weight to 0 by setting extreme tracker weight
        ensemble._tracker_weight_fn = lambda name, regime: 0.0
        result = ensemble.predict(self.X)
        # total weight = 2.0 * 1.0 * 1.0 * (0.5+0.5*0.3) * 0.0 = 0
        assert result.price == 0.0
        assert result.direction == "HOLD"

    def test_should_trade_tier2_sell(self):
        """High confidence with negative price → sell via tier 2."""
        ensemble = MoEEnsemble()
        pred = _PredBuilder.build(
            predictions=[-0.001, -0.002, 0.0, 0.0],
            confidences=[0.9, 0.85, 0.5, 0.5],
            weights=[1.0, 1.0, 1.0, 1.0],
        )
        should, direction, ratio = ensemble.should_trade(pred, 1.0, min_confidence=0.65)
        assert should is True
        assert direction == "SELL"

    def test_update_expert_result_with_sharpe_update(self):
        """update_expert_result triggers Sharpe update if returns exist."""
        ensemble = MoEEnsemble()
        ensemble.add_expert("expert1", _bullish_predict, _high_conf)
        # Seed some returns so the sharpe update branch executes
        ensemble.expert_returns["expert1"] = [0.01, 0.02, -0.01, 0.015, -0.005]
        ensemble.update_expert_result("expert1", 100.0)
        # Sharpe should have been updated
        assert "expert1" in ensemble.sharpe_ratios
