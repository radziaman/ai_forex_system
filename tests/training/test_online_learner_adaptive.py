import time
from training.online_learner import OnlineLearner


class TestOnlineLearnerAdaptive:
    def test_default_cooldown_unchanged(self):
        learner = OnlineLearner(retrain_cooldown_hours=4.0)
        assert learner.retrain_cooldown == 4.0 * 3600

    def test_adaptive_cooldown_reduces_with_drift(self):
        learner = OnlineLearner(retrain_cooldown_hours=4.0)
        for i in range(5):
            learner.on_drift_detected("EURUSD", drift_count=i + 1)
        cd = learner._get_adaptive_cooldown("EURUSD")
        assert cd < 4.0 * 3600

    def test_adaptive_cooldown_increases_with_stability(self):
        learner = OnlineLearner(retrain_cooldown_hours=4.0)
        cd = learner._get_adaptive_cooldown("EURUSD")
        assert cd >= 4.0 * 3600

    def test_should_retrain_respects_adaptive_cooldown(self):
        learner = OnlineLearner(retrain_cooldown_hours=4.0)
        learner._last_retrain["EURUSD"] = time.time()
        assert not learner.should_retrain("EURUSD", total_trades=100)
        learner.on_drift_detected("EURUSD", drift_count=5)
        learner._drift_signals["EURUSD"] = 5
        # Set last_retrain far enough in the past that the adaptive cooldown
        # (which shrinks with drift) has expired
        learner._last_retrain["EURUSD"] = time.time() - 10000
        result = learner.should_retrain("EURUSD", total_trades=100)
        assert result

    def test_cooldown_caps_at_minimum(self):
        learner = OnlineLearner(retrain_cooldown_hours=4.0)
        learner._last_retrain["EURUSD"] = time.time()
        learner.on_drift_detected("EURUSD", drift_count=100)
        cd = learner._get_adaptive_cooldown("EURUSD")
        assert cd >= 300.0

    def test_cooldown_caps_at_maximum(self):
        learner = OnlineLearner(retrain_cooldown_hours=4.0)
        learner._last_retrain["EURUSD"] = 0
        cd = learner._get_adaptive_cooldown("EURUSD")
        assert cd <= 12 * 3600
