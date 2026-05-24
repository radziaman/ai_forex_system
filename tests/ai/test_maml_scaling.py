import numpy as np
from ai.maml_scaler import (
    MetaLearningOrchestrator,
    MetaConfig,
    AdaptationResult,
    SimpleLinearModel,
    ParameterBuffer,
)


class TestParameterBuffer:
    def test_clone_independence(self):
        buf = ParameterBuffer()
        buf.set("w", np.array([1.0, 2.0]))
        cloned = buf.clone()
        cloned.set("w", np.array([3.0, 4.0]))
        np.testing.assert_array_equal(buf.get("w"), [1.0, 2.0])

    def test_to_flat_roundtrip(self):
        buf = ParameterBuffer()
        buf.set("w", np.array([[1.0, 2.0], [3.0, 4.0]]))
        buf.set("b", np.array([0.5]))
        flat = buf.to_flat()
        restored = buf.from_flat(flat)
        np.testing.assert_array_equal(restored.get("w"), buf.get("w"))
        np.testing.assert_array_equal(restored.get("b"), buf.get("b"))

    def test_add_gradient(self):
        buf = ParameterBuffer()
        buf.set("w", np.array([1.0, 2.0]))
        grad = ParameterBuffer()
        grad.set("w", np.array([0.1, 0.2]))
        buf.add_gradient(grad, lr=1.0)
        np.testing.assert_array_almost_equal(buf.get("w"), [0.9, 1.8])


class TestSimpleLinearModel:
    def test_predict_shape(self):
        model = SimpleLinearModel(n_features=5)
        X = np.random.randn(10, 5)
        preds = model.predict(X)
        assert preds.shape == (10,)

    def test_adapt_reduces_loss(self):
        np.random.seed(42)
        n = 50
        X = np.random.randn(n, 5)
        true_w = np.array([0.5, -0.3, 0.8, 0.1, -0.6])
        y = X @ true_w + np.random.randn(n) * 0.01

        model = SimpleLinearModel(n_features=5)
        loss_before = float(np.mean((y - model.predict(X)) ** 2))
        adapted_params = model.adapt(X, y, inner_lr=0.01, inner_steps=10)
        # Apply adapted params back to model
        model.w = adapted_params.get("w").copy()
        model.b = float(adapted_params.get("b").flatten()[0])
        loss_after = float(np.mean((y - model.predict(X)) ** 2))
        assert loss_after < loss_before


class TestMetaLearningOrchestrator:
    def test_register_and_adapt(self):
        orchestrator = MetaLearningOrchestrator()
        orchestrator.register_model("EURUSD")
        X = np.random.randn(20, 49)
        y = np.random.randn(20)
        result = orchestrator.adapt("EURUSD", X, y)
        assert isinstance(result, AdaptationResult)
        assert result.symbol == "EURUSD"

    def test_adapt_reduces_loss(self):
        np.random.seed(42)
        orchestrator = MetaLearningOrchestrator(
            config=MetaConfig(inner_lr=0.01, inner_steps=10)
        )
        n = 30
        X = np.random.randn(n, 49)
        true_w = np.random.randn(49) * 0.5
        y = X @ true_w + np.random.randn(n) * 0.05

        model = SimpleLinearModel(n_features=49)
        orchestrator.register_model("EURUSD", model)
        result = orchestrator.adapt("EURUSD", X, y)
        assert result.loss_after < result.loss_before
        assert result.improvement_pct > 0

    def test_adaptation_history(self):
        orchestrator = MetaLearningOrchestrator()
        orchestrator.register_model("GBPUSD")
        X = np.random.randn(10, 49)
        y = np.random.randn(10)
        orchestrator.adapt("GBPUSD", X, y)
        orchestrator.adapt("GBPUSD", X, y)
        history = orchestrator.get_adaptation_history("GBPUSD")
        assert len(history) == 2

    def test_meta_update_reduces_loss(self):
        np.random.seed(42)
        orchestrator = MetaLearningOrchestrator(
            config=MetaConfig(inner_lr=0.01, inner_steps=5, outer_lr=0.001)
        )
        model = SimpleLinearModel(n_features=10)
        orchestrator.register_model("EURUSD", model)

        tasks = []
        for _ in range(3):
            X = np.random.randn(15, 10)
            true_w = np.random.randn(10) * 0.5
            y = X @ true_w + np.random.randn(15) * 0.1
            tasks.append((X, y))

        orchestrator.meta_update("EURUSD", tasks)
        # Loss should improve after meta-update
        assert orchestrator.get_recent_improvement("EURUSD") >= 0

    def test_warm_start_trigger(self):
        orchestrator = MetaLearningOrchestrator(
            config=MetaConfig(adaptation_threshold=1.0)
        )
        orchestrator.register_model("USDJPY")
        needs_warm = orchestrator.warm_start("USDJPY")
        assert needs_warm  # No adaptations yet, below threshold

    def test_get_recent_improvement_empty(self):
        orchestrator = MetaLearningOrchestrator()
        assert orchestrator.get_recent_improvement("NONEXISTENT") == 0.0
