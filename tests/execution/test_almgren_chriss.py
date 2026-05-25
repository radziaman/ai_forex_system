from execution.almgren_chriss import AlmgrenChrissModel


class TestAlmgrenChrissModel:
    def test_impact_non_negative(self):
        model = AlmgrenChrissModel(volatility=0.15, adv=1e9, spread=0.0001)
        result = model.calculate_impact("EURUSD", 100_000, 1.10, "BUY")
        assert result.permanent_impact_bps >= 0.0
        assert result.temporary_impact_bps >= 0.0
        assert result.total_cost_bps >= 0.0

    def test_larger_volume_higher_impact(self):
        model = AlmgrenChrissModel(volatility=0.15, adv=1e9, spread=0.0001)
        r1 = model.calculate_impact("EURUSD", 100_000, 1.10, "BUY")
        r2 = model.calculate_impact("EURUSD", 1_000_000, 1.10, "BUY")
        assert r2.total_cost_bps > r1.total_cost_bps

    def test_sell_side_same_impact_as_buy(self):
        model = AlmgrenChrissModel(volatility=0.15, adv=1e9, spread=0.0001)
        buy = model.calculate_impact("EURUSD", 100_000, 1.10, "BUY")
        sell = model.calculate_impact("EURUSD", 100_000, 1.10, "SELL")
        assert abs(buy.total_cost_bps - sell.total_cost_bps) < 1e-10

    def test_optimal_trajectory_returns_slices(self):
        model = AlmgrenChrissModel(volatility=0.15, adv=1e9, spread=0.0001)
        slices = model.optimal_trajectory(
            volume=100_000, price=1.10, n_slices=10, duration_minutes=30
        )
        assert len(slices) == 10
        assert abs(sum(s["volume"] for s in slices) - 100_000) < 1.0

    def test_optimal_trajectory_front_loaded(self):
        model = AlmgrenChrissModel(volatility=0.15, adv=1e9, spread=0.0001)
        slices = model.optimal_trajectory(
            volume=100_000, price=1.10, n_slices=10, duration_minutes=30
        )
        assert slices[0]["volume"] >= slices[-1]["volume"]

    def test_impact_decays_with_longer_horizon(self):
        model = AlmgrenChrissModel(volatility=0.15, adv=1e9, spread=0.0001)
        fast = model.optimal_trajectory(
            volume=100_000, price=1.10, n_slices=5, duration_minutes=5
        )
        slow = model.optimal_trajectory(
            volume=100_000, price=1.10, n_slices=20, duration_minutes=60
        )
        # Compare volume-weighted average impact per unit volume
        fast_weighted = sum(s["impact_bps"] * s["volume"] for s in fast)
        slow_weighted = sum(s["impact_bps"] * s["volume"] for s in slow)
        total_vol = sum(s["volume"] for s in fast) or 1.0
        avg_fast = fast_weighted / total_vol
        avg_slow = slow_weighted / total_vol
        assert avg_fast > avg_slow

    def test_integration_with_cost_model(self):
        from execution.cost_model import CostModel

        ac = AlmgrenChrissModel(volatility=0.15, adv=1e9, spread=0.0001)
        model = CostModel(impact_model=ac)
        result = model.calculate("EURUSD", "BUY", 100_000, 1.10)
        assert result.total > 0
        assert result.market_impact > 0
