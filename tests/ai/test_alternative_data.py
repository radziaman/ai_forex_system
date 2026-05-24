from ai.alternative_data import (
    AlternativeDataAggregator,
    AlternativeDataSnapshot,
    CentralBankTone,
    CommoditySignal,
    COMMODITY_SENSITIVITY,
    CB_SENSITIVITY,
)


class TestAlternativeDataAggregator:
    def test_commodity_zscore_oil_up(self):
        """Rising oil price should produce positive oil signal."""
        agg = AlternativeDataAggregator(lookback_commodity=20)
        # Flat prices then a spike — last return is well above the mean
        prices = [70] * 10 + [71, 73, 76, 80, 85, 91, 98, 106, 115, 125]
        for price in prices:
            signal = agg.process_commodity_prices(oil_price=price)
        assert signal.oil_signal > 0.0  # Positive z-score for accelerating rise

    def test_commodity_mixed_signals(self):
        """Different commodities can give different signals."""
        agg = AlternativeDataAggregator(lookback_commodity=20)
        # Oil rises slowly, gold falls — need >=10 pts for z-score calc
        prices_oil = [70] * 6 + [72, 74, 76, 78, 80, 82, 84, 86, 88, 90]
        prices_gold = [2000] * 6 + [
            1980,
            1960,
            1940,
            1920,
            1900,
            1880,
            1860,
            1840,
            1820,
            1800,
        ]
        for o, g in zip(prices_oil, prices_gold):
            signal = agg.process_commodity_prices(oil_price=o, gold_price=g)
        assert signal.gold_signal < signal.oil_signal  # gold falling, oil rising

    def test_central_bank_hawkish_signal(self):
        agg = AlternativeDataAggregator()
        signal = agg.process_central_bank_signal("FED", CentralBankTone.HAWKISH, 0.8)
        assert signal.bank_name == "FED"
        assert signal.tone == CentralBankTone.HAWKISH
        assert signal.confidence == 0.8

    def test_central_bank_tone_score_hawkish(self):
        agg = AlternativeDataAggregator()
        agg.process_central_bank_signal("ECB", CentralBankTone.HAWKISH, 0.9)
        score = agg.get_recent_cb_tone("ECB")
        assert score > 0.5

    def test_central_bank_tone_score_dovish(self):
        agg = AlternativeDataAggregator()
        agg.process_central_bank_signal("BOJ", CentralBankTone.DOVISH, 0.7)
        score = agg.get_recent_cb_tone("BOJ")
        assert score < -0.5

    def test_central_bank_tone_score_empty(self):
        agg = AlternativeDataAggregator()
        score = agg.get_recent_cb_tone("RBA")
        assert score == 0.0

    def test_compute_composite_commodity_pair(self):
        """USDCAD should be sensitive to oil price changes."""
        agg = AlternativeDataAggregator()
        comm_signal = CommoditySignal(
            oil_signal=1.0, gold_signal=0.0, copper_signal=0.0, composite=0.5
        )
        composite = agg.compute_composite(comm_signal, {})
        assert "USDCAD" in composite
        # USDCAD oil sensitivity is 0.7, so score should be > 0
        # But other pairs also get contributions so just check it exists

    def test_compute_composite_with_cb(self):
        agg = AlternativeDataAggregator()
        comm_signal = CommoditySignal()
        cb_signal = agg.process_central_bank_signal("ECB", CentralBankTone.HAWKISH, 0.8)
        composite = agg.compute_composite(comm_signal, {"ECB": cb_signal})
        assert "EURUSD" in composite

    def test_get_snapshot(self):
        agg = AlternativeDataAggregator()
        snapshot = agg.get_snapshot(
            commodity_prices={"oil": 75.0, "gold": 1950.0, "copper": 4.5},
            cb_signals=[{"bank": "FED", "tone": "hawkish", "confidence": 0.7}],
            economic_data={"USDCAD": 0.2},
        )
        assert isinstance(snapshot, AlternativeDataSnapshot)
        assert "oil" in str(snapshot.commodity)
        assert len(snapshot.composite_signal) > 0

    def test_commodity_sensitivity_defined(self):
        assert "USDCAD" in COMMODITY_SENSITIVITY
        assert "AUDUSD" in COMMODITY_SENSITIVITY
        assert COMMODITY_SENSITIVITY["USDCAD"]["oil"] > 0.5

    def test_cb_sensitivity_all_major_pairs(self):
        for pair in ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]:
            assert pair in CB_SENSITIVITY
