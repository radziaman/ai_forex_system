"""Tests for MacroEconomicEngine."""

import pytest

from src.data.macroeconomic_features import MacroEconomicEngine


@pytest.fixture
def engine() -> MacroEconomicEngine:
    return MacroEconomicEngine()


class TestRateDifferential:
    def test_eurusd_negative(self, engine):
        feats = engine.compute_divergence_features("EURUSD")
        # EUR 4.00 - USD 5.50 = -1.50
        assert feats["rate_differential"] == pytest.approx(-1.50)

    def test_usdjpy_positive(self, engine):
        feats = engine.compute_divergence_features("USDJPY")
        # USD 5.50 - JPY 0.10 = 5.40
        assert feats["rate_differential"] == pytest.approx(5.40)

    def test_invalid_symbol(self, engine):
        feats = engine.compute_divergence_features("INVALID")
        assert feats["rate_differential"] == 0.0


class TestRateSlope1m:
    def test_usdjpy_positive_slope(self, engine):
        feats = engine.compute_divergence_features("USDJPY")
        # USD slope -0.05, JPY slope +0.10 => -0.15
        assert feats["rate_slope_1m"] == pytest.approx(-0.15)

    def test_eurusd_flat_slope(self, engine):
        feats = engine.compute_divergence_features("EURUSD")
        # EUR -0.05, USD -0.05 => 0.00
        assert feats["rate_slope_1m"] == pytest.approx(0.0)


class TestForwardDivergence:
    def test_forward_includes_slope(self, engine):
        feats = engine.compute_divergence_features("GBPUSD")
        # GBP 5.25 - USD 5.50 = -0.25
        # slope: GBP -0.05, USD -0.05 => 0.00
        # forward = -0.25 + 0.00 * 12 = -0.25
        expected = feats["rate_differential"] + feats["rate_slope_1m"] * 12.0
        assert feats["forward_divergence"] == pytest.approx(expected)


class TestHawkishDovishScore:
    def test_hawkish_usd(self, engine):
        feats = engine.compute_divergence_features("USDJPY")
        assert feats["hawkish_dovish_score"] == 1.0

    def test_dovish_jpy(self, engine):
        feats = engine.compute_divergence_features("JPYUSD")
        # JPY 0.10 <= 0.5 => dovish -1
        assert feats["hawkish_dovish_score"] == -1.0

    def test_neutral_eur(self, engine):
        feats = engine.compute_divergence_features("EURUSD")
        # EUR 4.00 is between 0.5 and 5.0 => neutral 0
        assert feats["hawkish_dovish_score"] == 0.0


class TestYieldCarryAnnualized:
    def test_eurusd_carry(self, engine):
        feats = engine.compute_divergence_features("EURUSD")
        expected = feats["rate_differential"] * 10000.0
        assert feats["yield_carry_annualized"] == pytest.approx(expected)

    def test_usdjpy_carry(self, engine):
        feats = engine.compute_divergence_features("USDJPY")
        expected = feats["rate_differential"] * 10000.0
        assert feats["yield_carry_annualized"] == pytest.approx(expected)
