"""Tests for the Satellite / Alternative Data Engine."""

import pytest

from src.data.alternative_data import AlternativeDataEngine


class TestAlternativeDataEngine:
    def test_init(self):
        engine = AlternativeDataEngine()
        assert engine.api_key is not None

    def test_compute_fx_impact_scores_keys(self):
        engine = AlternativeDataEngine()
        scores = engine.compute_fx_impact_scores()
        expected_keys = {
            "aud_agriculture_score",
            "cad_oil_score",
            "jpy_safe_haven_score",
            "usd_fed_divergence_score",
        }
        assert set(scores.keys()) == expected_keys

    def test_scores_are_normalized_0_to_1(self):
        engine = AlternativeDataEngine()
        scores = engine.compute_fx_impact_scores()
        for key, value in scores.items():
            assert 0.0 <= value <= 1.0, f"{key} = {value} out of range"

    def test_get_oil_shipping_score_deterministic(self):
        engine = AlternativeDataEngine()
        s1 = engine.get_oil_shipping_score()
        s2 = engine.get_oil_shipping_score()
        # Same second => same value
        assert s1 == pytest.approx(s2)
        assert 0.0 <= s1 <= 1.0
