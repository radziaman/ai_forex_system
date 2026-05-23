"""Tests for the Options Market Data Framework."""

from src.data.options_data import OptionsDataProvider


class TestOptionsDataProvider:
    def test_init_default_is_mock(self):
        provider = OptionsDataProvider()
        assert not provider.is_available()

    def test_init_with_source_is_available(self):
        provider = OptionsDataProvider(source="bloomberg")
        assert provider.is_available()

    def test_get_25d_risk_reversal(self):
        provider = OptionsDataProvider()
        rr = provider.get_25d_risk_reversal("EURUSD")
        assert isinstance(rr, float)

    def test_get_butterfly_spread(self):
        provider = OptionsDataProvider()
        bf = provider.get_butterfly_spread("EURUSD")
        assert isinstance(bf, float)

    def test_get_skew_index(self):
        provider = OptionsDataProvider()
        sk = provider.get_skew_index("EURUSD")
        assert isinstance(sk, float)
        assert 0.0 <= sk <= 1.0

    def test_get_term_structure(self):
        provider = OptionsDataProvider()
        term = provider.get_term_structure("EURUSD")
        assert isinstance(term, dict)
        assert len(term) > 0
        for expiry, iv in term.items():
            assert isinstance(expiry, str)
            assert isinstance(iv, float)
            assert iv > 0
