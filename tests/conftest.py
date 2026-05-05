"""
pytest configuration to filter out known system warnings.
"""
import warnings
import urllib3
import pytest


def pytest_configure(config):
    """Configure pytest to filter warnings."""
    # Suppress urllib3 NotOpenSSLWarning for LibreSSL compatibility
    warnings.filterwarnings(
        "ignore",
        "urllib3 v2 only supports OpenSSL",
        category=urllib3.exceptions.NotOpenSSLWarning,
    )
    urllib3.disable_warnings(urllib3.exceptions.NotOpenSSLWarning)


@pytest.fixture(autouse=True)
def filter_warnings():
    """Filter warnings during tests."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield
