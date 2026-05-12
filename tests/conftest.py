"""
pytest configuration — path setup + warning filters.
"""

import sys
import os
import warnings
import urllib3
import pytest

# Add src/ to path so all package imports resolve
_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _src not in sys.path:
    sys.path.insert(0, _src)


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
