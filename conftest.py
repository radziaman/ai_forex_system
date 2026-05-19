"""
Root conftest.py - filter warnings at the earliest point possible.
"""

import warnings

# Filter urllib3 NotOpenSSLWarning for LibreSSL compatibility
warnings.filterwarnings(
    "ignore",
    message="urllib3 v2 only supports OpenSSL",
    category=DeprecationWarning,
)

try:
    import urllib3

    warnings.filterwarnings("ignore", category=urllib3.exceptions.NotOpenSSLWarning)
    urllib3.disable_warnings(urllib3.exceptions.NotOpenSSLWarning)
except ImportError:
    pass
