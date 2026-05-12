"""Test configuration for service tests — ensures src/ is on sys.path."""

import sys
import os

# Add src/ to path so 'from infrastructure...' and 'from services...' resolve
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)
