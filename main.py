"""
Legacy entry point — delegates to src/main.py
"""
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from src.main import main as _main
    _main()
