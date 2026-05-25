#!/usr/bin/env python3
"""Run the unified dashboard server.

Usage:
    python -m scripts.run_dashboard          # default 0.0.0.0:8000
    python -m scripts.run_dashboard --port 9000
"""

from __future__ import annotations

import argparse
import uvicorn
from dashboard import app


def main():
    parser = argparse.ArgumentParser(description="Run RTS Dashboard")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind (default: 8000)",
    )
    args = parser.parse_args()

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        ws_ping_interval=20,
        ws_ping_timeout=20,
    )


if __name__ == "__main__":
    main()
