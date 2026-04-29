"""
cTrader Integration Helper for IC Markets
Run this script to set up your cTrader API credentials.
Supports both manual entry and automated OAuth2 flow.
"""
import json
import os
import sys
from pathlib import Path
from typing import Optional


def get_manual_credentials() -> dict:
    """Get credentials manually from user input."""
    print("=" * 60)
    print("cTrader API Setup for IC Markets (Manual Mode)")
    print("=" * 60)
    print("\nYou need to create a cTrader app at: https://connect.ctrader.com/")
    print("Make sure to add redirect URI: http://localhost:8080")
    print("\n")

    config = {}

    # App credentials
    config["app_id"] = input("Enter your App ID: ").strip()
    config["app_secret"] = input("Enter your App Secret: ").strip()

    # Get access token
    print("\n" + "=" * 60)
    print("To get an access token, visit:")
    print(f"https://connect.ctrader.com/apps/{config['app_id']}/")
    print("And authorize the app to get the access token.")
    print("=" * 60 + "\n")

    config["access_token"] = input("Enter your Access Token: ").strip()
    config["account_id"] = input("Enter your cTrader Account ID: ").strip()

    # IC Markets specific settings
    demo_input = input("Use Demo account? (y/n): ").strip().lower()
    config["demo"] = demo_input == 'y'
    config["use_websocket"] = False  # Not available in current API version

    config["ic_markets"] = {
        "name": "IC Markets",
        "server_type": "live" if not config["demo"] else "demo",
        "host": "live.ctrader.com" if not config["demo"] else "demo.ctrader.com",
        "port": 5035,
    }

    return config


def save_config(config: dict, config_path: Path):
    """Save configuration to JSON file."""
    config_path.parent.mkdir(exist_ok=True)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nConfig saved to {config_path}")
    print("\nYou can now run:")
    print("  - python scripts/live_trading.py")
    print("  - python scripts/ic_markets_signals.py")


def setup_ctrader_config():
    """Interactive setup for cTrader API credentials"""
    try:
        config = get_manual_credentials()
        config_path = Path("config/ctrader_config.json")
        save_config(config, config_path)
        return True
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        return False
    except Exception as e:
        print(f"\nError during setup: {e}")
        return False


if __name__ == "__main__":
    setup_ctrader_config()
