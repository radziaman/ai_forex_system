"""
cTrader Integration Helper for IC Markets
Run this script to set up your cTrader API credentials.
"""
import json
import os
from pathlib import Path

def setup_ctrader_config():
    """Interactive setup for cTrader API credentials"""
    print("=" * 60)
    print("cTrader API Setup for IC Markets")
    print("=" * 60)
    print("\nYou need to create a cTrader app at: https://connect.ctrader.com/")
    print("Make sure to add redirect URI: http://localhost:8080")
    print("\n")
    
    config = {}
    
    # App credentials
    config["app_id"] = input("Enter your App ID: ").strip()
    config["app_secret"] = input("Enter your App Secret: ").strip()
    
    # Get access token (OAuth2 flow)
    print("\n" + "="*60)
    print("To get an access token, visit:")
    print(f"https://connect.ctrader.com/apps/{config['app_id']}/")
    print("And authorize the app to get the access token.")
    print("="*60 + "\n")
    
    config["access_token"] = input("Enter your Access Token: ").strip()
    config["account_id"] = input("Enter your cTrader Account ID: ").strip()
    
    # IC Markets specific settings
    config["demo"] = input("Use Demo account? (y/n): ").strip().lower() == 'y'
    config["use_websocket"] = True
    
    config["ic_markets"] = {
        "name": "IC Markets",
        "server_type": "live" if not config["demo"] else "demo"
    }
    
    # Save config
    config_path = Path("config/ctrader_config.json")
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\nConfig saved to {config_path}")
    print("\nYou can now run: python main.py live")

if __name__ == "__main__":
    setup_ctrader_config()
