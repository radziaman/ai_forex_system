#!/usr/bin/env python3
"""
OAuth Helper for RTS - AI FX Trading System
Automates the cTrader OAuth flow to get valid access tokens.

Usage:
    python scripts/oauth_helper.py
"""

import webbrowser
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import time

# cTrader App Credentials
CLIENT_ID = "15217_h8WxunXX70m6O6qsnIx9ZO3GZraTdO0wnLjL3dTKyYG6fkbUca"
CLIENT_SECRET = "Zb8tEW4Axzq0AJqCNS8ubniYzsgp2kxuRkYBRD9XXOcLAj5aOT"
REDIRECT_URI = "http://localhost:8080"
SCOPE = "trading"
AUTH_URL = "https://id.ctrader.com/my/settings/openapi/grantingaccess/"

class OAuthCallbackHandler(BaseHTTPRequestHandler):
    """Handle OAuth callback and extract authorization code."""

    def do_GET(self):
        """Process the OAuth callback."""
        query = urllib.parse.urlparse(self.path).query
        params = urllib.parse.parse_qs(query)

        if 'code' in params:
            self.server.auth_code = params['code'][0]
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"""
            <html><body>
            <h1>Authorization Successful!</h1>
            <p>You can close this window and return to the terminal.</p>
            <script>setTimeout(() => window.close(), 3000);</script>
            </body></html>
            """)
        else:
            self.send_response(400)
            self.end_headers()

    def log_message(self, format, *args):
        """Suppress log messages."""
        pass


def get_authorization_code():
    """Start local server and open browser for OAuth flow."""
    print("\n=== cTrader OAuth Flow ===\n")
    print("1. Opening browser for authorization...")

    # Build auth URL
    params = {
        'client_id': CLIENT_ID,
        'redirect_uri': REDIRECT_URI,
        'scope': SCOPE,
        'product': 'web'
    }
    auth_url = AUTH_URL + '?' + urllib.parse.urlencode(params)

    # Start local server
    server = HTTPServer(('localhost', 8080), OAuthCallbackHandler)
    server.auth_code = None

    # Open browser
    webbrowser.open(auth_url)

    # Wait for callback
    print("2. Waiting for authorization in browser...")
    server.timeout = 300  # 5 minute timeout
    server.handle_request()

    if server.auth_code:
        print(f"3. Authorization code received: {server.auth_code[:20]}...")
        return server.auth_code
    else:
        print("ERROR: No authorization code received!")
        return None


def exchange_code_for_token(auth_code):
    """Exchange authorization code for access token."""
    import requests

    print("\n4. Exchanging code for access token...")

    token_url = "https://openapi.ctrader.com/apps/token"
    params = {
        'grant_type': 'authorization_code',
        'code': auth_code,
        'redirect_uri': REDIRECT_URI,
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET
    }

    try:
        response = requests.get(token_url, params=params, headers={'Accept': 'application/json'})
        if response.status_code == 200:
            token_data = response.json()
            print("\n✅ SUCCESS! Token received:")
            print(f"   Access Token: {token_data.get('access_token', 'N/A')[:50]}...")
            print(f"   Expires In: {token_data.get('expires_in', 'N/A')} seconds")
            print(f"   Refresh Token: {token_data.get('refresh_token', 'N/A')[:50]}...")
            return token_data
        else:
            print(f"ERROR: Token exchange failed - {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def save_token_to_env(token_data):
    """Save token to .env file for the application."""
    env_file = "/Users/radziaman/Antigravity/ai_forex_system/.env"

    with open(env_file, 'w') as f:
        f.write("# RTS - AI FX Trading System Environment Variables\n")
        f.write(f"CTRADER_ACCESS_TOKEN={token_data.get('access_token', '')}\n")
        f.write(f"CTRADER_REFRESH_TOKEN={token_data.get('refresh_token', '')}\n")
        f.write(f"CTRADER_CLIENT_ID={CLIENT_ID}\n")
        f.write(f"CTRADER_CLIENT_SECRET={CLIENT_SECRET}\n")

    print(f"\n✅ Tokens saved to {env_file}")
    print("   Make sure to add .env to .gitignore!")


def main():
    """Run the OAuth flow."""
    print("RTS - AI FX Trading System: OAuth Helper")
    print("=" * 50)

    # Step 1: Get authorization code
    auth_code = get_authorization_code()
    if not auth_code:
        return

    # Step 2: Exchange for token
    token_data = exchange_code_for_token(auth_code)
    if not token_data:
        return

    # Step 3: Save to .env
    save_token_to_env(token_data)

    print("\n" + "=" * 50)
    print("Next steps:")
    print("1. Update your cTrader script with the new access token")
    print("2. Or use environment variables: source .env")
    print("3. Run: python src/api/ctrader_ready.py")


if __name__ == "__main__":
    main()
