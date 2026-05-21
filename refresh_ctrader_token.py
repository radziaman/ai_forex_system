#!/usr/bin/env python3
"""cTrader Token Refresher — exchange authorization code for API tokens."""
import re, requests, sys

ENV_FILE = ".env"

# Load current .env
with open(ENV_FILE, "r") as f:
    env = f.read()

app_id = re.search(r"CTRADER_APP_ID=(\S+)", env)
app_secret = re.search(r"CTRADER_APP_SECRET=(\S+)", env)

if not app_id or not app_secret:
    print("❌ CTRADER_APP_ID or CTRADER_APP_SECRET not found in .env")
    sys.exit(1)

app_id = app_id.group(1)
app_secret = app_secret.group(1)

print("=== cTrader Token Refresher ===")
print()
print("Step 1: Open this URL in your browser:")
print(
    f"  https://id.ctrader.com/my/settings/openapi/grantingaccess/?client_id={app_id}&redirect_uri=https://spotware.com&scope=trading&product=web"
)
print()
print("Step 2: Authorize the application")
print("Step 3: Copy the 'code' parameter from the redirect URL")
print()

code = input("Enter authorization code: ").strip()

if not code:
    print("❌ No code entered")
    sys.exit(1)

print("\nExchanging code for tokens...")
resp = requests.post(
    "https://openapi.ctrader.com/apps/token",
    data={
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": "https://spotware.com",
        "client_id": app_id,
        "client_secret": app_secret,
    },
    timeout=30,
)

data = resp.json()
access_token = data.get("accessToken") or data.get("access_token", "")
refresh_token = data.get("refreshToken") or data.get("refresh_token", "")

if access_token and len(access_token) > 100:
    env = re.sub(
        r"CTRADER_ACCESS_TOKEN=.*",
        f"CTRADER_ACCESS_TOKEN={access_token}",
        env,
    )
    env = re.sub(
        r"CTRADER_REFRESH_TOKEN=.*",
        f"CTRADER_REFRESH_TOKEN={refresh_token}",
        env,
    )

    with open(ENV_FILE, "w") as f:
        f.write(env)

    print(f"\n✅ Tokens saved to {ENV_FILE}")
    print(f"   Access token:  {access_token[:40]}... ({len(access_token)} chars)")
    print(f"   Refresh token: {refresh_token[:40]}... ({len(refresh_token)} chars)")
    print()
    print("Now start the system:")
    print(f"  cd {sys.path[0] or '.'}")
    print(
        f"  nohup .venv/bin/python -m src.agentic.main_agentic --mode live > /tmp/agentic_live.log 2>&1 &"
    )
else:
    print(f"\n❌ Failed to get valid tokens. Response: {data}")
    if resp.status_code == 400:
        print(
            "The authorization code may have expired. Get a fresh one from the URL above."
        )
