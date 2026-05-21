#!/bin/bash
# cTrader Token Refresher
# Usage: bash refresh_ctrader_token.sh

cd "$(dirname "$0")"
source .venv/bin/activate

echo "=== cTrader Token Refresher ==="
echo ""
echo "Step 1: Get authorization code"
echo "Open this URL in your browser:"
echo "  https://id.ctrader.com/my/settings/openapi/grantingaccess/?client_id=${CTRADER_APP_ID}&redirect_uri=https://spotware.com&scope=trading&product=web"
echo ""
echo "After authorizing, you'll be redirected to a URL."
echo "Copy the 'code' parameter from the URL."
echo ""

read -p "Step 2: Enter authorization code: " AUTH_CODE

python3 -c "
import requests, re

# Read current .env
with open('.env', 'r') as f:
    env = f.read()

app_id = re.search(r'CTRADER_APP_ID=(\S+)', env).group(1)
app_secret = re.search(r'CTRADER_APP_SECRET=(\S+)', env).group(1)

url = 'https://openapi.ctrader.com/apps/token'
params = {
    'grant_type': 'authorization_code',
    'code': '$AUTH_CODE',
    'redirect_uri': 'https://spotware.com',
    'client_id': app_id,
    'client_secret': app_secret,
}

print('Exchanging code for tokens...')
resp = requests.post(url, data=params, timeout=30)
data = resp.json()

access_token = data.get('accessToken') or data.get('access_token', '')
refresh_token = data.get('refreshToken') or data.get('refresh_token', '')

if access_token and len(access_token) > 100:
    env = re.sub(r'CTRADER_ACCESS_TOKEN=.*', f'CTRADER_ACCESS_TOKEN={access_token}', env)
    env = re.sub(r'CTRADER_REFRESH_TOKEN=.*', f'CTRADER_REFRESH_TOKEN={refresh_token}', env)
    
    with open('.env', 'w') as f:
        f.write(env)
    
    print(f'✅ Tokens saved to .env')
    print(f'   Access token: {access_token[:30]}... ({len(access_token)} chars)')
    print(f'   Refresh token: {refresh_token[:30]}... ({len(refresh_token)} chars)')
    print('')
    echo 'Now start the system:'
    echo '  nohup .venv/bin/python -m src.agentic.main_agentic --mode live > /tmp/agentic_live.log 2>&1 &'
else:
    print(f'❌ Failed to get tokens. Response: {data}')
"
