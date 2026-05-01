# cTrader IC Markets Integration Guide

## Prerequisites

1. **Create cTrader App**:
   - Go to https://connect.ctrader.com/
   - Click "Create App"
   - Name: `RTS - AI FX Trading System`
   - Redirect URI: `http://localhost:8080`
   - Save `App ID` and `App Secret`

2. **Get Access Token**:
   - Visit: `https://connect.ctrader.com/apps/YOUR_APP_ID/`
   - Authorize the app
   - Copy the `Access Token`

3. **IC Markets Account**:
   - Login to IC Markets cTrader portal
   - Get your `Account ID` (not login name)

## Configuration

Run the setup script:
```bash
cd /Users/radziaman/Antigravity/ai_forex_system
source venv/bin/activate
python scripts/setup_ctrader.py
```

Enter when prompted:
- App ID: (from step 1)
- App Secret: (from step 1)
- Access Token: (from step 2)
- Account ID: (from step 3)
- Demo: `y` (for testing) or `n` (for live)

## OAuth2 Flow (Manual)

If the automated flow doesn't work:

1. Visit this URL (replace `YOUR_APP_ID`):
```
https://connect.ctrader.com/apps/YOUR_APP_ID/oauth2/auth?response_type=token&redirect_uri=http://localhost:8080&scope=trading
```

2. Authorize and you'll be redirected to:
```
http://localhost:8080/#access_token=YOUR_ACCESS_TOKEN&token_type=bearer&expires_in=86400
```

3. Copy `YOUR_ACCESS_TOKEN` from the URL

## Testing Connection

```bash
python scripts/test_ctrader_connection.py
```

## Live Trading

```bash
python scripts/live_trading.py
```

## IC Markets Specific Settings

- **Live Server**: `live.ctrader.com`
- **Demo Server**: `demo.ctrader.com`
- **Port**: `5035`
- **Protocol**: WebSocket (recommended) or TCP

## Symbol Mapping (IC Markets)

| Symbol | cTrader ID | Pip Value |
|---------|-------------|-----------|
| EURUSD  | 1           | 0.0001    |
| GBPUSD  | 2           | 0.0001    |
| USDJPY  | 3           | 0.01      |
| XAUUSD  | 4           | 0.01      |
| AUDUSD  | 7           | 0.0001    |

## Troubleshooting

1. **"protobuf" errors**: Already fixed - using protobuf 3.20.1
2. **Connection refused**: Check firewall allows port 5035
3. **Authentication failed**: Verify access token is valid (expires in 24h)
4. **Account not found**: Make sure Account ID is correct (numeric, not login name)

## Security Notes

- Never commit `config/ctrader_config.json` to git
- Add to `.gitignore`:
  ```
  config/ctrader_config.json
  ```
- Use environment variables for production:
  ```bash
  export CTRADER_APP_ID="your_app_id"
  export CTRADER_APP_SECRET="your_app_secret"
  export CTRADER_ACCESS_TOKEN="your_access_token"
  ```
