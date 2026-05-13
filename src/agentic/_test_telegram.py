"""Verify Telegram bot token and send direct test message."""
import requests, os

# Read from .env
token = ""
chat_id = ""
with open(".env") as f:
    for line in f:
        line = line.strip()
        if line.startswith("TELEGRAM_BOT_TOKEN"):
            token = line.split("=", 1)[1].strip().strip('"').strip("'")
        elif line.startswith("TELEGRAM_CHAT_ID"):
            chat_id = line.split("=", 1)[1].strip().strip('"').strip("'")

print(f"Chat ID: {chat_id}")

# 1. Verify bot token
resp = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=10)
if resp.status_code == 200:
    bot = resp.json()["result"]
    print(f"Bot: @{bot['username']} (id={bot['id']}) — token VALID")
else:
    print(f"Token INVALID: {resp.status_code} {resp.text[:200]}")
    exit(1)

# 2. Send direct synchronous test
payload = {
    "chat_id": chat_id,
    "text": "\u2705 Direct test — no background thread\nIf you see this, Telegram works.",
    "parse_mode": "HTML",
}
resp2 = requests.post(
    f"https://api.telegram.org/bot{token}/sendMessage",
    json=payload, timeout=10
)
print(f"Send: HTTP {resp2.status_code}")
if resp2.status_code == 200:
    print("MESSAGE SENT — check Telegram now")
elif resp2.status_code == 403:
    print("BOT BLOCKED — user must send /start to the bot first, or chat_id is wrong")
elif resp2.status_code == 400:
    print(f"Bad request: {resp2.text[:200]}")
elif resp2.status_code == 404:
    print("Bot token invalid or bot not found")
else:
    print(f"Response: {resp2.text[:200]}")
