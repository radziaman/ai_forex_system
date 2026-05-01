"""Secrets management — loads credentials from environment variables, never git-tracked files."""
import os
from pathlib import Path
from typing import Optional


def load_env_file(env_path: Optional[str] = None) -> None:
    """Load .env file if it exists."""
    path = Path(env_path or ".env")
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            if key not in os.environ:
                os.environ[key] = value


class Secrets:
    """Centralized secrets — never stored in code, always from env/.env."""

    def __init__(self, env_file: Optional[str] = None):
        load_env_file(env_file)

    @property
    def ctrader_app_id(self) -> str:
        return os.getenv("CTRADER_APP_ID", "")

    @property
    def ctrader_app_secret(self) -> str:
        return os.getenv("CTRADER_APP_SECRET", "")

    @property
    def ctrader_access_token(self) -> str:
        return os.getenv("CTRADER_ACCESS_TOKEN", "")

    @property
    def ctrader_refresh_token(self) -> str:
        return os.getenv("CTRADER_REFRESH_TOKEN", "")

    @property
    def ctrader_account_id(self) -> int:
        return int(os.getenv("CTRADER_ACCOUNT_ID", "0"))

    @property
    def telegram_bot_token(self) -> str:
        return os.getenv("TELEGRAM_BOT_TOKEN", "")

    @property
    def telegram_chat_id(self) -> str:
        return os.getenv("TELEGRAM_CHAT_ID", "")

    @property
    def redis_url(self) -> str:
        return os.getenv("REDIS_URL", "")

    @property
    def log_level(self) -> str:
        return os.getenv("LOG_LEVEL", "INFO")

    @property
    def is_demo(self) -> bool:
        return os.getenv("CTRADER_DEMO", "true").lower() == "true"

    @property
    def oanda_api_key(self) -> str:
        return os.getenv("OANDA_API_KEY", "")

    @property
    def oanda_account_id(self) -> str:
        return os.getenv("OANDA_ACCOUNT_ID", "")

    @property
    def oanda_practice(self) -> bool:
        return os.getenv("OANDA_PRACTICE", "true").lower() == "true"

    @property
    def provider(self) -> str:
        return os.getenv("TRADING_PROVIDER", "ctrader").lower()

    def validate(self) -> list:
        missing = []
        provider = self.provider
        if provider == "ctrader":
            if not self.ctrader_app_id: missing.append("CTRADER_APP_ID")
            if not self.ctrader_app_secret: missing.append("CTRADER_APP_SECRET")
            if not self.ctrader_account_id: missing.append("CTRADER_ACCOUNT_ID")
        elif provider == "oanda":
            if not self.oanda_api_key: missing.append("OANDA_API_KEY")
            if not self.oanda_account_id: missing.append("OANDA_ACCOUNT_ID")
        return missing
