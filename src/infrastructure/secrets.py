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
    def provider(self) -> str:
        return os.getenv("TRADING_PROVIDER", "ctrader").lower()

    @property
    def lmax_username(self) -> str:
        return os.getenv("LMAX_USERNAME", "")

    @property
    def lmax_password(self) -> str:
        return os.getenv("LMAX_PASSWORD", "")

    @property
    def lmax_demo(self) -> bool:
        return os.getenv("LMAX_DEMO", "true").lower() == "true"

    def validate(self) -> list:
        missing = []
        p = self.provider
        if p == "ctrader" or p == "dukascopy":
            if not self.ctrader_app_id: missing.append("CTRADER_APP_ID")
            if not self.ctrader_app_secret: missing.append("CTRADER_APP_SECRET")
            if not self.ctrader_account_id: missing.append("CTRADER_ACCOUNT_ID")
        elif p == "lmax":
            if not self.lmax_username: missing.append("LMAX_USERNAME")
            if not self.lmax_password: missing.append("LMAX_PASSWORD")
        return missing
