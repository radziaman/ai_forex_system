"""
Economic Calendar integration with event detection.
Fetches high-impact events and suppresses trading during critical periods.
"""
import time
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


HIGH_IMPACT_EVENTS = {
    "Non-Farm Payrolls": "NFP",
    "Interest Rate Decision": "RATE",
    "GDP": "GDP",
    "CPI": "CPI",
    "FOMC": "FOMC",
    "Retail Sales": "RETAIL",
    "Industrial Production": "INDPROD",
    "Unemployment Rate": "UNEMP",
    "ISM Manufacturing": "ISM",
    "ISM Services": "ISM_SVC",
    "Initial Jobless Claims": "JOBLESS",
    "PPI": "PPI",
    "Trade Balance": "TRADE",
    "Michigan Consumer Sentiment": "MICH",
    "Existing Home Sales": "HOME",
    "Consumer Confidence": "CONF",
    "Factory Orders": "FACTORY",
    "Durable Goods": "DURABLE",
    "Housing Starts": "HOUSING",
    "Building Permits": "BUILD",
    "Philadelphia Fed Index": "PHIL_FED",
    "Empire State Index": "EMPIRE",
    "Existing Home Sales": "HOME_SALES",
    "New Home Sales": "NEW_HOME",
    "Chicago PMI": "CHI_PMI",
    "Dallas Fed Index": "DAL_FED",
    "Richmond Fed Index": "RICH_FED",
    "Kansas City Fed Index": "KC_FED",
}

CURRENCY_KEYWORDS = {
    "USD": ["FOMC", "Federal Reserve", "Non-Farm", "Unemployment", "GDP", "CPI", "ISM", "Treasury"],
    "EUR": ["ECB", "Eurozone", "German", "French", "Italian", "EU"],
    "GBP": ["BOE", "Bank of England", "UK ", "British", "London"],
    "JPY": ["BOJ", "Bank of Japan", "Japanese", "Tokyo CPI"],
    "AUD": ["RBA", "Reserve Bank of Australia", "Australian"],
    "CAD": ["BOC", "Bank of Canada", "Canadian"],
    "NZD": ["RBNZ", "Reserve Bank of New Zealand", "New Zealand"],
    "CHF": ["SNB", "Swiss National Bank", "Swiss"],
}


@dataclass
class EconomicEvent:
    timestamp: float
    title: str
    currency: str
    impact: str
    forecast: Optional[str] = None
    previous: Optional[str] = None
    actual: Optional[str] = None
    event_id: str = ""

    @property
    def is_high_impact(self) -> bool:
        return self.impact.lower() in ("high", "nonfarm", "employment")

    @property
    def suppress_minutes_before(self) -> int:
        return 30 if self.is_high_impact else 15

    @property
    def suppress_minutes_after(self) -> int:
        return 30 if self.is_high_impact else 15


class EconomicCalendar:
    def __init__(
        self,
        cache_path: str = "data/alternative_data/economic_calendar.json",
        cache_ttl_hours: int = 6,
    ):
        self.cache_path = cache_path
        self.cache_ttl = cache_ttl_hours * 3600
        self.events: List[EconomicEvent] = []
        self._last_fetch = 0.0

    def fetch(
        self,
        days_forward: int = 7,
        days_backward: int = 1,
        currencies: Optional[List[str]] = None,
    ) -> List[EconomicEvent]:
        cached = self._load_cache()
        if cached is not None:
            self.events = cached
            return self.events

        self.events = self._fetch_from_sources(days_forward, days_backward, currencies)
        self._save_cache(self.events)
        return self.events

    def _fetch_from_sources(
        self, days_forward: int, days_backward: int, currencies: Optional[List[str]] = None,
    ) -> List[EconomicEvent]:
        events = []
        events.extend(self._fetch_forexfactory(days_forward, days_backward))
        if not events:
            events.extend(self._fetch_fred_calendar(days_forward))
        if not events:
            events.extend(self._generate_fallback_events(days_forward))
        if currencies:
            events = [e for e in events if e.currency in currencies]
        return events

    def _fetch_forexfactory(
        self, days_forward: int, days_backward: int,
    ) -> List[EconomicEvent]:
        if not REQUESTS_AVAILABLE:
            return self._generate_fallback_events(days_forward)
        events = []
        today = datetime.now()
        for offset in range(-days_backward, days_forward + 1):
            date = today + timedelta(days=offset)
            date_str = date.strftime("%Y-%m-%d")
            url = (
                "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
                if offset in (-1, 0, 1, 2, 3, 4, 5, 6, 7)
                else ""
            )
            if not url:
                continue
            try:
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    for item in data if isinstance(data, list) else data.get("data", []):
                        ev = self._parse_forexfactory_item(item, date_str)
                        if ev:
                            events.append(ev)
            except Exception:
                continue
        return events

    def _parse_forexfactory_item(self, item: Dict, date_str: str) -> Optional[EconomicEvent]:
        try:
            title = item.get("title", item.get("event", ""))
            impact = item.get("impact", item.get("importance", "low"))
            currency = item.get("country", item.get("currency", "USD"))
            time_str = item.get("time", "")
            if time_str:
                dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
            else:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
            forecast = item.get("forecast", item.get("estimate"))
            previous = item.get("previous")
            return EconomicEvent(
                timestamp=dt.timestamp(),
                title=title,
                currency=currency,
                impact=str(impact).lower(),
                forecast=str(forecast) if forecast else None,
                previous=str(previous) if previous else None,
                event_id=f"{title}_{dt.timestamp()}",
            )
        except Exception:
            return None

    def _fetch_fred_calendar(self, days_forward: int) -> List[EconomicEvent]:
        if not REQUESTS_AVAILABLE:
            return []
        events = []
        fred_api_key = os.getenv("FRED_API_KEY", "")
        if not fred_api_key:
            return []
        url = f"https://api.stlouisfed.org/fred/releases?api_key={fred_api_key}&file_type=json"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json().get("releases", [])
                for release in data[:50]:
                    rel_id = release.get("id")
                    rel_url = (
                        f"https://api.stlouisfed.org/fred/release/dates?"
                        f"api_key={fred_api_key}&release_id={rel_id}&"
                        f"include_release_dates_with_no_data=false&file_type=json"
                    )
                    r2 = requests.get(rel_url, timeout=10)
                    if r2.status_code == 200:
                        dates = r2.json().get("release_dates", [])
                        for rd in dates[:3]:
                            date_str = rd.get("date")
                            if date_str:
                                dt = datetime.strptime(date_str, "%Y-%m-%d")
                                if dt < datetime.now() + timedelta(days=days_forward):
                                    events.append(EconomicEvent(
                                        timestamp=dt.timestamp(),
                                        title=release.get("name", "Economic Release"),
                                        currency="USD",
                                        impact="high" if any(
                                            kw in release.get("name", "").lower()
                                            for kw in ["gdp", "cpi", "employment", "fed", "ism"]
                                        ) else "low",
                                        event_id=f"fred_{rel_id}_{date_str}",
                                    ))
        except Exception:
            pass
        return events

    def _generate_fallback_events(self, days_forward: int) -> List[EconomicEvent]:
        today = datetime.now()
        events = []
        high_impact_dates = {
            0: "USD", 1: "EUR", 2: "USD", 3: "JPY", 4: "GBP",
            5: "USD", 6: "USD", 7: "EUR", 8: "USD", 9: "JPY",
            10: "GBP", 11: "USD", 12: "EUR", 13: "USD", 14: "JPY",
            15: "USD", 16: "GBP", 17: "USD", 18: "EUR", 19: "USD",
            20: "JPY", 21: "USD", 22: "GBP", 23: "USD",
        }
        for day_offset in range(days_forward + 1):
            d = today + timedelta(days=day_offset)
            weekday = d.weekday()
            if weekday in high_impact_dates:
                for hour_offset in [8, 10, 13, 15]:
                    for minute in [0, 30]:
                        event_dt = d.replace(hour=hour_offset, minute=minute, second=0)
                        if event_dt > today:
                            currency = high_impact_dates.get(weekday, "USD")
                            events.append(EconomicEvent(
                                timestamp=event_dt.timestamp(),
                                title=f"High-Impact Economic Data ({currency})",
                                currency=currency,
                                impact="high",
                                event_id=f"fallback_{event_dt.timestamp()}",
                            ))
        return events

    def is_suppressed(self, timestamp: Optional[float] = None) -> Tuple[bool, Optional[EconomicEvent]]:
        ts = timestamp or time.time()
        for event in self.events:
            before = event.suppress_minutes_before * 60
            after = event.suppress_minutes_after * 60
            if event.timestamp - before <= ts <= event.timestamp + after:
                return True, event
        return False, None

    def get_upcoming_events(self, hours: int = 24) -> List[EconomicEvent]:
        now = time.time()
        cutoff = now + hours * 3600
        return sorted(
            [e for e in self.events if now <= e.timestamp <= cutoff],
            key=lambda e: e.timestamp,
        )

    def get_events_by_currency(self, currency: str, limit: int = 20) -> List[EconomicEvent]:
        return sorted(
            [e for e in self.events if e.currency == currency],
            key=lambda e: e.timestamp,
        )[:limit]

    def _load_cache(self) -> Optional[List[EconomicEvent]]:
        if not os.path.exists(self.cache_path):
            return None
        try:
            mtime = os.path.getmtime(self.cache_path)
            if time.time() - mtime > self.cache_ttl:
                return None
            with open(self.cache_path) as f:
                data = json.load(f)
            return [EconomicEvent(**item) for item in data]
        except Exception:
            return None

    def _save_cache(self, events: List[EconomicEvent]):
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            data = [
                {
                    "timestamp": e.timestamp,
                    "title": e.title,
                    "currency": e.currency,
                    "impact": e.impact,
                    "forecast": e.forecast,
                    "previous": e.previous,
                    "event_id": e.event_id,
                }
                for e in events
            ]
            with open(self.cache_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass
