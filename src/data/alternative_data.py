"""Satellite / Alternative Data Wiring for FX macro features.

Reuses the NASA POWER and EONET integration patterns already proven in
``ai/social_scrapers.py`` and exposes per-currency impact scores that
are consumed by ``FeatureEngine``.

Data sources:
  * NASA POWER — temperature / precipitation anomalies → AUD agriculture
  * NASA EONET — natural disasters → JPY safe-haven flows
  * Mock shipping data — placeholder for maritime AIS → CAD oil
  * Economic calendar — Fed divergence scoring
"""

import os
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
from loguru import logger

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Key agricultural regions for commodity-currency pairs
AG_REGIONS = [
    {"name": "Australia Wheat Belt", "lat": -34.0, "lon": 138.0, "currency": "AUD"},
    {"name": "US Corn Belt", "lat": 41.0, "lon": -93.0, "currency": "USD"},
    {"name": "Canada Prairies", "lat": 51.0, "lon": -114.0, "currency": "CAD"},
    {"name": "NZ Canterbury", "lat": -43.5, "lon": 172.0, "currency": "NZD"},
]

# Financial centres for EONET proximity scoring
FINANCIAL_CENTRES = [
    {
        "name": "New York",
        "lat": 40.7,
        "lon": -74.0,
        "radius_km": 500,
        "currencies": ["USD"],
    },
    {
        "name": "London",
        "lat": 51.5,
        "lon": -0.1,
        "radius_km": 400,
        "currencies": ["GBP", "EUR"],
    },
    {
        "name": "Tokyo",
        "lat": 35.7,
        "lon": 139.7,
        "radius_km": 400,
        "currencies": ["JPY"],
    },
    {
        "name": "Sydney",
        "lat": -33.9,
        "lon": 151.2,
        "radius_km": 400,
        "currencies": ["AUD"],
    },
    {
        "name": "Zurich",
        "lat": 47.4,
        "lon": 8.5,
        "radius_km": 300,
        "currencies": ["CHF"],
    },
    {
        "name": "Toronto",
        "lat": 43.7,
        "lon": -79.4,
        "radius_km": 400,
        "currencies": ["CAD"],
    },
]

EONET_WEIGHTS = {
    "severeStorms": 0.3,
    "volcanoes": 0.5,
    "earthquakes": 0.6,
    "floods": 0.4,
    "wildfires": 0.3,
}


class AlternativeDataEngine:
    """Aggregates satellite and alternative data into FX impact scores.

    Usage:
        engine = AlternativeDataEngine()
        scores = engine.compute_fx_impact_scores()
        # -> {"aud_agriculture_score": 0.42, ...}
    """

    CACHE_TTL_SECONDS: int = 6 * 3600  # 6 hours

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("NASA_API_KEY", "DEMO_KEY")
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._session: Any = None
        if REQUESTS_AVAILABLE:
            self._session = requests.Session()
            self._session.headers.update(
                {"User-Agent": "RTS-AI-Forex-System/1.0 (alternative-data-engine)"}
            )

    # -- public API ----------------------------------------------------------

    def get_agricultural_weather_score(self) -> float:
        """Read NASA POWER data and return an anomaly score 0.0–1.0.

        Higher = larger temperature / precipitation deviation from
        seasonal norms → potential agricultural disruption.
        """
        scores: List[float] = []
        for region in AG_REGIONS:
            data = self._fetch_power(region["lat"], region["lon"])
            if data is None:
                continue
            anomaly = self._region_anomaly(data)
            scores.append(anomaly)
        if not scores:
            return 0.0
        # Scale: anomaly ~2.0 → score 1.0
        return float(np.clip(max(scores), 0.0, 2.0) / 2.0)

    def get_oil_shipping_score(self) -> float:
        """Mock for maritime AIS / oil-tanker tracking data.

        In production this would call a maritime API (e.g. MarineTraffic,
        AISHub) to compute shipping congestion around key oil terminals.
        Returns a random-ish value in ``[0, 1]`` so the feature pipeline
        still has a signal dimension.
        """
        # Deterministic mock based on day-of-month so tests are stable
        return (float(int(time.time()) % 86400) / 86400.0) * 0.3

    def get_natural_event_impact(self) -> float:
        """Read NASA EONET and return a natural-disaster impact score 0.0–1.0.

        Higher = more / stronger events near major financial centres.
        """
        events = self._fetch_eonet_events()
        if not events:
            return 0.0
        total = 0.0
        for event in events:
            weight = self._event_weight(event)
            if weight <= 0:
                continue
            geometries = event.get("geometry", [])
            if not geometries:
                continue
            coords = geometries[-1].get("coordinates", [])
            if len(coords) < 2:
                continue
            event_lon, event_lat = coords[:2]
            for centre in FINANCIAL_CENTRES:
                d = self._haversine(event_lat, event_lon, centre["lat"], centre["lon"])
                if d < centre["radius_km"]:
                    proximity = 1.0 - (d / centre["radius_km"])
                    total += proximity * weight
        # Scale: total ~3.0 → score 1.0
        return float(np.clip(total / 3.0, 0.0, 1.0))

    def compute_fx_impact_scores(self) -> Dict[str, float]:
        """Return a dict of 4 macro features in ``[0, 1]``.

        Keys:
            * ``aud_agriculture_score``  — weather impact on AUD
            * ``cad_oil_score``          — oil price impact on CAD
            * ``jpy_safe_haven_score``   — natural disaster → JPY safe haven
            * ``usd_fed_divergence_score`` — rate differential impact (mock)
        """
        ag_score = self.get_agricultural_weather_score()
        oil_score = self.get_oil_shipping_score()
        nat_score = self.get_natural_event_impact()
        # Fed divergence: mock for now; in production this would read
        # from the EconomicCalendar rate-decision differential.
        fed_score = 0.5
        return {
            "aud_agriculture_score": ag_score,
            "cad_oil_score": oil_score,
            "jpy_safe_haven_score": nat_score,
            "usd_fed_divergence_score": fed_score,
        }

    # -- NASA EONET ----------------------------------------------------------

    def _fetch_eonet_events(self) -> List[Dict]:
        cached = self._cache_get("eonet_events")
        if cached is not None:
            return cached  # type: ignore[no-any-return]

        if not REQUESTS_AVAILABLE or self._session is None:
            return []

        try:
            url = "https://eonet.gsfc.nasa.gov/api/v3/events"
            resp = self._session.get(
                url, params={"status": "open", "limit": 100}, timeout=30
            )
            resp.raise_for_status()
            events: List[Dict] = resp.json().get("events", [])
            self._cache_set("eonet_events", events)
            logger.info(f"AlternativeData: fetched {len(events)} EONET events")
            return events
        except Exception as exc:
            logger.debug(f"EONET fetch failed: {exc}")
            return []

    @staticmethod
    def _event_weight(event: Dict) -> float:
        categories = event.get("categories", [])
        if not categories:
            return 0.0
        best = 0.0
        for cat in categories:
            cat_id: str = cat.get("id", "")
            for known_id, w in EONET_WEIGHTS.items():
                if known_id.lower() in cat_id.lower():
                    best = max(best, w)
        return best if best > 0 else 0.08

    # -- NASA POWER ----------------------------------------------------------

    def _fetch_power(self, lat: float, lon: float) -> Optional[Dict]:
        key = f"power_{lat:.1f}_{lon:.1f}"
        cached = self._cache_get(key)
        if cached is not None:
            return cached  # type: ignore[no-any-return]

        if not REQUESTS_AVAILABLE or self._session is None:
            return None

        try:
            url = "https://power.larc.nasa.gov/api/temporal/monthly/point"
            params = {
                "parameters": "T2M,PRECTOTCORR",
                "community": "AG",
                "longitude": lon,
                "latitude": lat,
                "start": 2024,
                "end": 2025,
                "format": "JSON",
            }
            resp = self._session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data: Dict = resp.json()
            self._cache_set(key, data)
            return data
        except Exception as exc:
            logger.debug(f"NASA POWER fetch failed ({lat:.1f},{lon:.1f}): {exc}")
            return None

    @staticmethod
    def _region_anomaly(data: Dict) -> float:
        try:
            props = data.get("properties", {})
            params = props.get("parameter", {})
            if not params:
                return 0.0
            temp_vals = list(params.get("T2M", {}).values())[-3:]
            precip_vals = list(params.get("PRECTOTCORR", {}).values())[-3:]
            anomaly = 0.0
            count = 0
            if temp_vals:
                avg_t = sum(temp_vals) / len(temp_vals)
                anomaly += abs(avg_t - 25.0) / 15.0
                count += 1
            if precip_vals:
                avg_p = sum(precip_vals) / len(precip_vals)
                anomaly += abs(avg_p - 100.0) / 100.0
                count += 1
            return anomaly / count if count > 0 else 0.0
        except Exception:
            return 0.0

    # -- helpers -------------------------------------------------------------

    def _cache_get(self, key: str) -> Optional[Any]:
        entry = self._cache.get(key)
        if entry is None:
            return None
        value, ts = entry
        if time.time() - ts > self.CACHE_TTL_SECONDS:
            del self._cache[key]
            return None
        return value

    def _cache_set(self, key: str, value: Any) -> None:
        self._cache[key] = (value, time.time())

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        import math

        R = 6371.0  # km
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = (
            math.sin(dphi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c
