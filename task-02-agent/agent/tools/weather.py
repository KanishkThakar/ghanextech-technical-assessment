"""Weather tool: Open-Meteo, which needs no API key.

Two calls: geocoding turns a place name into coordinates, then the forecast
endpoint returns current conditions and the next few days.
"""

from __future__ import annotations

from typing import Any, Dict

import requests

from .base import Tool, ToolError

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT_SECONDS = 15

# Open-Meteo returns WMO weather codes; these are the ones worth naming.
WEATHER_CODES = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    71: "slight snow",
    73: "moderate snow",
    75: "heavy snow",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


def _get(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        response = requests.get(url, params=params, timeout=TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        raise ToolError(f"Could not reach the weather service: {exc}") from exc
    if response.status_code >= 400:
        raise ToolError(f"The weather service returned {response.status_code}.")
    try:
        return response.json()
    except ValueError as exc:
        raise ToolError("The weather service returned something that is not JSON.") from exc


def get_weather(location: str, days: int = 3) -> Dict[str, Any]:
    """Current weather and a short forecast for a place name."""
    location = (location or "").strip()
    if not location:
        raise ToolError("A location is required, e.g. 'Mumbai'.")
    days = max(1, min(int(days or 3), 7))

    geo = _get(GEOCODE_URL, {"name": location, "count": 1, "language": "en", "format": "json"})
    results = geo.get("results") or []
    if not results:
        raise ToolError(f"Could not find a place called '{location}'. Try a city name.")
    place = results[0]

    forecast = _get(
        FORECAST_URL,
        {
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "forecast_days": days,
            "timezone": "auto",
        },
    )

    current = forecast.get("current") or {}
    daily = forecast.get("daily") or {}
    days_out = []
    for i, date in enumerate(daily.get("time", [])):
        code = (daily.get("weather_code") or [None])[i]
        daily_max = daily.get("temperature_2m_max") or []
        daily_min = daily.get("temperature_2m_min") or []
        rain = daily.get("precipitation_probability_max") or []
        days_out.append(
            {
                "date": date,
                "condition": WEATHER_CODES.get(code, f"code {code}"),
                "max_c": daily_max[i] if i < len(daily_max) else None,
                "min_c": daily_min[i] if i < len(daily_min) else None,
                "rain_chance_pct": rain[i] if i < len(rain) else None,
            }
        )

    return {
        "location": f"{place.get('name')}, {place.get('country')}",
        "current": {
            "temperature_c": current.get("temperature_2m"),
            "feels_like_c": current.get("apparent_temperature"),
            "humidity_pct": current.get("relative_humidity_2m"),
            "wind_kmh": current.get("wind_speed_10m"),
            "condition": WEATHER_CODES.get(current.get("weather_code"), "unknown"),
        },
        "forecast": days_out,
    }


WEATHER = Tool(
    name="get_weather",
    description=(
        "Get the current weather and a short daily forecast for a place. Use it whenever "
        "the user asks about weather, temperature, rain or what to wear somewhere."
    ),
    parameters={
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City or place name, e.g. 'Mumbai'."},
            "days": {
                "type": "integer",
                "description": "How many forecast days to return, 1 to 7. Default 3.",
                "minimum": 1,
                "maximum": 7,
            },
        },
        "required": ["location"],
        "additionalProperties": False,
    },
    func=get_weather,
)
