"""
client.py

A small registry of free (or free-tier) AI APIs and the environment
variable names to store their API keys (the "id" the project can read).

This file does NOT contain any secret keys. Store your actual keys in
environment variables (recommended) and set them on your system or in
VS Code launch configuration.

Usage:
    - Inspect available services: run `python client.py`
    - Get an API key in code: `get_api_key('huggingface')`

Add more services to the FREE_APIS dict as you need.
"""

import os
from typing import Optional

# Key: service id used in code. Value: metadata including env var name.
FREE_APIS = {
    "huggingface": {
        "display_name": "Hugging Face Inference API",
        "base_url": "https://api-inference.huggingface.co/models",
        "note": "Free tier available; requires API token",
        "api_key_env": "HUGGINGFACE_API_KEY",
    },
    "replicate": {
        "display_name": "Replicate",
        "base_url": "https://api.replicate.com/v1",
        "note": "Free credits for new users; requires API token",
        "api_key_env": "REPLICATE_API_TOKEN",
    },
    "libretranslate": {
        "display_name": "LibreTranslate (self-host or public instances)",
        "base_url": "https://libretranslate.com/",
        "note": "Open-source translation service; public instances may be free",
        "api_key_env": "LIBRETRANSLATE_API_KEY",
    },
    "assemblyai": {
        "display_name": "AssemblyAI (speech-to-text)",
        "base_url": "https://api.assemblyai.com/v2",
        "note": "Offers free credits; requires API token",
        "api_key_env": "ASSEMBLYAI_API_KEY",
    },
    "newsapi": {
        "display_name": "NewsAPI.org (news search)",
        "base_url": "https://newsapi.org/v2/everything",
        "note": "Free tier available for developers; useful for searching news articles",
        "api_key_env": "NEWSAPI_API_KEY",
    },
    "duckduckgo": {
        "display_name": "DuckDuckGo Instant Answer (search)",
        "base_url": "https://api.duckduckgo.com/",
        "note": "No API key required; limited instant answers and search metadata",
        "api_key_env": None,
    },
    "open-meteo": {
        "display_name": "Open-Meteo (weather)",
        "base_url": "https://api.open-meteo.com/v1/forecast",
        "note": "Truly free weather API with no key required",
        "api_key_env": None,
    },
}


def get_api_key(service_id: str) -> Optional[str]:
    """Return the API key read from the configured environment variable.

    - `service_id` must exist in `FREE_APIS`.
    - Returns `None` when the service does not require a key or if the
      environment variable is not set.
    """
    meta = FREE_APIS.get(service_id)
    if meta is None:
        raise KeyError(f"Unknown service id: {service_id!r}")
    env_name = meta.get("api_key_env")
    if not env_name:
        return None
    return os.environ.get(env_name)


def search_news(service_id: str, query: str, **kwargs) -> dict:
    """Search news / web using a supported service and return parsed JSON.

    Supported `service_id` values: `newsapi`, `duckduckgo`.

    Returns the raw JSON response (service-specific). Raises `KeyError`
    for unknown services and `RuntimeError` when an API key is required
    but missing.
    """
    import requests

    meta = FREE_APIS.get(service_id)
    if meta is None:
        raise KeyError(f"Unknown service id: {service_id!r}")

    base = meta.get("base_url")
    if service_id == "newsapi":
        api_key = get_api_key("newsapi")
        if not api_key:
            raise RuntimeError("NewsAPI requires an API key. Set NEWSAPI_API_KEY in env.")
        params = {"q": query, "pageSize": kwargs.get("pageSize", 20), "language": kwargs.get("language", "en")}
        headers = {"Authorization": api_key} 
        resp = requests.get(base, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()

    if service_id == "duckduckgo":
        params = {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
        params.update({k: v for k, v in kwargs.items() if k not in ("format",)})
        resp = requests.get(base, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()

    raise KeyError(f"search_news not implemented for service: {service_id}")


def show_available_apis() -> None:
    """Prints the known services and their environment variable names."""
    for sid, meta in FREE_APIS.items():
        env = meta.get("api_key_env") or "(no key required)"
        note = meta.get("note", "")
        print(f"{sid}: {meta['display_name']} - env var: {env}")
        if note:
            print(f"    note: {note}")


if __name__ == "__main__":
    show_available_apis()
