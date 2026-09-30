"""Pluggable web-research tools. No API key needed for the default path."""
from __future__ import annotations

import re

import httpx
from ddgs import DDGS

from .config import settings


def web_search(query: str, max_results: int | None = None) -> list[dict]:
    """Search the web. Uses Tavily when TAVILY_API_KEY is set, else DuckDuckGo."""
    n = max_results or settings.max_search_results
    if settings.tavily_api_key:
        return _tavily_search(query, n)
    return _ddg_search(query, n)


def _ddg_search(query: str, n: int) -> list[dict]:
    results: list[dict] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=n):
            results.append(
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                }
            )
    return results


def _tavily_search(query: str, n: int) -> list[dict]:
    resp = httpx.post(
        "https://api.tavily.com/search",
        json={
            "api_key": settings.tavily_api_key,
            "query": query,
            "max_results": n,
            "search_depth": "advanced",
        },
        timeout=settings.request_timeout_s,
    )
    resp.raise_for_status()
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": (r.get("content", "") or "")[:500],
        }
        for r in resp.json().get("results", [])
    ]


def fetch_page(url: str, max_chars: int = 6000) -> str:
    """Fetch a page and return plain text (best-effort, never raises)."""
    try:
        resp = httpx.get(
            url,
            timeout=settings.request_timeout_s,
            follow_redirects=True,
            headers={"User-Agent": "research-agent/0.1"},
        )
        resp.raise_for_status()
        text = re.sub(r"<script.*?</script>", " ", resp.text, flags=re.S | re.I)
        text = re.sub(r"<style.*?</style>", " ", text, flags=re.S | re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars]
    except Exception as e:  # noqa: BLE001 - research must be resilient
        return f"[fetch failed: {e}]"
