"""Thin HTTP client for the Diffbot Knowledge Graph DQL endpoint."""

from __future__ import annotations

from typing import Any

import httpx

DEFAULT_BASE_URL = "https://kg.diffbot.com"
DQL_PATH = "/kg/v3/dql"


def _params(token: str, query: str, *, size: int, offset: int) -> dict[str, Any]:
    return {
        "token": token,
        "query": query,
        "size": size,
        "from": offset,
        "format": "json",
    }


class DiffbotKGClient:
    """Minimal sync + async client for Diffbot KG DQL search.

    Wraps a single endpoint so the retriever stays focused on document mapping.
    """

    def __init__(
        self,
        token: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
    ) -> None:
        self._token = token
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def search(self, query: str, *, size: int, offset: int = 0) -> dict[str, Any]:
        """Run a DQL query synchronously and return the parsed JSON body."""
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.get(
                f"{self._base_url}{DQL_PATH}",
                params=_params(self._token, query, size=size, offset=offset),
            )
            resp.raise_for_status()
            return resp.json()

    async def asearch(
        self, query: str, *, size: int, offset: int = 0
    ) -> dict[str, Any]:
        """Run a DQL query asynchronously and return the parsed JSON body."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(
                f"{self._base_url}{DQL_PATH}",
                params=_params(self._token, query, size=size, offset=offset),
            )
            resp.raise_for_status()
            return resp.json()
