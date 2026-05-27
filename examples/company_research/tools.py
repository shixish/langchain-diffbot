"""Tools the agent can call.

We give the agent three Diffbot-backed surfaces:

- `search_kg(dql)` — Diffbot Knowledge Graph via DQL.
- `web_search(query)` — Diffbot web search.
- `extract_url(url)` — Diffbot Analyze extract on a single URL.

Each tool is a thin `@tool` wrapper that:
  1. Calls the package's pre-built Diffbot class.
  2. Shapes / truncates the response so a single tool call doesn't blow
     past the model's per-minute input-token budget. Diffbot KG entities,
     web search results, and extracted pages can each run thousands of
     tokens — without shaping, a multi-step agent will hit rate limits
     fast. The same projection-allowlist + content-truncation pattern is
     applied to each surface so the example demonstrates how to keep an
     agent loop token-efficient end-to-end.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from diffbot.errors import APIError
from langchain_core.tools import tool

from langchain_diffbot import (
    DiffbotExtractTool,
    DiffbotKnowledgeGraphRetriever,
    DiffbotWebSearchRetriever,
)

# Projection allowlist for KG entities. Only these top-level fields ride
# along in metadata — full entities can easily be thousands of tokens each.
_KG_FIELDS = [
    "id",
    "type",
    "name",
    "homepageUri",
    "nbEmployees",
    "industries",
    "location",
    "employments",
    "date",
]

_WEB_SEARCH_K = 5
_WEB_SEARCH_CONTENT_CHARS = 800
_EXTRACT_CONTENT_CHARS = 4000


@lru_cache(maxsize=1)
def _kg_retriever() -> DiffbotKnowledgeGraphRetriever:
    # Lazy so importing this module doesn't require DIFFBOT_API_TOKEN.
    return DiffbotKnowledgeGraphRetriever(k=5, fields=_KG_FIELDS)


@lru_cache(maxsize=1)
def _web_retriever() -> DiffbotWebSearchRetriever:
    return DiffbotWebSearchRetriever(
        k=_WEB_SEARCH_K,
        fields=["title", "pageUrl", "score"],
    )


@lru_cache(maxsize=1)
def _extract_tool() -> DiffbotExtractTool:
    return DiffbotExtractTool()


@tool
def search_kg(dql_query: str) -> list[dict]:
    """Search the Diffbot Knowledge Graph with a DQL query.

    DQL (Diffbot Query Language) syntax cheatsheet:

    - Filter by type: `type:Organization`, `type:Person`, `type:Article`
    - Exact match: `name:"Diffbot"`
    - Nested fields use dots: `location.city.name:"Austin"`
    - Combine filters with spaces (AND): `type:Organization industries:"Robotics"`
    - Sort with `sortBy:<field> desc` (e.g. `sortBy:nbEmployees desc`)

    Examples:
        - `type:Organization location.city.name:"Austin" industries:"Robotics"`
        - `type:Person employments.{employer.name:"Diffbot" isCurrent:true}`
        - `type:Article tags.label:"Artificial Intelligence" sortBy:date desc`

    Returns a list of entity dicts. Each entity has `summary` (description/summary
    text), `id`, `type`, `name`, and a few projected fields like `homepageUri`,
    `nbEmployees`, `industries`, `location`, `employments`, `date`. Other KG
    fields are intentionally omitted to keep responses small — refine the DQL
    query if you need different information.
    """
    try:
        docs = _kg_retriever().invoke(dql_query)
    except APIError as exc:
        # Surface DQL syntax errors back to the model so it can refine and retry.
        return [
            {
                "error": (
                    f"Diffbot rejected the query ({exc.status_code}): "
                    f"{exc.message or 'see body'}. Refine the DQL and try again."
                )
            }
        ]
    return [{"summary": d.page_content, **d.metadata} for d in docs]


@tool
def web_search(query: str) -> list[dict]:
    """Search the web via Diffbot. Use when the KG comes up short or you need current info.

    Args:
        query: natural-language search string.

    Returns up to 5 results, each with `title`, `pageUrl`, `score`, and a
    truncated `content` snippet (~800 chars). If you need the full page,
    pass the `pageUrl` to `extract_url`.
    """
    docs = _web_retriever().invoke(query)
    return [
        {
            **d.metadata,
            "content": d.page_content[:_WEB_SEARCH_CONTENT_CHARS],
        }
        for d in docs
    ]


@tool
def extract_url(url: str) -> dict[str, Any]:
    """Fetch and read a single web page via Diffbot's Analyze API.

    Args:
        url: page to extract.

    Returns a dict with `content` (markdown), `title`, `pageUrl`, `type`. The
    content is truncated (~4000 chars) to stay inside per-minute token
    budgets — call this on specific URLs you want to drill into, not on
    everything.
    """
    raw = _extract_tool().invoke({"url": url})
    if "error" in raw:
        return raw
    return {
        **raw,
        "content": (raw.get("content") or "")[:_EXTRACT_CONTENT_CHARS],
    }
