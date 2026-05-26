"""Tools the agent can call."""

from __future__ import annotations

from functools import lru_cache

import httpx
from langchain_core.tools import tool
from langchain_diffbot import DiffbotKnowledgeGraphRetriever

# Projection allowlist: only these top-level entity fields ride along in
# metadata. Without it a single tool response can blow past the model's
# per-minute input-token budget.
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


@lru_cache(maxsize=1)
def _get_retriever() -> DiffbotKnowledgeGraphRetriever:
    # Lazy so importing this module doesn't require DIFFBOT_API_TOKEN.
    return DiffbotKnowledgeGraphRetriever(k=5, fields=_KG_FIELDS)


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
        docs = _get_retriever().invoke(dql_query)
    except httpx.HTTPStatusError as exc:
        # Surface DQL syntax errors back to the model so it can refine and retry.
        return [{"error": f"Diffbot rejected the query ({exc.response.status_code}). Refine the DQL and try again."}]
    return [{"summary": d.page_content, **d.metadata} for d in docs]
