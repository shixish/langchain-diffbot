"""Diffbot tools — agent-callable wrappers around individual SDK methods.

Each tool is a thin BaseTool around one `diffbot` method. Args schemas mirror
the SDK signatures one-for-one so agents calling these tools see the same
shape as a direct SDK call.
"""

from __future__ import annotations

from typing import Any

from diffbot.errors import ExtractionError
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_diffbot._base import _BaseDiffbotComponent


class _DiffbotExtractInput(BaseModel):
    url: str = Field(description="URL to extract structured content from.")
    api: str = Field(
        default="analyze",
        description=(
            "Diffbot extract API to call. Defaults to `analyze` "
            "(auto-detects content type)."
        ),
    )
    fmt: str = Field(
        default="markdown",
        description="Output format. `markdown` uses Diffbot's LLM-optimized mode.",
    )


class DiffbotExtractTool(_BaseDiffbotComponent, BaseTool):
    """Tool that extracts structured content from a URL via Diffbot's analyze API.

    Returns a small dict so the agent doesn't have to wade through the full
    raw response. On extraction failure (a 200 response with an `errorCode`
    body) returns a structured error dict instead of raising, so the agent
    can react. Auth / rate-limit errors propagate as exceptions — those are
    infra problems, not per-call signals.
    """

    name: str = "diffbot_extract"
    description: str = (
        "Extract structured content (title, text, type, resolved URL) from a "
        "single web page. Use for reading the contents of a known URL."
    )
    args_schema: type[BaseModel] = _DiffbotExtractInput

    @staticmethod
    def _shape_response(raw: dict[str, Any]) -> dict[str, Any]:
        objects = raw.get("objects") or []
        first = objects[0] if objects else {}
        return {
            "content": first.get("text") or raw.get("markdown") or "",
            "title": first.get("title") or raw.get("title"),
            "pageUrl": first.get("pageUrl") or raw.get("url"),
            "resolvedPageUrl": first.get("resolvedPageUrl"),
            "type": first.get("type") or raw.get("type"),
        }

    def _run(
        self,
        url: str,
        api: str = "analyze",
        fmt: str = "markdown",
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> dict[str, Any]:
        try:
            with self._sync_db() as db:
                raw = db.extract(url, api=api, fmt=fmt)
        except ExtractionError as e:
            return {"error": str(e), "errorCode": e.error_code}
        return self._shape_response(raw)

    async def _arun(
        self,
        url: str,
        api: str = "analyze",
        fmt: str = "markdown",
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> dict[str, Any]:
        try:
            async with self._async_db() as db:
                raw = await db.extract(url, api=api, fmt=fmt)
        except ExtractionError as e:
            return {"error": str(e), "errorCode": e.error_code}
        return self._shape_response(raw)


class _DiffbotWebSearchInput(BaseModel):
    text: str = Field(description="Natural-language search query.")
    num_results: int | None = Field(
        default=None, description="Max results to return. Server default if unset."
    )
    max_tokens: int | None = Field(
        default=None, description="Optional total content-token cap."
    )


class DiffbotWebSearchTool(_BaseDiffbotComponent, BaseTool):
    """Tool that performs a Diffbot web search and returns the raw result list.

    Use this when the agent needs the search results as-is (with score, title,
    pageUrl, content). For LangChain `Document` output use
    `DiffbotWebSearchRetriever` instead.
    """

    name: str = "diffbot_web_search"
    description: str = (
        "Search the web via Diffbot. Returns a list of results, each with "
        "title, pageUrl, score, and content."
    )
    args_schema: type[BaseModel] = _DiffbotWebSearchInput

    def _run(
        self,
        text: str,
        num_results: int | None = None,
        max_tokens: int | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> list[dict[str, Any]]:
        with self._sync_db() as db:
            body = db.web_search(text, num_results=num_results, max_tokens=max_tokens)
        return body.get("search_results", [])

    async def _arun(
        self,
        text: str,
        num_results: int | None = None,
        max_tokens: int | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> list[dict[str, Any]]:
        async with self._async_db() as db:
            body = await db.web_search(
                text, num_results=num_results, max_tokens=max_tokens
            )
        return body.get("search_results", [])


class _DiffbotEntitiesInput(BaseModel):
    text: str = Field(description="Text to extract entities and sentiment from.")
    lang: str = Field(default="auto", description="Language hint (`auto` to detect).")


class DiffbotEntitiesTool(_BaseDiffbotComponent, BaseTool):
    """Tool that identifies entities and sentiment in text via Diffbot NLP.

    Returns the SDK response dict as-is — it's small (entity list + sentiment).
    Entity IDs in the response can be looked up in the KG via
    `DiffbotKnowledgeGraphTool` or `DiffbotKnowledgeGraphRetriever` using
    `id:or("id1","id2",...)`.
    """

    name: str = "diffbot_entities"
    description: str = (
        "Identify entities (people, organizations, places, ...) and sentiment "
        "in a piece of text. Returns entity IDs that can be looked up in the "
        "Diffbot Knowledge Graph."
    )
    args_schema: type[BaseModel] = _DiffbotEntitiesInput

    def _run(
        self,
        text: str,
        lang: str = "auto",
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> dict[str, Any]:
        with self._sync_db() as db:
            return db.entities(text, lang=lang)

    async def _arun(
        self,
        text: str,
        lang: str = "auto",
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> dict[str, Any]:
        async with self._async_db() as db:
            return await db.entities(text, lang=lang)


class _DiffbotDQLInput(BaseModel):
    query: str = Field(
        description='DQL query, e.g. `type:Organization name:"Diffbot"`.'
    )
    size: int = Field(default=10, description="Max results.")
    from_: int = Field(default=0, description="Result offset.")
    filter: str | None = Field(
        default=None, description="Optional DQL filter expression."
    )


class DiffbotKnowledgeGraphTool(_BaseDiffbotComponent, BaseTool):
    """Tool that runs a DQL query against the Diffbot Knowledge Graph.

    Returns the raw response dict (with `data`, `hits`, etc.). For LangChain
    `Document` output use `DiffbotKnowledgeGraphRetriever` instead.

    Best for agents that have been instructed in DQL syntax.
    """

    name: str = "diffbot_knowledge_graph"
    description: str = (
        "Query the Diffbot Knowledge Graph with a DQL expression "
        '(e.g. `type:Organization location.city.name:"Boston"`). '
        "Returns the raw response — use only if you know DQL syntax."
    )
    args_schema: type[BaseModel] = _DiffbotDQLInput

    def _run(
        self,
        query: str,
        size: int = 10,
        from_: int = 0,
        filter: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> dict[str, Any]:
        with self._sync_db() as db:
            body = db.dql(query, size=size, from_=from_, filter=filter)
        if not isinstance(body, dict):
            msg = "Unexpected non-JSON DQL response."
            raise TypeError(msg)
        return body

    async def _arun(
        self,
        query: str,
        size: int = 10,
        from_: int = 0,
        filter: str | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> dict[str, Any]:
        async with self._async_db() as db:
            body = await db.dql(query, size=size, from_=from_, filter=filter)
        if not isinstance(body, dict):
            msg = "Unexpected non-JSON DQL response."
            raise TypeError(msg)
        return body
