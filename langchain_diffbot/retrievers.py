"""Diffbot Knowledge Graph retriever."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, SecretStr, model_validator

from langchain_diffbot._client import DEFAULT_BASE_URL, DiffbotKGClient

DEFAULT_CONTENT_FIELDS: tuple[str, ...] = ("description", "summary", "name")
"""Default ordered priority for selecting `page_content` from an entity."""

DocumentMapper = Callable[[dict[str, Any]], Document]


def _default_entity_to_document(
    entity: dict[str, Any],
    *,
    content_fields: Sequence[str],
    fields: Sequence[str] | None,
) -> Document:
    """Map a Diffbot KG entity to a LangChain `Document`.

    `page_content` is the first non-empty value among `content_fields`.
    `metadata` is the remaining top-level fields, optionally narrowed to
    `fields` (the projection allowlist).
    """
    page_content = ""
    content_field_used: str | None = None
    for f in content_fields:
        value = entity.get(f)
        if value:
            page_content = value if isinstance(value, str) else str(value)
            content_field_used = f
            break

    if fields is None:
        metadata = {
            k: v for k, v in entity.items() if k != content_field_used
        }
    else:
        metadata = {
            k: entity[k]
            for k in fields
            if k in entity and k != content_field_used
        }
    return Document(page_content=page_content, metadata=metadata)


class DiffbotKnowledgeGraphRetriever(BaseRetriever):
    """Retriever backed by the Diffbot Knowledge Graph DQL endpoint.

    The `query` passed to `invoke` is a
    [DQL](https://docs.diffbot.com/reference/dql-quickstart) expression
    (e.g. `type:Organization industries:"Artificial Intelligence"`).

    Example:
        ```python
        from langchain_diffbot import DiffbotKnowledgeGraphRetriever

        retriever = DiffbotKnowledgeGraphRetriever(k=5)
        retriever.invoke('type:Organization location.city.name:"Boston"')
        ```

    Shaping the output (recommended for agent/tool use, where large entity
    payloads can blow past LLM input-token limits):

        ```python
        retriever = DiffbotKnowledgeGraphRetriever(
            k=5,
            fields=["id", "type", "name", "homepageUri", "nbEmployees"],
        )
        ```

    For full control, pass `document_mapper`:

        ```python
        def mapper(entity):
            return Document(
                page_content=entity.get("summary", ""),
                metadata={"id": entity["id"], "name": entity["name"]},
            )

        retriever = DiffbotKnowledgeGraphRetriever(document_mapper=mapper)
        ```
    """

    diffbot_api_token: SecretStr = Field(default=SecretStr(""))
    """Diffbot API token. Falls back to the `DIFFBOT_API_TOKEN` env var."""

    base_url: str = DEFAULT_BASE_URL
    """Override for the Diffbot KG base URL (useful for tests / proxies)."""

    k: int = 10
    """Default number of results to return. Can be overridden per `invoke` call."""

    timeout: float = 30.0
    """HTTP timeout in seconds for each request."""

    fields: list[str] | None = None
    """Allowlist of top-level entity keys to keep in `metadata`.

    `None` (default) keeps every field. Set this to a small list like
    `["id", "type", "name", "homepageUri"]` to drastically shrink Document
    payloads — important when the retriever feeds an LLM tool call, since
    full Diffbot KG entities can run thousands of tokens each.

    Ignored when `document_mapper` is set.
    """

    content_fields: list[str] = Field(default_factory=lambda: list(DEFAULT_CONTENT_FIELDS))
    """Ordered priority for selecting `page_content` from an entity.

    The first key in this list with a non-empty value wins. The chosen key
    is excluded from `metadata` to avoid duplicating data.

    Ignored when `document_mapper` is set.
    """

    document_mapper: DocumentMapper | None = None
    """Optional override mapping a raw entity dict to a `Document`.

    When provided, the retriever bypasses `fields`/`content_fields` and calls
    this function for each entity. Use this for total control over the
    Document shape (e.g. nested-field projection, custom formatting).
    """

    @model_validator(mode="after")
    def _resolve_token(self) -> DiffbotKnowledgeGraphRetriever:
        if not self.diffbot_api_token.get_secret_value():
            env_token = os.environ.get("DIFFBOT_API_TOKEN", "")
            if not env_token:
                msg = (
                    "A Diffbot API token is required. Pass `diffbot_api_token=...` "
                    "or set the `DIFFBOT_API_TOKEN` environment variable."
                )
                raise ValueError(msg)
            self.diffbot_api_token = SecretStr(env_token)
        return self

    def _client(self) -> DiffbotKGClient:
        return DiffbotKGClient(
            token=self.diffbot_api_token.get_secret_value(),
            base_url=self.base_url,
            timeout=self.timeout,
        )

    def _resolve_k(self, kwargs: dict[str, Any]) -> int:
        # Honor a per-call `k` override threaded through `invoke(..., k=N)`.
        k = kwargs.get("k", self.k)
        if not isinstance(k, int) or k <= 0:
            msg = f"`k` must be a positive integer, got {k!r}."
            raise ValueError(msg)
        return k

    def _hit_to_document(self, hit: dict[str, Any]) -> Document:
        # Diffbot returns each result as {"score": ..., "entity": {...}, "entity_ctx": ...}.
        # Older shapes (and our tests) sometimes embed entity fields at the top
        # level, so fall back to the hit itself when there's no nested entity.
        entity = hit.get("entity", hit)
        if self.document_mapper is not None:
            return self.document_mapper(entity)
        doc = _default_entity_to_document(
            entity,
            content_fields=self.content_fields,
            fields=self.fields,
        )
        if "score" in hit:
            doc.metadata["score"] = hit["score"]
        return doc

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> list[Document]:
        k = self._resolve_k(kwargs)
        body = self._client().search(query, size=k)
        return [self._hit_to_document(h) for h in body.get("data", [])[:k]]

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> list[Document]:
        k = self._resolve_k(kwargs)
        body = await self._client().asearch(query, size=k)
        return [self._hit_to_document(h) for h in body.get("data", [])[:k]]
