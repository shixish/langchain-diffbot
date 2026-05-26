---
name: Use diffbot-python SDK
overview: Make langchain-diffbot a thin LangChain layer over diffbot-python — expose every SDK surface as the closest LangChain primitive, with passthrough kwargs and bring-your-own-client.
todos:
  - id: dep-add
    content: Add diffbot-python git dependency to pyproject.toml (uv source pinned to GitHub main; follow-up to switch to PyPI when published).
    status: pending
  - id: client-base
    content: Replace _client.py with a tiny lifecycle helper — a BaseDiffbotComponent (or mixin) that owns `client`/`async_client`/`diffbot_api_token`/`timeout` fields and yields a context-managed SDK client when none is supplied.
    status: pending
  - id: kg-retriever
    content: Refactor DiffbotKnowledgeGraphRetriever to call `Diffbot.dql` directly. Drop `base_url`. Expose all SDK dql kwargs as retriever fields (size→k, from_, filter, exportspec, format, extra). Keep fields/content_fields/document_mapper output shaping.
    status: pending
  - id: web-search-retriever
    content: Add DiffbotWebSearchRetriever — wraps `web_search`, fields k (=num_results), max_tokens, plus fields/content_fields/document_mapper for shaping.
    status: pending
  - id: extract-tool
    content: Add DiffbotExtractTool — wraps `extract(url, api, fmt)`. ExtractionError → structured tool output; AuthError/APIError propagate.
    status: pending
  - id: web-search-tool
    content: Add DiffbotWebSearchTool — wraps `web_search` for direct agent use (parallel surface to the retriever; returns raw result list, no Document conversion).
    status: pending
  - id: entities-tool
    content: Add DiffbotEntitiesTool — wraps `entities(text, lang)` for NLP entity + sentiment extraction.
    status: pending
  - id: kg-tool
    content: Add DiffbotKnowledgeGraphTool — wraps `dql` for agents that know DQL syntax. Returns raw JSON, not Documents.
    status: pending
  - id: extract-loader
    content: Add DiffbotExtractLoader — DocumentLoader over a list of URLs, calling `extract` per URL; lazy_load + alazy_load.
    status: pending
  - id: crawl-loader
    content: Add DiffbotCrawlLoader — DocumentLoader that streams `crawl()` events and yields Documents for URL_PROCESSED events; lazy_load + alazy_load.
    status: pending
  - id: chat-model
    content: Add ChatDiffbot (BaseChatModel) — wraps `ask(messages)` streaming RAG endpoint. _stream + _astream native; _generate/_agenerate aggregate.
    status: pending
  - id: exports
    content: Update __init__.py __all__ and tests/unit_tests/test_imports.py for every new public surface.
    status: pending
  - id: tests
    content: Adapt existing KG retriever tests (new upstream URL, no base_url) and add respx-backed unit tests for each new surface. Cover client-passthrough (provided client is used as-is) and ExtractionError mapping.
    status: pending
  - id: claudemd
    content: Update CLAUDE.md — transport story moves to "diffbot-python wraps httpx"; document the thin-layer + bring-your-own-client design; list every new surface in Layout.
    status: pending
isProject: false
---

# Adopt diffbot-python in langchain-diffbot

## Goals

- **Thin layer**: every LangChain class calls `Diffbot` / `DiffbotAsync` directly, mirrors the SDK's kwargs by name, and adds nothing the SDK doesn't already do — except the LangChain plumbing (Pydantic fields, `Document` conversion, output shaping for retrievers).
- **Complete surface**: expose every SDK method as the closest LangChain primitive (retriever, tool, loader, chat model). If the SDK gains a method, adding a thin LangChain wrapper for it should be a ~30-line change.
- **Bring-your-own-client**: every class accepts an optional pre-built `Diffbot` / `DiffbotAsync`. Anything the SDK supports (custom `analyze_url`, `transport=`, shared connection pool, custom headers) just works — we don't have to re-expose each knob.
- Preserve current design principles: native sync + native async (no executor fallback); output shaping on retrievers to keep payloads LLM-sized.

## SDK facts (verified against `diffbot/diffbot-python@main`)

Verified by reading `src/diffbot/{__init__,client,kg,web_search,extract,errors,ask,crawl,nlp}.py`.

- **Transport**: SDK uses `httpx`. `respx` continues to work for unit tests — mocks target the real upstream URLs.
- **Two classes**: `from diffbot import Diffbot, DiffbotAsync`. Same method names on each; **not** a single client with `*_async` methods.
- **Context managers**: both implement `__enter__`/`__exit__` (and `__aenter__`/`__aexit__`). They open an `httpx.Client` / `httpx.AsyncClient` in `__init__` and need `close()` / `aclose()`.
- **Constructor**: `Diffbot(token, *, timeout=30.0, analyze_url=..., llm_url=..., crawler_url=..., web_search_url=..., nlp_url=..., transport=None)`. `timeout`, `transport`, and per-API URL overrides are all here — **but no `kg_url`** (DQL endpoint is hardcoded in `kg.py`).
- **Errors** (`diffbot.errors`): `DiffbotError` → `APIError` → `AuthError`, `RateLimitError`; plus `ExtractionError` (raised when `extract` returns 200 with an `errorCode` body) and `ValidationError`. `APIError` carries `.status_code`, `.message`, `.request_id`.

### Method inventory

| SDK method | Signature (sync; async mirrors) | LangChain surface(s) |
|---|---|---|
| `extract` | `extract(url, api="analyze", fmt="markdown") -> dict` | `DiffbotExtractTool`, `DiffbotExtractLoader` |
| `ask` | `ask(messages) -> Iterator[str]` (streams) | `ChatDiffbot` (BaseChatModel) |
| `crawl` | `crawl(site, **kwargs) -> Iterator[CrawlEvent]` | `DiffbotCrawlLoader` (DocumentLoader) |
| `crawl_list_jobs` | `() -> list[dict]` | — (admin; reachable via `.client`) |
| `crawl_get_job` | `(job_name) -> dict` | — (admin; reachable via `.client`) |
| `crawl_delete_job` | `(job_name) -> None` | — (admin; reachable via `.client`) |
| `dql` | `dql(query, *, size=10, from_=0, format="json", filter=None, exportspec=None, extra=None, raw=False) -> dict | bytes` | `DiffbotKnowledgeGraphRetriever`, `DiffbotKnowledgeGraphTool` |
| `dql_parallel` | `(queries, *, workers=8) -> list` (sync only) | — (utility; reachable via `.client`) |
| `dql_refresh_ontology` | `(dest) -> None` | — (utility; reachable via `.client`) |
| `web_search` | `web_search(text, *, num_results=None, max_tokens=None) -> dict` | `DiffbotWebSearchRetriever`, `DiffbotWebSearchTool` |
| `entities` | `entities(text, *, lang="auto") -> dict` | `DiffbotEntitiesTool` |

Admin/utility methods (`crawl_list_jobs`, `crawl_get_job`, `crawl_delete_job`, `dql_parallel`, `dql_refresh_ontology`) aren't natural fits for any LangChain primitive. We don't wrap them — but because every class exposes `.client` and `.async_client`, users can call them directly when needed.

## Thin-layer architecture

### Bring-your-own-client

Every LangChain class has the same two optional fields, plus the convenience fields:

```python
class _BaseDiffbotComponent(BaseModel):
    client: Optional[Diffbot] = Field(default=None, exclude=True)
    async_client: Optional[DiffbotAsync] = Field(default=None, exclude=True)
    diffbot_api_token: Optional[SecretStr] = None  # falls back to DIFFBOT_API_TOKEN
    timeout: float = 30.0
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _sync_client(self) -> ContextManager[Diffbot]:
        # if self.client provided, yield it without closing; else construct & close
        ...
    def _async_client(self) -> AsyncContextManager[DiffbotAsync]: ...
```

Behavior:
- `client=…` provided → use it; **do not close it** on our way out (user owns the lifecycle).
- Not provided → construct a fresh `Diffbot(token=..., timeout=...)` inside a `with` block per call. Matches today's per-call pattern in `DiffbotKGClient`.
- For high-throughput use, the user passes a long-lived `client` once and we reuse it — no need to add caching logic on our side.

### Direct SDK calls

Each LangChain class's sync and async methods call the SDK directly inside the lifecycle helper. No intermediate wrapper class. Example:

```python
def _get_relevant_documents(self, query: str, *, run_manager) -> list[Document]:
    with self._sync_client() as db:
        body = db.dql(query, size=self.k, from_=self.from_, filter=self.filter, ...)
    return [self._hit_to_document(h) for h in body.get("data", [])[: self.k]]
```

This means `_client.py` is no longer a wrapper that mirrors every SDK method. It either:
- Becomes `_base.py` exporting `_BaseDiffbotComponent`, **or**
- Is deleted; the base lives in `langchain_diffbot/__init__.py` private surface.

I'd lean toward `_base.py` for clarity.

### Passthrough kwargs

Where the SDK takes kwargs, we mirror them as Pydantic fields with the same names. The only translations we keep:
- `size` → `k` on the KG retriever (LangChain convention; `k` wins because every other retriever uses it).
- `num_results` → `k` on the web search retriever (same reason).

All other SDK kwargs keep their original names: `from_`, `filter`, `exportspec`, `format`, `extra`, `raw`, `max_tokens`, `api`, `fmt`, `lang`. No renames, no enums, no "convenience" reshaping.

Per-call overrides use LangChain's `.with_config({"configurable": {...}})` pattern; we don't add a parallel kwargs API.

## Per-surface designs

### `DiffbotKnowledgeGraphRetriever` (existing — refactor)

- Drop `base_url` (SDK has no `kg_url` knob; see §"Base URL").
- New fields: `from_`, `filter`, `exportspec`, `format`, `extra` (all passthrough to `dql`).
- Keep: `k`, `timeout`, `fields`, `content_fields`, `document_mapper`, `diffbot_api_token`.
- Add: `client`, `async_client` (from `_BaseDiffbotComponent`).
- `_get_relevant_documents` and `_aget_relevant_documents` call `db.dql(...)` directly.

### `DiffbotWebSearchRetriever` (new)

- Fields: `k` (→ `num_results`), `max_tokens`, plus `fields`/`content_fields`/`document_mapper` mirroring the KG retriever for consistency.
- Default `content_fields = ["content", "snippet"]`.
- Default metadata: `title`, `pageUrl`, `score`.
- Maps `response["search_results"][:k]` to `Document`s.

### `DiffbotExtractTool` (new)

- BaseTool with `args_schema`: `url: str`, `api: str = "analyze"`, `fmt: str = "markdown"` — SDK signature verbatim.
- Returns a small dict shaped from the response: `content`, `title`, `pageUrl`, `resolvedPageUrl`, `type`. (Verify field paths during implementation against a real response.)
- Error handling:
  - `ExtractionError` → return `{"error": str(e), "errorCode": e.error_code}` so the agent gets a readable signal.
  - `AuthError` / `APIError` → re-raise (infra/config problems, not per-call errors).

### `DiffbotWebSearchTool` (new)

- Parallel to the retriever — same SDK call, but as a Tool that returns the raw `search_results` list (no `Document` conversion).
- args_schema: `text: str`, `num_results: int | None`, `max_tokens: int | None`.

### `DiffbotEntitiesTool` (new)

- args_schema: `text: str`, `lang: str = "auto"`.
- Returns the SDK response dict as-is (small; agent can read `entities` and `sentiment` directly).

### `DiffbotKnowledgeGraphTool` (new)

- For agents that know DQL syntax (e.g. instructed to write `type:Person name:"Ada"`).
- args_schema mirrors `dql` kwargs: `query: str`, `size: int = 10`, `from_: int = 0`, `filter: str | None`, `format: str = "json"`.
- Returns raw response dict. (For Document-shaped results, use the retriever instead.)

### `DiffbotExtractLoader` (new)

- DocumentLoader over a list of URLs.
- Init: `urls: list[str]`, `api: str = "analyze"`, `fmt: str = "markdown"`, plus `_BaseDiffbotComponent` fields.
- `lazy_load()` iterates URLs, calls `db.extract(url, ...)`, yields a `Document` per URL.
- `alazy_load()` async equivalent using `DiffbotAsync` (native — gives us hundreds-of-URLs-in-flight on one event loop).
- Document mapping: `page_content` = markdown content, metadata = `{url, title, type, pageUrl, resolvedPageUrl}`.

### `DiffbotCrawlLoader` (new)

- DocumentLoader that drives a crawl and yields a `Document` per `URL_PROCESSED` event.
- Init: `site: str`, plus crawl kwargs as passthrough (`hops`, etc.) and `_BaseDiffbotComponent` fields.
- `lazy_load()` iterates `db.crawl(site, **kwargs)`, filters to `URL_PROCESSED` events, yields Documents.
- `alazy_load()` async equivalent via `DiffbotAsync.crawl`.
- Non-URL_PROCESSED events (status/progress) are dropped silently in the default mapper, but users can pass a custom `event_mapper` to handle them.

### `ChatDiffbot` (new)

- `BaseChatModel` over `ask(messages)`. The SDK already streams tokens, so `_stream` / `_astream` are the natural fit; `_generate` / `_agenerate` aggregate the stream into a single `ChatGeneration`.
- Field set: standard chat-model fields + `_BaseDiffbotComponent`.
- Converts LangChain `BaseMessage` list ↔ Diffbot `[{"role", "content"}]`.

## Base URL: drop the retriever's knob

The SDK has no `kg_url` constructor knob (verified — `KG_DQL_ENDPOINT` is hardcoded in `kg.py`). So `DiffbotKnowledgeGraphRetriever.base_url` can't pass through.

- **Decision**: drop `base_url` from the retriever. The bring-your-own-client escape hatch covers the legitimate use cases:
  - For tests: pass `client=Diffbot(token=..., transport=httpx.MockTransport(...))`.
  - For proxies / local mocks: same, with a transport that points wherever needed.
- **Follow-up** (out of scope): upstream a PR adding `kg_url` to the `Diffbot` constructor — same shape as `analyze_url` / `web_search_url`. Then users can override the KG endpoint without touching transport.

## File layout (post-change)

```
langchain_diffbot/
├── __init__.py             # re-export every public class
├── _base.py                # _BaseDiffbotComponent — client lifecycle helper
├── chat_models.py          # ChatDiffbot
├── document_loaders.py     # DiffbotExtractLoader, DiffbotCrawlLoader
├── retrievers.py           # DiffbotKnowledgeGraphRetriever, DiffbotWebSearchRetriever
├── tools.py                # DiffbotExtractTool, DiffbotWebSearchTool, DiffbotEntitiesTool, DiffbotKnowledgeGraphTool
└── py.typed
```

`_client.py` is removed (or renamed to `_base.py` with content reduced to ~40 lines).

## Tests

- The SDK uses `httpx`, so existing `respx` fixtures keep working — point mocks at the real upstream URLs:
  - KG: `https://kg.diffbot.com/kg/v3/dql`
  - Web search: `https://llm.diffbot.com/api/v1/web_search`
  - Extract: `https://api.diffbot.com/v3/analyze`
  - Ask: `https://llm.diffbot.com/rag/v1/chat/completions`
  - Crawl: `https://api.diffbot.com/v3/crawl`
  - NLP: per `web_search.NLP_BASE` — verify URL during implementation.
- Per-surface unit tests:
  - KG retriever: happy path, `fields` allowlist, `content_fields` priority, `document_mapper` override, removal of `base_url`.
  - Web search retriever: happy path + shaping.
  - Each tool: happy path + `args_schema` round-trip.
  - Extract tool: `ExtractionError` → structured dict; `AuthError` → raises.
  - Loaders: lazy iteration over multiple URLs/events.
  - Chat model: stream + aggregate; message format conversion.
- One cross-cutting test: **client passthrough** — construct any class with a pre-built `Diffbot(token=..., transport=httpx.MockTransport(...))` and confirm we use it without closing it after the call.
- Integration tests unchanged — already gated on `DIFFBOT_API_TOKEN`.

## CLAUDE.md updates

- **Stack**: replace "HTTP: `httpx`" with "HTTP: `diffbot-python` (which wraps `httpx`, so `respx` continues to work for unit tests)".
- **Architectural decisions**: add a new section "Thin layer over diffbot-python" documenting:
  - Direct SDK calls inside per-class methods (no wrapper).
  - Bring-your-own-client pattern.
  - Kwarg passthrough policy (only `size`/`num_results` → `k` translations).
- **Layout**: enumerate the new modules.

## Out of scope (deferred)

- **Admin/utility SDK methods** (`crawl_list_jobs`, `crawl_get_job`, `crawl_delete_job`, `dql_parallel`, `dql_refresh_ontology`). Users can call them via the exposed `.client` / `.async_client` on any of our classes — no LangChain primitive fits cleanly.
- **`kg_url` upstream PR**: file as a follow-up if users hit it.
- **PyPI pin**: switch from git source to a version pin once `diffbot-python` is published.

## Files expected to change/add

- Modify:
  - [/Users/andrew/Code/Diffbot/langchain-diffbot/pyproject.toml](/Users/andrew/Code/Diffbot/langchain-diffbot/pyproject.toml)
  - [/Users/andrew/Code/Diffbot/langchain-diffbot/langchain_diffbot/retrievers.py](/Users/andrew/Code/Diffbot/langchain-diffbot/langchain_diffbot/retrievers.py)
  - [/Users/andrew/Code/Diffbot/langchain-diffbot/langchain_diffbot/__init__.py](/Users/andrew/Code/Diffbot/langchain-diffbot/langchain_diffbot/__init__.py)
  - [/Users/andrew/Code/Diffbot/langchain-diffbot/tests/unit_tests/test_imports.py](/Users/andrew/Code/Diffbot/langchain-diffbot/tests/unit_tests/test_imports.py)
  - [/Users/andrew/Code/Diffbot/langchain-diffbot/tests/unit_tests/test_retrievers.py](/Users/andrew/Code/Diffbot/langchain-diffbot/tests/unit_tests/test_retrievers.py)
  - [/Users/andrew/Code/Diffbot/langchain-diffbot/CLAUDE.md](/Users/andrew/Code/Diffbot/langchain-diffbot/CLAUDE.md)
- Add:
  - [/Users/andrew/Code/Diffbot/langchain-diffbot/langchain_diffbot/_base.py](/Users/andrew/Code/Diffbot/langchain-diffbot/langchain_diffbot/_base.py)
  - [/Users/andrew/Code/Diffbot/langchain-diffbot/langchain_diffbot/tools.py](/Users/andrew/Code/Diffbot/langchain-diffbot/langchain_diffbot/tools.py)
  - [/Users/andrew/Code/Diffbot/langchain-diffbot/langchain_diffbot/document_loaders.py](/Users/andrew/Code/Diffbot/langchain-diffbot/langchain_diffbot/document_loaders.py)
  - [/Users/andrew/Code/Diffbot/langchain-diffbot/langchain_diffbot/chat_models.py](/Users/andrew/Code/Diffbot/langchain-diffbot/langchain_diffbot/chat_models.py)
  - [/Users/andrew/Code/Diffbot/langchain-diffbot/tests/unit_tests/test_base.py](/Users/andrew/Code/Diffbot/langchain-diffbot/tests/unit_tests/test_base.py)
  - [/Users/andrew/Code/Diffbot/langchain-diffbot/tests/unit_tests/test_web_search_retriever.py](/Users/andrew/Code/Diffbot/langchain-diffbot/tests/unit_tests/test_web_search_retriever.py)
  - [/Users/andrew/Code/Diffbot/langchain-diffbot/tests/unit_tests/test_tools.py](/Users/andrew/Code/Diffbot/langchain-diffbot/tests/unit_tests/test_tools.py)
  - [/Users/andrew/Code/Diffbot/langchain-diffbot/tests/unit_tests/test_document_loaders.py](/Users/andrew/Code/Diffbot/langchain-diffbot/tests/unit_tests/test_document_loaders.py)
  - [/Users/andrew/Code/Diffbot/langchain-diffbot/tests/unit_tests/test_chat_models.py](/Users/andrew/Code/Diffbot/langchain-diffbot/tests/unit_tests/test_chat_models.py)
- Remove:
  - [/Users/andrew/Code/Diffbot/langchain-diffbot/langchain_diffbot/_client.py](/Users/andrew/Code/Diffbot/langchain-diffbot/langchain_diffbot/_client.py)
