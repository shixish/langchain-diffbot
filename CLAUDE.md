# langchain-diffbot

LangChain integration package for the Diffbot Knowledge Graph and Extract APIs.

## Stack

- **Build/deps**: `uv` (not Poetry/pip). `pyproject.toml` is PEP 621 with `hatchling` as the build backend. Dependency groups (`test`, `lint`, `typing`) are declared under `[dependency-groups]` and invoked via `uv run --group <name> ...`.
- **HTTP**: `httpx` — gives us one library with both sync (`httpx.Client`) and async (`httpx.AsyncClient`) clients.
- **Models**: `pydantic` v2.
- **LangChain**: targets `langchain-core >=1.0,<2.0`. During local dev `langchain-core` and `langchain-tests` are resolved from a sibling `../langchain/` checkout via `[tool.uv.sources]` — release builds use the published versions.
- **Test runner**: `pytest` with `asyncio_mode = "auto"`.

## Architectural decisions

### Both sync and async are implemented natively

`DiffbotKnowledgeGraphRetriever` defines both `_get_relevant_documents` and `_aget_relevant_documents`, each delegating to the matching method on `DiffbotKGClient` (`search` / `asearch`).

LangChain would let us implement only one and inherit a thread-pool fallback for the other, but for an HTTP-bound integration that fallback caps concurrency at the default executor size (~12 workers) and breaks cancellation propagation. Native async lets a single event loop hold hundreds of in-flight Diffbot calls. The duplication cost is ~15 lines of one-line delegation, so we pay it.

When adding new retrievers / tools / loaders, define both surfaces the same way.

### No `_async`/`_sync` codegen split

Some LangChain integration packages keep async as the source of truth and generate sync mirrors via `unasync`. We don't — the implementations here are thin enough (each method one line of delegation) that hand-mirroring is cheaper than maintaining a codegen script and CI drift check.

Revisit if any single feature area grows past ~100 lines of non-trivial async logic that needs a sync twin.

### Output shaping on the retriever

`DiffbotKnowledgeGraphRetriever` accepts `fields` (metadata allowlist), `content_fields` (priority list for `page_content`), and `document_mapper` (full override). Diffbot KG entities can run thousands of tokens each — without shaping, a single retrieval can blow past LLM input limits when fed into a tool call. Defaults preserve everything; agent-style users are expected to pass `fields=[...]`.

## Commands

```
make format          # ruff format + ruff check --fix
make lint            # ruff check + ruff format --check
make typing          # mypy on the package
make test            # unit tests
make test_integration  # integration tests (needs DIFFBOT_API_TOKEN)
```

## Layout

```
langchain_diffbot/
├── __init__.py        # public re-exports — every user-facing class listed in __all__
├── _client.py         # DiffbotKGClient (sync + async httpx wrapper, leading underscore = private)
├── retrievers.py      # DiffbotKnowledgeGraphRetriever
└── py.typed           # PEP 561 marker
tests/
├── unit_tests/        # no network — use respx to mock httpx
└── integration_tests/ # hit real Diffbot; require DIFFBOT_API_TOKEN
```

New public surfaces go in their own top-level module (`tools.py`, `document_loaders.py`, etc.) and get re-exported from `__init__.py`. `tests/unit_tests/test_imports.py` asserts the public surface.
