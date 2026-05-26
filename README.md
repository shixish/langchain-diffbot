# langchain-diffbot

LangChain integration for the [Diffbot Knowledge Graph](https://docs.diffbot.com/) and Extract APIs.

## Installation

```bash
pip install langchain-diffbot
```

## Authentication

Get an API token at https://app.diffbot.com/get-started/ and export it:

```bash
export DIFFBOT_API_TOKEN="..."
```

## Quickstart — Knowledge Graph retriever

```python
from langchain_diffbot import DiffbotKnowledgeGraphRetriever

retriever = DiffbotKnowledgeGraphRetriever(k=5)
docs = retriever.invoke("type:Organization industries:\"Artificial Intelligence\" location.city.name:\"Boston\"")
for d in docs:
    print(d.metadata["name"], "—", d.page_content[:120])
```

The query string is a [DQL (Diffbot Query Language)](https://docs.diffbot.com/reference/dql-quickstart) expression.

## Shaping the output

Diffbot KG entities are large — nested `suppliers`, `employments`, `tags`,
`articles`, etc. Dumping them straight into an LLM prompt can blow past
per-minute input-token limits in a single call. The retriever exposes three
knobs to keep `Document` payloads tight:

```python
from langchain_core.documents import Document
from langchain_diffbot import DiffbotKnowledgeGraphRetriever

# 1. Project only the top-level fields you care about. Drops everything else
#    from `metadata`. Recommended for agent / tool-use scenarios.
retriever = DiffbotKnowledgeGraphRetriever(
    k=5,
    fields=["id", "type", "name", "homepageUri", "nbEmployees"],
)

# 2. Choose which field becomes `page_content`. First non-empty value wins.
retriever = DiffbotKnowledgeGraphRetriever(
    content_fields=["summary", "description", "name"],
)

# 3. For total control, pass a `document_mapper` that turns a raw entity
#    dict into whatever Document shape you want.
def mapper(entity: dict) -> Document:
    return Document(
        page_content=entity.get("summary", ""),
        metadata={"id": entity["id"], "name": entity["name"]},
    )

retriever = DiffbotKnowledgeGraphRetriever(document_mapper=mapper)
```

`fields` and `content_fields` are ignored when `document_mapper` is set.

## Using it in a chain

The retriever is a standard `BaseRetriever`, so it slots into LCEL like any
other LangChain retriever:

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_diffbot import DiffbotKnowledgeGraphRetriever

retriever = DiffbotKnowledgeGraphRetriever(
    k=5,
    fields=["id", "name", "homepageUri", "nbEmployees", "industries"],
)

prompt = ChatPromptTemplate.from_template(
    "Answer using only this Diffbot KG context:\n\n{context}\n\nQuestion: {question}"
)


def _format(docs):
    return "\n---\n".join(
        f"{d.metadata.get('name')} (id={d.metadata.get('id')}): {d.page_content}"
        for d in docs
    )


chain = (
    {"context": retriever | _format, "question": RunnablePassthrough()}
    | prompt
    | ChatAnthropic(model="claude-sonnet-4-6")
    | StrOutputParser()
)

chain.invoke('type:Organization location.city.name:"Boston" industries:"Biotech"')
```

## Examples

The [`examples/`](./examples) folder has runnable demos:

- [`examples/quickstart.ipynb`](./examples/quickstart.ipynb) — notebook covering the retriever, output shaping, async, and a Claude-powered research agent.
- [`examples/company_research/`](./examples/company_research) — the same agent as a one-shot CLI: `cd examples && python -m company_research "your question"`.

Both need `langchain` + `langchain-anthropic` on top of the base package. Install the extra:

```bash
pip install "langchain-diffbot[examples]"
```

## Development

```bash
uv sync --all-groups
uv run pytest tests/unit_tests
```

Integration tests hit the live Diffbot API and require `DIFFBOT_API_TOKEN`:

```bash
uv run pytest tests/integration_tests
```
