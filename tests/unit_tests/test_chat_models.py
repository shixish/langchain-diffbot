"""Unit tests for `ChatDiffbot` (no network)."""

from __future__ import annotations

import httpx
import respx
from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage

from langchain_diffbot import ChatDiffbot

ASK_URL = "https://llm.diffbot.com/rag/v1/chat/completions"

SSE_BODY = (
    b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n'
    b'data: {"choices": [{"delta": {"content": ", "}}]}\n'
    b'data: {"choices": [{"delta": {"content": "world"}}]}\n'
    b"data: [DONE]\n"
)


@respx.mock
def test_stream_yields_chunks() -> None:
    respx.post(ASK_URL).mock(return_value=httpx.Response(200, content=SSE_BODY))
    llm = ChatDiffbot(diffbot_api_token="t")
    chunks = list(llm.stream([HumanMessage(content="hi")]))
    contents = [c.content for c in chunks if isinstance(c, AIMessageChunk)]
    assert "".join(contents) == "Hello, world"


@respx.mock
def test_invoke_aggregates_stream() -> None:
    respx.post(ASK_URL).mock(return_value=httpx.Response(200, content=SSE_BODY))
    llm = ChatDiffbot(diffbot_api_token="t")
    result = llm.invoke([HumanMessage(content="hi")])
    assert result.content == "Hello, world"


@respx.mock
def test_message_role_mapping() -> None:
    route = respx.post(ASK_URL).mock(return_value=httpx.Response(200, content=SSE_BODY))
    llm = ChatDiffbot(diffbot_api_token="t")
    llm.invoke([SystemMessage(content="you are a bot"), HumanMessage(content="hi")])
    import json

    payload = json.loads(route.calls.last.request.content)
    roles = [m["role"] for m in payload["messages"]]
    assert roles == ["system", "user"]


@respx.mock
async def test_astream_yields_chunks() -> None:
    respx.post(ASK_URL).mock(return_value=httpx.Response(200, content=SSE_BODY))
    llm = ChatDiffbot(diffbot_api_token="t")
    parts: list[str] = []
    async for chunk in llm.astream([HumanMessage(content="hi")]):
        if isinstance(chunk, AIMessageChunk) and isinstance(chunk.content, str):
            parts.append(chunk.content)
    assert "".join(parts) == "Hello, world"
