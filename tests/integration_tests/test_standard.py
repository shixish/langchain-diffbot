"""Conformance against the shared `langchain-tests` retriever suite."""

from __future__ import annotations

import os
from typing import Any

import pytest
from langchain_core.retrievers import BaseRetriever
from langchain_tests.integration_tests import RetrieversIntegrationTests

from langchain_diffbot import DiffbotKnowledgeGraphRetriever

pytestmark = pytest.mark.skipif(
    not os.environ.get("DIFFBOT_API_TOKEN"),
    reason="DIFFBOT_API_TOKEN not set",
)


class TestDiffbotKnowledgeGraphRetriever(RetrieversIntegrationTests):
    @property
    def retriever_constructor(self) -> type[BaseRetriever]:
        return DiffbotKnowledgeGraphRetriever

    @property
    def retriever_constructor_params(self) -> dict[str, Any]:
        return {}

    @property
    def retriever_query_example(self) -> str:
        # Broad enough to reliably return >=3 results from the live KG.
        return "type:Organization"
