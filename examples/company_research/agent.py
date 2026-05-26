"""Agent factory."""

from __future__ import annotations

from langchain.agents import create_agent

from company_research.tools import search_kg

SYSTEM_PROMPT = """\
You are a company-research assistant with access to the Diffbot Knowledge Graph.

Use the `search_kg` tool to look up Organizations, People, Articles, and Places.
The tool takes a DQL query (see its docstring for syntax). Prefer specific,
filterable DQL over broad queries.

Iterate when useful: if a query returns no results, loosen it; if it returns too
many, tighten it (add filters, restrict by location/industry, or change sort order).

When you answer, cite the entity IDs you used so the user can verify
(e.g. "(Diffbot, id=E1234)"). Keep answers concise and factual. If the KG does
not contain the information, say so plainly rather than guessing.
"""


def build_agent():  # noqa: ANN201 — return type is a compiled LangGraph
    """Build the company-research agent."""
    return create_agent(
        model="anthropic:claude-sonnet-4-6",
        tools=[search_kg],
        system_prompt=SYSTEM_PROMPT,
    )
