# `langchain-diffbot` examples

Two ways to see the package in action. Both expect `DIFFBOT_API_TOKEN` and (for the agent code) `ANTHROPIC_API_KEY` in the environment — copy `.env.example` to `.env` and fill in the values, or `export` them in your shell.

## Notebook

`quickstart.ipynb` walks through the retriever (basic + output shaping + async) and then builds a Claude-powered research agent on top of it.

```bash
pip install "langchain-diffbot[examples]" jupyter
jupyter lab examples/quickstart.ipynb
```

## CLI

`company_research/` is the same agent as a one-shot CLI. Useful for shell scripting or quick spot checks.

```bash
pip install "langchain-diffbot[examples]"
cd examples
python -m company_research "What companies in Austin work on robotics?"
python -m company_research --quiet "Who are the executives at Diffbot?"
```

The agent picks its own DQL queries, iterates if the first one is too broad or too narrow, and cites entity IDs in the answer. Drop `--quiet` to see tool calls and intermediate KG responses.
