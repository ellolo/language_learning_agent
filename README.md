# Language learning agent

A compact demo that wires **LangGraph** to **LangChain agents** (as in Langchain 1.0): the learner sends a short message, a target language, and a **CEFR level** (A1–C2). A router classifies the request as **grammar**, **reading**, or **out of scope**.

- **Reading** — an agent produces short leveled text on the topic they asked for.  
- **Grammar** — an agent generates a single exercise at their level.  
- **Unknown** — the workflow returns a short message that only grammar and reading are supported.

Agents may call **Tavily** for web search when it helps. Supported target languages: English, Spanish, French, German, Italian.

---

## Prerequisites

- [Python](https://www.python.org/) 3.13 or newer  
- [uv](https://docs.astral.sh/uv/) (recommended for installing dependencies and running commands)

## Installation

From the repository root:

```bash
uv sync
```

```bash
cp example.env .env
```

Configure API keys in `.env` (see the table below).

The included [`langgraph.json`](langgraph.json) loads `.env` automatically for `langgraph dev`.

### Environment variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `OPENAI_API_KEY` | Yes | Chat model (default: `gpt-4o-mini`) |
| `TAVILY_API_KEY` | Yes | Web search for the grammar and reading agents |
| `LANGSMITH_API_KEY` | No | LangSmith tracing |
| `LANGSMITH_TRACING` | No | Set to `true` to enable tracing when LangSmith is configured |


## Usage

### LangGraph Studio

[`langgraph.json`](langgraph.json) registers the compiled graph as **`agent`**, exported from [`src/language_agent.py`](src/language_agent.py) as `language_workflow`.

Start the dev server (hot reload and Studio):

```bash
uv run langgraph dev
```

Use the printed URL to open **LangGraph Studio** or attach your IDE. If the default port is taken:

```bash
uv run langgraph dev --port 8123
```

### Calling the graph from Python

Invoke with fields that match `InputState` in `src/language_agent.py`: `user_message`, `user_level`, and optionally `target_language`.

From the repo root, add `src` to the path (same idea as [`notebooks/language_agent.ipynb`](notebooks/language_agent.ipynb)):

```python
import sys
sys.path.insert(0, "src")

from language_agent import language_workflow

result = language_workflow.invoke({
    "user_message": "I want a short text about trains for beginners.",
    "user_level": "A2",
    "target_language": "German",
})
print(result["response"])
```

Or set `PYTHONPATH=src` once and import normally.

### Notebooks

Examples live under [`notebooks/`](notebooks/). With Jupyter installed:

```bash
uv run jupyter lab
```

Add Jupyter if needed: `uv add jupyterlab`.

## Project layout

| Path | Role |
|------|------|
| `src/language_agent.py` | Graph, agents, and `language_workflow` |
| `langgraph.json` | LangGraph CLI: graphs and env file |
| `pyproject.toml`, `uv.lock` | Dependencies (`langgraph-cli[inmem]` supplies `langgraph dev`) |
