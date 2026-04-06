# Language learning

This repo contains **two layers** of LangChain / LangGraph code:

1. **`conversational_agent`**: a chat agent that talks to the learner in English (then optionally in the target language), records **name**, **CEFR level**, and **target language**, and can call the `language_workflow` as a tool, to generate learning material on demand (text to read or grammar exercises). The agent is implemented as a LangChain 1.0 agent in `conversational_agent.py`

2. **`language_workflow`**: a **router graph** workflow that generates learning material matching the input preferences. Inputs are:
    - `user_message`: the messaege of the user specifying what material he wants.
    - `user_level`: the proficiency level of the user, as  **CEFR** level (A1–C2)
    - `target_language`: the language the user wants to learn. Supported languages: English, Spanish, French, German, Italian

   The workflow first classifies the input request as **grammar** (the user wants a grammar exercise), **reading** (the user wants text to read), or **unknown** (the user request cannot be satisfied), then runs the matching specialist agent to generate the learning material. Agents may use **Tavily** web search to generate the material. The workflow is implemented using LangGraph Graph API. The specialized agents are implemented using LangChain 1.0 agents. 

Note that the The conversational agent’s **language learning** tool invokes the `language_workflow` internally, so you usually interact with **`conversational_agent`**, instead of calling directly the `language_workflow`.

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
| `TAVILY_API_KEY` | Yes | Web search for the grammar and reading subgraph |
| `LANGSMITH_API_KEY` | No | LangSmith tracing |
| `LANGSMITH_TRACING` | No | Set to `true` to enable tracing when LangSmith is configured |


## Usage

### LangGraph Studio

[`langgraph.json`](langgraph.json) registers the graph id **`agent`**, exported from [`src/conversational_agent.py`](src/conversational_agent.py) as **`conversational_agent`**. That build uses **no custom checkpointer** so `langgraph dev` and LangGraph Cloud can supply persistence themselves.

Start the dev server (hot reload and Studio):

```bash
uv run langgraph dev
```

Use the printed URL to open **LangGraph Studio** or attach your IDE. If the default port is taken:

```bash
uv run langgraph dev --port 8123
```

### Notebooks and local scripts (in-memory threads)

For **`thread_id`**-style checkpointing outside the API, import **`conversational_agent_with_checkpointer`** from the same module (it wraps **`InMemorySaver()`**). The default **`conversational_agent`** export is intended for Studio and must not bundle a custom checkpointer, or `langgraph dev` will error.

```python
import sys
sys.path.insert(0, "src")

from conversational_agent import conversational_agent_with_checkpointer

# Pass config={"configurable": {"thread_id": "..."}} on invoke/stream as needed.
```

### Calling the learning workflow graph directly

You can invoke the **router graph** only (no conversation wrapper) with fields that match `InputState` in [`src/language_agent.py`](src/language_agent.py): `user_message`, `user_level`, and optionally `target_language`.

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

### Notebooks

Examples notebooks for using the conversational agent and the learning workflow can be found at" [`notebooks/`](notebooks/). 

To run the notebooks, first install Jupyter (`uv add jupyterlab`), then run:

```bash
uv run jupyter lab
```

## Project layout

| Path | Role |
|------|------|
| `src/conversational_agent.py` | Conversational agent (`conversational_agent`, `conversational_agent_with_checkpointer`) and tool that calls the learning graph |
| `src/language_agent.py` | Router graph, grammar/reading agents, exported `language_workflow` |
| `notebooks` | Example notebooks for using the agent and the workflow |
| `langgraph.json` | LangGraph CLI: graphs and env file |
| `pyproject.toml`, `uv.lock` | Dependencies (`langgraph-cli[inmem]` supplies `langgraph dev`) |
