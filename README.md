# traqo

Structured tracing for LLM applications. JSONL files, hierarchical spans, zero infrastructure.

```python
from traqo import Tracer, trace
from pathlib import Path

@trace()
async def classify(text: str) -> str:
    response = await llm.chat(text)
    return response

with Tracer(Path("traces/run.jsonl")):
    await classify("Is this a bug?")
```

Your traces are just `.jsonl` files. Read them with `grep`, query them with DuckDB, or hand them to an AI assistant.

## Why traqo?

- **Zero infrastructure** -- no server, no database, no account. `pip install traqo` and go.
- **AI-first** -- JSONL is text. AI assistants read your traces directly, no browser needed.
- **Hierarchical spans** -- not flat logs. Reconstruct the full call tree across functions and files.
- **Zero dependencies** -- stdlib only. Integrations are optional extras.
- **Transparent** -- traces are portable files. No vendor lock-in, no proprietary format.

## Install

```bash
pip install traqo                   # Core (zero dependencies)
pip install traqo[openai]           # + OpenAI integration
pip install traqo[anthropic]        # + Anthropic integration
pip install traqo[langchain]        # + LangChain integration
pip install traqo[all]              # Everything
```

## Quick Start

### 1. Trace a function

```python
from traqo import Tracer, trace
from pathlib import Path

@trace()
async def summarize(text: str) -> str:
    # your logic here
    return summary

@trace()
async def pipeline(docs: list[str]) -> list[str]:
    return [await summarize(doc) for doc in docs]

async with Tracer(Path("traces/my_run.jsonl")):
    results = await pipeline(["doc1", "doc2"])
```

### 2. Auto-trace LLM calls

```python
from traqo.integrations.openai import traced_openai
from openai import OpenAI

client = traced_openai(OpenAI(), operation="summarize")
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Summarize this..."}],
)
# Token usage, duration, input/output all captured automatically
```

Works the same way for Anthropic and LangChain:

```python
from traqo.integrations.anthropic import traced_anthropic
from traqo.integrations.langchain import traced_model
```

### 3. Read your traces

```bash
# Last line is always trace_end with summary stats
tail -1 traces/my_run.jsonl | jq .

# All LLM calls
grep '"type":"llm_call"' traces/my_run.jsonl | jq .

# Errors
grep '"status":"error"' traces/**/*.jsonl

# Token costs
grep '"type":"llm_call"' traces/**/*.jsonl | jq '.token_usage'
```

## API Reference

### `Tracer(path, *, metadata=None, capture_content=True)`

Creates a trace session writing to a JSONL file. Use as a context manager.

```python
with Tracer(
    Path("traces/run.jsonl"),
    metadata={"run_id": "abc123", "model": "gpt-4o"},
    capture_content=False,  # Omit LLM input/output (keep tokens, duration)
):
    await my_pipeline()
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `Path` | required | JSONL file path. Parent dirs created automatically. |
| `metadata` | `dict` | `{}` | Arbitrary metadata written to `trace_start`. |
| `capture_content` | `bool` | `True` | If `False`, LLM inputs/outputs omitted. |

**Methods:**

| Method | Description |
|---|---|
| `log(name, data)` | Write a custom event |
| `llm_event(model=, input_messages=, output_text=, token_usage=, duration_s=, operation=)` | Write an `llm_call` event |
| `span(name, inputs)` | Manual span context manager |
| `child(name, path)` | Create a child tracer writing to a separate file |

### `@trace(name=None, *, capture_input=True, capture_output=True)`

Decorator that wraps a function in a span. Works with sync and async functions.

```python
@trace()
async def my_step(data: list) -> dict:
    return process(data)

@trace("custom_name", capture_input=False)
def sensitive_step(secret: str) -> str:
    return handle(secret)
```

When no tracer is active, `@trace` is a pure passthrough with zero overhead.

### `get_tracer() -> Tracer | None`

Returns the active tracer for the current context, or `None`.

```python
from traqo import get_tracer

tracer = get_tracer()
if tracer:
    tracer.log("checkpoint", {"count": len(results)})
```

### `disable()` / `enable()`

```python
import traqo
traqo.disable()  # All tracing becomes no-op
traqo.enable()   # Re-enable
```

Or via environment variable: `TRAQO_DISABLED=1`

## Child Tracers

For concurrent agents or workers that produce many events. Each child writes to its own file, linked to the parent.

```python
with Tracer(Path("traces/pipeline.jsonl")) as tracer:
    child = tracer.child("reentrancy_agent", Path("traces/agents/reentrancy.jsonl"))
    with child:
        await run_agent(...)
```

The parent trace records `child_started` / `child_ended` events and includes child summaries in `trace_end`.

## JSONL Format

Every line is a self-contained JSON object. Six event types:

| Type | When | Key Fields |
|---|---|---|
| `trace_start` | Tracer enters | `tracer_version`, `metadata` |
| `span_start` | Function/span begins | `id`, `parent_id`, `name`, `input` |
| `span_end` | Function/span ends | `id`, `duration_s`, `status`, `output`, `error` |
| `llm_call` | LLM invocation | `model`, `input`, `output`, `token_usage`, `duration_s` |
| `event` | Custom checkpoint | `name`, `data` |
| `trace_end` | Tracer exits | `duration_s`, `stats`, `children` |

## Query with DuckDB

```sql
SELECT model, count(*) as calls,
       sum(token_usage.input_tokens) as total_in,
       sum(token_usage.output_tokens) as total_out,
       avg(duration_s) as avg_duration
FROM read_json('traces/**/*.jsonl')
WHERE type = 'llm_call'
GROUP BY model;
```

## vs Alternatives

| Dimension | traqo | Opik (self-hosted) | Langfuse (self-hosted) |
|---|---|---|---|
| Infrastructure | None (filesystem) | Docker + ClickHouse + MySQL | Docker + Postgres |
| Setup | `pip install traqo` | Docker compose + config | Docker compose + config |
| Monthly cost | $0 | $50-200 | $50-200 |
| Data format | JSONL (portable) | ClickHouse tables | Postgres tables |
| Query method | grep / DuckDB / AI | SQL + UI | SQL + UI |
| Dependencies | Zero | Many | Many |

## License

MIT
