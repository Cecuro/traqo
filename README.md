# traqo

Structured tracing for applications. JSONL files, hierarchical spans, zero infrastructure.

```python
from traqo import Tracer, trace
from pathlib import Path

@trace()
def classify(text: str) -> str:
    response = llm.chat(text)
    return response

with Tracer(Path("traces/run.jsonl"), input={"query": "Is this a bug?"}):
    result = classify("Is this a bug?")
```

Your traces are just `.jsonl` files. Read them with `grep`, query them with DuckDB, or hand them to an AI assistant.

## Why traqo?

- **Zero infrastructure** -- no server, no database, no account. `pip install traqo` and go.
- **AI-first** -- JSONL is text. AI assistants read your traces directly, no browser needed.
- **Hierarchical spans** -- not flat logs. Reconstruct the full call tree across functions and files.
- **Everything is a span** -- LLM calls, DB queries, HTTP requests. All spans with metadata.
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
def summarize(text: str) -> str:
    # your logic here
    return summary

@trace()
def pipeline(docs: list[str]) -> list[str]:
    return [summarize(doc) for doc in docs]

with Tracer(
    Path("traces/my_run.jsonl"),
    input={"docs": ["doc1", "doc2"]},
    tags=["production"],
) as tracer:
    results = pipeline(["doc1", "doc2"])
    tracer.set_output({"count": len(results)})
```

`@trace()` works with `async def` and `async with` too — it detects and handles both automatically.

### 2. Auto-trace LLM calls

```python
from traqo.integrations.openai import traced_openai
from openai import OpenAI

client = traced_openai(OpenAI(), operation="summarize")
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Summarize this..."}],
)
# Token usage, model, input/output all captured automatically as span metadata
```

Works the same way for Anthropic and LangChain:

```python
from traqo.integrations.anthropic import traced_anthropic
from traqo.integrations.langchain import traced_model
```

### 3. Use metadata, tags, and kind

```python
with Tracer(Path("traces/run.jsonl"), tags=["prod"]) as tracer:
    with tracer.span(
        "classify",
        input={"text": "Is this a bug?"},
        metadata={"model": "gpt-4o", "provider": "openai"},
        tags=["llm"],
        kind="llm",
    ) as span:
        result = call_llm(...)
        span.set_metadata("token_usage", {"input_tokens": 100, "output_tokens": 50})
        span.set_output(result)
```

### 4. Access the current span from anywhere

```python
from traqo import trace, get_current_span

@trace()
def classify(text: str) -> str:
    span = get_current_span()
    if span:
        span.set_metadata("confidence", 0.95)
        span.set_metadata("model", "gpt-4o")
    return result
```

### 5. Read your traces

```bash
# Last line is always trace_end with summary stats
tail -1 traces/my_run.jsonl | jq .

# All LLM spans
grep '"kind":"llm"' traces/my_run.jsonl | jq .

# Filter by tag
grep '"tags"' traces/my_run.jsonl | jq .

# Errors
grep '"status":"error"' traces/**/*.jsonl

# Token usage from span metadata
grep '"token_usage"' traces/**/*.jsonl | jq '.metadata.token_usage'
```

## Trace Viewer UI

Browse and inspect traces in your browser. Zero dependencies — uses Python's built-in HTTP server.

```bash
traqo ui ./traces                  # Serve traces on http://localhost:7600
traqo ui ./traces --port 8080     # Custom port
python -m traqo ui ./traces       # Alternative invocation
```

Features: folder navigation, search/filter, span tree with waterfall timing, JSON viewer with syntax highlighting, token usage visualization, keyboard shortcuts (Escape to go back, ? for help).

## API Reference

### `Tracer(path, *, input=None, metadata=None, tags=None, thread_id=None, capture_content=True, backends=None)`

Creates a trace session writing to a JSONL file. Use as a context manager.

```python
with Tracer(
    Path("traces/run.jsonl"),
    input={"query": "What is the weather?"},
    metadata={"run_id": "abc123"},
    tags=["production", "chatbot"],
    thread_id="conv-456",
    capture_content=False,  # Integrations omit LLM input/output
) as tracer:
    result = my_pipeline()
    tracer.set_output({"response": result})
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `Path` | required | JSONL file path. Parent dirs created automatically. |
| `input` | `Any` | `None` | Trace input, written to `trace_start`. |
| `metadata` | `dict` | `{}` | Arbitrary metadata written to `trace_start`. |
| `tags` | `list[str]` | `[]` | Tags for filtering/categorization, written to `trace_start`. |
| `thread_id` | `str` | `None` | Conversation/thread grouping ID, written to `trace_start`. |
| `capture_content` | `bool` | `True` | If `False`, integration wrappers (OpenAI, Anthropic, LangChain) omit LLM message inputs/outputs. The `@trace` decorator has separate `capture_input`/`capture_output` flags. |
| `backends` | `list[Backend]` | `None` | Storage backends notified on events and trace completion. The local JSONL file is always written regardless. |

**Methods:**

| Method | Description |
|---|---|
| `span(name, *, input=, metadata=, tags=, kind=)` | Span context manager. Yields a `Span` object. |
| `set_output(value)` | Set trace-level output (written to `trace_end`). |
| `log(name, data)` | Write a custom event. |
| `child(name, path)` | Create a child tracer writing to a separate file. |

### `Span`

Mutable handle yielded by `tracer.span()`. Set output and metadata during execution.

```python
with tracer.span("my_step", input=data, tags=["important"], kind="tool") as span:
    result = do_work()
    span.set_output(result)
    span.set_metadata("latency_ms", 42)
    span.update_metadata({"extra": "info"})
```

| Method | Description |
|---|---|
| `set_output(value)` | Set span output (written to `span_end`) |
| `set_metadata(key, value)` | Set a metadata key |
| `update_metadata(dict)` | Merge a dict into metadata |

### `@trace(name=None, *, capture_input=True, capture_output=True, metadata=None, tags=None, kind=None)`

Decorator that wraps a function in a span. Works with sync and async functions.

```python
@trace()
async def my_step(data: list) -> dict:
    return process(data)

@trace("custom_name", capture_input=False, kind="tool")
def sensitive_step(secret: str) -> str:
    return handle(secret)

@trace(metadata={"component": "auth"}, tags=["auth"], kind="tool")
def login(user: str) -> bool:
    return authenticate(user)
```

When no tracer is active, `@trace` is a pure passthrough with zero overhead.

### `get_current_span() -> Span | None`

Returns the current active span, or `None`. Use inside `@trace`-decorated functions to set metadata dynamically.

```python
from traqo import trace, get_current_span

@trace()
def my_function(text: str) -> str:
    span = get_current_span()
    if span:
        span.set_metadata("custom_key", "custom_value")
    return process(text)
```

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
        run_agent(...)
```

The parent trace records `child_started` / `child_ended` events and includes child summaries in `trace_end`.

## JSONL Format

Every line is a self-contained JSON object. Five event types:

| Type | When | Key Fields |
|---|---|---|
| `trace_start` | Tracer enters | `tracer_version`, `input`, `metadata`, `tags`, `thread_id` |
| `span_start` | Span begins | `id`, `parent_id`, `name`, `input`, `metadata`, `tags`, `kind` |
| `span_end` | Span ends | `id`, `duration_s`, `status`, `output`, `metadata`, `tags`, `kind` |
| `event` | Custom checkpoint | `name`, `data` |
| `trace_end` | Tracer exits | `duration_s`, `output`, `stats`, `children` |

The `kind` field categorizes spans (e.g. `"llm"`, `"tool"`, `"retriever"`). The `tags` field is a list of strings for filtering. Both are omitted when not set.

The `metadata` dict is the universal extension point. LLM-specific data like `model`, `provider`, and `token_usage` are stored there.

## Query with DuckDB

```sql
-- All LLM spans with token usage
SELECT metadata->>'model' as model,
       count(*) as calls,
       sum((metadata->'token_usage'->>'input_tokens')::int) as total_in,
       sum((metadata->'token_usage'->>'output_tokens')::int) as total_out,
       avg(duration_s) as avg_duration
FROM read_json('traces/**/*.jsonl')
WHERE kind = 'llm'
GROUP BY model;

-- All traces for a conversation thread
SELECT * FROM read_json('traces/**/*.jsonl')
WHERE thread_id = 'conv-123'
AND type = 'trace_start';
```

## License

MIT
