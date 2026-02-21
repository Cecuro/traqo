---
name: traqo-tracing
description: Read, analyze, and write traqo JSONL traces for application observability. Use when: (1) reading or debugging .jsonl trace files, (2) investigating token usage or costs, (3) analyzing pipeline execution flow or errors, (4) adding tracing instrumentation to Python code, (5) querying trace data with grep or DuckDB. Triggers on phrases like "read the trace", "what happened in the pipeline", "token usage", "why did it fail", "add tracing", "trace this function", "check the logs".
---

# traqo Trace Analysis

Analyze JSONL traces produced by the `traqo` Python package.

## Trace File Structure

Last line is always `trace_end` with summary stats. Start there.

## Event Types

| Event | Key Fields |
|-------|------------|
| `trace_start` | `tracer_version`, `metadata` (run_id, model, etc.) |
| `span_start` | `id`, `parent_id`, `name`, `input`, `metadata`, `kind` |
| `span_end` | `id`, `parent_id`, `name`, `duration_s`, `status`, `output`, `metadata`, `kind` |
| `event` | `name`, `data` (arbitrary dict) |
| `trace_end` | `duration_s`, `stats`, `children` (child trace paths) |

Every event has `id` and `parent_id` for tree reconstruction. The `kind` field categorizes spans (e.g. `"llm"`, `"tool"`, `"retriever"`). LLM-specific data (`model`, `provider`, `token_usage`) lives in `metadata`.

## Event Structure

```json
{"type":"span_start","id":"x1y2z3","parent_id":"a1b2c3","ts":"2026-02-20T10:00:01Z","name":"classify","kind":"llm","input":[{"role":"user","content":"..."}],"metadata":{"provider":"openai","model":"gpt-4o"}}
```

```json
{"type":"span_end","id":"x1y2z3","parent_id":"a1b2c3","ts":"2026-02-20T10:00:03Z","name":"classify","kind":"llm","duration_s":1.8,"status":"ok","output":"...","metadata":{"provider":"openai","model":"gpt-4o","token_usage":{"input_tokens":1500,"output_tokens":800}}}
```

```json
{"type":"trace_end","ts":"2026-02-20T10:05:00Z","duration_s":300.0,"stats":{"spans":15,"events":5,"total_input_tokens":45000,"total_output_tokens":12000,"errors":0},"children":[{"name":"agent_a","path":"traces/agent_a.jsonl","duration_s":45.2,"spans":3,"total_input_tokens":5000,"total_output_tokens":2000}]}
```

## Common Queries

### Navigation
```bash
# Overview (always start here)
tail -1 trace.jsonl | jq .

# Follow child traces
tail -1 trace.jsonl | jq '.children[].path'
```

### Token Usage
```bash
# Per-span tokens from metadata
grep '"token_usage"' trace.jsonl | jq '.metadata.token_usage'

# Total from summary
tail -1 trace.jsonl | jq '.stats | {total_input_tokens, total_output_tokens}'
```

### Errors
```bash
grep '"status":"error"' traces/**/*.jsonl | jq '{name: .name, error: .error}'
```

### LLM Spans
```bash
# All LLM spans
grep '"kind":"llm"' trace.jsonl | jq '{name, metadata}'

# LLM spans with model and duration
grep '"kind":"llm"' trace.jsonl | grep span_end | jq '{name, model: .metadata.model, duration_s}'

# Spans for a specific provider
grep '"provider":"openai"' trace.jsonl | jq .
```

### Span Tree
```bash
# All spans
grep '"type":"span_start"' trace.jsonl | jq '{id, parent_id, name, kind}'

# Everything inside a specific span
grep '"parent_id":"<span_id>"' trace.jsonl | jq .
```

### DuckDB
```sql
SELECT metadata->>'model' as model,
       count(*) as calls,
       sum((metadata->'token_usage'->>'input_tokens')::int) as total_in,
       sum((metadata->'token_usage'->>'output_tokens')::int) as total_out,
       avg(duration_s) as avg_duration
FROM read_json('traces/**/*.jsonl')
WHERE kind = 'llm'
GROUP BY model;
```

## Adding Tracing to Code

### Decorate a function
```python
from traqo import trace

@trace()
async def my_function(data):
    return process(data)

@trace(metadata={"component": "auth"}, kind="tool")
def login(user):
    return authenticate(user)
```

### Wrap an LLM client
```python
from traqo.integrations.openai import traced_openai
client = traced_openai(OpenAI(), operation="classify")

from traqo.integrations.anthropic import traced_anthropic
client = traced_anthropic(AsyncAnthropic(), operation="analyze")

from traqo.integrations.langchain import traced_model
model = traced_model(ChatOpenAI(), operation="summarize")
```

### Use spans with metadata
```python
from traqo import get_tracer

tracer = get_tracer()
if tracer:
    with tracer.span("my_step", input=data, metadata={"model": "gpt-4o"}, kind="llm") as span:
        result = call_llm()
        span.set_metadata("token_usage", {"input_tokens": 100, "output_tokens": 50})
        span.set_output(result)
```

### Log a custom event
```python
from traqo import get_tracer
tracer = get_tracer()
if tracer:
    tracer.log("checkpoint", {"count": len(results)})
```

### Activate tracing
```python
from traqo import Tracer
from pathlib import Path

with Tracer(Path("traces/run.jsonl"), metadata={"run_id": "abc123"}):
    await main()
```

### Child tracer for concurrent agents
```python
child = tracer.child("my_agent", Path("traces/agents/my_agent.jsonl"))
with child:
    await run_agent()
```
