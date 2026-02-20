---
name: traqo-tracing
description: Read, analyze, and write traqo JSONL traces for LLM application observability. Use when: (1) reading or debugging .jsonl trace files, (2) investigating LLM token usage or costs, (3) analyzing pipeline execution flow or errors, (4) adding tracing instrumentation to Python code, (5) querying trace data with grep or DuckDB. Triggers on phrases like "read the trace", "what happened in the pipeline", "token usage", "why did it fail", "add tracing", "trace this function", "check the logs".
---

# traqo Trace Analysis

Analyze JSONL traces produced by the `traqo` Python package.

## Trace File Structure

Last line is always `trace_end` with summary stats. Start there.

## Event Types

| Event | Key Fields |
|-------|------------|
| `trace_start` | `tracer_version`, `metadata` (run_id, model, etc.) |
| `span_start` | `id`, `parent_id`, `name`, `input` |
| `span_end` | `id`, `parent_id`, `name`, `duration_s`, `status`, `output`, `error` |
| `llm_call` | `model`, `input`, `output`, `token_usage`, `duration_s`, `operation` |
| `event` | `name`, `data` (arbitrary dict) |
| `trace_end` | `duration_s`, `stats`, `children` (child trace paths) |

Every event has `id` and `parent_id` for tree reconstruction.

## Event Structure

```json
{"type":"llm_call","id":"x1y2z3","parent_id":"a1b2c3","ts":"2026-02-20T10:00:01Z","model":"gpt-4o","input":[{"role":"user","content":"..."}],"output":"...","duration_s":1.8,"token_usage":{"input_tokens":1500,"output_tokens":800},"operation":"classify"}
```

```json
{"type":"trace_end","ts":"2026-02-20T10:05:00Z","duration_s":300.0,"stats":{"spans":15,"llm_calls":8,"events":5,"total_input_tokens":45000,"total_output_tokens":12000,"errors":0},"children":[{"name":"agent_a","path":"traces/agent_a.jsonl","duration_s":45.2,"llm_calls":3}]}
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
# Per-call tokens
grep '"type":"llm_call"' trace.jsonl | jq '.token_usage'

# Total from summary
tail -1 trace.jsonl | jq '.stats | {total_input_tokens, total_output_tokens}'
```

### Errors
```bash
grep '"status":"error"' traces/**/*.jsonl | jq '{name: .name, error: .error}'
```

### LLM Calls
```bash
# All LLM calls with model and duration
grep '"type":"llm_call"' trace.jsonl | jq '{model, duration_s, operation}'

# Full input/output for a specific operation
grep '"type":"llm_call"' trace.jsonl | jq 'select(.operation=="classify")'
```

### Span Tree
```bash
# All spans
grep '"type":"span_start"' trace.jsonl | jq '{id, parent_id, name}'

# Everything inside a specific span
grep '"parent_id":"<span_id>"' trace.jsonl | jq .
```

### DuckDB
```sql
SELECT model, count(*) as calls,
       sum(token_usage.input_tokens) as total_in,
       sum(token_usage.output_tokens) as total_out,
       avg(duration_s) as avg_duration
FROM read_json('traces/**/*.jsonl')
WHERE type = 'llm_call'
GROUP BY model;
```

## Adding Tracing to Code

### Decorate a function
```python
from traqo import trace

@trace()
async def my_function(data):
    return process(data)
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
