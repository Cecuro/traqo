---
name: traqo-tracing
description: >-
  Read, analyze, and visualize traqo JSONL traces for application observability.
  Use when: (1) reading or debugging .jsonl/.jsonl.gz trace files, (2) investigating
  token usage or costs, (3) analyzing pipeline execution flow or errors,
  (4) adding tracing instrumentation to Python code, (5) querying trace data
  with shell commands, (6) launching the trace viewer UI. Triggers on phrases
  like "read the trace", "what happened in the pipeline", "token usage",
  "why did it fail", "add tracing", "trace this function", "check the logs",
  "show me the traces", "open the dashboard", "visualize the run".
---

# traqo Trace Analysis

Analyze JSONL traces produced by the `traqo` Python package.

## Trace File Structure

Traces may be stored as raw `.jsonl`, compressed `.jsonl.gz`, or split into `.jsonl.gz` (main) + `.content.jsonl.zst` (externalized large span inputs). Last line is always `trace_end` with summary stats. Start there.

For compressed traces, large `span_start` inputs (>10 KB) are replaced with `{"_ref": "<span_id>", "_size": N}` stubs. The full input lives in the `.content.jsonl.zst` sidecar file.

## Event Types

| Event | Key Fields |
|-------|------------|
| `trace_start` | `tracer_version`, `input`, `metadata`, `tags`, `thread_id` |
| `span_start` | `id`, `parent_id`, `name`, `input`, `metadata`, `tags`, `kind` |
| `span_end` | `id`, `parent_id`, `name`, `duration_s`, `status`, `output`, `metadata`, `tags`, `kind` |
| `event` | `name`, `data` (arbitrary dict) |
| `trace_end` | `duration_s`, `output`, `stats`, `children` |

Every event has `id` and `parent_id` for tree reconstruction. The `kind` field categorizes spans (e.g. `"llm"`, `"tool"`, `"retriever"`). LLM-specific data (`model`, `provider`, `token_usage`) lives in `metadata`. `tags` is a list of strings for filtering. `thread_id` groups traces into conversations.

## Event Structure

```json
{"type":"trace_start","ts":"2026-02-20T10:00:00Z","tracer_version":"0.2.0","input":{"query":"hello"},"tags":["production"],"thread_id":"conv-123","metadata":{"run_id":"abc"}}
```

```json
{"type":"span_start","id":"x1y2z3","parent_id":"a1b2c3","ts":"2026-02-20T10:00:01Z","name":"classify","kind":"llm","tags":["gpt-4o"],"input":[{"role":"user","content":"..."}],"metadata":{"provider":"openai","model":"gpt-4o"}}
```

```json
{"type":"span_end","id":"x1y2z3","parent_id":"a1b2c3","ts":"2026-02-20T10:00:03Z","name":"classify","kind":"llm","duration_s":1.8,"status":"ok","output":"...","metadata":{"provider":"openai","model":"gpt-4o","token_usage":{"input_tokens":1500,"output_tokens":800,"cache_read_tokens":1200,"cache_creation_tokens":0}}}
```

```json
{"type":"trace_end","ts":"2026-02-20T10:05:00Z","duration_s":300.0,"output":{"response":"..."},"stats":{"spans":15,"events":5,"total_input_tokens":45000,"total_output_tokens":12000,"total_cache_read_tokens":30000,"total_cache_creation_tokens":5000,"errors":0},"children":[{"name":"agent_a","path":"traces/agent_a.jsonl","duration_s":45.2,"spans":3,"total_input_tokens":5000,"total_output_tokens":2000}]}
```

## Reading Traces

### File formats

Traces on disk may be raw `.jsonl` or compressed `.jsonl.gz` (with an optional `.content.jsonl.zst` sidecar). Check what you have first:

```bash
ls traces/    # Look for .jsonl, .jsonl.gz, .content.jsonl.zst
```

For **raw `.jsonl`** — use `grep`, `tail`, `jq` directly.
For **compressed `.jsonl.gz`** — use `zcat` (Linux) or `gzcat` (macOS) to decompress, then pipe to `grep`/`jq`. Or use `zgrep` for grep-like searches.

```bash
# Raw
tail -1 trace.jsonl | jq .

# Compressed (macOS)
gzcat trace.jsonl.gz | tail -1 | jq .

# Compressed (Linux)
zcat trace.jsonl.gz | tail -1 | jq .
```

**Tip:** For large or compressed traces, prefer `traqo ui` over shell commands.

### Navigation
```bash
# Overview (always start here — last line is trace_end with stats)
gzcat trace.jsonl.gz | tail -1 | jq .

# Follow child traces
gzcat trace.jsonl.gz | tail -1 | jq '.children[].path'
```

### Token Usage
```bash
# Per-span tokens from metadata (input_tokens includes cached)
zgrep '"token_usage"' trace.jsonl.gz | jq '.metadata.token_usage'

# Total from summary (includes cache breakdown)
gzcat trace.jsonl.gz | tail -1 | jq '.stats | {total_input_tokens, total_output_tokens, total_cache_read_tokens, total_cache_creation_tokens}'
```

### Errors
```bash
zgrep '"status":"error"' traces/**/*.jsonl.gz | jq '{name: .name, error: .error}'
```

### LLM Spans
```bash
# All LLM spans
zgrep '"kind":"llm"' trace.jsonl.gz | jq '{name, metadata}'

# LLM spans with model and duration
zgrep '"kind":"llm"' trace.jsonl.gz | grep span_end | jq '{name, model: .metadata.model, duration_s}'
```

### Span Tree
```bash
zgrep '"type":"span_start"' trace.jsonl.gz | jq '{id, parent_id, name, kind}'
```

### Python reader API
```python
from traqo.reader import iter_llm_spans, aggregate_tokens
from pathlib import Path

# Iterate LLM spans (handles .jsonl and .jsonl.gz transparently)
for span in iter_llm_spans(Path("trace.jsonl.gz")):
    print(f"{span.model}: {span.input_tokens}in/{span.output_tokens}out ({span.duration_s}s)")

# Aggregate tokens by model
aggregate_tokens(Path("trace.jsonl.gz"))
# {'gpt-4o': {'input': 45000, 'output': 12000}}
```

### Retrieving externalized content

When a `span_start` input was too large (>10 KB), it gets replaced with a reference stub:
```json
{"type":"span_start","id":"abc123","input":{"_ref":"abc123","_size":245000}}
```

The full input lives in the `.content.jsonl.zst` sidecar file. To retrieve it:

```python
from traqo.compress import read_content
from pathlib import Path

# read_content streams the zst file and stops at the matching span — ~1 MB memory
data = read_content(Path("trace.content.jsonl.zst"), "abc123")
# Returns the original input dict, or None if not found
```

In the **trace viewer UI** (`traqo ui`), externalized inputs show a "Load full input" button that fetches on click via the `/api/content` endpoint — no manual work needed.

## Claude Code Integration

Convert Claude Code session transcripts into traqo traces.

```bash
# Sync a single session
traqo cc-sync path/to/session.jsonl

# Sync all sessions from ~/.claude/projects/
traqo cc-sync --all --output-dir ./traces

# As a Claude Code Stop hook (~/.claude/settings.json)
# { "hooks": { "Stop": [{ "type": "command", "command": "traqo cc-sync --hook" }] } }
```

Produces one trace per session with turn spans, LLM spans (with token usage including cache breakdown), tool call spans, and subagent hierarchy.

## Trace Viewer UI

Built-in React web dashboard. Bundled with the pip package — no extra install needed.

```bash
# Local traces
traqo ui traces/

# Custom port
traqo ui traces/ --port 8080

# S3 or GCS
traqo ui s3://bucket/prefix
traqo ui gs://bucket/prefix
```

Features: span tree with waterfall timing, tag/status filtering, search, token usage charts, cache token totals, keyboard shortcuts (↑/↓ navigate, Esc back, ? help). Handles compressed `.jsonl.gz` traces and loads externalized span inputs on demand via "Load full input" button. Suggest the UI when the user wants to visually explore or browse traces.

## Adding Tracing to Code

### Decorate a function
```python
from traqo import trace

@trace()
async def my_function(data):
    return process(data)

@trace(metadata={"component": "auth"}, tags=["auth"], kind="tool")
def login(user):
    return authenticate(user)
```

### Access current span from decorated function
```python
from traqo import trace, get_current_span

@trace()
def classify(text):
    span = get_current_span()
    if span:
        span.set_metadata("confidence", 0.95)
    return result
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

### Trace a Claude Agent SDK session
```python
from claude_agent_sdk import query, ClaudeAgentOptions
from traqo.integrations.claude_agent_sdk import traqo_agent

# Standalone
async with traqo_agent("code-review", output_dir="./traces", tags=["review"]) as hooks:
    async for msg in query(prompt="Review this PR", options=ClaudeAgentOptions(hooks=hooks)):
        print(msg)

# Nested inside a parent pipeline trace
with Tracer(Path("traces/pipeline.jsonl"), tags=["ci"]) as tracer:
    async with traqo_agent("code-review", tags=["review"]) as hooks:
        async for msg in query(prompt="Review", options=ClaudeAgentOptions(hooks=hooks)):
            ...
```

### Use spans with metadata
```python
from traqo import get_tracer

tracer = get_tracer()
if tracer:
    with tracer.span("my_step", input=data, metadata={"model": "gpt-4o"}, tags=["llm"], kind="llm") as span:
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

with Tracer(
    Path("traces/run.jsonl"),
    input={"query": "hello"},
    metadata={"run_id": "abc123"},
    tags=["production"],
    thread_id="conv-456",
) as tracer:
    result = await main()
    tracer.set_output({"response": result})
```

### Child tracer for concurrent agents
```python
child = tracer.child("my_agent", Path("traces/agents/my_agent.jsonl"))
with child:
    await run_agent()
```
