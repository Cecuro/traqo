# Changelog

## 0.1.0 (2026-02-20)

Initial release.

### Core
- `Tracer` context manager with JSONL output, `trace_start`/`trace_end` lifecycle events
- `@trace` decorator for sync and async functions with automatic span nesting
- `get_tracer()` for manual access to the active tracer
- Hierarchical spans via `contextvars` span stack -- parent/child relationships tracked automatically
- `llm_event()` for manual LLM call logging with token usage tracking
- `log()` for custom business logic events
- `span()` manual span context manager
- Child tracers with separate files and parent linkage
- `capture_content` flag to omit LLM inputs/outputs (privacy/size control)
- `disable()`/`enable()` global toggle + `TRAQO_DISABLED` env var
- Thread-safe writes via `threading.Lock`
- Async context manager support (`async with Tracer(...)`)
- Zero runtime dependencies (stdlib only)

### Integrations
- `traqo.integrations.openai` -- `traced_openai()` wraps OpenAI sync/async clients
- `traqo.integrations.anthropic` -- `traced_anthropic()` wraps Anthropic sync/async clients
- `traqo.integrations.langchain` -- `traced_model()` wraps LangChain `BaseChatModel`

### JSONL Format
- Six event types: `trace_start`, `span_start`, `span_end`, `llm_call`, `event`, `trace_end`
- `trace_end` includes aggregate stats (spans, llm_calls, events, tokens, errors) and child summaries
- All events include `id`, `parent_id`, `ts` for tree reconstruction
