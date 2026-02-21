# Changelog

## 0.2.0 (2026-02-21)

**Breaking changes** — everything is a span.

### Changed
- **`llm_call` event type removed** — LLM calls are now regular spans with `kind="llm"` and metadata (`model`, `provider`, `token_usage`)
- **`tracer.llm_event()` removed** — use `tracer.span()` with metadata instead
- **`tracer.span()` yields a `Span` object** instead of a string span_id. Use `span.set_output()`, `span.set_metadata()`, `span.update_metadata()`
- **`tracer.span()` signature changed** — now takes keyword args: `span(name, *, input=, metadata=, tags=, kind=)`
- **`trace_end` stats** — `llm_calls` field removed, replaced by `total_input_tokens` / `total_output_tokens`
- **Child summary** — `llm_calls` replaced by `spans`, `total_input_tokens`, `total_output_tokens`
- **5 event types** instead of 6: `trace_start`, `span_start`, `span_end`, `event`, `trace_end`

### Added
- `Span` class — mutable handle yielded by `tracer.span()`, with `set_output()`, `set_metadata()`, `update_metadata()`
- `kind` field on spans — categorize spans as `"llm"`, `"tool"`, `"retriever"`, etc.
- `metadata` dict on spans — universal extension point for model, provider, token_usage, and any custom data
- `tags` on `Tracer` and `span()` — list of strings for filtering/categorization
- `input` on `Tracer` — trace-level input written to `trace_start`
- `tracer.set_output()` — trace-level output written to `trace_end`
- `thread_id` on `Tracer` — conversation/thread grouping ID written to `trace_start`
- `get_current_span()` — access the active span from within `@trace`-decorated functions
- `@trace(metadata=, tags=, kind=)` — pass metadata, tags, and kind through the decorator
- Token accumulation from `metadata.token_usage` convention — tokens in span metadata are automatically summed in `trace_end` stats
- Integrations (OpenAI, Anthropic, LangChain) now use span-based tracing with `kind="llm"`

## 0.1.0 (2026-02-20)

Initial release.

### Core
- `Tracer` context manager with JSONL output, `trace_start`/`trace_end` lifecycle events
- `@trace` decorator for sync and async functions with automatic span nesting
- `get_tracer()` for manual access to the active tracer
- Hierarchical spans via `contextvars` span stack — parent/child relationships tracked automatically
- `log()` for custom business logic events
- `span()` manual span context manager
- Child tracers with separate files and parent linkage
- `capture_content` flag to omit LLM inputs/outputs (privacy/size control)
- `disable()`/`enable()` global toggle + `TRAQO_DISABLED` env var
- Thread-safe writes via `threading.Lock`
- Async context manager support (`async with Tracer(...)`)
- Zero runtime dependencies (stdlib only)

### Integrations
- `traqo.integrations.openai` — `traced_openai()` wraps OpenAI sync/async clients
- `traqo.integrations.anthropic` — `traced_anthropic()` wraps Anthropic sync/async clients
- `traqo.integrations.langchain` — `traced_model()` wraps LangChain `BaseChatModel`
