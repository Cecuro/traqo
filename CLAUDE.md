# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv run pytest tests/ -v              # Run all tests
uv run pytest tests/test_tracer.py -v # Run a single test file
uv run pytest tests/ -k "test_nan"   # Run tests matching a pattern
uv run ruff check --fix . && uv run ruff format .  # Lint + format
uv run pyright traqo/                # Type check
```

**Before finishing any task**, run ruff and pyright to catch issues:
```bash
uv run ruff check --fix . && uv run ruff format . && uv run pyright traqo/
```

Zero runtime dependencies — only stdlib. Storage backends (S3, GCS) are optional extras.

## Development guidelines

- **Package management**: Only use `uv`, never pip.
- **Git hygiene**: Never use `git add -A` or `git add .` — add files explicitly by name. This prevents accidental commits of `.jsonl` trace files or `.env`.
- **File creation**: Don't create `.md` or `.txt` files unless explicitly requested.
- **Simple solutions**: Prefer straightforward code over abstractions. Remove unused code immediately.

## Frontend (Trace Viewer UI)

The UI is a React + TypeScript app built with Vite, located in `frontend/`.

```bash
cd frontend
npm ci                  # Install dependencies
npm run dev             # Dev server on http://localhost:5173 (proxies /api to :7600)
npm run build           # Production build → ../traqo/ui/static/
```

During development, run the Python backend alongside the Vite dev server:
```bash
traqo ui ./traces       # Backend on :7600
cd frontend && npm run dev  # Frontend on :5173 with API proxy
```

The built output (`traqo/ui/static/`) is gitignored — it's generated in CI by `publish.yml` before packaging. The Python server (`traqo/ui/server.py`) serves the built static files and provides SPA fallback routing.

## Releasing

1. Bump version in **both** `pyproject.toml` and `traqo/_version.py` (they must match).
2. Verify `skills/traqo-tracing/SKILL.md` reflects current features (event types, integrations, CLI commands, UI).
3. Commit the version bump (`chore: bump version to X.Y.Z`).
4. Push to `main`.
5. Create a GitHub release with tag `vX.Y.Z` via `gh release create vX.Y.Z --title "vX.Y.Z" --notes "..."`.
6. The `publish.yml` workflow automatically builds the React frontend (`npm ci && npm run build` in `frontend/`), then builds and publishes to PyPI.

### Agent skill

The traqo-tracing skill lives in `skills/traqo-tracing/SKILL.md`. Install it via:
```bash
npx skills add Cecuro/traqo --yes --global
```

## Architecture

traqo is a structured JSONL tracing library. It writes trace events (one JSON object per line) to local files. No server, no network, no storage costs.

### Core flow

`Tracer` (context manager) → opens a JSONL file → `span()` creates nested spans → each span writes `span_start`/`span_end` events → `Tracer.__exit__` writes `trace_end` with aggregated stats.

Parent-child span nesting is tracked via a `ContextVar` span stack (`_span_stack` in `tracer.py`). Each span reads the current stack top as its `parent_id`, then pushes itself. This is async-safe — concurrent tasks get isolated stacks.

### Key modules

- **`tracer.py`** — `Tracer` and `Span` classes, context management, file I/O. The `Span` object is a mutable handle yielded by `tracer.span()` — set output and metadata during execution, written to `span_end` on exit.
- **`decorator.py`** — `@trace()` wraps sync/async functions. Pure passthrough when no tracer is active (zero overhead). Uses `inspect.signature` to capture args.
- **`serialize.py`** — Single serialization path via `_serialize_value()`. Handles datetime, UUID, Enum, dataclass, Pydantic, numpy, circular refs (sibling-tolerant add/discard), NaN→None. No limits on string length, depth, or collection size. `to_json()` pre-processes through `_serialize_value()` then `json.dumps(allow_nan=False)`. The `json_default` function delegates to `_serialize_value`.
- **`integrations/`** — OpenAI and Anthropic use a proxy-wrapper pattern (intercept API calls, delegate via `__getattr__`). LangChain has two approaches: `TraqoCallback` (callback handler writing spans directly) and `TracedChatModel` (BaseChatModel subclass wrapping `_generate`).
- **`backend.py`** — `Backend` protocol (runtime_checkable) with `on_event()`, `on_trace_complete()`, `close()`. Shared `ThreadPoolExecutor` for background uploads. `flush_backends()` waits + recreates pool (for servers). `shutdown_backends()` waits + tears down (for atexit). An atexit handler is registered on first `submit_background()` call.
- **`backends/`** — `S3Backend` (requires `traqo[s3]`), `GCSBackend` (requires `traqo[gcs]`), `LocalBackend` (stdlib only, copies to target dir with collision-safe filenames). All are batch-upload backends: `on_event()` is a no-op, upload happens in `on_trace_complete()`.
- **`ui/server.py`** — Built-in HTTP server for the trace viewer. Serves the React SPA from `ui/static/` (built from `frontend/`) and provides `/api/traces` and `/api/trace` endpoints. Supports local dirs, S3, and GCS sources.

### Token tracking convention

Spans store `metadata["token_usage"] = {"input_tokens": N, "output_tokens": N}`. The tracer's `_accumulate_tokens()` sums these into `_stats_input_tokens`/`_stats_output_tokens`, reported in `trace_end.stats`.

### Storage backends

Backends are additive — the local JSONL file is always written as the source of truth. Backends receive notifications via `on_event()` (after each write) and `on_trace_complete()` (after file close). `close()` is NOT called from `Tracer.__exit__()` — backends are long-lived and may be shared. Cleanup happens via atexit or explicit `shutdown_backends()`. Child tracers inherit their parent's backends. Backend validation happens in `Tracer.__init__()`.

### 5 event types

`trace_start`, `span_start`, `span_end`, `event` (point-in-time log via `tracer.log()`), `trace_end`.

## Integration patterns

- **OpenAI/Anthropic**: `traced_openai(client)` / `traced_anthropic(client)` return wrapped clients. Span kind is `"llm"`.
- **LangChain callback**: `TraqoCallback()` passed via `config={"callbacks": [callback]}`. Maps LangChain `run_id` → traqo `span_id`. Falls back to `_get_parent_id()` to nest under `@trace` decorator spans.
- **LangChain wrapper**: `traced_model(model)` returns a `TracedChatModel`. Has `bind_tools()` override because `BaseChatModel.bind_tools` raises `NotImplementedError` and `__getattr__` doesn't fire for methods defined on the class.
