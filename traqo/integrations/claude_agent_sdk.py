"""Claude Agent SDK integration — async context manager that traces agent runs.

Wraps a Claude Agent SDK session with traqo tracing. The ``traqo_agent`` context
manager yields a ``hooks`` dict that plugs directly into ``ClaudeAgentOptions``.
When the agent finishes, the Stop hook converts the Claude transcript into a
traqo trace via ``cc_sync``.

Usage::

    from claude_agent_sdk import query, ClaudeAgentOptions
    from traqo.integrations.claude_agent_sdk import traqo_agent

    # Standalone
    async with traqo_agent("code-review", output_dir="./traces") as hooks:
        async for msg in query(
            prompt="Fix the bug", options=ClaudeAgentOptions(hooks=hooks)
        ):
            print(msg)

    # Nested — auto-registers as child trace in active parent
    from traqo import Tracer

    with Tracer("pipeline", trace_dir="./traces") as t:
        async with traqo_agent("code-review", tags=["review"]) as hooks:
            async for msg in query(
                prompt="Fix it", options=ClaudeAgentOptions(hooks=hooks)
            ):
                ...
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from claude_agent_sdk import HookMatcher as _HookMatcher
    from claude_agent_sdk import StopHookInput as _StopHookInput
    from claude_agent_sdk.types import HookContext as _HookContext
    from claude_agent_sdk.types import SyncHookJSONOutput as _SyncHookJSONOutput
except ImportError as err:
    raise ImportError(
        "claude-agent-sdk not installed. Install with: pip install claude-agent-sdk"
    ) from err

from traqo.tracer import get_tracer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SDK bug workaround: keep stdin open for hook callbacks
# ---------------------------------------------------------------------------
# The SDK's process_query() closes stdin immediately after sending a string
# prompt.  This breaks Stop hooks because the CLI subprocess checks
# ``this.inputClosed`` before sending the hook callback request, causing a
# "Stream closed" error.
#
# Fix: patch process_query to run end_input() in a background task that waits
# for hooks to complete before closing stdin.  This lets receive_messages()
# start immediately while stdin stays open for the control protocol.

_hook_done_event: asyncio.Event | None = None
_patch_installed = False


def _install_hooks_fix() -> asyncio.Event:
    """Patch the SDK to defer stdin closure until hooks complete.

    Returns an asyncio.Event that MUST be set after all hook callbacks finish.
    """
    global _hook_done_event, _patch_installed
    if _hook_done_event is not None:
        return _hook_done_event

    _hook_done_event = asyncio.Event()

    if _patch_installed:
        return _hook_done_event

    from claude_agent_sdk._internal.client import InternalClient
    from claude_agent_sdk._internal.message_parser import parse_message
    from claude_agent_sdk._internal.query import Query
    from claude_agent_sdk._internal.transport.subprocess_cli import (
        SubprocessCLITransport,
    )

    _orig_process_query = InternalClient.process_query

    async def _patched_process_query(
        self: Any,
        prompt: Any,
        options: Any,
        transport: Any = None,
    ):  # type: ignore[override]
        """Wraps process_query: defers end_input for string prompts with hooks."""
        from dataclasses import asdict, replace

        has_hooks = bool(options.hooks)

        # Only patch the specific case: string prompt + hooks
        if not (isinstance(prompt, str) and has_hooks):
            async for msg in _orig_process_query(self, prompt, options, transport):
                yield msg
            return

        # --- Reproduce process_query logic with deferred end_input ---
        configured_options = options
        if options.can_use_tool:
            if isinstance(prompt, str):
                raise ValueError("can_use_tool requires streaming mode.")
            if options.permission_prompt_tool_name:
                raise ValueError(
                    "can_use_tool and permission_prompt_tool_name are mutually exclusive."
                )
            configured_options = replace(options, permission_prompt_tool_name="stdio")

        if transport is not None:
            chosen_transport = transport
        else:
            chosen_transport = SubprocessCLITransport(
                prompt=prompt, options=configured_options
            )

        await chosen_transport.connect()

        sdk_mcp_servers = {}
        if configured_options.mcp_servers and isinstance(
            configured_options.mcp_servers, dict
        ):
            for sname, config in configured_options.mcp_servers.items():
                if isinstance(config, dict) and config.get("type") == "sdk":
                    sdk_mcp_servers[sname] = config["instance"]

        agents_dict = None
        if configured_options.agents:
            agents_dict = {
                aname: {k: v for k, v in asdict(adef).items() if v is not None}
                for aname, adef in configured_options.agents.items()
            }

        query_obj = Query(
            transport=chosen_transport,
            is_streaming_mode=True,
            can_use_tool=configured_options.can_use_tool,
            hooks=self._convert_hooks_to_internal_format(configured_options.hooks)
            if configured_options.hooks
            else None,
            sdk_mcp_servers=sdk_mcp_servers,
            agents=agents_dict,
        )

        try:
            await query_obj.start()
            await query_obj.initialize()

            # Write user message
            user_message = {
                "type": "user",
                "session_id": "",
                "message": {"role": "user", "content": prompt},
                "parent_tool_use_id": None,
            }
            await chosen_transport.write(json.dumps(user_message) + "\n")

            # KEY FIX: run end_input in a background task that waits for hooks
            async def _deferred_end_input() -> None:
                assert _hook_done_event is not None
                try:
                    await asyncio.wait_for(_hook_done_event.wait(), timeout=120.0)
                except asyncio.TimeoutError:
                    logger.warning("traqo: timeout waiting for hook, closing stdin")
                await chosen_transport.end_input()

            if query_obj._tg:
                query_obj._tg.start_soon(_deferred_end_input)

            async for data in query_obj.receive_messages():
                message = parse_message(data)
                if message is not None:
                    yield message
        finally:
            await query_obj.close()

    InternalClient.process_query = _patched_process_query  # type: ignore[assignment]
    _patch_installed = True
    return _hook_done_event


def _uninstall_hooks_fix() -> None:
    """Reset the done event (keep the patch installed — it's idempotent)."""
    global _hook_done_event
    _hook_done_event = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _read_trace_end_stats(path: Path) -> dict[str, Any]:
    """Read the trace_end event from a traqo JSONL file and return its stats."""
    last_line = ""
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                last_line = stripped
    if not last_line:
        return {}
    try:
        event = json.loads(last_line)
        if event.get("type") == "trace_end":
            return event.get("stats", {})
    except (json.JSONDecodeError, KeyError):
        pass
    return {}


@asynccontextmanager
async def traqo_agent(
    name: str,
    *,
    output_dir: str | Path | None = None,
    thread_id: str | None = None,
    tags: list[str] | None = None,
):
    """Async context manager that returns Claude Agent SDK hooks for tracing.

    Args:
        name: Name for this agent trace.
        output_dir: Directory for trace output. Uses TRAQO_TRACE_DIR or ./traces
                    if not specified.
        thread_id: Optional thread ID for the trace.
        tags: Optional tags for the trace.

    Yields:
        A hooks dict suitable for ``ClaudeAgentOptions(hooks=hooks)``.
        Contains a ``Stop`` hook that converts the session transcript to a
        traqo trace file via ``cc_sync.sync_session()``.
    """
    import os

    start_time = datetime.now(timezone.utc)
    parent = get_tracer()
    resolved_dir = Path(
        output_dir
        if output_dir is not None
        else os.environ.get("TRAQO_TRACE_DIR", "./traces")
    )

    # State shared between the hook callback and __aexit__
    output_path: Path | None = None

    # Install the SDK fix and get the event to signal when our hook is done.
    done_event = _install_hooks_fix()

    if parent is not None:
        parent.write_event(
            {
                "type": "event",
                "id": uuid.uuid4().hex[:12],
                "name": "child_started",
                "ts": start_time.isoformat(),
                "data": {"child_name": name, "source": "claude_agent_sdk"},
            }
        )
        with parent._lock:
            parent._stats_events += 1

    async def _stop_hook(
        input_data: _StopHookInput,
        tool_use_id: str | None,
        context: _HookContext,
    ) -> _SyncHookJSONOutput:
        nonlocal output_path
        from traqo.cc_sync import sync_session

        try:
            transcript_path_str = input_data.get("transcript_path", "")
            session_id = input_data.get("session_id", "")
            if not transcript_path_str:
                logger.warning("traqo_agent: no transcript_path in stop hook input")
                return {}

            transcript_path = Path(transcript_path_str).expanduser()
            # Brief delay to ensure the transcript is fully flushed to disk.
            # The CLI fires the Stop hook before the transcript write is synced.
            await asyncio.sleep(0.5)
            # Run sync I/O in a thread to avoid blocking the event loop.
            result = await asyncio.to_thread(
                sync_session,
                transcript_path,
                resolved_dir,
                session_id=session_id or None,
                name=name,
                thread_id=thread_id,
                tags=tags,
                force=True,
            )
            if result:
                output_path = result
                logger.info("traqo_agent: synced %s → %s", name, result.name)
        finally:
            # Signal that the hook is done so stdin can be closed.
            done_event.set()
        return {}

    hooks: dict[str, list[Any]] = {
        "Stop": [_HookMatcher(hooks=[_stop_hook])],
    }

    try:
        yield hooks
    finally:
        # Ensure the event is set even if the hook never fired (e.g. agent error).
        done_event.set()
        _uninstall_hooks_fix()

        # Roll up stats into parent if nested
        if parent is not None and output_path is not None:
            stats = _read_trace_end_stats(output_path)
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            summary = {
                "name": name,
                "file": output_path.stem + ".jsonl.gz",
                "duration_s": round(duration, 3),
                "spans": stats.get("spans", 0),
                "total_input_tokens": stats.get("total_input_tokens", 0),
                "total_output_tokens": stats.get("total_output_tokens", 0),
                "total_reasoning_tokens": stats.get("total_reasoning_tokens", 0),
            }
            parent._children.append(summary)
            with parent._lock:
                parent._stats_spans += stats.get("spans", 0)
                parent._stats_input_tokens += stats.get("total_input_tokens", 0)
                parent._stats_output_tokens += stats.get("total_output_tokens", 0)
                parent._stats_cache_read_tokens += stats.get(
                    "total_cache_read_tokens", 0
                )
                parent._stats_cache_creation_tokens += stats.get(
                    "total_cache_creation_tokens", 0
                )
                parent._stats_reasoning_tokens += stats.get("total_reasoning_tokens", 0)
                parent._stats_errors += stats.get("errors", 0)
                parent._stats_events += 1
            parent.write_event(
                {
                    "type": "event",
                    "id": uuid.uuid4().hex[:12],
                    "name": "child_ended",
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "data": {
                        "child_name": name,
                        "child_file": output_path.stem + ".jsonl.gz",
                        "duration_s": summary["duration_s"],
                        "spans": summary["spans"],
                        "total_input_tokens": summary["total_input_tokens"],
                        "total_output_tokens": summary["total_output_tokens"],
                        "total_reasoning_tokens": summary["total_reasoning_tokens"],
                    },
                }
            )
