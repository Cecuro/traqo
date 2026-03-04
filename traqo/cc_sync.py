"""Convert Claude Code session transcripts to traqo trace format.

Claude Code writes session transcripts as JSONL files at:
    ~/.claude/projects/[project-path-encoded]/[session-id].jsonl

This module parses those transcripts and converts them into traqo's native
JSONL trace format (trace_start, span_start, span_end, trace_end), producing
one trace per session.

Usage as a Claude Code Stop hook (in ~/.claude/settings.json):
    {
      "hooks": {
        "Stop": [{
          "hooks": [{
            "type": "command",
            "command": "traqo cc-sync --hook"
          }]
        }]
      }
    }

Manual usage:
    traqo cc-sync path/to/session.jsonl
    traqo cc-sync --all
    traqo cc-sync --all --output-dir ./traces
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from traqo._version import __version__

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class AssistantMessage:
    """A deduplicated assistant message (merged from streaming chunks)."""

    message_id: str
    model: str
    content_blocks: list[dict[str, Any]]
    usage: dict[str, int]
    request_id: str
    timestamp: str


@dataclass
class Turn:
    """A parsed conversation turn (user prompt → responses → tool calls)."""

    index: int
    prompt: str
    timestamp: str
    assistant_messages: list[AssistantMessage]
    tool_results: dict[str, Any]  # tool_use_id → result content
    subagent_progress: dict[str, list[dict[str, Any]]]  # agentId → records
    duration_ms: int | None = None


@dataclass
class ParsedSession:
    """A fully parsed Claude Code session."""

    session_id: str
    slug: str
    version: str
    git_branch: str
    cwd: str
    turns: list[Turn]
    first_timestamp: str
    last_timestamp: str
    model: str = ""
    api_errors: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Transcript parsing
# ---------------------------------------------------------------------------


def _is_user_prompt(record: dict[str, Any]) -> bool:
    """Check if a record is a real user prompt (not meta/tool_result/command)."""
    if record.get("type") != "user":
        return False
    if record.get("isMeta"):
        return False
    msg = record.get("message", {})
    content = msg.get("content")
    if not isinstance(content, str):
        return False
    # Filter out local commands and system messages
    if content.startswith("<command-name>") or content.startswith("<local-command"):
        return False
    return not record.get("isCompactSummary")


def _is_tool_result(record: dict[str, Any]) -> bool:
    """Check if a record is a tool result user message."""
    if record.get("type") != "user":
        return False
    msg = record.get("message", {})
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    return any(block.get("type") == "tool_result" for block in content)


def _merge_assistant_chunks(
    chunks: list[dict[str, Any]],
) -> AssistantMessage:
    """Merge streaming chunks with the same message.id into one AssistantMessage."""
    # Merge content blocks from all chunks (each chunk may have different blocks)
    all_content: list[dict[str, Any]] = []
    for chunk in chunks:
        msg = chunk.get("message", {})
        for block in msg.get("content", []):
            all_content.append(block)

    # Use last chunk for usage (most accurate) and metadata
    last = chunks[-1]
    last_msg = last.get("message", {})
    usage_raw = last_msg.get("usage", {})
    usage = {
        "input_tokens": usage_raw.get("input_tokens", 0),
        "output_tokens": usage_raw.get("output_tokens", 0),
        "cache_read_input_tokens": usage_raw.get("cache_read_input_tokens", 0),
        "cache_creation_input_tokens": usage_raw.get("cache_creation_input_tokens", 0),
    }

    return AssistantMessage(
        message_id=last_msg.get("id", ""),
        model=last_msg.get("model", ""),
        content_blocks=all_content,
        usage=usage,
        request_id=last.get("requestId", ""),
        timestamp=last.get("timestamp", ""),
    )


def parse_transcript(path: Path) -> ParsedSession:
    """Parse a Claude Code JSONL transcript into structured data."""
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.debug("skipping malformed line in %s", path)
                continue

    # Extract session metadata from first usable record
    session_id = ""
    slug = ""
    version = ""
    git_branch = ""
    cwd = ""
    model = ""
    first_ts = ""
    last_ts = ""
    api_errors: list[dict[str, Any]] = []

    for r in records:
        if r.get("type") in ("user", "assistant"):
            if not session_id:
                session_id = r.get("sessionId", "")
            if not slug:
                slug = r.get("slug", "")
            if not version:
                version = r.get("version", "")
            if not git_branch:
                git_branch = r.get("gitBranch", "")
            if not cwd:
                cwd = r.get("cwd", "")
        if r.get("type") == "assistant" and not model:
            model = r.get("message", {}).get("model", "")
        ts = r.get("timestamp")
        if ts:
            if not first_ts:
                first_ts = ts
            last_ts = ts
        # Collect API errors
        if r.get("type") == "system" and r.get("subtype") == "api_error":
            api_errors.append(r)

    # Group assistant chunks by message.id
    assistant_chunks: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        if r.get("type") != "assistant":
            continue
        msg_id = r.get("message", {}).get("id", "")
        if not msg_id:
            continue
        assistant_chunks.setdefault(msg_id, []).append(r)

    # Deduplicate: merge each group into a single AssistantMessage
    deduped: dict[str, AssistantMessage] = {}
    for msg_id, chunks in assistant_chunks.items():
        deduped[msg_id] = _merge_assistant_chunks(chunks)

    # Track which message.ids we've already seen (for ordering in turns)
    seen_msg_ids: set[str] = set()

    # Build turns
    turns: list[Turn] = []
    current_turn: Turn | None = None
    turn_idx = 0

    for r in records:
        rtype = r.get("type")

        # New turn boundary: real user prompt
        if _is_user_prompt(r):
            turn_idx += 1
            prompt_text = r.get("message", {}).get("content", "")
            current_turn = Turn(
                index=turn_idx,
                prompt=prompt_text,
                timestamp=r.get("timestamp", ""),
                assistant_messages=[],
                tool_results={},
                subagent_progress={},
            )
            turns.append(current_turn)
            seen_msg_ids.clear()
            continue

        if current_turn is None:
            continue

        # Tool results
        if _is_tool_result(r):
            content = r.get("message", {}).get("content", [])
            for block in content:
                if block.get("type") == "tool_result":
                    tid = block.get("tool_use_id", "")
                    if tid:
                        current_turn.tool_results[tid] = block.get("content")
            continue

        # Assistant message (add deduped version, in first-seen order)
        if rtype == "assistant":
            msg_id = r.get("message", {}).get("id", "")
            if msg_id and msg_id not in seen_msg_ids and msg_id in deduped:
                seen_msg_ids.add(msg_id)
                current_turn.assistant_messages.append(deduped[msg_id])
            continue

        # Turn duration
        if rtype == "system" and r.get("subtype") == "turn_duration":
            current_turn.duration_ms = r.get("durationMs")
            continue

        # Subagent progress
        if rtype == "progress":
            data = r.get("data", {})
            agent_id = data.get("agentId", "")
            if agent_id:
                current_turn.subagent_progress.setdefault(agent_id, []).append(r)
            continue

    return ParsedSession(
        session_id=session_id,
        slug=slug,
        version=version,
        git_branch=git_branch,
        cwd=cwd,
        turns=turns,
        first_timestamp=first_ts,
        last_timestamp=last_ts,
        model=model,
        api_errors=api_errors,
    )


# ---------------------------------------------------------------------------
# Trace event generation
# ---------------------------------------------------------------------------


def _short_id() -> str:
    return uuid.uuid4().hex[:12]


def _extract_text(content_blocks: list[dict[str, Any]]) -> str:
    """Extract concatenated text from content blocks."""
    parts = []
    for block in content_blocks:
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(parts)


def _extract_tool_uses(content_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract tool_use blocks from content."""
    return [b for b in content_blocks if b.get("type") == "tool_use"]


def generate_trace_events(
    session: ParsedSession,
    *,
    name: str | None = None,
    thread_id: str | None = None,
    tags: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Convert a ParsedSession into a list of traqo trace events.

    Args:
        session: Parsed Claude Code session.
        name: Override the trace name (default: derived from session slug).
        thread_id: Override the thread_id (default: session_id).
        tags: Override the tags list (default: ["claude-code"]).
    """
    events: list[dict[str, Any]] = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_cache_read = 0
    total_cache_create = 0
    span_count = 0
    error_count = 0

    # Derive trace name
    if name is not None:
        trace_name = name
    elif session.slug:
        trace_name = f"Claude Code: {session.slug}"
    else:
        trace_name = f"Claude Code: {session.session_id[:8]}"

    # trace_start
    metadata: dict[str, Any] = {
        "session_id": session.session_id,
        "version": session.version,
        "provider": "anthropic",
    }
    if session.git_branch:
        metadata["git_branch"] = session.git_branch
    if session.cwd:
        metadata["project"] = Path(session.cwd).name
        metadata["working_directory"] = session.cwd
    if session.model:
        metadata["model"] = session.model

    first_prompt = session.turns[0].prompt if session.turns else None

    events.append(
        {
            "type": "trace_start",
            "ts": session.first_timestamp,
            "tracer_version": __version__,
            "name": trace_name,
            "thread_id": thread_id if thread_id is not None else session.session_id,
            "tags": tags if tags is not None else ["claude-code"],
            "metadata": metadata,
            **({"input": first_prompt} if first_prompt else {}),
        }
    )

    for turn in session.turns:
        turn_id = _short_id()
        preview = turn.prompt[:80].replace("\n", " ")
        turn_name = f"Turn {turn.index}: {preview}"

        # Turn span (kind=chain)
        events.append(
            {
                "type": "span_start",
                "id": turn_id,
                "parent_id": None,
                "name": turn_name,
                "ts": turn.timestamp,
                "kind": "chain",
                "input": turn.prompt,
            }
        )
        span_count += 1

        # Process assistant messages and tool calls in order
        for am in turn.assistant_messages:
            text_output = _extract_text(am.content_blocks)
            tool_uses = _extract_tool_uses(am.content_blocks)
            thinking_parts = [
                b.get("thinking", "")
                for b in am.content_blocks
                if b.get("type") == "thinking" and b.get("thinking")
            ]
            thinking_text = "\n".join(thinking_parts)
            has_thinking = bool(thinking_text)

            cache_read = am.usage.get("cache_read_input_tokens", 0)
            cache_create = am.usage.get("cache_creation_input_tokens", 0)
            token_usage = {
                "input_tokens": am.usage.get("input_tokens", 0)
                + cache_read
                + cache_create,
                "output_tokens": am.usage.get("output_tokens", 0),
            }
            total_input_tokens += token_usage["input_tokens"]
            total_output_tokens += token_usage["output_tokens"]
            total_cache_read += cache_read
            total_cache_create += cache_create

            if cache_read:
                token_usage["cache_read_tokens"] = cache_read
            if cache_create:
                token_usage["cache_creation_tokens"] = cache_create

            llm_metadata: dict[str, Any] = {
                "model": am.model,
                "token_usage": token_usage,
                "provider": "anthropic",
            }
            if am.request_id:
                llm_metadata["request_id"] = am.request_id
            if has_thinking:
                llm_metadata["has_thinking"] = True

            # Only emit an LLM span if there's text output or thinking.
            # Tool-only responses (no text) are represented by the tool
            # spans alone — avoids cluttering the trace with empty LLM spans.
            if text_output or has_thinking:
                llm_id = _short_id()
                llm_duration = None
                if am.timestamp and turn.timestamp:
                    try:
                        am_dt = datetime.fromisoformat(
                            am.timestamp.replace("Z", "+00:00")
                        )
                        turn_dt = datetime.fromisoformat(
                            turn.timestamp.replace("Z", "+00:00")
                        )
                        llm_duration = round((am_dt - turn_dt).total_seconds(), 3)
                        if llm_duration < 0:
                            llm_duration = None
                    except (ValueError, TypeError):
                        pass

                events.append(
                    {
                        "type": "span_start",
                        "id": llm_id,
                        "parent_id": turn_id,
                        "name": am.model or "claude",
                        "ts": am.timestamp or turn.timestamp,
                        "kind": "llm",
                        "metadata": llm_metadata,
                    }
                )
                llm_end: dict[str, Any] = {
                    "type": "span_end",
                    "id": llm_id,
                    "parent_id": turn_id,
                    "name": am.model or "claude",
                    "ts": am.timestamp or turn.timestamp,
                    "kind": "llm",
                    "status": "ok",
                    "metadata": llm_metadata,
                }
                if text_output and thinking_text:
                    llm_end["output"] = {
                        "text": text_output,
                        "reasoning": thinking_text,
                    }
                elif text_output:
                    llm_end["output"] = text_output
                elif thinking_text:
                    llm_end["output"] = {"reasoning": thinking_text}
                if llm_duration is not None:
                    llm_end["duration_s"] = llm_duration
                events.append(llm_end)
                span_count += 1

            # Tool spans for each tool_use in this message
            for tu in tool_uses:
                tool_id = _short_id()
                tool_name = tu.get("name", "unknown_tool")
                tool_input = tu.get("input", {})
                tool_use_id = tu.get("id", "")
                tool_output = turn.tool_results.get(tool_use_id)

                tool_meta: dict[str, Any] = {}
                # Attach LLM metadata to tool spans when there's no
                # separate LLM span (tool-only responses)
                if not text_output and not has_thinking:
                    tool_meta["token_usage"] = token_usage
                    tool_meta["model"] = am.model

                tool_start: dict[str, Any] = {
                    "type": "span_start",
                    "id": tool_id,
                    "parent_id": turn_id,
                    "name": tool_name,
                    "ts": am.timestamp or turn.timestamp,
                    "kind": "tool",
                    "input": tool_input,
                }
                if tool_meta:
                    tool_start["metadata"] = tool_meta
                events.append(tool_start)

                tool_end: dict[str, Any] = {
                    "type": "span_end",
                    "id": tool_id,
                    "parent_id": turn_id,
                    "name": tool_name,
                    "ts": am.timestamp or turn.timestamp,
                    "kind": "tool",
                    "status": "ok",
                }
                if tool_meta:
                    tool_end["metadata"] = tool_meta
                if tool_output is not None:
                    tool_end["output"] = tool_output
                events.append(tool_end)
                span_count += 1

        # Subagent spans
        for agent_id, progress_records in turn.subagent_progress.items():
            agent_span_id = _short_id()
            short_agent = agent_id[-8:] if len(agent_id) > 8 else agent_id

            # Extract prompt from first progress record
            first_data = progress_records[0].get("data", {}) if progress_records else {}
            agent_prompt = first_data.get("prompt", "")

            first_ts = (
                progress_records[0].get("timestamp", turn.timestamp)
                if progress_records
                else turn.timestamp
            )
            last_ts = (
                progress_records[-1].get("timestamp", turn.timestamp)
                if progress_records
                else turn.timestamp
            )

            events.append(
                {
                    "type": "span_start",
                    "id": agent_span_id,
                    "parent_id": turn_id,
                    "name": f"Subagent: {short_agent}",
                    "ts": first_ts,
                    "kind": "agent",
                    **({"input": agent_prompt} if agent_prompt else {}),
                }
            )

            # Collect tool results from user-type progress records
            agent_tool_results: dict[str, Any] = {}
            for pr in progress_records:
                data = pr.get("data", {})
                nested_msg = data.get("message", {})
                if nested_msg.get("type") == "user":
                    inner_content = nested_msg.get("message", {}).get("content", "")
                    if isinstance(inner_content, list):
                        for block in inner_content:
                            if block.get("type") == "tool_result":
                                tid = block.get("tool_use_id", "")
                                if tid:
                                    agent_tool_results[tid] = block.get("content")

            # Process assistant progress records (LLM + tool spans)
            for pr in progress_records:
                data = pr.get("data", {})
                nested_msg = data.get("message", {})
                if nested_msg.get("type") != "assistant":
                    continue

                inner = nested_msg.get("message", {})
                nested_model = inner.get("model", "")
                nested_usage = inner.get("usage", {})
                nested_content = inner.get("content", [])

                nested_cache_read = nested_usage.get("cache_read_input_tokens", 0)
                nested_cache_create = nested_usage.get("cache_creation_input_tokens", 0)
                nested_token_usage = {
                    "input_tokens": nested_usage.get("input_tokens", 0)
                    + nested_cache_read
                    + nested_cache_create,
                    "output_tokens": nested_usage.get("output_tokens", 0),
                }
                total_input_tokens += nested_token_usage["input_tokens"]
                total_output_tokens += nested_token_usage["output_tokens"]
                total_cache_read += nested_cache_read
                total_cache_create += nested_cache_create

                nested_text = _extract_text(nested_content)
                nested_tool_uses = _extract_tool_uses(nested_content)
                nested_ts = nested_msg.get("timestamp", first_ts)

                # Only emit LLM span if there's text output (skip
                # tool-only responses to avoid noise)
                if nested_text:
                    nested_id = _short_id()
                    nested_meta: dict[str, Any] = {
                        "model": nested_model,
                        "token_usage": nested_token_usage,
                        "provider": "anthropic",
                    }
                    events.append(
                        {
                            "type": "span_start",
                            "id": nested_id,
                            "parent_id": agent_span_id,
                            "name": nested_model or "claude",
                            "ts": nested_ts,
                            "kind": "llm",
                            "metadata": nested_meta,
                        }
                    )
                    events.append(
                        {
                            "type": "span_end",
                            "id": nested_id,
                            "parent_id": agent_span_id,
                            "name": nested_model or "claude",
                            "ts": nested_ts,
                            "kind": "llm",
                            "status": "ok",
                            "metadata": nested_meta,
                            "output": nested_text,
                        }
                    )
                    span_count += 1

                # Tool spans for subagent tool_use blocks
                for tu in nested_tool_uses:
                    tool_id = _short_id()
                    tool_name = tu.get("name", "unknown_tool")
                    tool_input = tu.get("input", {})
                    tool_use_id = tu.get("id", "")
                    tool_output = agent_tool_results.get(tool_use_id)

                    tool_meta: dict[str, Any] = {}
                    if not nested_text:
                        tool_meta["token_usage"] = nested_token_usage
                        tool_meta["model"] = nested_model

                    tool_start: dict[str, Any] = {
                        "type": "span_start",
                        "id": tool_id,
                        "parent_id": agent_span_id,
                        "name": tool_name,
                        "ts": nested_ts,
                        "kind": "tool",
                        "input": tool_input,
                    }
                    if tool_meta:
                        tool_start["metadata"] = tool_meta
                    events.append(tool_start)

                    tool_end_evt: dict[str, Any] = {
                        "type": "span_end",
                        "id": tool_id,
                        "parent_id": agent_span_id,
                        "name": tool_name,
                        "ts": nested_ts,
                        "kind": "tool",
                        "status": "ok",
                    }
                    if tool_meta:
                        tool_end_evt["metadata"] = tool_meta
                    if tool_output is not None:
                        tool_end_evt["output"] = tool_output
                    events.append(tool_end_evt)
                    span_count += 1

            # Agent duration
            agent_duration = None
            try:
                start_dt = datetime.fromisoformat(first_ts.replace("Z", "+00:00"))
                end_dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
                agent_duration = round((end_dt - start_dt).total_seconds(), 3)
            except (ValueError, TypeError):
                pass

            agent_end: dict[str, Any] = {
                "type": "span_end",
                "id": agent_span_id,
                "parent_id": turn_id,
                "name": f"Subagent: {short_agent}",
                "ts": last_ts,
                "kind": "agent",
                "status": "ok",
            }
            if agent_duration is not None:
                agent_end["duration_s"] = agent_duration
            events.append(agent_end)
            span_count += 1

        # Turn span end
        turn_end: dict[str, Any] = {
            "type": "span_end",
            "id": turn_id,
            "parent_id": None,
            "name": turn_name,
            "ts": turn.timestamp,
            "kind": "chain",
            "status": "ok",
        }
        if turn.duration_ms is not None:
            turn_end["duration_s"] = round(turn.duration_ms / 1000.0, 3)
        events.append(turn_end)

    # Compute total duration
    total_duration = 0.0
    if session.first_timestamp and session.last_timestamp:
        try:
            start = datetime.fromisoformat(
                session.first_timestamp.replace("Z", "+00:00")
            )
            end = datetime.fromisoformat(session.last_timestamp.replace("Z", "+00:00"))
            total_duration = round((end - start).total_seconds(), 3)
        except (ValueError, TypeError):
            pass

    error_count = len(session.api_errors)

    # trace_end
    events.append(
        {
            "type": "trace_end",
            "ts": session.last_timestamp or datetime.now(timezone.utc).isoformat(),
            "duration_s": total_duration,
            "status": "error" if error_count > 0 else "ok",
            "stats": {
                "spans": span_count,
                "events": 0,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_cache_read_tokens": total_cache_read,
                "total_cache_creation_tokens": total_cache_create,
                "total_reasoning_tokens": 0,
                "errors": error_count,
            },
        }
    )

    return events


# ---------------------------------------------------------------------------
# State tracking
# ---------------------------------------------------------------------------


def _state_dir(output_dir: Path) -> Path:
    return output_dir / ".cc-sync-state"


def _load_state(output_dir: Path, session_id: str) -> dict[str, Any]:
    state_file = _state_dir(output_dir) / f"{session_id}.json"
    if state_file.exists():
        try:
            return json.loads(state_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_state(output_dir: Path, session_id: str, state: dict[str, Any]) -> None:
    sd = _state_dir(output_dir)
    sd.mkdir(parents=True, exist_ok=True)
    state_file = sd / f"{session_id}.json"
    state_file.write_text(json.dumps(state), encoding="utf-8")


# ---------------------------------------------------------------------------
# Sync logic
# ---------------------------------------------------------------------------


def _to_json_line(event: dict[str, Any]) -> str:
    """Serialize an event dict to a JSON string."""
    return json.dumps(event, default=str, ensure_ascii=False)


def sync_session(
    transcript_path: Path,
    output_dir: Path,
    *,
    session_id: str | None = None,
    name: str | None = None,
    thread_id: str | None = None,
    tags: list[str] | None = None,
    force: bool = False,
) -> Path | None:
    """Sync a single Claude Code session transcript to traqo format.

    Args:
        transcript_path: Path to the Claude Code JSONL transcript.
        output_dir: Directory to write the traqo trace file.
        session_id: Override session ID (default: transcript filename stem).
        name: Override the trace name.
        thread_id: Override the thread_id in trace_start.
        tags: Override the tags list in trace_start.
        force: Skip the state check and always rewrite.

    Returns the output path if written, or None if no new data.
    """
    if not transcript_path.exists():
        logger.warning("transcript not found: %s", transcript_path)
        return None

    file_size = transcript_path.stat().st_size

    # Derive session_id from filename if not provided
    if session_id is None:
        session_id = transcript_path.stem

    # Check state
    if not force:
        state = _load_state(output_dir, session_id)
        if state.get("file_size") == file_size:
            logger.debug("no new data for session %s, skipping", session_id)
            return None

    # Parse and convert
    session = parse_transcript(transcript_path)
    if not session.turns:
        logger.debug("no turns found in %s, skipping", transcript_path)
        return None

    events = generate_trace_events(session, name=name, thread_id=thread_id, tags=tags)

    # Write output
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"cc-{session_id}.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for event in events:
            f.write(_to_json_line(event) + "\n")

    # Save state
    _save_state(output_dir, session_id, {"file_size": file_size})

    logger.info(
        "synced %s → %s (%d turns, %d spans)",
        transcript_path.name,
        output_path.name,
        len(session.turns),
        len(events) - 2,
    )
    return output_path


def _find_claude_projects_dir() -> Path | None:
    """Find the ~/.claude/projects/ directory."""
    claude_dir = Path.home() / ".claude" / "projects"
    return claude_dir if claude_dir.is_dir() else None


def sync_all(
    output_dir: Path | None = None,
    project_filter: str | None = None,
) -> list[Path]:
    """Sync all Claude Code sessions found in ~/.claude/projects/.

    Args:
        output_dir: Where to write traqo trace files. Defaults to TRAQO_TRACE_DIR or ./traces.
        project_filter: If set, only sync sessions for project dirs containing this string.

    Returns:
        List of output paths that were written.
    """
    if output_dir is None:
        output_dir = Path(os.environ.get("TRAQO_TRACE_DIR", "./traces"))

    projects_dir = _find_claude_projects_dir()
    if projects_dir is None:
        logger.warning("~/.claude/projects/ not found")
        return []

    results: list[Path] = []
    for project_dir in sorted(projects_dir.iterdir()):
        if not project_dir.is_dir():
            continue
        if project_filter and project_filter not in project_dir.name:
            continue

        # Find session JSONL files (direct children, not subagents)
        for jsonl_file in sorted(project_dir.glob("*.jsonl")):
            result = sync_session(jsonl_file, output_dir)
            if result:
                results.append(result)

    if results:
        logger.info("synced %d sessions", len(results))
    else:
        logger.info("no new sessions to sync")
    return results


def run_hook() -> None:
    """Entry point for Claude Code Stop hook. Reads JSON from stdin."""
    try:
        hook_input = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, OSError) as e:
        logger.error("failed to read hook input: %s", e)
        sys.exit(1)

    transcript_path = hook_input.get("transcript_path", "")
    session_id = hook_input.get("session_id", "")

    if not transcript_path:
        logger.error("no transcript_path in hook input")
        sys.exit(1)

    # Expand ~ in path
    transcript = Path(transcript_path).expanduser()
    output_dir = Path(os.environ.get("TRAQO_TRACE_DIR", "./traces"))

    result = sync_session(transcript, output_dir, session_id=session_id or None)
    if result:
        logger.info("hook: synced %s", result.name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for `traqo cc-sync`."""
    parser = argparse.ArgumentParser(
        prog="traqo cc-sync",
        description="Sync Claude Code session transcripts to traqo trace format",
    )
    parser.add_argument(
        "file",
        nargs="?",
        help="Path to a Claude Code session transcript (.jsonl)",
    )
    parser.add_argument(
        "--hook",
        action="store_true",
        help="Run as a Claude Code Stop hook (reads JSON from stdin)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="sync_all",
        help="Sync all sessions from ~/.claude/projects/",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Output directory for trace files (default: TRAQO_TRACE_DIR or ./traces)",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Filter to project dirs containing this string (with --all)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args(argv)

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s", stream=sys.stderr)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(os.environ.get("TRAQO_TRACE_DIR", "./traces"))
    )

    if args.hook:
        run_hook()
    elif args.sync_all:
        results = sync_all(output_dir=output_dir, project_filter=args.project)
        if results:
            print(f"Synced {len(results)} session(s) to {output_dir}", file=sys.stderr)
    elif args.file:
        path = Path(args.file)
        result = sync_session(path, output_dir)
        if result:
            print(f"Synced to {result}", file=sys.stderr)
        else:
            print("No data to sync", file=sys.stderr)
    else:
        parser.print_help()
        sys.exit(1)
