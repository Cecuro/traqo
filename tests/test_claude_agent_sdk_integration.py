"""Tests for traqo.integrations.claude_agent_sdk — Claude Agent SDK tracing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

claude_agent_sdk = pytest.importorskip(
    "claude_agent_sdk", reason="claude-agent-sdk not installed"
)

from tests.conftest import read_events  # noqa: E402
from traqo import Tracer  # noqa: E402
from traqo.cc_sync import (  # noqa: E402
    AssistantMessage,
    ParsedSession,
    Turn,
    generate_trace_events,
)
from traqo.integrations.claude_agent_sdk import (  # noqa: E402
    _read_trace_end_stats,
    traqo_agent,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_fake_transcript(path: Path) -> None:
    """Write a minimal Claude Code transcript JSONL for testing."""
    records = [
        {
            "type": "user",
            "sessionId": "agent-sess-1",
            "version": "2.1.63",
            "gitBranch": "main",
            "slug": "test-agent",
            "cwd": "/tmp/project",
            "message": {"role": "user", "content": "Fix the bug"},
            "uuid": "user-1",
            "timestamp": "2026-03-01T10:00:00.000Z",
        },
        {
            "type": "assistant",
            "sessionId": "agent-sess-1",
            "version": "2.1.63",
            "gitBranch": "main",
            "slug": "test-agent",
            "cwd": "/tmp/project",
            "requestId": "req_1",
            "message": {
                "model": "claude-sonnet-4-6",
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "I fixed the bug."}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cache_read_input_tokens": 200,
                    "cache_creation_input_tokens": 0,
                    "service_tier": "standard",
                },
            },
            "uuid": "asst-1",
            "timestamp": "2026-03-01T10:00:05.000Z",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _write_traqo_trace(path: Path, stats: dict[str, Any]) -> None:
    """Write a minimal traqo trace JSONL with given trace_end stats."""
    events = [
        {"type": "trace_start", "ts": "2026-03-01T10:00:00.000Z", "name": "test"},
        {"type": "trace_end", "ts": "2026-03-01T10:00:05.000Z", "stats": stats},
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


async def _call_stop_hook(
    hooks: dict[str, Any], input_data: dict[str, Any]
) -> dict[str, Any]:
    """Simulate the Agent SDK calling the Stop hook callback."""
    matcher = hooks["Stop"][0]
    callback = matcher.hooks[0]
    return await callback(input_data, None, None)


# ---------------------------------------------------------------------------
# _read_trace_end_stats tests
# ---------------------------------------------------------------------------


class TestReadTraceEndStats:
    def test_reads_stats_from_last_line(self, tmp_path: Path) -> None:
        stats = {"spans": 5, "total_input_tokens": 100, "total_output_tokens": 50}
        trace_path = tmp_path / "trace.jsonl"
        _write_traqo_trace(trace_path, stats)

        result = _read_trace_end_stats(trace_path)
        assert result["spans"] == 5
        assert result["total_input_tokens"] == 100
        assert result["total_output_tokens"] == 50

    def test_returns_empty_for_non_trace_end(self, tmp_path: Path) -> None:
        path = tmp_path / "trace.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps({"type": "event", "name": "something"}) + "\n")

        assert _read_trace_end_stats(path) == {}

    def test_returns_empty_for_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        assert _read_trace_end_stats(path) == {}


# ---------------------------------------------------------------------------
# cc_sync override tests
# ---------------------------------------------------------------------------


class TestCcSyncOverrides:
    def test_name_override(self) -> None:
        session = ParsedSession(
            session_id="s1",
            slug="original-slug",
            version="2.1.63",
            git_branch="main",
            cwd="/tmp",
            model="claude-sonnet-4-6",
            turns=[
                Turn(
                    index=1,
                    prompt="Q",
                    timestamp="2026-03-01T10:00:00.000Z",
                    assistant_messages=[
                        AssistantMessage(
                            message_id="m1",
                            model="claude-sonnet-4-6",
                            content_blocks=[{"type": "text", "text": "A"}],
                            usage={"input_tokens": 10, "output_tokens": 5},
                            request_id="r1",
                            timestamp="2026-03-01T10:00:01.000Z",
                        )
                    ],
                    tool_results={},
                    subagent_progress={},
                )
            ],
            first_timestamp="2026-03-01T10:00:00.000Z",
            last_timestamp="2026-03-01T10:00:01.000Z",
        )
        events = generate_trace_events(session, name="my-agent")
        assert events[0]["name"] == "my-agent"

    def test_thread_id_override(self) -> None:
        session = ParsedSession(
            session_id="s1",
            slug="slug",
            version="2.1.63",
            git_branch="main",
            cwd="/tmp",
            model="claude-sonnet-4-6",
            turns=[
                Turn(
                    index=1,
                    prompt="Q",
                    timestamp="2026-03-01T10:00:00.000Z",
                    assistant_messages=[
                        AssistantMessage(
                            message_id="m1",
                            model="claude-sonnet-4-6",
                            content_blocks=[{"type": "text", "text": "A"}],
                            usage={"input_tokens": 10, "output_tokens": 5},
                            request_id="r1",
                            timestamp="2026-03-01T10:00:01.000Z",
                        )
                    ],
                    tool_results={},
                    subagent_progress={},
                )
            ],
            first_timestamp="2026-03-01T10:00:00.000Z",
            last_timestamp="2026-03-01T10:00:01.000Z",
        )
        events = generate_trace_events(session, thread_id="custom-thread")
        assert events[0]["thread_id"] == "custom-thread"

    def test_tags_override(self) -> None:
        session = ParsedSession(
            session_id="s1",
            slug="slug",
            version="2.1.63",
            git_branch="main",
            cwd="/tmp",
            model="claude-sonnet-4-6",
            turns=[
                Turn(
                    index=1,
                    prompt="Q",
                    timestamp="2026-03-01T10:00:00.000Z",
                    assistant_messages=[
                        AssistantMessage(
                            message_id="m1",
                            model="claude-sonnet-4-6",
                            content_blocks=[{"type": "text", "text": "A"}],
                            usage={"input_tokens": 10, "output_tokens": 5},
                            request_id="r1",
                            timestamp="2026-03-01T10:00:01.000Z",
                        )
                    ],
                    tool_results={},
                    subagent_progress={},
                )
            ],
            first_timestamp="2026-03-01T10:00:00.000Z",
            last_timestamp="2026-03-01T10:00:01.000Z",
        )
        events = generate_trace_events(session, tags=["review", "agent-sdk"])
        assert events[0]["tags"] == ["review", "agent-sdk"]


# ---------------------------------------------------------------------------
# traqo_agent context manager tests
# ---------------------------------------------------------------------------


class TestTraqoAgent:
    @pytest.mark.asyncio
    async def test_yields_hooks_dict_matching_sdk_format(self, tmp_path: Path) -> None:
        """Hooks dict matches ClaudeAgentOptions format: Stop → [HookMatcher]."""
        from claude_agent_sdk import HookMatcher

        async with traqo_agent("test-agent", output_dir=tmp_path) as hooks:
            assert "Stop" in hooks
            assert len(hooks["Stop"]) == 1
            matcher = hooks["Stop"][0]
            assert isinstance(matcher, HookMatcher)
            assert len(matcher.hooks) == 1
            assert callable(matcher.hooks[0])

    @pytest.mark.asyncio
    async def test_stop_callback_is_async(self, tmp_path: Path) -> None:
        """The stop callback is async (Agent SDK requires async callbacks)."""
        import asyncio

        async with traqo_agent("test-agent", output_dir=tmp_path) as hooks:
            callback = hooks["Stop"][0].hooks[0]
            assert asyncio.iscoroutinefunction(callback)

    @pytest.mark.asyncio
    async def test_stop_hook_syncs_transcript(self, tmp_path: Path) -> None:
        transcript_path = tmp_path / "transcript.jsonl"
        output_dir = tmp_path / "traces"
        _write_fake_transcript(transcript_path)

        async with traqo_agent(
            "my-agent", output_dir=output_dir, tags=["test"]
        ) as hooks:
            # Simulate the Agent SDK calling the stop hook
            result = await _call_stop_hook(
                hooks,
                {
                    "transcript_path": str(transcript_path),
                    "session_id": "agent-sess-1",
                    "hook_event_name": "Stop",
                    "cwd": "/tmp/project",
                    "stop_hook_active": True,
                },
            )
            # Stop hooks return {} (no modifications)
            assert result == {}

        # Verify the trace was written
        output_file = output_dir / "cc-agent-sess-1.jsonl"
        assert output_file.exists()

        events = read_events(output_file)
        assert events[0]["type"] == "trace_start"
        assert events[0]["name"] == "my-agent"
        assert events[0]["tags"] == ["test"]
        assert events[-1]["type"] == "trace_end"

    @pytest.mark.asyncio
    async def test_nested_parent_gets_child_events(self, tmp_path: Path) -> None:
        parent_path = tmp_path / "parent.jsonl"
        transcript_path = tmp_path / "transcript.jsonl"
        output_dir = tmp_path / "traces"
        _write_fake_transcript(transcript_path)

        with Tracer(path=parent_path):
            async with traqo_agent("review-agent", output_dir=output_dir) as hooks:
                await _call_stop_hook(
                    hooks,
                    {
                        "transcript_path": str(transcript_path),
                        "session_id": "agent-sess-1",
                        "hook_event_name": "Stop",
                        "cwd": "/tmp/project",
                        "stop_hook_active": True,
                    },
                )

        parent_events = read_events(parent_path)
        names = [e.get("name") for e in parent_events if e["type"] == "event"]
        assert "child_started" in names
        assert "child_ended" in names

    @pytest.mark.asyncio
    async def test_nested_parent_stats_rollup(self, tmp_path: Path) -> None:
        parent_path = tmp_path / "parent.jsonl"
        transcript_path = tmp_path / "transcript.jsonl"
        output_dir = tmp_path / "traces"
        _write_fake_transcript(transcript_path)

        with Tracer(path=parent_path):
            async with traqo_agent("review-agent", output_dir=output_dir) as hooks:
                await _call_stop_hook(
                    hooks,
                    {
                        "transcript_path": str(transcript_path),
                        "session_id": "agent-sess-1",
                        "hook_event_name": "Stop",
                        "cwd": "/tmp/project",
                        "stop_hook_active": True,
                    },
                )

        parent_events = read_events(parent_path)
        trace_end = parent_events[-1]
        assert trace_end["type"] == "trace_end"
        # The fake transcript has 100 input + 200 cache_read = 300 total input, 50 output
        assert trace_end["stats"]["total_input_tokens"] == 300
        assert trace_end["stats"]["total_output_tokens"] == 50
        assert trace_end["stats"]["spans"] > 0
        assert len(trace_end.get("children", [])) == 1
        assert trace_end["children"][0]["name"] == "review-agent"

    @pytest.mark.asyncio
    async def test_standalone_no_parent(self, tmp_path: Path) -> None:
        """traqo_agent works without a parent tracer (standalone mode)."""
        transcript_path = tmp_path / "transcript.jsonl"
        output_dir = tmp_path / "traces"
        _write_fake_transcript(transcript_path)

        async with traqo_agent("standalone", output_dir=output_dir) as hooks:
            await _call_stop_hook(
                hooks,
                {
                    "transcript_path": str(transcript_path),
                    "session_id": "agent-sess-1",
                    "hook_event_name": "Stop",
                    "cwd": "/tmp/project",
                    "stop_hook_active": True,
                },
            )

        output_file = output_dir / "cc-agent-sess-1.jsonl"
        assert output_file.exists()

    @pytest.mark.asyncio
    async def test_no_transcript_path_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Stop hook with missing transcript_path logs a warning."""
        async with traqo_agent("test", output_dir=tmp_path) as hooks:
            with caplog.at_level("WARNING"):
                await _call_stop_hook(
                    hooks,
                    {
                        "session_id": "s1",
                        "hook_event_name": "Stop",
                        "cwd": "/tmp",
                        "stop_hook_active": True,
                    },
                )

        assert "no transcript_path" in caplog.text

    @pytest.mark.asyncio
    async def test_stop_hook_returns_empty_dict(self, tmp_path: Path) -> None:
        """Stop hook always returns {} (no agent behavior modification)."""
        transcript_path = tmp_path / "transcript.jsonl"
        _write_fake_transcript(transcript_path)

        async with traqo_agent("test", output_dir=tmp_path) as hooks:
            result = await _call_stop_hook(
                hooks,
                {
                    "transcript_path": str(transcript_path),
                    "session_id": "agent-sess-1",
                    "hook_event_name": "Stop",
                    "cwd": "/tmp",
                    "stop_hook_active": True,
                },
            )
        assert result == {}

    @pytest.mark.asyncio
    async def test_multi_agent_pipeline(self, tmp_path: Path) -> None:
        """Multiple agent runs inside one parent trace roll up correctly."""
        parent_path = tmp_path / "parent.jsonl"
        output_dir = tmp_path / "traces"

        # Two different transcripts for two agents
        t1 = tmp_path / "t1.jsonl"
        t2 = tmp_path / "t2.jsonl"
        _write_fake_transcript(t1)
        _write_fake_transcript(t2)

        stop_input_base: dict[str, Any] = {
            "hook_event_name": "Stop",
            "cwd": "/tmp/project",
            "stop_hook_active": True,
        }

        with Tracer(path=parent_path):
            async with traqo_agent(
                "agent-1", output_dir=output_dir, tags=["review"]
            ) as hooks:
                await _call_stop_hook(
                    hooks,
                    {
                        **stop_input_base,
                        "transcript_path": str(t1),
                        "session_id": "sess-a",
                    },
                )

            async with traqo_agent(
                "agent-2", output_dir=output_dir, tags=["test"]
            ) as hooks:
                await _call_stop_hook(
                    hooks,
                    {
                        **stop_input_base,
                        "transcript_path": str(t2),
                        "session_id": "sess-b",
                    },
                )

        parent_events = read_events(parent_path)
        trace_end = parent_events[-1]
        assert trace_end["type"] == "trace_end"

        # Two children registered
        assert len(trace_end["children"]) == 2
        assert trace_end["children"][0]["name"] == "agent-1"
        assert trace_end["children"][1]["name"] == "agent-2"

        # Stats rolled up from both (each transcript: 300 input, 50 output)
        assert trace_end["stats"]["total_input_tokens"] == 600
        assert trace_end["stats"]["total_output_tokens"] == 100

        # Both child trace files were created
        assert (output_dir / "cc-sess-a.jsonl").exists()
        assert (output_dir / "cc-sess-b.jsonl").exists()

        # Verify child traces have correct overridden names/tags
        child_a_events = read_events(output_dir / "cc-sess-a.jsonl")
        assert child_a_events[0]["name"] == "agent-1"
        assert child_a_events[0]["tags"] == ["review"]

        child_b_events = read_events(output_dir / "cc-sess-b.jsonl")
        assert child_b_events[0]["name"] == "agent-2"
        assert child_b_events[0]["tags"] == ["test"]
