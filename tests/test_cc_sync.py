"""Tests for traqo.cc_sync — Claude Code transcript to traqo trace conversion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from traqo.cc_sync import (
    AssistantMessage,
    ParsedSession,
    Turn,
    generate_trace_events,
    parse_transcript,
    sync_session,
)

# ---------------------------------------------------------------------------
# Helpers for building synthetic transcript records
# ---------------------------------------------------------------------------


def _user_prompt(
    content: str,
    *,
    session_id: str = "test-session-1",
    parent_uuid: str | None = None,
    timestamp: str = "2026-03-01T10:00:00.000Z",
    slug: str = "test-slug",
    version: str = "2.1.63",
    git_branch: str = "main",
    cwd: str = "/tmp/project",
) -> dict[str, Any]:
    return {
        "type": "user",
        "parentUuid": parent_uuid,
        "isSidechain": False,
        "userType": "external",
        "sessionId": session_id,
        "version": version,
        "gitBranch": git_branch,
        "slug": slug,
        "cwd": cwd,
        "message": {"role": "user", "content": content},
        "uuid": f"user-{content[:8]}",
        "timestamp": timestamp,
    }


def _assistant(
    content_blocks: list[dict[str, Any]],
    *,
    message_id: str = "msg_001",
    model: str = "claude-opus-4-6",
    request_id: str = "req_001",
    input_tokens: int = 100,
    output_tokens: int = 50,
    cache_read: int = 0,
    cache_create: int = 0,
    stop_reason: str | None = None,
    timestamp: str = "2026-03-01T10:00:05.000Z",
    parent_uuid: str | None = None,
    session_id: str = "test-session-1",
) -> dict[str, Any]:
    return {
        "type": "assistant",
        "parentUuid": parent_uuid,
        "isSidechain": False,
        "userType": "external",
        "sessionId": session_id,
        "version": "2.1.63",
        "gitBranch": "main",
        "slug": "test-slug",
        "cwd": "/tmp/project",
        "requestId": request_id,
        "message": {
            "model": model,
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "content": content_blocks,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read_input_tokens": cache_read,
                "cache_creation_input_tokens": cache_create,
                "service_tier": "standard",
            },
        },
        "uuid": f"asst-{message_id}-{len(content_blocks)}",
        "timestamp": timestamp,
    }


def _tool_result(
    tool_use_id: str,
    content: str,
    *,
    timestamp: str = "2026-03-01T10:00:06.000Z",
    session_id: str = "test-session-1",
) -> dict[str, Any]:
    return {
        "type": "user",
        "parentUuid": None,
        "sessionId": session_id,
        "version": "2.1.63",
        "message": {
            "role": "user",
            "content": [
                {
                    "tool_use_id": tool_use_id,
                    "type": "tool_result",
                    "content": content,
                    "is_error": False,
                }
            ],
        },
        "uuid": f"result-{tool_use_id}",
        "timestamp": timestamp,
    }


def _turn_duration(
    duration_ms: int = 5000,
    *,
    timestamp: str = "2026-03-01T10:00:10.000Z",
) -> dict[str, Any]:
    return {
        "type": "system",
        "subtype": "turn_duration",
        "durationMs": duration_ms,
        "isMeta": False,
        "timestamp": timestamp,
    }


def _meta_user(
    content: str, *, timestamp: str = "2026-03-01T09:59:00.000Z"
) -> dict[str, Any]:
    return {
        "type": "user",
        "isMeta": True,
        "sessionId": "test-session-1",
        "message": {"role": "user", "content": content},
        "uuid": "meta-1",
        "timestamp": timestamp,
    }


def _command_user(
    command: str, *, timestamp: str = "2026-03-01T09:59:00.000Z"
) -> dict[str, Any]:
    return {
        "type": "user",
        "sessionId": "test-session-1",
        "message": {
            "role": "user",
            "content": f"<command-name>{command}</command-name>\n<command-message>{command}</command-message>",
        },
        "uuid": "cmd-1",
        "timestamp": timestamp,
    }


def _progress(
    agent_id: str,
    nested_msg: dict[str, Any] | None = None,
    *,
    timestamp: str = "2026-03-01T10:00:07.000Z",
    prompt: str = "Research this topic",
) -> dict[str, Any]:
    return {
        "type": "progress",
        "parentUuid": None,
        "data": {
            "type": "agent_progress",
            "agentId": agent_id,
            "prompt": prompt,
            "message": nested_msg or {},
        },
        "uuid": f"progress-{agent_id}",
        "timestamp": timestamp,
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# parse_transcript tests
# ---------------------------------------------------------------------------


class TestParseTranscript:
    def test_single_turn_basic(self, tmp_path: Path) -> None:
        """Parse a minimal transcript with one user prompt and one assistant response."""
        records = [
            _user_prompt("Hello, what is 2+2?"),
            _assistant(
                [{"type": "text", "text": "2+2 is 4."}],
                stop_reason="end_turn",
                output_tokens=10,
            ),
            _turn_duration(3000),
        ]
        path = tmp_path / "session.jsonl"
        _write_jsonl(path, records)

        session = parse_transcript(path)
        assert session.session_id == "test-session-1"
        assert session.slug == "test-slug"
        assert session.model == "claude-opus-4-6"
        assert len(session.turns) == 1

        turn = session.turns[0]
        assert turn.prompt == "Hello, what is 2+2?"
        assert turn.index == 1
        assert turn.duration_ms == 3000
        assert len(turn.assistant_messages) == 1
        assert turn.assistant_messages[0].model == "claude-opus-4-6"

    def test_streaming_dedup(self, tmp_path: Path) -> None:
        """Multiple assistant chunks with same message.id merge into one."""
        records = [
            _user_prompt("Do something"),
            # Chunk 1: thinking
            _assistant(
                [{"type": "thinking", "thinking": "Let me think..."}],
                message_id="msg_streamed",
                output_tokens=5,
                timestamp="2026-03-01T10:00:01.000Z",
            ),
            # Chunk 2: tool_use
            _assistant(
                [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "Bash",
                        "input": {"command": "ls"},
                    }
                ],
                message_id="msg_streamed",
                output_tokens=20,
                timestamp="2026-03-01T10:00:02.000Z",
            ),
            # Chunk 3: text (with final output_tokens)
            _assistant(
                [{"type": "text", "text": "Here's the result."}],
                message_id="msg_streamed",
                output_tokens=50,
                stop_reason="end_turn",
                timestamp="2026-03-01T10:00:03.000Z",
            ),
        ]
        path = tmp_path / "session.jsonl"
        _write_jsonl(path, records)

        session = parse_transcript(path)
        assert len(session.turns) == 1
        turn = session.turns[0]

        # Should be one deduped assistant message
        assert len(turn.assistant_messages) == 1
        am = turn.assistant_messages[0]

        # Content blocks merged from all chunks
        assert len(am.content_blocks) == 3
        assert am.content_blocks[0]["type"] == "thinking"
        assert am.content_blocks[1]["type"] == "tool_use"
        assert am.content_blocks[2]["type"] == "text"

        # Usage from last chunk
        assert am.usage["output_tokens"] == 50

    def test_two_turns(self, tmp_path: Path) -> None:
        """Two user prompts create two turns."""
        records = [
            _user_prompt("First question", timestamp="2026-03-01T10:00:00.000Z"),
            _assistant(
                [{"type": "text", "text": "First answer"}],
                message_id="msg_1",
                timestamp="2026-03-01T10:00:01.000Z",
            ),
            _turn_duration(1000, timestamp="2026-03-01T10:00:02.000Z"),
            _user_prompt("Second question", timestamp="2026-03-01T10:01:00.000Z"),
            _assistant(
                [{"type": "text", "text": "Second answer"}],
                message_id="msg_2",
                timestamp="2026-03-01T10:01:01.000Z",
            ),
            _turn_duration(2000, timestamp="2026-03-01T10:01:03.000Z"),
        ]
        path = tmp_path / "session.jsonl"
        _write_jsonl(path, records)

        session = parse_transcript(path)
        assert len(session.turns) == 2
        assert session.turns[0].prompt == "First question"
        assert session.turns[0].index == 1
        assert session.turns[1].prompt == "Second question"
        assert session.turns[1].index == 2

    def test_tool_use_result_pairing(self, tmp_path: Path) -> None:
        """Tool use blocks are paired with matching tool results."""
        records = [
            _user_prompt("Check git status"),
            _assistant(
                [
                    {
                        "type": "tool_use",
                        "id": "toolu_abc",
                        "name": "Bash",
                        "input": {"command": "git status"},
                    },
                ],
                message_id="msg_tools",
                timestamp="2026-03-01T10:00:01.000Z",
            ),
            _tool_result("toolu_abc", "On branch main\nnothing to commit"),
            _assistant(
                [{"type": "text", "text": "The branch is clean."}],
                message_id="msg_after_tool",
                timestamp="2026-03-01T10:00:07.000Z",
            ),
        ]
        path = tmp_path / "session.jsonl"
        _write_jsonl(path, records)

        session = parse_transcript(path)
        assert len(session.turns) == 1
        turn = session.turns[0]
        assert "toolu_abc" in turn.tool_results
        assert turn.tool_results["toolu_abc"] == "On branch main\nnothing to commit"
        assert len(turn.assistant_messages) == 2

    def test_meta_and_command_messages_filtered(self, tmp_path: Path) -> None:
        """isMeta and command messages don't create turns."""
        records = [
            _meta_user("Caveat: ignore this"),
            _command_user("/clear"),
            _user_prompt("Real question"),
            _assistant(
                [{"type": "text", "text": "Real answer"}],
                message_id="msg_real",
            ),
        ]
        path = tmp_path / "session.jsonl"
        _write_jsonl(path, records)

        session = parse_transcript(path)
        assert len(session.turns) == 1
        assert session.turns[0].prompt == "Real question"

    def test_subagent_progress(self, tmp_path: Path) -> None:
        """Progress records are grouped by agentId under the current turn."""
        nested_assistant = {
            "type": "assistant",
            "message": {
                "model": "claude-sonnet-4-6",
                "content": [{"type": "text", "text": "Subagent found info."}],
                "usage": {"input_tokens": 50, "output_tokens": 25},
            },
            "timestamp": "2026-03-01T10:00:08.000Z",
        }
        records = [
            _user_prompt("Research this"),
            _assistant(
                [
                    {
                        "type": "tool_use",
                        "id": "toolu_agent",
                        "name": "Agent",
                        "input": {"prompt": "Research"},
                    }
                ],
                message_id="msg_agent",
            ),
            _progress("agent-abc123", nested_assistant, prompt="Research this topic"),
        ]
        path = tmp_path / "session.jsonl"
        _write_jsonl(path, records)

        session = parse_transcript(path)
        assert len(session.turns) == 1
        turn = session.turns[0]
        assert "agent-abc123" in turn.subagent_progress
        assert len(turn.subagent_progress["agent-abc123"]) == 1

    def test_malformed_lines_skipped(self, tmp_path: Path) -> None:
        """Malformed JSON lines are skipped without error."""
        path = tmp_path / "session.jsonl"
        with open(path, "w") as f:
            f.write("not valid json\n")
            f.write(json.dumps(_user_prompt("Works fine")) + "\n")
            f.write(
                json.dumps(
                    _assistant([{"type": "text", "text": "OK"}], message_id="msg_ok")
                )
                + "\n"
            )

        session = parse_transcript(path)
        assert len(session.turns) == 1

    def test_empty_transcript(self, tmp_path: Path) -> None:
        """Empty transcript produces no turns."""
        path = tmp_path / "session.jsonl"
        _write_jsonl(path, [])

        session = parse_transcript(path)
        assert len(session.turns) == 0


# ---------------------------------------------------------------------------
# generate_trace_events tests
# ---------------------------------------------------------------------------


class TestGenerateTraceEvents:
    def test_basic_structure(self) -> None:
        """Single turn produces trace_start, chain span, llm span, trace_end."""
        session = ParsedSession(
            session_id="sess-1",
            slug="test-session",
            version="2.1.63",
            git_branch="main",
            cwd="/tmp/project",
            model="claude-opus-4-6",
            turns=[
                Turn(
                    index=1,
                    prompt="Hello",
                    timestamp="2026-03-01T10:00:00.000Z",
                    assistant_messages=[
                        _make_assistant_msg(
                            "msg_1",
                            "claude-opus-4-6",
                            "Hi there!",
                            input_tokens=10,
                            output_tokens=5,
                        ),
                    ],
                    tool_results={},
                    subagent_progress={},
                    duration_ms=2000,
                ),
            ],
            first_timestamp="2026-03-01T10:00:00.000Z",
            last_timestamp="2026-03-01T10:00:02.000Z",
        )

        events = generate_trace_events(session)

        # trace_start, span_start(chain), span_start(llm), span_end(llm), span_end(chain), trace_end
        assert events[0]["type"] == "trace_start"
        assert events[0]["name"] == "Claude Code: test-session"
        assert events[0]["thread_id"] == "sess-1"
        assert "claude-code" in events[0]["tags"]

        assert events[1]["type"] == "span_start"
        assert events[1]["kind"] == "chain"
        chain_id = events[1]["id"]

        assert events[2]["type"] == "span_start"
        assert events[2]["kind"] == "llm"
        assert events[2]["parent_id"] == chain_id

        assert events[3]["type"] == "span_end"
        assert events[3]["kind"] == "llm"
        assert events[3]["output"] == "Hi there!"

        assert events[4]["type"] == "span_end"
        assert events[4]["kind"] == "chain"
        assert events[4]["duration_s"] == 2.0

        assert events[5]["type"] == "trace_end"
        assert events[5]["stats"]["total_input_tokens"] == 10
        assert events[5]["stats"]["total_output_tokens"] == 5
        assert events[5]["stats"]["spans"] == 2  # chain + llm

    def test_tool_spans(self) -> None:
        """Tool use blocks create tool spans with matched results."""

        session = ParsedSession(
            session_id="sess-2",
            slug="tool-test",
            version="2.1.63",
            git_branch="main",
            cwd="/tmp/project",
            model="claude-opus-4-6",
            turns=[
                Turn(
                    index=1,
                    prompt="Check status",
                    timestamp="2026-03-01T10:00:00.000Z",
                    assistant_messages=[
                        AssistantMessage(
                            message_id="msg_1",
                            model="claude-opus-4-6",
                            content_blocks=[
                                {
                                    "type": "tool_use",
                                    "id": "toolu_1",
                                    "name": "Bash",
                                    "input": {"command": "git status"},
                                },
                            ],
                            usage={"input_tokens": 50, "output_tokens": 20},
                            request_id="req_1",
                            timestamp="2026-03-01T10:00:01.000Z",
                        ),
                    ],
                    tool_results={"toolu_1": "On branch main"},
                    subagent_progress={},
                ),
            ],
            first_timestamp="2026-03-01T10:00:00.000Z",
            last_timestamp="2026-03-01T10:00:02.000Z",
        )

        events = generate_trace_events(session)
        tool_starts = [
            e
            for e in events
            if e.get("type") == "span_start" and e.get("kind") == "tool"
        ]
        tool_ends = [
            e for e in events if e.get("type") == "span_end" and e.get("kind") == "tool"
        ]

        assert len(tool_starts) == 1
        assert tool_starts[0]["name"] == "Bash"
        assert tool_starts[0]["input"] == {"command": "git status"}

        assert len(tool_ends) == 1
        assert tool_ends[0]["output"] == "On branch main"

        # No LLM span since assistant only had tool_use (no text)
        llm_spans = [
            e
            for e in events
            if e.get("type") == "span_start" and e.get("kind") == "llm"
        ]
        assert len(llm_spans) == 0

        # Token usage attached to tool span metadata instead
        assert tool_starts[0]["metadata"]["token_usage"]["input_tokens"] == 50
        assert tool_starts[0]["metadata"]["model"] == "claude-opus-4-6"

    def test_token_accumulation(self) -> None:
        """Token counts from multiple assistant messages accumulate correctly."""
        session = ParsedSession(
            session_id="sess-3",
            slug="tokens",
            version="2.1.63",
            git_branch="main",
            cwd="/tmp",
            model="claude-opus-4-6",
            turns=[
                Turn(
                    index=1,
                    prompt="Q1",
                    timestamp="2026-03-01T10:00:00.000Z",
                    assistant_messages=[
                        _make_assistant_msg(
                            "m1",
                            "claude-opus-4-6",
                            "A1",
                            input_tokens=100,
                            output_tokens=50,
                        ),
                        _make_assistant_msg(
                            "m2",
                            "claude-opus-4-6",
                            "A2",
                            input_tokens=200,
                            output_tokens=100,
                        ),
                    ],
                    tool_results={},
                    subagent_progress={},
                ),
            ],
            first_timestamp="2026-03-01T10:00:00.000Z",
            last_timestamp="2026-03-01T10:00:10.000Z",
        )

        events = generate_trace_events(session)
        trace_end = events[-1]
        assert trace_end["type"] == "trace_end"
        assert trace_end["stats"]["total_input_tokens"] == 300
        assert trace_end["stats"]["total_output_tokens"] == 150

    def test_cache_tokens_in_metadata(self) -> None:
        """Cache token info stored in LLM span metadata."""

        session = ParsedSession(
            session_id="sess-cache",
            slug="cache",
            version="2.1.63",
            git_branch="main",
            cwd="/tmp",
            model="claude-opus-4-6",
            turns=[
                Turn(
                    index=1,
                    prompt="Q",
                    timestamp="2026-03-01T10:00:00.000Z",
                    assistant_messages=[
                        AssistantMessage(
                            message_id="m1",
                            model="claude-opus-4-6",
                            content_blocks=[{"type": "text", "text": "A"}],
                            usage={
                                "input_tokens": 10,
                                "output_tokens": 5,
                                "cache_read_input_tokens": 5000,
                                "cache_creation_input_tokens": 2000,
                            },
                            request_id="r1",
                            timestamp="2026-03-01T10:00:01.000Z",
                        ),
                    ],
                    tool_results={},
                    subagent_progress={},
                ),
            ],
            first_timestamp="2026-03-01T10:00:00.000Z",
            last_timestamp="2026-03-01T10:00:01.000Z",
        )

        events = generate_trace_events(session)
        llm_end = [
            e for e in events if e.get("type") == "span_end" and e.get("kind") == "llm"
        ][0]
        usage = llm_end["metadata"]["token_usage"]
        # input_tokens includes cache tokens (10 + 5000 + 2000)
        assert usage["input_tokens"] == 7010
        assert usage["cache_read_tokens"] == 5000
        assert usage["cache_creation_tokens"] == 2000

        # trace_end total also includes cache tokens
        trace_end = events[-1]
        assert trace_end["stats"]["total_input_tokens"] == 7010

    def test_subagent_spans_with_tools(self) -> None:
        """Subagent progress records create agent spans with LLM + tool spans."""
        session = ParsedSession(
            session_id="sess-agent",
            slug="agent-test",
            version="2.1.63",
            git_branch="main",
            cwd="/tmp",
            model="claude-opus-4-6",
            turns=[
                Turn(
                    index=1,
                    prompt="Research this",
                    timestamp="2026-03-01T10:00:00.000Z",
                    assistant_messages=[],
                    tool_results={},
                    subagent_progress={
                        "agent-abc12345": [
                            # Assistant with tool_use
                            {
                                "type": "progress",
                                "timestamp": "2026-03-01T10:00:02.000Z",
                                "data": {
                                    "type": "agent_progress",
                                    "agentId": "agent-abc12345",
                                    "prompt": "Research topic",
                                    "message": {
                                        "type": "assistant",
                                        "message": {
                                            "model": "claude-opus-4-6",
                                            "content": [
                                                {
                                                    "type": "tool_use",
                                                    "id": "toolu_sub1",
                                                    "name": "Read",
                                                    "input": {
                                                        "file_path": "/tmp/data.txt"
                                                    },
                                                }
                                            ],
                                            "usage": {
                                                "input_tokens": 30,
                                                "output_tokens": 10,
                                            },
                                        },
                                        "timestamp": "2026-03-01T10:00:03.000Z",
                                    },
                                },
                            },
                            # User tool result
                            {
                                "type": "progress",
                                "timestamp": "2026-03-01T10:00:04.000Z",
                                "data": {
                                    "type": "agent_progress",
                                    "agentId": "agent-abc12345",
                                    "message": {
                                        "type": "user",
                                        "message": {
                                            "content": [
                                                {
                                                    "type": "tool_result",
                                                    "tool_use_id": "toolu_sub1",
                                                    "content": "file contents here",
                                                }
                                            ]
                                        },
                                        "timestamp": "2026-03-01T10:00:04.000Z",
                                    },
                                },
                            },
                            # Assistant with text response
                            {
                                "type": "progress",
                                "timestamp": "2026-03-01T10:00:05.000Z",
                                "data": {
                                    "type": "agent_progress",
                                    "agentId": "agent-abc12345",
                                    "message": {
                                        "type": "assistant",
                                        "message": {
                                            "model": "claude-opus-4-6",
                                            "content": [
                                                {
                                                    "type": "text",
                                                    "text": "Based on the file, here is my analysis.",
                                                }
                                            ],
                                            "usage": {
                                                "input_tokens": 50,
                                                "output_tokens": 20,
                                            },
                                        },
                                        "timestamp": "2026-03-01T10:00:06.000Z",
                                    },
                                },
                            },
                        ],
                    },
                ),
            ],
            first_timestamp="2026-03-01T10:00:00.000Z",
            last_timestamp="2026-03-01T10:00:06.000Z",
        )

        events = generate_trace_events(session)
        agent_starts = [
            e
            for e in events
            if e.get("type") == "span_start" and e.get("kind") == "agent"
        ]
        assert len(agent_starts) == 1
        agent_id = agent_starts[0]["id"]
        assert "abc12345" in agent_starts[0]["name"]

        # Should have 1 LLM span (only the text response; tool-only is skipped)
        nested_llm = [
            e
            for e in events
            if e.get("type") == "span_end"
            and e.get("kind") == "llm"
            and e.get("parent_id") == agent_id
        ]
        assert len(nested_llm) == 1
        assert nested_llm[0]["output"] == "Based on the file, here is my analysis."

        # Should have 1 tool span with input and output
        nested_tools = [
            e
            for e in events
            if e.get("type") == "span_end"
            and e.get("kind") == "tool"
            and e.get("parent_id") == agent_id
        ]
        assert len(nested_tools) == 1
        assert nested_tools[0]["name"] == "Read"
        assert nested_tools[0]["output"] == "file contents here"

        tool_starts = [
            e
            for e in events
            if e.get("type") == "span_start"
            and e.get("kind") == "tool"
            and e.get("parent_id") == agent_id
        ]
        assert tool_starts[0]["input"] == {"file_path": "/tmp/data.txt"}
        # Tool-only response: token usage folded onto tool span
        assert tool_starts[0]["metadata"]["token_usage"]["input_tokens"] == 30
        assert tool_starts[0]["metadata"]["model"] == "claude-opus-4-6"

        # Token accumulation includes all subagent tokens
        trace_end = events[-1]
        assert trace_end["stats"]["total_input_tokens"] == 80
        assert trace_end["stats"]["total_output_tokens"] == 30


# ---------------------------------------------------------------------------
# sync_session tests
# ---------------------------------------------------------------------------


class TestSyncSession:
    def test_writes_output_file(self, tmp_path: Path) -> None:
        """sync_session writes a valid traqo JSONL file."""
        transcript = tmp_path / "transcript.jsonl"
        output_dir = tmp_path / "traces"
        records = [
            _user_prompt("Hello"),
            _assistant([{"type": "text", "text": "Hi"}], message_id="m1"),
        ]
        _write_jsonl(transcript, records)

        result = sync_session(transcript, output_dir, session_id="test-sess")
        assert result is not None
        assert result.name == "cc-test-sess.jsonl"
        assert result.exists()

        # Verify it's valid JSONL with trace_start and trace_end
        lines = result.read_text().strip().split("\n")
        events = [json.loads(line) for line in lines]
        assert events[0]["type"] == "trace_start"
        assert events[-1]["type"] == "trace_end"

    def test_skip_when_unchanged(self, tmp_path: Path) -> None:
        """Second sync with no new data returns None."""
        transcript = tmp_path / "transcript.jsonl"
        output_dir = tmp_path / "traces"
        records = [
            _user_prompt("Hello"),
            _assistant([{"type": "text", "text": "Hi"}], message_id="m1"),
        ]
        _write_jsonl(transcript, records)

        result1 = sync_session(transcript, output_dir, session_id="test-sess")
        assert result1 is not None

        result2 = sync_session(transcript, output_dir, session_id="test-sess")
        assert result2 is None

    def test_rewrite_when_new_data(self, tmp_path: Path) -> None:
        """Second sync with new data rewrites the output."""
        transcript = tmp_path / "transcript.jsonl"
        output_dir = tmp_path / "traces"
        records = [
            _user_prompt("Hello"),
            _assistant([{"type": "text", "text": "Hi"}], message_id="m1"),
        ]
        _write_jsonl(transcript, records)

        result1 = sync_session(transcript, output_dir, session_id="test-sess")
        assert result1 is not None
        original_size = result1.stat().st_size

        # Append more data
        records.append(_user_prompt("Follow up", timestamp="2026-03-01T10:01:00.000Z"))
        records.append(
            _assistant(
                [{"type": "text", "text": "Sure"}],
                message_id="m2",
                timestamp="2026-03-01T10:01:01.000Z",
            )
        )
        _write_jsonl(transcript, records)

        result2 = sync_session(transcript, output_dir, session_id="test-sess")
        assert result2 is not None
        assert result2.stat().st_size > original_size

    def test_empty_transcript_skipped(self, tmp_path: Path) -> None:
        """Transcript with no turns returns None."""
        transcript = tmp_path / "empty.jsonl"
        output_dir = tmp_path / "traces"
        _write_jsonl(
            transcript,
            [{"type": "file-history-snapshot", "messageId": "x", "snapshot": {}}],
        )

        result = sync_session(transcript, output_dir)
        assert result is None

    def test_missing_transcript(self, tmp_path: Path) -> None:
        """Missing transcript file returns None."""
        result = sync_session(tmp_path / "nonexistent.jsonl", tmp_path / "traces")
        assert result is None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_assistant_msg(
    msg_id: str,
    model: str,
    text: str,
    *,
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> AssistantMessage:
    return AssistantMessage(
        message_id=msg_id,
        model=model,
        content_blocks=[{"type": "text", "text": text}],
        usage={"input_tokens": input_tokens, "output_tokens": output_tokens},
        request_id=f"req_{msg_id}",
        timestamp="2026-03-01T10:00:01.000Z",
    )
