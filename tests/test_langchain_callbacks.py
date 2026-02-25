"""Tests for LangChain callback handler — retriever, chain, and agent callbacks."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

try:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
    from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult
except ImportError:
    pytest.skip("langchain-core not installed", allow_module_level=True)

from tests.conftest import read_events
from traqo import Tracer
from traqo.integrations.langchain import (
    TracedChatModel,
    TraqoCallback,
    _extract_output,
    _interrupt_value,
    _is_langgraph_interrupt,
    _parse_usage_metadata,
    track_langgraph,
)


class TestRetrieverCallbacks:
    def test_retriever_start_end(self, trace_file: Path):
        cb = TraqoCallback()
        run_id = uuid4()

        with Tracer(path=trace_file):
            cb.on_retriever_start(
                serialized={"id": ["langchain", "retrievers", "FAISS"]},
                query="what is traqo?",
                run_id=run_id,
            )

            # Simulate documents returned
            class FakeDoc:
                def __init__(self, content, metadata):
                    self.page_content = content
                    self.metadata = metadata

            docs = [
                FakeDoc("traqo is a tracing lib", {"source": "readme.md"}),
                FakeDoc("traqo writes JSONL", {"source": "docs.md"}),
            ]
            cb.on_retriever_end(documents=docs, run_id=run_id)

        events = read_events(trace_file)
        starts = [
            e
            for e in events
            if e["type"] == "span_start" and e.get("kind") == "retriever"
        ]
        ends = [
            e
            for e in events
            if e["type"] == "span_end" and e.get("kind") == "retriever"
        ]

        assert len(starts) == 1
        assert starts[0]["name"] == "FAISS"
        assert starts[0]["input"] == "what is traqo?"

        assert len(ends) == 1
        assert ends[0]["status"] == "ok"
        assert len(ends[0]["output"]) == 2
        assert ends[0]["output"][0]["page_content"] == "traqo is a tracing lib"
        assert ends[0]["output"][0]["metadata"] == {"source": "readme.md"}

    def test_retriever_error(self, trace_file: Path):
        cb = TraqoCallback()
        run_id = uuid4()

        with Tracer(path=trace_file):
            cb.on_retriever_start(
                serialized={"id": ["langchain", "retrievers", "Pinecone"]},
                query="test query",
                run_id=run_id,
            )
            cb.on_retriever_error(
                error=ConnectionError("index not found"),
                run_id=run_id,
            )

        events = read_events(trace_file)
        ends = [
            e
            for e in events
            if e["type"] == "span_end" and e.get("kind") == "retriever"
        ]
        assert len(ends) == 1
        assert ends[0]["status"] == "error"
        assert ends[0]["error"]["type"] == "ConnectionError"
        assert "index not found" in ends[0]["error"]["message"]

    def test_retriever_name_fallback(self, trace_file: Path):
        cb = TraqoCallback()
        run_id = uuid4()

        with Tracer(path=trace_file):
            cb.on_retriever_start(
                serialized={"name": "my_retriever"},
                query="test",
                run_id=run_id,
            )
            cb.on_retriever_end(documents=[], run_id=run_id)

        events = read_events(trace_file)
        starts = [
            e
            for e in events
            if e["type"] == "span_start" and e.get("kind") == "retriever"
        ]
        assert starts[0]["name"] == "my_retriever"

    def test_retriever_parent_nesting(self, trace_file: Path):
        cb = TraqoCallback()
        chain_id = uuid4()
        retriever_id = uuid4()

        with Tracer(path=trace_file):
            cb.on_chain_start(
                serialized={"id": ["langchain", "chains", "RetrievalQA"]},
                inputs={"query": "test"},
                run_id=chain_id,
            )
            cb.on_retriever_start(
                serialized={"id": ["langchain", "retrievers", "FAISS"]},
                query="test",
                run_id=retriever_id,
                parent_run_id=chain_id,
            )
            cb.on_retriever_end(
                documents=[], run_id=retriever_id, parent_run_id=chain_id
            )
            cb.on_chain_end(outputs={"result": "answer"}, run_id=chain_id)

        events = read_events(trace_file)
        chain_start = [
            e for e in events if e["type"] == "span_start" and e.get("kind") == "chain"
        ][0]
        retriever_start = [
            e
            for e in events
            if e["type"] == "span_start" and e.get("kind") == "retriever"
        ][0]
        assert retriever_start["parent_id"] == chain_start["id"]


class TestChainCallbacks:
    def test_chain_start_end(self, trace_file: Path):
        cb = TraqoCallback()
        run_id = uuid4()

        with Tracer(path=trace_file):
            cb.on_chain_start(
                serialized={"id": ["langchain", "chains", "LLMChain"]},
                inputs={"text": "hello world"},
                run_id=run_id,
            )
            cb.on_chain_end(
                outputs={"result": "processed hello world"},
                run_id=run_id,
            )

        events = read_events(trace_file)
        starts = [
            e for e in events if e["type"] == "span_start" and e.get("kind") == "chain"
        ]
        ends = [
            e for e in events if e["type"] == "span_end" and e.get("kind") == "chain"
        ]

        assert len(starts) == 1
        assert starts[0]["name"] == "LLMChain"
        assert starts[0]["input"] == {"text": "hello world"}

        assert len(ends) == 1
        assert ends[0]["status"] == "ok"
        assert ends[0]["output"] == {"result": "processed hello world"}

    def test_chain_error(self, trace_file: Path):
        cb = TraqoCallback()
        run_id = uuid4()

        with Tracer(path=trace_file):
            cb.on_chain_start(
                serialized={"id": ["langchain", "chains", "LLMChain"]},
                inputs={"text": "test"},
                run_id=run_id,
            )
            cb.on_chain_error(
                error=ValueError("chain failed"),
                run_id=run_id,
            )

        events = read_events(trace_file)
        ends = [
            e for e in events if e["type"] == "span_end" and e.get("kind") == "chain"
        ]
        assert len(ends) == 1
        assert ends[0]["status"] == "error"
        assert ends[0]["error"]["type"] == "ValueError"

    def test_chain_name_from_name_key(self, trace_file: Path):
        cb = TraqoCallback()
        run_id = uuid4()

        with Tracer(path=trace_file):
            cb.on_chain_start(
                serialized={"name": "CustomChain"},
                inputs={},
                run_id=run_id,
            )
            cb.on_chain_end(outputs={}, run_id=run_id)

        events = read_events(trace_file)
        starts = [
            e for e in events if e["type"] == "span_start" and e.get("kind") == "chain"
        ]
        assert starts[0]["name"] == "CustomChain"

    def test_nested_chains(self, trace_file: Path):
        cb = TraqoCallback()
        outer_id = uuid4()
        inner_id = uuid4()

        with Tracer(path=trace_file):
            cb.on_chain_start(
                serialized={"id": ["langchain", "chains", "SequentialChain"]},
                inputs={"input": "test"},
                run_id=outer_id,
            )
            cb.on_chain_start(
                serialized={"id": ["langchain", "chains", "LLMChain"]},
                inputs={"text": "test"},
                run_id=inner_id,
                parent_run_id=outer_id,
            )
            cb.on_chain_end(outputs={"output": "inner"}, run_id=inner_id)
            cb.on_chain_end(outputs={"output": "outer"}, run_id=outer_id)

        events = read_events(trace_file)
        outer_start = [
            e
            for e in events
            if e["type"] == "span_start" and e["name"] == "SequentialChain"
        ][0]
        inner_start = [
            e for e in events if e["type"] == "span_start" and e["name"] == "LLMChain"
        ][0]
        assert inner_start["parent_id"] == outer_start["id"]

    def test_capture_content_false(self, trace_file: Path):
        cb = TraqoCallback()
        run_id = uuid4()

        with Tracer(path=trace_file, capture_content=False):
            cb.on_chain_start(
                serialized={"id": ["langchain", "chains", "LLMChain"]},
                inputs={"secret": "data"},
                run_id=run_id,
            )
            cb.on_chain_end(outputs={"result": "sensitive"}, run_id=run_id)

        events = read_events(trace_file)
        starts = [
            e for e in events if e["type"] == "span_start" and e.get("kind") == "chain"
        ]
        ends = [
            e for e in events if e["type"] == "span_end" and e.get("kind") == "chain"
        ]
        assert "input" not in starts[0]
        assert "output" not in ends[0]


class TestSerializedNone:
    """LangChain v0.3+ passes serialized=None for Runnables; name comes via kwargs."""

    def test_chain_start_serialized_none_uses_kwarg_name(self, trace_file: Path):
        cb = TraqoCallback()
        run_id = uuid4()

        with Tracer(path=trace_file):
            cb.on_chain_start(
                serialized=None,
                inputs={"text": "hello"},
                run_id=run_id,
                name="RunnableSequence",
            )
            cb.on_chain_end(outputs={"result": "done"}, run_id=run_id)

        events = read_events(trace_file)
        starts = [
            e for e in events if e["type"] == "span_start" and e.get("kind") == "chain"
        ]
        ends = [
            e for e in events if e["type"] == "span_end" and e.get("kind") == "chain"
        ]
        assert len(starts) == 1
        assert starts[0]["name"] == "RunnableSequence"
        assert starts[0]["input"] == {"text": "hello"}
        assert len(ends) == 1
        assert ends[0]["status"] == "ok"

    def test_chain_start_serialized_none_no_name_falls_back(self, trace_file: Path):
        cb = TraqoCallback()
        run_id = uuid4()

        with Tracer(path=trace_file):
            cb.on_chain_start(serialized=None, inputs={}, run_id=run_id)
            cb.on_chain_end(outputs={}, run_id=run_id)

        events = read_events(trace_file)
        starts = [
            e for e in events if e["type"] == "span_start" and e.get("kind") == "chain"
        ]
        assert starts[0]["name"] == "chain"

    def test_retriever_start_serialized_none_uses_kwarg_name(self, trace_file: Path):
        """BaseRetriever passes serialized=None with name kwarg."""
        cb = TraqoCallback()
        run_id = uuid4()

        with Tracer(path=trace_file):
            cb.on_retriever_start(
                serialized=None,
                query="what is traqo?",
                run_id=run_id,
                name="FAISS",
            )
            cb.on_retriever_end(documents=[], run_id=run_id)

        events = read_events(trace_file)
        starts = [
            e
            for e in events
            if e["type"] == "span_start" and e.get("kind") == "retriever"
        ]
        assert len(starts) == 1
        assert starts[0]["name"] == "FAISS"

    def test_tool_start_serialized_none_uses_kwarg_name(self, trace_file: Path):
        cb = TraqoCallback()
        run_id = uuid4()

        with Tracer(path=trace_file):
            cb.on_tool_start(
                serialized=None,
                input_str="2 + 2",
                run_id=run_id,
                name="calculator",
            )
            cb.on_tool_end(output="4", run_id=run_id)

        events = read_events(trace_file)
        starts = [
            e for e in events if e["type"] == "span_start" and e.get("kind") == "tool"
        ]
        assert len(starts) == 1
        assert starts[0]["name"] == "calculator"

    def test_kwarg_name_preferred_over_serialized(self, trace_file: Path):
        """kwargs['name'] takes precedence (matches LangChain's StdOutCallbackHandler)."""
        cb = TraqoCallback()
        run_id = uuid4()

        with Tracer(path=trace_file):
            cb.on_chain_start(
                serialized={"id": ["langchain", "chains", "LLMChain"]},
                inputs={},
                run_id=run_id,
                name="my_custom_name",
            )
            cb.on_chain_end(outputs={}, run_id=run_id)

        events = read_events(trace_file)
        starts = [
            e for e in events if e["type"] == "span_start" and e.get("kind") == "chain"
        ]
        assert starts[0]["name"] == "my_custom_name"

    def test_chain_error_after_serialized_none_start(self, trace_file: Path):
        """Full lifecycle: serialized=None start followed by error."""
        cb = TraqoCallback()
        run_id = uuid4()

        with Tracer(path=trace_file) as t:
            cb.on_chain_start(
                serialized=None,
                inputs={},
                run_id=run_id,
                name="RunnableLambda",
            )
            cb.on_chain_error(error=ValueError("failed"), run_id=run_id)
            assert t._stats_errors == 1
            assert t._stats_spans == 1

        events = read_events(trace_file)
        ends = [
            e for e in events if e["type"] == "span_end" and e.get("kind") == "chain"
        ]
        assert ends[0]["status"] == "error"
        assert ends[0]["name"] == "RunnableLambda"

    def test_interrupt_after_serialized_none_start(self, trace_file: Path):
        cb = TraqoCallback()
        run_id = uuid4()

        with Tracer(path=trace_file):
            cb.on_chain_start(
                serialized=None,
                inputs={"messages": []},
                run_id=run_id,
                name="CompiledStateGraph",
            )
            cb.on_chain_error(error=GraphInterrupt("approval needed"), run_id=run_id)

        events = read_events(trace_file)
        ends = [
            e for e in events if e["type"] == "span_end" and e.get("kind") == "chain"
        ]
        assert ends[0]["status"] == "interrupted"
        assert ends[0]["name"] == "CompiledStateGraph"


class TestSafeCallbackDecorator:
    """Callback errors should be logged, not raised."""

    def test_callback_exception_does_not_propagate(self, trace_file: Path):
        cb = TraqoCallback()
        run_id = uuid4()

        with Tracer(path=trace_file):
            # Pass a non-dict, non-None serialized to trigger an AttributeError
            # inside _name_from_serialized when it tries .get() on an int.
            # The @_safe_callback decorator should catch this.
            cb.on_chain_start(
                serialized=42,  # type: ignore[arg-type]
                inputs={"text": "hello"},
                run_id=run_id,
            )

        # Should reach here without raising — the decorator caught it
        events = read_events(trace_file)
        chain_starts = [
            e for e in events if e["type"] == "span_start" and e.get("kind") == "chain"
        ]
        # The span was NOT written because the callback errored out
        assert len(chain_starts) == 0

    def test_end_callback_exception_does_not_propagate(self, trace_file: Path):
        cb = TraqoCallback()
        run_id = uuid4()

        with Tracer(path=trace_file):
            cb.on_chain_start(serialized=None, inputs={}, run_id=run_id, name="test")
            # Force an error in on_chain_end by passing a bad outputs type
            # that causes issues during write_event serialization.
            # Even if it doesn't error here, the point is the decorator is in place.
            cb.on_chain_end(outputs=None, run_id=run_id)

        # Should reach here without raising


class TestAgentCallbacks:
    def test_agent_action_and_finish(self, trace_file: Path):
        cb = TraqoCallback()
        run_id = uuid4()

        class FakeAction:
            tool = "search"
            tool_input = "what is traqo?"
            log = "Thought: I need to search\nAction: search"

        class FakeFinish:
            return_values = {"output": "traqo is a tracing library"}
            log = "Final Answer: traqo is a tracing library"

        with Tracer(path=trace_file):
            cb.on_agent_action(action=FakeAction(), run_id=run_id)
            cb.on_agent_finish(finish=FakeFinish(), run_id=run_id)

        events = read_events(trace_file)
        starts = [
            e for e in events if e["type"] == "span_start" and e.get("kind") == "agent"
        ]
        ends = [
            e for e in events if e["type"] == "span_end" and e.get("kind") == "agent"
        ]

        assert len(starts) == 1
        assert starts[0]["name"] == "search"
        assert starts[0]["input"]["tool"] == "search"
        assert starts[0]["input"]["tool_input"] == "what is traqo?"

        assert len(ends) == 1
        assert ends[0]["status"] == "ok"
        assert ends[0]["output"]["return_values"] == {
            "output": "traqo is a tracing library"
        }
        assert "Final Answer" in ends[0]["output"]["log"]

    def test_agent_no_tracer(self):
        """Agent callbacks should be no-ops without an active tracer."""
        cb = TraqoCallback()
        run_id = uuid4()

        class FakeAction:
            tool = "search"
            tool_input = "test"
            log = ""

        # Should not raise
        cb.on_agent_action(action=FakeAction(), run_id=run_id)
        cb.on_agent_finish(
            finish=type("F", (), {"return_values": {}, "log": ""})(), run_id=run_id
        )


class TestSpanCounting:
    def test_retriever_increments_span_count(self, trace_file: Path):
        cb = TraqoCallback()
        run_id = uuid4()

        with Tracer(path=trace_file) as t:
            cb.on_retriever_start(
                serialized={"id": ["FAISS"]},
                query="test",
                run_id=run_id,
            )
            cb.on_retriever_end(documents=[], run_id=run_id)
            assert t._stats_spans == 1

    def test_chain_increments_span_count(self, trace_file: Path):
        cb = TraqoCallback()
        run_id = uuid4()

        with Tracer(path=trace_file) as t:
            cb.on_chain_start(
                serialized={"id": ["LLMChain"]},
                inputs={},
                run_id=run_id,
            )
            cb.on_chain_end(outputs={}, run_id=run_id)
            assert t._stats_spans == 1

    def test_agent_increments_span_count(self, trace_file: Path):
        cb = TraqoCallback()
        run_id = uuid4()

        class FakeAction:
            tool = "test"
            tool_input = ""
            log = ""

        with Tracer(path=trace_file) as t:
            cb.on_agent_action(action=FakeAction(), run_id=run_id)
            cb.on_agent_finish(
                finish=type("F", (), {"return_values": {}, "log": ""})(), run_id=run_id
            )
            assert t._stats_spans == 1

    def test_error_increments_error_count(self, trace_file: Path):
        cb = TraqoCallback()

        with Tracer(path=trace_file) as t:
            # Retriever error
            r_id = uuid4()
            cb.on_retriever_start(serialized={"id": ["R"]}, query="q", run_id=r_id)
            cb.on_retriever_error(error=RuntimeError("fail"), run_id=r_id)

            # Chain error
            c_id = uuid4()
            cb.on_chain_start(serialized={"id": ["C"]}, inputs={}, run_id=c_id)
            cb.on_chain_error(error=RuntimeError("fail"), run_id=c_id)

            assert t._stats_errors == 2
            assert t._stats_spans == 2


# ---------------------------------------------------------------------------
# LLM callbacks (on_chat_model_start / on_llm_start / on_llm_end / on_llm_error)
# ---------------------------------------------------------------------------


class TestLLMCallbacks:
    def test_chat_model_start_end_cycle(self, trace_file: Path):
        cb = TraqoCallback()
        run_id = uuid4()

        msg = AIMessage(
            content="Hello world",
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )
        gen = ChatGeneration(message=msg)
        llm_result = LLMResult(generations=[[gen]])

        with Tracer(path=trace_file):
            cb.on_chat_model_start(
                serialized={
                    "kwargs": {"model_name": "gpt-4"},
                    "id": ["langchain", "chat_models", "ChatOpenAI"],
                },
                messages=[[HumanMessage(content="Hi")]],
                run_id=run_id,
            )
            cb.on_llm_end(response=llm_result, run_id=run_id)

        events = read_events(trace_file)
        starts = [
            e for e in events if e["type"] == "span_start" and e.get("kind") == "llm"
        ]
        ends = [e for e in events if e["type"] == "span_end" and e.get("kind") == "llm"]
        trace_end = [e for e in events if e["type"] == "trace_end"][0]

        assert len(starts) == 1
        assert starts[0]["name"] == "gpt-4"
        assert starts[0]["metadata"]["provider"] == "langchain"
        assert starts[0]["input"][0]["role"] == "human"
        assert starts[0]["input"][0]["content"] == "Hi"

        assert len(ends) == 1
        assert ends[0]["status"] == "ok"
        assert ends[0]["metadata"]["token_usage"]["input_tokens"] == 10
        assert ends[0]["metadata"]["token_usage"]["output_tokens"] == 5
        assert ends[0]["output"] == "Hello world"

        assert trace_end["stats"]["total_input_tokens"] == 10
        assert trace_end["stats"]["total_output_tokens"] == 5

    def test_chat_model_start_llm_error(self, trace_file: Path):
        cb = TraqoCallback()
        run_id = uuid4()

        with Tracer(path=trace_file):
            cb.on_chat_model_start(
                serialized={"kwargs": {"model_name": "gpt-4"}, "id": ["ChatOpenAI"]},
                messages=[[HumanMessage(content="Hi")]],
                run_id=run_id,
            )
            cb.on_llm_error(error=RuntimeError("rate limited"), run_id=run_id)

        events = read_events(trace_file)
        ends = [e for e in events if e["type"] == "span_end" and e.get("kind") == "llm"]
        assert len(ends) == 1
        assert ends[0]["status"] == "error"
        assert ends[0]["error"]["type"] == "RuntimeError"
        assert "rate limited" in ends[0]["error"]["message"]

    def test_llm_start_end_non_chat(self, trace_file: Path):
        cb = TraqoCallback()
        run_id = uuid4()

        msg = AIMessage(content="Completed text")
        gen = ChatGeneration(message=msg)
        llm_result = LLMResult(generations=[[gen]])

        with Tracer(path=trace_file):
            cb.on_llm_start(
                serialized={
                    "kwargs": {"model_name": "text-davinci-003"},
                    "id": ["langchain", "llms", "OpenAI"],
                },
                prompts=["Complete this: Hello"],
                run_id=run_id,
            )
            cb.on_llm_end(response=llm_result, run_id=run_id)

        events = read_events(trace_file)
        starts = [
            e for e in events if e["type"] == "span_start" and e.get("kind") == "llm"
        ]
        ends = [e for e in events if e["type"] == "span_end" and e.get("kind") == "llm"]

        assert len(starts) == 1
        assert starts[0]["name"] == "text-davinci-003"
        assert starts[0]["input"] == ["Complete this: Hello"]

        assert len(ends) == 1
        assert ends[0]["status"] == "ok"
        assert ends[0]["output"] == "Completed text"

    def test_llm_end_with_reasoning_and_cache_tokens(self, trace_file: Path):
        cb = TraqoCallback()
        run_id = uuid4()

        msg = AIMessage(
            content="Reasoned response",
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
                "output_token_details": {"reasoning": 20},
                "input_token_details": {"cache_read": 30, "cache_creation": 10},
            },
        )
        gen = ChatGeneration(message=msg)
        llm_result = LLMResult(generations=[[gen]])

        with Tracer(path=trace_file):
            cb.on_chat_model_start(
                serialized={
                    "kwargs": {"model_name": "o1-preview"},
                    "id": ["ChatOpenAI"],
                },
                messages=[[HumanMessage(content="Think")]],
                run_id=run_id,
            )
            cb.on_llm_end(response=llm_result, run_id=run_id)

        events = read_events(trace_file)
        end = [e for e in events if e["type"] == "span_end" and e.get("kind") == "llm"][
            0
        ]
        usage = end["metadata"]["token_usage"]
        assert usage["input_tokens"] == 100
        assert usage["output_tokens"] == 50
        assert usage["reasoning_tokens"] == 20
        assert usage["cache_read_tokens"] == 30
        assert usage["cache_creation_tokens"] == 10


# ---------------------------------------------------------------------------
# Tool callbacks (on_tool_start / on_tool_end / on_tool_error)
# ---------------------------------------------------------------------------


class TestToolCallbacks:
    def test_tool_start_end_cycle(self, trace_file: Path):
        cb = TraqoCallback()
        run_id = uuid4()

        with Tracer(path=trace_file):
            cb.on_tool_start(
                serialized={"name": "calculator"},
                input_str="2 + 2",
                run_id=run_id,
            )
            cb.on_tool_end(output="4", run_id=run_id)

        events = read_events(trace_file)
        starts = [
            e for e in events if e["type"] == "span_start" and e.get("kind") == "tool"
        ]
        ends = [
            e for e in events if e["type"] == "span_end" and e.get("kind") == "tool"
        ]

        assert len(starts) == 1
        assert starts[0]["name"] == "calculator"
        assert starts[0]["input"] == "2 + 2"

        assert len(ends) == 1
        assert ends[0]["status"] == "ok"
        assert ends[0]["output"] == "4"

    def test_tool_start_error(self, trace_file: Path):
        cb = TraqoCallback()
        run_id = uuid4()

        with Tracer(path=trace_file):
            cb.on_tool_start(
                serialized={"name": "web_search"},
                input_str="query",
                run_id=run_id,
            )
            cb.on_tool_error(error=TimeoutError("search timed out"), run_id=run_id)

        events = read_events(trace_file)
        ends = [
            e for e in events if e["type"] == "span_end" and e.get("kind") == "tool"
        ]
        assert len(ends) == 1
        assert ends[0]["status"] == "error"
        assert ends[0]["error"]["type"] == "TimeoutError"
        assert "search timed out" in ends[0]["error"]["message"]


# ---------------------------------------------------------------------------
# TracedChatModel wrapper
# ---------------------------------------------------------------------------


class _MockChatModel(BaseChatModel):
    """Minimal BaseChatModel for testing TracedChatModel."""

    model_name: str = "mock-model"

    @property
    def _llm_type(self) -> str:
        return "mock"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        msg = AIMessage(
            content="Mock response",
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )
        gen = ChatGeneration(message=msg)
        return ChatResult(generations=[gen])


class TestTracedChatModel:
    def test_generate_creates_span(self, trace_file: Path):
        mock_model = _MockChatModel()
        traced = TracedChatModel(wrapped=mock_model)

        with Tracer(path=trace_file):
            result = traced._generate([HumanMessage(content="Hi")])

        assert len(result.generations) == 1
        assert result.generations[0].message.content == "Mock response"

        events = read_events(trace_file)
        starts = [e for e in events if e["type"] == "span_start"]
        ends = [e for e in events if e["type"] == "span_end"]
        trace_end = [e for e in events if e["type"] == "trace_end"][0]

        assert len(starts) == 1
        assert starts[0]["kind"] == "llm"
        assert starts[0]["metadata"]["provider"] == "langchain"
        assert starts[0]["metadata"]["model"] == "mock-model"
        assert starts[0]["input"][0]["role"] == "human"

        assert len(ends) == 1
        assert ends[0]["status"] == "ok"
        assert ends[0]["output"] == "Mock response"
        assert ends[0]["metadata"]["token_usage"]["input_tokens"] == 10
        assert ends[0]["metadata"]["token_usage"]["output_tokens"] == 5

        assert trace_end["stats"]["total_input_tokens"] == 10
        assert trace_end["stats"]["total_output_tokens"] == 5

    def test_generate_no_tracer_passthrough(self):
        mock_model = _MockChatModel()
        traced = TracedChatModel(wrapped=mock_model)

        result = traced._generate([HumanMessage(content="Hi")])

        assert len(result.generations) == 1
        assert result.generations[0].message.content == "Mock response"


# ---------------------------------------------------------------------------
# Token parsing and output extraction helpers
# ---------------------------------------------------------------------------


class TestTokenParsing:
    def test_parse_usage_metadata_dict(self):
        meta = {
            "input_tokens": 100,
            "output_tokens": 50,
            "output_token_details": {"reasoning": 20},
            "input_token_details": {"cache_read": 30, "cache_creation": 10},
        }
        usage = _parse_usage_metadata(meta)
        assert usage["input_tokens"] == 100
        assert usage["output_tokens"] == 50
        assert usage["reasoning_tokens"] == 20
        assert usage["cache_read_tokens"] == 30
        assert usage["cache_creation_tokens"] == 10

    def test_parse_usage_metadata_object(self):
        meta = MagicMock()
        meta.input_tokens = 20
        meta.output_tokens = 15
        detail_out = MagicMock()
        detail_out.reasoning = 5
        meta.output_token_details = detail_out
        detail_in = MagicMock()
        detail_in.cache_read = 3
        detail_in.cache_creation = 0  # 0 is falsy — should not be included
        meta.input_token_details = detail_in

        usage = _parse_usage_metadata(meta)
        assert usage["input_tokens"] == 20
        assert usage["output_tokens"] == 15
        assert usage["reasoning_tokens"] == 5
        assert usage["cache_read_tokens"] == 3
        assert "cache_creation_tokens" not in usage

    def test_extract_output_text(self):
        msg = AIMessage(content="Hello world")
        gen = ChatGeneration(message=msg)
        result = ChatResult(generations=[gen])
        output = _extract_output(result)
        assert output == "Hello world"

    def test_extract_output_tool_calls(self):
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "search", "args": {"q": "test"}, "id": "tc1"}],
        )
        gen = ChatGeneration(message=msg)
        result = ChatResult(generations=[gen])
        output = _extract_output(result)
        assert output == [{"name": "search", "args": {"q": "test"}}]

    def test_extract_output_structured_content(self):
        msg = AIMessage(
            content=[
                {"type": "reasoning", "summary": [{"text": "thinking..."}]},
                {"type": "text", "text": "Hello"},
            ]
        )
        gen = ChatGeneration(message=msg)
        result = ChatResult(generations=[gen])
        output = _extract_output(result)
        assert output == {"reasoning": "thinking...", "text": "Hello"}


# ---------------------------------------------------------------------------
# LangGraph interrupt handling
# ---------------------------------------------------------------------------


class GraphInterrupt(Exception):
    """Fake LangGraph GraphInterrupt for testing."""

    pass


class NodeInterrupt(Exception):
    """Fake LangGraph NodeInterrupt for testing."""

    pass


class TestInterruptDetection:
    def test_detects_graph_interrupt(self):
        assert _is_langgraph_interrupt(GraphInterrupt("paused"))

    def test_detects_node_interrupt(self):
        assert _is_langgraph_interrupt(NodeInterrupt("waiting"))

    def test_ignores_regular_errors(self):
        assert not _is_langgraph_interrupt(ValueError("oops"))
        assert not _is_langgraph_interrupt(RuntimeError("fail"))

    def test_interrupt_value_from_args(self):
        err = GraphInterrupt("please approve")
        assert _interrupt_value(err) == "please approve"

    def test_interrupt_value_from_value_attr(self):
        err = GraphInterrupt()
        err.value = {"question": "Continue?"}
        assert _interrupt_value(err) == {"question": "Continue?"}

    def test_interrupt_value_none(self):
        err = GraphInterrupt()
        assert _interrupt_value(err) is None


class TestChainErrorInterruptHandling:
    def test_graph_interrupt_recorded_as_interrupted(self, trace_file: Path):
        cb = TraqoCallback()
        run_id = uuid4()

        with Tracer(path=trace_file):
            cb.on_chain_start(
                serialized={"id": ["langgraph", "graph", "CompiledGraph"]},
                inputs={"messages": [{"role": "user", "content": "hi"}]},
                run_id=run_id,
            )
            cb.on_chain_error(
                error=GraphInterrupt("human approval needed"),
                run_id=run_id,
            )

        events = read_events(trace_file)
        ends = [
            e for e in events if e["type"] == "span_end" and e.get("kind") == "chain"
        ]
        assert len(ends) == 1
        assert ends[0]["status"] == "interrupted"
        assert ends[0]["output"] == "human approval needed"
        assert "error" not in ends[0]

    def test_node_interrupt_recorded_as_interrupted(self, trace_file: Path):
        cb = TraqoCallback()
        run_id = uuid4()

        with Tracer(path=trace_file):
            cb.on_chain_start(
                serialized={"id": ["langgraph", "graph", "CompiledGraph"]},
                inputs={},
                run_id=run_id,
            )
            err = NodeInterrupt()
            err.value = {"question": "Approve this action?", "options": ["yes", "no"]}
            cb.on_chain_error(error=err, run_id=run_id)

        events = read_events(trace_file)
        ends = [
            e for e in events if e["type"] == "span_end" and e.get("kind") == "chain"
        ]
        assert len(ends) == 1
        assert ends[0]["status"] == "interrupted"
        assert ends[0]["output"]["question"] == "Approve this action?"

    def test_interrupt_no_error_count(self, trace_file: Path):
        """Interrupts should not increment the error counter."""
        cb = TraqoCallback()
        run_id = uuid4()

        with Tracer(path=trace_file) as t:
            cb.on_chain_start(
                serialized={"id": ["CompiledGraph"]},
                inputs={},
                run_id=run_id,
            )
            cb.on_chain_error(error=GraphInterrupt("pause"), run_id=run_id)
            assert t._stats_errors == 0
            assert t._stats_spans == 1

    def test_interrupt_capture_content_false(self, trace_file: Path):
        """Interrupt payload should be omitted when capture_content=False."""
        cb = TraqoCallback()
        run_id = uuid4()

        with Tracer(path=trace_file, capture_content=False):
            cb.on_chain_start(
                serialized={"id": ["CompiledGraph"]},
                inputs={},
                run_id=run_id,
            )
            cb.on_chain_error(
                error=GraphInterrupt("secret approval data"),
                run_id=run_id,
            )

        events = read_events(trace_file)
        ends = [
            e for e in events if e["type"] == "span_end" and e.get("kind") == "chain"
        ]
        assert ends[0]["status"] == "interrupted"
        assert "output" not in ends[0]

    def test_regular_error_still_works(self, trace_file: Path):
        """Non-interrupt errors should still be recorded normally."""
        cb = TraqoCallback()
        run_id = uuid4()

        with Tracer(path=trace_file) as t:
            cb.on_chain_start(
                serialized={"id": ["LLMChain"]},
                inputs={},
                run_id=run_id,
            )
            cb.on_chain_error(error=ValueError("real error"), run_id=run_id)
            assert t._stats_errors == 1

        events = read_events(trace_file)
        ends = [
            e for e in events if e["type"] == "span_end" and e.get("kind") == "chain"
        ]
        assert ends[0]["status"] == "error"
        assert ends[0]["error"]["type"] == "ValueError"


# ---------------------------------------------------------------------------
# track_langgraph() helper
# ---------------------------------------------------------------------------


class _FakeGraph:
    """Minimal mock of a compiled LangGraph for testing track_langgraph."""

    def __init__(self):
        self.last_config = None

    def invoke(self, input: Any, config: dict | None = None, **kwargs: Any) -> dict:
        self.last_config = config
        return {"result": "ok"}

    async def ainvoke(
        self, input: Any, config: dict | None = None, **kwargs: Any
    ) -> dict:
        self.last_config = config
        return {"result": "ok"}

    def stream(self, input: Any, config: dict | None = None, **kwargs: Any):
        self.last_config = config
        yield {"chunk": 1}
        yield {"chunk": 2}

    async def astream(self, input: Any, config: dict | None = None, **kwargs: Any):
        self.last_config = config
        yield {"chunk": 1}
        yield {"chunk": 2}


class TestTrackLanggraph:
    def test_injects_callback_on_invoke(self):
        graph = _FakeGraph()
        cb = TraqoCallback()
        track_langgraph(graph, callback=cb)

        result = graph.invoke({"messages": []})
        assert result == {"result": "ok"}
        assert cb in graph.last_config["callbacks"]

    def test_creates_callback_if_none(self):
        graph = _FakeGraph()
        track_langgraph(graph)

        graph.invoke({"messages": []})
        cbs = graph.last_config["callbacks"]
        assert len(cbs) == 1
        assert isinstance(cbs[0], TraqoCallback)

    def test_preserves_existing_callbacks(self):
        graph = _FakeGraph()
        cb = TraqoCallback()
        existing_cb = MagicMock()

        track_langgraph(graph, callback=cb)

        graph.invoke({}, config={"callbacks": [existing_cb]})
        cbs = graph.last_config["callbacks"]
        assert existing_cb in cbs
        assert cb in cbs

    def test_no_duplicate_callback(self):
        graph = _FakeGraph()
        cb = TraqoCallback()
        track_langgraph(graph, callback=cb)

        # Pass the same callback in config — should not duplicate
        graph.invoke({}, config={"callbacks": [cb]})
        cbs = graph.last_config["callbacks"]
        assert cbs.count(cb) == 1

    def test_stream_injects_callback(self):
        graph = _FakeGraph()
        cb = TraqoCallback()
        track_langgraph(graph, callback=cb)

        chunks = list(graph.stream({"messages": []}))
        assert chunks == [{"chunk": 1}, {"chunk": 2}]
        assert cb in graph.last_config["callbacks"]

    @pytest.mark.asyncio
    async def test_ainvoke_injects_callback(self):
        graph = _FakeGraph()
        cb = TraqoCallback()
        track_langgraph(graph, callback=cb)

        result = await graph.ainvoke({"messages": []})
        assert result == {"result": "ok"}
        assert cb in graph.last_config["callbacks"]

    @pytest.mark.asyncio
    async def test_astream_injects_callback(self):
        graph = _FakeGraph()
        cb = TraqoCallback()
        track_langgraph(graph, callback=cb)

        chunks = [c async for c in graph.astream({"messages": []})]
        assert chunks == [{"chunk": 1}, {"chunk": 2}]
        assert cb in graph.last_config["callbacks"]

    def test_returns_graph_for_chaining(self):
        graph = _FakeGraph()
        result = track_langgraph(graph)
        assert result is graph
