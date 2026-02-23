"""Tests for LangChain callback handler — retriever, chain, and agent callbacks."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError:
    pytest.skip("langchain-core not installed", allow_module_level=True)

from traqo import Tracer
from traqo.integrations.langchain import TraqoCallback
from tests.conftest import read_events


class TestRetrieverCallbacks:
    def test_retriever_start_end(self, trace_file: Path):
        cb = TraqoCallback()
        run_id = uuid4()

        with Tracer(trace_file):
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
        starts = [e for e in events if e["type"] == "span_start" and e.get("kind") == "retriever"]
        ends = [e for e in events if e["type"] == "span_end" and e.get("kind") == "retriever"]

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

        with Tracer(trace_file):
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
        ends = [e for e in events if e["type"] == "span_end" and e.get("kind") == "retriever"]
        assert len(ends) == 1
        assert ends[0]["status"] == "error"
        assert ends[0]["error"]["type"] == "ConnectionError"
        assert "index not found" in ends[0]["error"]["message"]

    def test_retriever_name_fallback(self, trace_file: Path):
        cb = TraqoCallback()
        run_id = uuid4()

        with Tracer(trace_file):
            cb.on_retriever_start(
                serialized={"name": "my_retriever"},
                query="test",
                run_id=run_id,
            )
            cb.on_retriever_end(documents=[], run_id=run_id)

        events = read_events(trace_file)
        starts = [e for e in events if e["type"] == "span_start" and e.get("kind") == "retriever"]
        assert starts[0]["name"] == "my_retriever"

    def test_retriever_parent_nesting(self, trace_file: Path):
        cb = TraqoCallback()
        chain_id = uuid4()
        retriever_id = uuid4()

        with Tracer(trace_file):
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
            cb.on_retriever_end(documents=[], run_id=retriever_id, parent_run_id=chain_id)
            cb.on_chain_end(outputs={"result": "answer"}, run_id=chain_id)

        events = read_events(trace_file)
        chain_start = [e for e in events if e["type"] == "span_start" and e.get("kind") == "chain"][0]
        retriever_start = [e for e in events if e["type"] == "span_start" and e.get("kind") == "retriever"][0]
        assert retriever_start["parent_id"] == chain_start["id"]


class TestChainCallbacks:
    def test_chain_start_end(self, trace_file: Path):
        cb = TraqoCallback()
        run_id = uuid4()

        with Tracer(trace_file):
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
        starts = [e for e in events if e["type"] == "span_start" and e.get("kind") == "chain"]
        ends = [e for e in events if e["type"] == "span_end" and e.get("kind") == "chain"]

        assert len(starts) == 1
        assert starts[0]["name"] == "LLMChain"
        assert starts[0]["input"] == {"text": "hello world"}

        assert len(ends) == 1
        assert ends[0]["status"] == "ok"
        assert ends[0]["output"] == {"result": "processed hello world"}

    def test_chain_error(self, trace_file: Path):
        cb = TraqoCallback()
        run_id = uuid4()

        with Tracer(trace_file):
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
        ends = [e for e in events if e["type"] == "span_end" and e.get("kind") == "chain"]
        assert len(ends) == 1
        assert ends[0]["status"] == "error"
        assert ends[0]["error"]["type"] == "ValueError"

    def test_chain_name_from_name_key(self, trace_file: Path):
        cb = TraqoCallback()
        run_id = uuid4()

        with Tracer(trace_file):
            cb.on_chain_start(
                serialized={"name": "CustomChain"},
                inputs={},
                run_id=run_id,
            )
            cb.on_chain_end(outputs={}, run_id=run_id)

        events = read_events(trace_file)
        starts = [e for e in events if e["type"] == "span_start" and e.get("kind") == "chain"]
        assert starts[0]["name"] == "CustomChain"

    def test_nested_chains(self, trace_file: Path):
        cb = TraqoCallback()
        outer_id = uuid4()
        inner_id = uuid4()

        with Tracer(trace_file):
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
        outer_start = [e for e in events if e["type"] == "span_start" and e["name"] == "SequentialChain"][0]
        inner_start = [e for e in events if e["type"] == "span_start" and e["name"] == "LLMChain"][0]
        assert inner_start["parent_id"] == outer_start["id"]

    def test_capture_content_false(self, trace_file: Path):
        cb = TraqoCallback()
        run_id = uuid4()

        with Tracer(trace_file, capture_content=False):
            cb.on_chain_start(
                serialized={"id": ["langchain", "chains", "LLMChain"]},
                inputs={"secret": "data"},
                run_id=run_id,
            )
            cb.on_chain_end(outputs={"result": "sensitive"}, run_id=run_id)

        events = read_events(trace_file)
        starts = [e for e in events if e["type"] == "span_start" and e.get("kind") == "chain"]
        ends = [e for e in events if e["type"] == "span_end" and e.get("kind") == "chain"]
        assert "input" not in starts[0]
        assert "output" not in ends[0]


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

        with Tracer(trace_file):
            cb.on_agent_action(action=FakeAction(), run_id=run_id)
            cb.on_agent_finish(finish=FakeFinish(), run_id=run_id)

        events = read_events(trace_file)
        starts = [e for e in events if e["type"] == "span_start" and e.get("kind") == "agent"]
        ends = [e for e in events if e["type"] == "span_end" and e.get("kind") == "agent"]

        assert len(starts) == 1
        assert starts[0]["name"] == "search"
        assert starts[0]["input"]["tool"] == "search"
        assert starts[0]["input"]["tool_input"] == "what is traqo?"

        assert len(ends) == 1
        assert ends[0]["status"] == "ok"
        assert ends[0]["output"]["return_values"] == {"output": "traqo is a tracing library"}
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
        cb.on_agent_finish(finish=type("F", (), {"return_values": {}, "log": ""})(), run_id=run_id)


class TestSpanCounting:
    def test_retriever_increments_span_count(self, trace_file: Path):
        cb = TraqoCallback()
        run_id = uuid4()

        with Tracer(trace_file) as t:
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

        with Tracer(trace_file) as t:
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

        with Tracer(trace_file) as t:
            cb.on_agent_action(action=FakeAction(), run_id=run_id)
            cb.on_agent_finish(finish=type("F", (), {"return_values": {}, "log": ""})(), run_id=run_id)
            assert t._stats_spans == 1

    def test_error_increments_error_count(self, trace_file: Path):
        cb = TraqoCallback()

        with Tracer(trace_file) as t:
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
