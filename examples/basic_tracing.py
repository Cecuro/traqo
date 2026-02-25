"""
Example: Basic traqo tracing — no API keys needed.

Shows core concepts:
  - Trace lifecycle (trace_start → spans → trace_end)
  - Nested spans with parent-child relationships
  - Metadata, tags, and kind fields
  - Token usage accumulation
  - Trace-level input/output

Run:
    uv run python examples/basic_tracing.py

Then inspect the generated file in examples/traces/ — each line is one event.
"""

import json
import os
import time

from traqo import Tracer


def main():
    trace_dir = os.path.join(os.path.dirname(__file__), "traces")

    with Tracer(
        "basic",
        trace_dir=trace_dir,
        input={"query": "What is Python?"},
        tags=["example", "basic"],
        thread_id="demo-001",
    ) as tracer:
        # Span 1: a "tool" span grouping related work
        with tracer.span(
            "search",
            input={"query": "Python programming language"},
            kind="tool",
            tags=["retrieval"],
        ) as search:
            time.sleep(0.05)  # simulate work
            search.set_output({"results": 3})
            search.set_metadata("source", "wikipedia")

        # Span 2: an "llm" span with token usage
        with tracer.span(
            "generate_answer",
            input=[{"role": "user", "content": "What is Python?"}],
            kind="llm",
            metadata={"provider": "openai", "model": "gpt-4o"},
        ) as llm:
            time.sleep(0.05)  # simulate LLM call
            llm.set_output("Python is a high-level programming language.")
            llm.set_metadata("token_usage", {"input_tokens": 12, "output_tokens": 8})

        # Point-in-time log event
        tracer.log("pipeline_complete", {"steps": 2})

        tracer.set_output({"answer": "Python is a high-level programming language."})

    # Print results
    trace_path = tracer._path
    print("Trace written to:", trace_path)
    print()
    with open(trace_path) as f:
        for line in f:
            if line.strip():
                event = json.loads(line)
                print(json.dumps(event, indent=2))
                print()


if __name__ == "__main__":
    main()
