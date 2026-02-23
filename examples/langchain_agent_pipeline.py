"""
Example: Multi-step LangChain agent pipeline with traqo tracing.

This shows how traqo captures a realistic agent workflow:
  1. A top-level trace wraps the entire session
  2. A "tool" span groups related work (like an agent step)
  3. Individual LLM calls are traced as nested "llm" spans
  4. Token usage, model info, and content are captured automatically

Run:
    export AZURE_OPENAI_API_KEY=...
    export AZURE_OPENAI_ENDPOINT=...
    uv run python examples/langchain_agent_pipeline.py

Output traces are written to examples/traces/
"""

import json
import os
import sys

from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI

from traqo import Tracer
from traqo.integrations.langchain import traced_model


def create_model(deployment: str = "gpt-5.1") -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_deployment=deployment,
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version="2025-03-01-preview",
    )


def main():
    trace_path = os.path.join(os.path.dirname(__file__), "traces", "agent_pipeline.jsonl")
    user_input = "The server crashed with an OOM error"

    # Wrap the base model — each operation gets its own traced wrapper
    base = create_model("gpt-5.1")
    classify_llm = traced_model(base, operation="classify")
    explain_llm = traced_model(base, operation="explain")

    # --- Trace the full pipeline ---
    with Tracer(
        trace_path,
        input={"text": user_input},
        tags=["example", "azure"],
        thread_id="demo-conversation-001",
    ) as tracer:

        # Agent step: "analyze" groups the two LLM calls
        with tracer.span("analyze", input={"text": user_input}, kind="tool", tags=["pipeline"]) as step:

            # LLM call 1: classify
            classification = classify_llm.invoke(
                [HumanMessage(content=f"Classify in one word (bug/feature/question): {user_input}")]
            ).content.strip().lower()

            # LLM call 2: explain
            explanation = explain_llm.invoke(
                [HumanMessage(content=f"In one sentence, why is this a {classification}: {user_input}")]
            ).content.strip()

            step.set_output({"class": classification, "reason": explanation})
            step.set_metadata("input_length", len(user_input))

        tracer.set_output({"class": classification, "reason": explanation})

    # Print the trace
    print("Trace written to:", trace_path)
    print()
    with open(trace_path) as f:
        for line in f:
            if line.strip():
                print(json.dumps(json.loads(line), indent=2))
                print()


if __name__ == "__main__":
    main()
