"""
Example: LangChain tracing with traqo — two integration patterns.

Pattern 1: TraqoCallback — auto-traces LLM calls and tool use inside agents.
Pattern 2: traced_model() — wraps a model for use in manual pipelines.

Run:
    export AZURE_OPENAI_API_KEY=...
    export AZURE_OPENAI_ENDPOINT=...
    uv run python examples/langchain_tracing.py
"""

import os

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent

from traqo import Tracer, trace
from traqo.integrations.langchain import TraqoCallback, traced_model


def create_model(deployment: str = "gpt-5.1") -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_deployment=deployment,
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version="2025-03-01-preview",
    )


# ── Pattern 1: TraqoCallback auto-traces agents ─────────────────────────


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    weather = {"london": "Rainy, 8°C", "tokyo": "Sunny, 22°C"}
    return weather.get(city.lower(), f"No weather data for {city}")


@tool
def get_population(city: str) -> str:
    """Get the population of a city."""
    populations = {"london": "8.8 million", "tokyo": "13.9 million"}
    return populations.get(city.lower(), f"No population data for {city}")


@trace(kind="tool")
def ask_agent(agent, question: str, callbacks: list) -> str:
    """Decorated with @trace — creates a parent span for the agent run."""
    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"callbacks": callbacks},
    )
    return result["messages"][-1].content


def demo_callback():
    print("=== Pattern 1: TraqoCallback ===")
    trace_path = os.path.join(
        os.path.dirname(__file__), "traces", "agent_callback.jsonl"
    )

    llm = create_model()
    agent = create_react_agent(llm, [get_weather, get_population])
    callback = TraqoCallback()

    with Tracer(trace_path, input={"city": "Tokyo"}, tags=["agent"]):
        answer = ask_agent(
            agent, "What's the weather and population of Tokyo?", [callback]
        )
        print(f"Answer: {answer}")

    print(f"Trace written to: {trace_path}\n")


# ── Pattern 2: traced_model() wraps models for pipelines ────────────────


def demo_traced_model():
    print("=== Pattern 2: traced_model ===")
    trace_path = os.path.join(
        os.path.dirname(__file__), "traces", "agent_pipeline.jsonl"
    )

    base = create_model()
    classify_llm = traced_model(base, operation="classify")
    explain_llm = traced_model(base, operation="explain")

    user_input = "The server crashed with an OOM error"

    with Tracer(trace_path, input={"text": user_input}, tags=["pipeline"]) as tracer:
        with tracer.span("analyze", input={"text": user_input}, kind="tool") as step:
            classification = classify_llm.invoke(
                [HumanMessage(content=f"Classify (bug/feature/question): {user_input}")]
            ).content.strip()
            explanation = explain_llm.invoke(
                [HumanMessage(content=f"Why is this a {classification}: {user_input}")]
            ).content.strip()
            step.set_output({"class": classification, "reason": explanation})

        tracer.set_output({"class": classification, "reason": explanation})

    print(f"Result: {classification} — {explanation}")
    print(f"Trace written to: {trace_path}\n")


if __name__ == "__main__":
    demo_callback()
    demo_traced_model()
