"""
Example: LangGraph ReAct agent with callback tracing and decorators.

Shows how traqo captures agent execution automatically:
  - TraqoCallback auto-traces every LLM call and tool execution
  - @trace decorator traces your own functions
  - No manual span management needed — everything is captured

Run:
    export AZURE_OPENAI_API_KEY=...
    export AZURE_OPENAI_ENDPOINT=...
    uv run python examples/langchain_agent.py

Output traces are written to examples/traces/
"""

import json
import os

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent

from traqo import Tracer, trace
from traqo.integrations.langchain import TraqoCallback


# --- Tools ---

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    weather = {
        "london": "Rainy, 8°C",
        "tokyo": "Sunny, 22°C",
        "new york": "Cloudy, 15°C",
    }
    return weather.get(city.lower(), f"No weather data for {city}")


@tool
def get_population(city: str) -> str:
    """Get the population of a city."""
    populations = {
        "london": "8.8 million",
        "tokyo": "13.9 million",
        "new york": "8.3 million",
    }
    return populations.get(city.lower(), f"No population data for {city}")


# --- Application functions decorated with @trace ---

@trace(kind="tool")
def run_agent(agent, question: str, callbacks: list) -> str:
    """Run the agent — traced automatically via decorator."""
    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"callbacks": callbacks},
    )
    return result["messages"][-1].content


@trace(kind="tool")
def city_report(agent, city: str, callbacks: list) -> dict:
    """Build a city report — calls the agent, traced via decorator."""
    answer = run_agent(agent, f"What's the weather and population of {city}?", callbacks)
    return {"city": city, "report": answer}


def main():
    trace_path = os.path.join(os.path.dirname(__file__), "traces", "agent_callback.jsonl")

    # Setup: raw model + callback (no wrapper needed)
    llm = AzureChatOpenAI(
        azure_deployment="gpt-5.1",
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version="2025-03-01-preview",
    )
    agent = create_react_agent(llm, [get_weather, get_population])
    callback = TraqoCallback()

    # Run inside a trace — decorators and callback handle the rest
    with Tracer(
        trace_path,
        input={"city": "Tokyo"},
        tags=["example", "agent", "callback"],
        thread_id="agent-demo-002",
    ):
        result = city_report(agent, "Tokyo", [callback])
        print(f"Report: {result['report']}")

    # Print the trace
    print("\nTrace written to:", trace_path)
    print()
    with open(trace_path) as f:
        for line in f:
            if line.strip():
                event = json.loads(line)
                etype = event["type"]

                if etype in ("span_start", "span_end"):
                    name = event["name"]
                    kind = event.get("kind", "")
                    parent = event.get("parent_id")
                    extra = ""
                    if etype == "span_end":
                        dur = event.get("duration_s", 0)
                        tokens = event.get("metadata", {}).get("token_usage", {})
                        if tokens:
                            extra = f"  {tokens['input_tokens']}in/{tokens['output_tokens']}out"
                        extra += f"  {dur:.2f}s"
                        status = event.get("status", "")
                        extra += f"  [{status}]"
                    depth = "    " if parent else "  "
                    print(f"{depth}{etype:12s}  {name:20s}  kind={kind:4s}{extra}")

                elif etype == "trace_start":
                    print(f"  {etype:12s}  tags={event.get('tags')}  thread_id={event.get('thread_id')}")

                elif etype == "trace_end":
                    s = event.get("stats", {})
                    print(f"  {etype:12s}  spans={s.get('spans')}  "
                          f"tokens={s.get('total_input_tokens')}in/{s.get('total_output_tokens')}out  "
                          f"errors={s.get('errors')}  {event.get('duration_s', 0):.2f}s")


if __name__ == "__main__":
    main()
