"""Example: tracing Claude Agent SDK sessions with traqo.

This example shows how to use ``traqo_agent`` to automatically convert
Claude Agent SDK transcripts into traqo trace files — both standalone
and nested inside a parent pipeline trace.

Requirements:
    pip install claude-agent-sdk traqo

Usage:
    python examples/claude_agent_sdk_tracing.py
"""

from __future__ import annotations

import asyncio

# from claude_agent_sdk import query, ClaudeAgentOptions
from traqo import Tracer
from traqo.integrations.claude_agent_sdk import traqo_agent


async def standalone_example() -> None:
    """Standalone: trace a single Agent SDK session."""
    async with traqo_agent(
        "code-review", output_dir="./traces", tags=["review"]
    ) as hooks:
        # Pass hooks to the Agent SDK:
        #
        #   async for msg in query(
        #       prompt="Review this PR for security issues",
        #       options=ClaudeAgentOptions(hooks=hooks),
        #   ):
        #       print(msg)
        #
        # The Stop hook fires automatically when the agent finishes,
        # converting the transcript to a traqo trace in ./traces/.
        print(f"hooks ready: {list(hooks.keys())}")


async def nested_pipeline_example() -> None:
    """Nested: multiple Agent SDK calls inside a parent pipeline trace."""
    with Tracer("ci-pipeline", trace_dir="./traces", tags=["ci"]) as tracer:
        # Step 1: code review agent
        async with traqo_agent("code-review", tags=["review"]) as hooks:
            # async for msg in query(
            #     prompt="Review the latest commit",
            #     options=ClaudeAgentOptions(hooks=hooks),
            # ):
            #     pass
            print(f"code-review hooks: {list(hooks.keys())}")

        # Step 2: test generation agent
        async with traqo_agent("test-gen", tags=["testing"]) as hooks:
            # async for msg in query(
            #     prompt="Generate tests for the changed files",
            #     options=ClaudeAgentOptions(hooks=hooks),
            # ):
            #     pass
            print(f"test-gen hooks: {list(hooks.keys())}")

        # The parent trace automatically rolls up token usage and span
        # counts from both child agents in trace_end.stats and .children.
        tracer.log("pipeline_complete", {"agents": ["code-review", "test-gen"]})


async def main() -> None:
    print("=== Standalone example ===")
    await standalone_example()

    print("\n=== Nested pipeline example ===")
    await nested_pipeline_example()


if __name__ == "__main__":
    asyncio.run(main())
