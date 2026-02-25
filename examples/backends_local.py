"""
Example: Storage backends — collect traces into a central directory.

Shows:
  - LocalBackend copies completed trace files to a target directory
  - Collision-safe filenames (timestamp + hex suffix)
  - organize_by_date groups files into YYYY-MM-DD subdirectories
  - S3/GCS backends work the same way (just swap the backend)

Run:
    uv run python examples/backends_local.py

Then inspect examples/traces/collected/ — copies are organized by date.

For cloud backends (same pattern, different backend):
    from traqo.backends.s3 import S3Backend
    backends = [S3Backend("my-bucket", prefix="traces/")]

    from traqo.backends.gcs import GCSBackend
    backends = [GCSBackend("my-bucket", prefix="traces/")]
"""

import os
import time

from traqo import Tracer
from traqo.backends.local import LocalBackend


def main():
    traces_dir = os.path.join(os.path.dirname(__file__), "traces")
    collected_dir = os.path.join(traces_dir, "collected")

    backend = LocalBackend(collected_dir, organize_by_date=True)

    with Tracer(
        "run",
        trace_dir=traces_dir,
        input={"query": "demo"},
        backends=[backend],
    ) as tracer:
        with tracer.span("step_1", kind="tool") as span:
            time.sleep(0.02)
            span.set_output({"status": "ok"})

        with tracer.span("step_2", kind="llm") as span:
            time.sleep(0.02)
            span.set_output("Generated answer.")
            span.set_metadata("token_usage", {"input_tokens": 10, "output_tokens": 5})

        tracer.set_output({"answer": "Generated answer."})

    # Show what was collected
    print("Original trace:", tracer._path)
    print()
    print("Collected copies:")
    for root, _, files in os.walk(collected_dir):
        for f in sorted(files):
            path = os.path.join(root, f)
            size = os.path.getsize(path)
            print(f"  {os.path.relpath(path, collected_dir)} ({size} bytes)")


if __name__ == "__main__":
    main()
