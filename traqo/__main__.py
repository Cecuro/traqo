"""Allow running as: python -m traqo <command>"""

from __future__ import annotations

import sys


def main() -> None:
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print("Usage: python -m traqo <command>")
        print()
        print("Commands:")
        print("  ui [TRACES_DIR] [--port PORT]   Start the trace viewer")
        print("  cc-sync [OPTIONS]               Sync Claude Code sessions to traces")
        sys.exit(0)

    command = args[0]

    if command == "ui":
        sys.argv = sys.argv[1:]  # shift so argparse in server.py sees the right args
        from traqo.ui.server import main as ui_main

        ui_main()
    elif command == "cc-sync":
        sys.argv = sys.argv[1:]
        from traqo.cc_sync import main as cc_sync_main

        cc_sync_main()
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        print("Run 'python -m traqo --help' for usage.", file=sys.stderr)
        sys.exit(1)


main()
