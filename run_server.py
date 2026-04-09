#!/usr/bin/env python3
"""
LeanAI Web Server
Starts the FastAPI server with the web UI.

Usage:
    python run_server.py              # default: localhost:8000
    python run_server.py --port 9000  # custom port
    python run_server.py --host 0.0.0.0  # accessible from network
"""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description="LeanAI Web Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn not installed. Run:")
        print("  pip install uvicorn fastapi")
        sys.exit(1)

    print(f"""
╔══════════════════════════════════════════════════════════╗
║                    LeanAI Web Server                     ║
║   Phase 5b: REST API + Web UI                            ║
╚══════════════════════════════════════════════════════════╝

  Web UI:   http://{args.host}:{args.port}/
  API Docs: http://{args.host}:{args.port}/docs
  Status:   http://{args.host}:{args.port}/status
""")

    uvicorn.run(
        "api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
