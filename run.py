"""
Unified MAdVerse launcher.

Modes:
- local:       MAdVerse standalone on port 8000
- integration: MAdVerse API for AdCraft_AI on port 8001
- deploy:      Hugging Face/production mode on $PORT (default 7860)
"""

import argparse
import os
import sys
from typing import Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _resolve_defaults(mode: str) -> Tuple[int, bool, bool]:
    if mode == "integration":
        return int(os.environ.get("PORT", 8001)), _bool_env("RELOAD", True), _bool_env("ACCESS_LOG", True)
    if mode == "deploy":
        return int(os.environ.get("PORT", 7860)), _bool_env("RELOAD", False), _bool_env("ACCESS_LOG", True)
    return int(os.environ.get("PORT", 8000)), _bool_env("RELOAD", False), _bool_env("ACCESS_LOG", False)


def _print_banner(mode: str, port: int) -> None:
    title = {
        "local": "MAdVerse AI - Local Mode",
        "integration": "MAdVerse AI - AdCraft Integration Mode",
        "deploy": "MAdVerse AI - Deployment Mode",
    }[mode]

    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print()
    print(f"  API URL:   http://localhost:{port}")
    print(f"  API docs:  http://localhost:{port}/docs")
    print(f"  Health:    http://localhost:{port}/api/health")
    if mode == "integration":
        print("  AdCraft_AI Frontend: http://localhost:5173")
    print()
    print("  Press Ctrl+C to stop the server")
    print("=" * 70)
    print()


def main(argv=None):
    try:
        import uvicorn
    except ImportError:
        print("Installing required packages...")
        os.system(f"{sys.executable} -m pip install fastapi uvicorn[standard] python-multipart deep-translator")
        import uvicorn

    parser = argparse.ArgumentParser(description="Run MAdVerse server")
    parser.add_argument(
        "--mode",
        choices=["local", "integration", "deploy"],
        default=os.environ.get("MADVERSE_MODE", "local"),
        help="Run mode (default: local)",
    )
    parser.add_argument("--port", type=int, default=None, help="Port override")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    args = parser.parse_args(argv)

    default_port, default_reload, default_access_log = _resolve_defaults(args.mode)
    port = args.port if args.port is not None else default_port

    reload_enabled = default_reload
    if args.reload:
        reload_enabled = True
    if args.no_reload:
        reload_enabled = False

    _print_banner(args.mode, port)

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=reload_enabled,
        log_level="info",
        access_log=default_access_log,
    )


if __name__ == "__main__":
    main()
