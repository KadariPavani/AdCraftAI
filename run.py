"""
AdCraft AI - Server Launcher
Run this to start the web application.

Usage:
    python run.py

Then open http://localhost:8000 in your browser.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    try:
        import uvicorn
    except ImportError:
        print("Installing required packages...")
        os.system(f"{sys.executable} -m pip install fastapi uvicorn[standard] python-multipart deep-translator")
        import uvicorn

    print()
    print("=" * 55)
    print("  AdCraft AI - Ad Generation Platform")
    print("=" * 55)
    print()
    print("  Open in browser: http://localhost:8000")
    print("  API docs:        http://localhost:8000/docs")
    print()
    print("  Press Ctrl+C to stop the server")
    print("=" * 55)
    print()

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
