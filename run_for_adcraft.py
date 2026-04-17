"""
MAdVerse Server Launcher for AdCraft Integration
Runs on port 8001 to work alongside AdCraft_AI frontend
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

    # Default to 8001 for integration with AdCraft_AI (which uses 5173 for frontend)
    port = int(os.environ.get("PORT", 8001))

    print()
    print("=" * 60)
    print("  MAdVerse AI Backend - AdCraft Integration Mode")
    print("=" * 60)
    print()
    print(f"  API URL:   http://localhost:{port}")
    print(f"  API docs:  http://localhost:{port}/docs")
    print(f"  Health:    http://localhost:{port}/api/health")
    print()
    print("  AdCraft_AI Frontend: http://localhost:5173")
    print()
    print("  Press Ctrl+C to stop the server")
    print("=" * 60)
    print()

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=True,  # Enable hot reload for development
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    main()
