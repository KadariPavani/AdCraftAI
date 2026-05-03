"""
Compatibility launcher for AdCraft_AI integration.

Use:
    python run_for_adcraft.py

This runs the unified launcher in integration mode (port 8001 by default).
"""

from run import main


if __name__ == "__main__":
    main(["--mode", "integration"])
