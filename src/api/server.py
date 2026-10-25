"""Entry point for the Engram API server.

Usage:
    engram-server              # after pip install
    python -m src.api.server   # from source
"""

import uvicorn


def main():
    """Launch the Engram web server."""
    uvicorn.run(
        "src.api.main:app",
        host="127.0.0.1",
        port=8000,
    )


if __name__ == "__main__":
    main()
