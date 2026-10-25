"""Entry point for the Verinode API server.

Usage:
    verinode-server              # after pip install
    python -m src.api.server   # from source
"""

import uvicorn


def main():
    """Launch the Verinode web server."""
    uvicorn.run(
        "src.api.main:app",
        host="127.0.0.1",
        port=8000,
    )


if __name__ == "__main__":
    main()
