"""
FastAPI application for Engram AI.

Run with:
    python run_server.py
    # or
    uvicorn src.api.main:app --reload
"""

import os
from contextlib import asynccontextmanager
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .state import AppState
import src.api.state as state_mod
from .routes import graph, nodes, ingest, query, verification, memory, maintenance


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load graph. Shutdown: close connections."""
    graph_path = os.environ.get("ENGRAM_GRAPH", "engram_graph.db")
    model = os.environ.get("ENGRAM_MODEL", "qwen3:14b")

    state_mod.app_state = AppState(
        graph_path=graph_path,
        model=model,
    )
    state_mod.app_state.load_graph()

    print(f"[Engram] Loaded graph from: {graph_path}")
    print(f"[Engram] Graph has {len(state_mod.app_state.graph)} nodes")
    print(f"[Engram] Model: {model}")

    yield

    # Shutdown
    if hasattr(state_mod.app_state.graph, "close"):
        state_mod.app_state.graph.close()
    print("[Engram] Shutdown complete.")


try:
    _version = version("engram-ai")
except PackageNotFoundError:
    _version = "0.1.0"

app = FastAPI(
    title="Engram AI",
    description="Knowledge graph with confidence tracking, provenance, and truth verification",
    version=_version,
    lifespan=lifespan,
)

# CORS for local React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server
        "http://localhost:3000",   # Alternative React port
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes
app.include_router(graph.router, prefix="/api")
app.include_router(nodes.router, prefix="/api")
app.include_router(ingest.router, prefix="/api")
app.include_router(query.router, prefix="/api")
app.include_router(verification.router, prefix="/api")
app.include_router(memory.router, prefix="/api")
app.include_router(maintenance.router, prefix="/api")


@app.get("/api/health")
def health():
    """Health check endpoint."""
    st = state_mod.app_state
    return {
        "status": "ok",
        "graph_nodes": len(st.graph) if st else 0,
        "model": st.model if st else None,
    }


# Serve React build in production (if dist/ exists)
# StaticFiles handles actual files (JS, CSS, images).
# The catch-all route below handles SPA client-side routing by
# serving index.html for any non-API path that isn't a real file.
web_dist = Path(__file__).parent.parent / "web" / "dist"
if web_dist.is_dir():
    # Mount static assets (JS/CSS bundles, favicon, etc.)
    app.mount("/assets", StaticFiles(directory=str(web_dist / "assets")), name="static-assets")

    @app.get("/{full_path:path}")
    async def spa_fallback(request: Request, full_path: str):
        """Serve index.html for any non-API route (SPA client-side routing)."""
        # Try to serve the exact file if it exists (e.g. favicon.ico, robots.txt)
        file_path = web_dist / full_path
        if full_path and file_path.is_file():
            return FileResponse(file_path)
        # Otherwise return index.html and let React Router handle the path
        return FileResponse(web_dist / "index.html")
