"""Launch Verinode in demo mode with the pre-built demo graph."""
import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `src.*` imports work
# regardless of where this script is invoked from.
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

os.environ.setdefault("VERINODE_GRAPH", "data/demo_graph.json")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=True)
