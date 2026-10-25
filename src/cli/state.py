"""
Engine state management for the Verinode CLI.

Handles graph loading/saving and lazy initialisation of
heavyweight components (embedding engine, pipeline, etc.)
so that read-only commands start instantly without loading
Ollama or sentence-transformers.

Supports both JSON (``.json``) and SQLite (``.db``) graph files.
The file extension determines which backend is used automatically.
"""

import sys
from pathlib import Path
from typing import Optional

import click

# Ensure ``src/`` is on the import path when running as ``python -m src.cli``
_src_root = str(Path(__file__).resolve().parent.parent.parent)
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)

from src.core.graph import KnowledgeGraph
from src.core.confidence import ConfidenceEngine
from src.core.reasoning import ReasoningEngine
from src.core.neuroplasticity import NeuroplasticityEngine


def _is_sqlite_path(path: str) -> bool:
    """Return True if *path* looks like a SQLite database file."""
    return Path(path).suffix.lower() in (".db", ".sqlite", ".sqlite3")


class EngineState:
    """
    Shared state across CLI commands.

    Heavy components (``EmbeddingEngine``, ``Pipeline``) are created
    lazily on first access so that read-only graph queries
    (``stats``, ``list``, ``inspect``, ``explain``) do not need
    Ollama or sentence-transformers.

    Supports both JSON (``.json``) and SQLite (``.db``) graph backends.
    The backend is auto-detected from the file extension.  With SQLite,
    mutations are persisted incrementally so explicit ``save_graph()``
    calls are no-ops (unless saving to a *different* path/format).
    """

    def __init__(self, graph_path: str, model: str):
        self.graph_path: str = graph_path
        self.model: str = model
        self.modified: bool = False
        self.is_sqlite: bool = _is_sqlite_path(graph_path)

        # Always-available (lightweight)
        self.graph: KnowledgeGraph = KnowledgeGraph()
        self.confidence_engine: ConfidenceEngine = ConfidenceEngine()
        self.reasoning_engine: ReasoningEngine = ReasoningEngine()
        self.neuroplasticity_engine: NeuroplasticityEngine = NeuroplasticityEngine(
            confidence_engine=self.confidence_engine,
        )

        # Lazy (heavyweight)
        self._embedding_engine = None
        self._pipeline = None
        self._contradiction_checker = None
        self._verification_engine = None
        self._llm_memory = None

    # ------------------------------------------------------------------
    # Graph persistence
    # ------------------------------------------------------------------

    def load_graph(self, path: Optional[str] = None) -> None:
        """Load graph from *path* (or ``self.graph_path``).

        Auto-detects JSON vs SQLite from the file extension.
        """
        p = path or self.graph_path
        if not Path(p).exists():
            return

        if _is_sqlite_path(p):
            from src.core.storage import SqliteGraph

            self.graph = SqliteGraph(p)
            self.is_sqlite = True
        else:
            self.graph = KnowledgeGraph.load(p)
            self.is_sqlite = False

        self.graph_path = p
        self.modified = False

    def save_graph(self, path: Optional[str] = None) -> None:
        """Save graph to *path* (or ``self.graph_path``).

        With SQLite graphs changes are already persisted, so this is
        a no-op when *path* matches the current graph path.  If saving
        to a different path or format, the appropriate conversion is
        performed.
        """
        p = path or self.graph_path
        target_is_sqlite = _is_sqlite_path(p)

        # If the graph is already a SqliteGraph pointing at the same file,
        # there is nothing to do — changes are auto-persisted.
        if self.is_sqlite and target_is_sqlite and Path(p).resolve() == Path(self.graph_path).resolve():
            self.modified = False
            return

        # Cross-format or different-file save
        if target_is_sqlite:
            from src.core.storage import SqliteGraph

            SqliteGraph.from_graph(self.graph, p)
        else:
            Path(p).parent.mkdir(parents=True, exist_ok=True)
            self.graph.save(p)
        self.modified = False

    # ------------------------------------------------------------------
    # Lazy component access
    # ------------------------------------------------------------------

    @property
    def embedding_engine(self):
        """Return (or lazily create) the ``EmbeddingEngine``."""
        if self._embedding_engine is None:
            from src.core.embeddings import EmbeddingEngine

            self._embedding_engine = EmbeddingEngine()
            # If the graph already has nodes, build the vector index
            if len(self.graph) > 0:
                self._embedding_engine.index_graph(self.graph)
        return self._embedding_engine

    @property
    def pipeline(self):
        """Return (or lazily create) the ``Pipeline``."""
        if self._pipeline is None:
            from src.core.pipeline import Pipeline
            from src.core.fact_extraction import FactExtractionEngine

            try:
                extractor = FactExtractionEngine(model=self.model)
            except Exception as exc:
                raise click.ClickException(
                    f"Cannot create fact extractor (is Ollama running?): {exc}"
                )

            self._pipeline = Pipeline(
                self.graph,
                extractor=extractor,
                embedding_engine=self.embedding_engine,
                confidence_engine=self.confidence_engine,
            )
        return self._pipeline

    @property
    def contradiction_checker(self):
        """Return (or lazily create) the ``ContradictionChecker``."""
        if self._contradiction_checker is None:
            from src.core.llm_contradiction import create_contradiction_checker

            self._contradiction_checker = create_contradiction_checker(
                model=self.model
            )
        return self._contradiction_checker

    @property
    def verification_engine(self):
        """Return (or lazily create) the ``VerificationEngine``."""
        if self._verification_engine is None:
            from src.core.verification import VerificationEngine

            self._verification_engine = VerificationEngine(
                self.graph,
                embedding_engine=self.embedding_engine,
                confidence_engine=self.confidence_engine,
                reasoning_engine=self.reasoning_engine,
                contradiction_checker=self.contradiction_checker,
            )
        return self._verification_engine

    @property
    def llm_memory(self):
        """Return (or lazily create) the ``LLMMemory``."""
        if self._llm_memory is None:
            from src.core.llm_memory import LLMMemory

            self._llm_memory = LLMMemory(
                self.graph,
                embedding_engine=self.embedding_engine,
                reasoning_engine=self.reasoning_engine,
                confidence_engine=self.confidence_engine,
                pipeline=self.pipeline,
                verification_engine=self.verification_engine,
            )
        return self._llm_memory

    def ensure_embeddings_indexed(self) -> None:
        """Rebuild the embedding index if it is empty but the graph has nodes."""
        if len(self.graph) > 0 and self.embedding_engine.index_size == 0:
            self.embedding_engine.index_graph(self.graph)
