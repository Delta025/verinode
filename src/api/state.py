"""
Shared application state for the FastAPI backend.

Follows the lazy initialization pattern from src/cli/state.py:
lightweight components init eagerly, heavyweight (embeddings, Ollama)
init lazily on first use.
"""

from pathlib import Path
from typing import Optional

from src.core.graph import KnowledgeGraph
from src.core.confidence import ConfidenceEngine
from src.core.reasoning import ReasoningEngine
from src.core.neuroplasticity import NeuroplasticityEngine


class AppState:
    """Application-wide state with lazy component initialization."""

    def __init__(
        self,
        graph_path: str = "verinode_graph.db",
        model: str = "qwen3:14b",
    ):
        self.graph_path = graph_path
        self.model = model

        # Lightweight — always available
        self.graph: KnowledgeGraph = KnowledgeGraph()
        self.confidence_engine = ConfidenceEngine()
        self.reasoning_engine = ReasoningEngine()
        self.neuroplasticity_engine = NeuroplasticityEngine(
            confidence_engine=self.confidence_engine,
        )

        # Heavyweight — lazy
        self._embedding_engine = None
        self._pipeline = None
        self._contradiction_checker = None
        self._reranker = None
        self._verification_engine = None
        self._llm_memory = None

    def load_graph(self):
        """Load graph from disk. Auto-detects JSON vs SQLite by extension."""
        path = Path(self.graph_path)
        if not path.exists():
            return

        if path.suffix in (".db", ".sqlite", ".sqlite3"):
            from src.core.storage import SqliteGraph
            self.graph = SqliteGraph(str(path))
        else:
            self.graph = KnowledgeGraph.load(str(path))

    @property
    def embedding_engine(self):
        if self._embedding_engine is None:
            from src.core.embeddings import EmbeddingEngine
            self._embedding_engine = EmbeddingEngine()
            if len(self.graph) > 0:
                self._embedding_engine.index_graph(self.graph)
        return self._embedding_engine

    @property
    def pipeline(self):
        if self._pipeline is None:
            import os
            from src.core.pipeline import Pipeline
            from src.core.fact_extraction import create_fact_extractor
            backend = os.environ.get("VERINODE_EXTRACTOR_BACKEND", "ollama")
            if backend == "ollama":
                extractor = create_fact_extractor(backend="ollama", model=self.model)
            else:
                extractor = create_fact_extractor(backend=backend)
            self._pipeline = Pipeline(
                self.graph,
                extractor=extractor,
                embedding_engine=self.embedding_engine,
                confidence_engine=self.confidence_engine,
                contradiction_checker=self.contradiction_checker,
            )
        return self._pipeline

    @property
    def contradiction_checker(self):
        if self._contradiction_checker is None:
            import os
            from src.core.llm_contradiction import create_contradiction_checker
            backend = os.environ.get("VERINODE_CHECKER_BACKEND", "ollama")
            # Only pass model name when using the matching backend;
            # Gemini needs its own model name (defaults to gemini-2.0-flash).
            model = self.model if backend == "ollama" else None
            self._contradiction_checker = create_contradiction_checker(
                model=model,
                backend=backend,
            )
        return self._contradiction_checker

    @property
    def reranker(self):
        if self._reranker is None:
            try:
                from src.core.reranker import CrossEncoderReranker
                self._reranker = CrossEncoderReranker()
            except ImportError:
                self._reranker = None
        return self._reranker

    @property
    def verification_engine(self):
        if self._verification_engine is None:
            from src.core.verification import VerificationEngine
            self._verification_engine = VerificationEngine(
                self.graph,
                embedding_engine=self.embedding_engine,
                confidence_engine=self.confidence_engine,
                reasoning_engine=self.reasoning_engine,
                contradiction_checker=self.contradiction_checker,
                reranker=self.reranker,
            )
        return self._verification_engine

    @property
    def llm_memory(self):
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

    def reset_lazy(self):
        """Clear all lazy-initialized heavy components.

        Call after replacing the graph so engines re-index on next access.
        """
        self._embedding_engine = None
        self._pipeline = None
        self._contradiction_checker = None
        self._reranker = None
        self._verification_engine = None
        self._llm_memory = None

    def ensure_embeddings_indexed(self):
        """Build embedding index if graph has nodes but index is empty."""
        if len(self.graph) > 0 and self.embedding_engine.index_size == 0:
            self.embedding_engine.index_graph(self.graph)


# Module-level singleton, set during app lifespan
app_state: Optional[AppState] = None
