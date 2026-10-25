"""
Vector embedding engine using sentence-transformers.

Provides semantic similarity search, automatic embedding generation,
and a FAISS-backed vector index for efficient nearest-neighbor queries.
"""

from typing import List, Optional, Dict, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
_sentence_transformers = None
_faiss = None


def _get_sentence_transformers():
    global _sentence_transformers
    if _sentence_transformers is None:
        try:
            import sentence_transformers
            _sentence_transformers = sentence_transformers
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for embeddings. "
                "Install with: pip install sentence-transformers"
            )
    return _sentence_transformers


def _get_faiss():
    """Return the faiss module, or None if not installed."""
    global _faiss
    if _faiss is None:
        try:
            import faiss
            _faiss = faiss
        except ImportError:
            return None
    return _faiss


class EmbeddingEngine:
    """
    Manages vector embeddings for knowledge nodes.

    Wraps sentence-transformers for embedding generation and FAISS
    for efficient nearest-neighbor search over the knowledge graph.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        dimension: int = 384,
        use_faiss: bool = True,
    ):
        """
        Initialize embedding engine.

        Args:
            model_name: sentence-transformers model to use.
            dimension: Embedding dimensionality (must match model output).
            use_faiss: Whether to use FAISS for indexing. Falls back to
                       brute-force numpy search if False or FAISS unavailable.
        """
        self.model_name = model_name
        self.dimension = dimension
        self._model = None

        # Index storage
        self._node_ids: List[str] = []          # position -> node_id
        self._id_to_position: Dict[str, int] = {}  # node_id -> position
        self._embeddings: Optional[np.ndarray] = None  # N x dimension matrix
        self._pending_embeddings: List[np.ndarray] = []  # buffer for batch adds

        # FAISS index (optional)
        self._faiss_index = None
        self._faiss_dirty = False  # True when index needs rebuild
        self._use_faiss = use_faiss

    @property
    def model(self):
        """Lazy-load the sentence-transformers model."""
        if self._model is None:
            st = _get_sentence_transformers()
            self._model = st.SentenceTransformer(self.model_name)
            logger.info("Loaded embedding model: %s", self.model_name)
        return self._model

    @property
    def index_size(self) -> int:
        """Number of embeddings currently indexed."""
        return len(self._node_ids)

    # ------------------------------------------------------------------
    # Embedding generation
    # ------------------------------------------------------------------

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate an embedding vector for a single text string.

        Args:
            text: Text to embed.

        Returns:
            1-D numpy array of shape (dimension,).
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32).flatten()

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.
            batch_size: Encoding batch size.

        Returns:
            2-D numpy array of shape (len(texts), dimension).
        """
        embeddings = self.model.encode(
            texts, batch_size=batch_size, convert_to_numpy=True
        )
        return embeddings.astype(np.float32)

    # ------------------------------------------------------------------
    # Graph integration
    # ------------------------------------------------------------------

    def embed_node(self, node) -> np.ndarray:
        """
        Generate and store an embedding for a KnowledgeNode.

        Sets `node.embedding` in place.

        Args:
            node: KnowledgeNode to embed.

        Returns:
            The embedding vector.
        """
        embedding = self.embed_text(node.text)
        node.embedding = embedding
        return embedding

    def index_graph(self, graph) -> int:
        """
        Build (or rebuild) the search index from all nodes in a graph.

        Nodes that already have embeddings will reuse them; nodes without
        embeddings will be embedded.

        Args:
            graph: KnowledgeGraph to index.

        Returns:
            Number of nodes indexed.
        """
        node_ids = []
        embeddings = []

        # Collect nodes that need embedding and batch-encode them
        needs_embed_ids = []
        needs_embed_texts = []
        for node_id, node in graph.nodes.items():
            node_ids.append(node_id)
            if node.embedding is None:
                needs_embed_ids.append(node_id)
                needs_embed_texts.append(node.text)

        if needs_embed_texts:
            batch_vecs = self.embed_texts(needs_embed_texts)
            for i, nid in enumerate(needs_embed_ids):
                graph.nodes[nid].embedding = batch_vecs[i]

        for node_id in node_ids:
            embeddings.append(graph.nodes[node_id].embedding)

        if not embeddings:
            self._node_ids = []
            self._id_to_position = {}
            self._embeddings = None
            self._pending_embeddings.clear()
            self._faiss_index = None
            self._faiss_dirty = False
            return 0

        self._node_ids = node_ids
        self._id_to_position = {nid: i for i, nid in enumerate(node_ids)}
        self._embeddings = np.vstack(embeddings).astype(np.float32)
        self._pending_embeddings.clear()

        # Build FAISS index if available
        if self._use_faiss:
            self._build_faiss_index()
            self._faiss_dirty = False

        return len(node_ids)

    def add_to_index(self, node) -> None:
        """
        Add a single node to the existing index.

        The FAISS index is NOT rebuilt immediately — call :meth:`flush_index`
        after a batch of adds, or let :meth:`search` / :meth:`find_similar`
        auto-flush before querying.  This makes batch ingestion O(N) instead
        of O(N**2).

        Args:
            node: KnowledgeNode (must have .embedding set).
        """
        if node.embedding is None:
            node.embedding = self.embed_text(node.text)

        embedding = node.embedding.astype(np.float32).reshape(1, -1)

        # Accumulate in a pending list to avoid O(N) vstack on every call.
        self._pending_embeddings.append(embedding)

        pos = len(self._node_ids)
        self._node_ids.append(node.id)
        self._id_to_position[node.id] = pos

        # Mark FAISS as stale (rebuild deferred to flush_index or next search)
        if self._use_faiss:
            self._faiss_dirty = True

    def remove_from_index(self, node_id: str) -> None:
        """
        Remove a node from the index.

        Args:
            node_id: ID of the node to remove.
        """
        if node_id not in self._id_to_position:
            return

        # Flush any pending adds first so positions are consistent
        self._flush_pending()

        pos = self._id_to_position[node_id]

        # Remove from arrays
        self._node_ids.pop(pos)
        if self._embeddings is not None:
            self._embeddings = np.delete(self._embeddings, pos, axis=0)
            if self._embeddings.shape[0] == 0:
                self._embeddings = None

        # Rebuild position map
        self._id_to_position = {nid: i for i, nid in enumerate(self._node_ids)}

        # Mark FAISS as dirty (will rebuild on next search or flush)
        if self._use_faiss:
            self._faiss_dirty = True

    # ------------------------------------------------------------------
    # Deferred index maintenance
    # ------------------------------------------------------------------

    def _flush_pending(self) -> None:
        """Merge pending embeddings into the main matrix."""
        if not self._pending_embeddings:
            return
        new_block = np.vstack(self._pending_embeddings)  # single vstack
        if self._embeddings is None:
            self._embeddings = new_block
        else:
            self._embeddings = np.vstack([self._embeddings, new_block])
        self._pending_embeddings.clear()

    def flush_index(self) -> None:
        """
        Merge any pending embeddings and rebuild the FAISS index.

        Call this after a batch of :meth:`add_to_index` calls — for example
        at the end of ``Pipeline.ingest_batch()``.  Search methods call this
        automatically, so an explicit flush is only needed if you want to
        control *when* the (small) rebuild cost is incurred.
        """
        self._flush_pending()
        if self._use_faiss and self._faiss_dirty:
            self._build_faiss_index()
            self._faiss_dirty = False

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.0,
        domain: Optional[str] = None,
        graph=None,
    ) -> List[Dict]:
        """
        Semantic search over indexed nodes.

        Args:
            query: Natural language query string.
            k: Maximum number of results to return.
            threshold: Minimum cosine similarity (0-1) to include a result.
            domain: If provided, restrict results to this domain.
            graph: KnowledgeGraph for domain filtering.

        Returns:
            List of dicts with keys: node_id, score, text.
        """
        if self._embeddings is None or len(self._node_ids) == 0:
            return []

        query_vec = self.embed_text(query).reshape(1, -1)
        scores, indices = self._search_vectors(query_vec, k=min(k * 3, len(self._node_ids)))

        results = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self._node_ids):
                continue
            node_id = self._node_ids[idx]

            # Domain filter
            if domain and graph:
                node = graph.get_node(node_id)
                if node and node.domain != domain:
                    continue

            if score >= threshold:
                results.append({
                    "node_id": node_id,
                    "score": float(score),
                })

            if len(results) >= k:
                break

        return results

    def find_similar(
        self,
        node_id: str,
        k: int = 5,
        threshold: float = 0.0,
    ) -> List[Dict]:
        """
        Find nodes similar to a given node.

        Args:
            node_id: ID of the reference node.
            k: Number of results.
            threshold: Minimum cosine similarity.

        Returns:
            List of dicts with keys: node_id, score.
        """
        if node_id not in self._id_to_position:
            return []

        # Flush pending so position is valid against the embedding matrix
        self._flush_pending()

        pos = self._id_to_position[node_id]
        query_vec = self._embeddings[pos].reshape(1, -1)

        # k+1 because the node itself will be in the results
        scores, indices = self._search_vectors(query_vec, k=k + 1)

        results = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self._node_ids):
                continue
            result_id = self._node_ids[idx]
            if result_id == node_id:
                continue
            if score >= threshold:
                results.append({
                    "node_id": result_id,
                    "score": float(score),
                })
            if len(results) >= k:
                break

        return results

    def cosine_similarity(self, node_id_a: str, node_id_b: str) -> float:
        """
        Compute cosine similarity between two indexed nodes.

        Args:
            node_id_a: First node ID.
            node_id_b: Second node ID.

        Returns:
            Cosine similarity in [-1, 1].
        """
        if node_id_a not in self._id_to_position or node_id_b not in self._id_to_position:
            raise ValueError("Both nodes must be in the index")

        # Flush pending so position mapping is consistent with the matrix
        self._flush_pending()

        vec_a = self._embeddings[self._id_to_position[node_id_a]]
        vec_b = self._embeddings[self._id_to_position[node_id_b]]

        dot = np.dot(vec_a, vec_b)
        norm = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
        if norm == 0:
            return 0.0
        return float(dot / norm)

    # ------------------------------------------------------------------
    # Public accessors (encapsulation over private internals)
    # ------------------------------------------------------------------

    def get_embedding(self, node_id: str) -> Optional[np.ndarray]:
        """
        Return a copy of the indexed embedding for a node.

        Args:
            node_id: The node ID to look up.

        Returns:
            Copy of the embedding vector, or None if not indexed.
        """
        if node_id not in self._id_to_position:
            return None
        # Flush pending so position mapping is consistent with the matrix
        self._flush_pending()
        pos = self._id_to_position[node_id]
        return self._embeddings[pos].copy()

    def has_embedding(self, node_id: str) -> bool:
        """Check whether a node ID is present in the index."""
        return node_id in self._id_to_position

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_faiss_index(self) -> None:
        """Build or rebuild the FAISS index from current embeddings."""
        if self._embeddings is None or self._embeddings.shape[0] == 0:
            self._faiss_index = None
            return

        faiss = _get_faiss()
        if faiss is None:
            self._faiss_index = None
            return

        # Normalize for cosine similarity (inner product on L2-normalized vecs)
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized = self._embeddings / norms

        index = faiss.IndexFlatIP(self.dimension)
        index.add(normalized)
        self._faiss_index = index

    def _search_vectors(
        self, query_vec: np.ndarray, k: int
    ) -> Tuple[List[float], List[int]]:
        """
        Run nearest-neighbor search. Uses FAISS if available, otherwise numpy.

        Automatically flushes any pending embeddings before searching to
        ensure newly-added nodes are visible.

        Args:
            query_vec: Query vector of shape (1, dimension).
            k: Number of neighbors.

        Returns:
            Tuple of (scores, indices).
        """
        # Auto-flush pending adds and rebuild FAISS if stale
        if self._pending_embeddings or self._faiss_dirty:
            self.flush_index()

        k = min(k, len(self._node_ids))
        if k == 0:
            return [], []

        # Normalize query
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_normalized = query_vec / norm
        else:
            query_normalized = query_vec

        if self._faiss_index is not None:
            distances, indices = self._faiss_index.search(
                query_normalized.astype(np.float32), k
            )
            return distances[0].tolist(), indices[0].tolist()

        # Fallback: brute-force cosine similarity with numpy
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized_embeddings = self._embeddings / norms

        similarities = (normalized_embeddings @ query_normalized.T).flatten()
        top_k = np.argsort(similarities)[::-1][:k]

        scores = similarities[top_k].tolist()
        indices = top_k.tolist()
        return scores, indices
