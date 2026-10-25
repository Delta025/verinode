"""
Inverted index for fast entity-overlap retrieval.

Replaces the O(N) full-graph scan in Pipeline._find_entity_matches()
and VerificationEngine._gather_candidates() with O(k) lookups via
a word-to-node-ids inverted index.

Also provides BM25-scored retrieval using the same posting lists,
with IDF weighting and document-length normalization for better
ranking of rare, discriminating terms over common words.
"""

import logging
import math
import re
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Shared stopwords and content-word extraction
# ------------------------------------------------------------------

STOP_WORDS: Set[str] = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "shall",
    "can",
    "of",
    "in",
    "to",
    "for",
    "with",
    "on",
    "at",
    "from",
    "by",
    "about",
    "as",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "between",
    "under",
    "over",
    "and",
    "but",
    "or",
    "nor",
    "not",
    "no",
    "so",
    "if",
    "then",
    "than",
    "too",
    "very",
    "just",
    "only",
    "also",
    "that",
    "this",
    "these",
    "those",
    "it",
    "its",
    "he",
    "she",
    "they",
    "them",
    "his",
    "her",
    "their",
    "we",
    "you",
    "i",
    "me",
    "my",
    "your",
    "our",
    "which",
    "who",
    "whom",
    "what",
    "where",
    "when",
    "how",
    "all",
    "each",
    "every",
    "both",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "any",
    "many",
    "much",
    "own",
    "same",
    "up",
    "out",
    "off",
    "down",
    "there",
    "here",
}


def extract_content_words(text: str) -> Set[str]:
    """
    Extract content words (non-stopwords, length >= 2) from text.

    Tokenizes on non-alphanumeric boundaries, lowercases, strips
    stopwords and single-character tokens.
    """
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {t for t in tokens if len(t) >= 2 and t not in STOP_WORDS}


class EntityIndex:
    """
    Inverted index mapping content words to node IDs.

    Supports O(k) entity-overlap retrieval where k is the number of
    content words in the query, instead of O(N) full-graph scanning.

    Usage::

        idx = EntityIndex()
        idx.index_graph(graph)          # bulk index all nodes
        idx.add_node(new_node)          # incremental index
        matches = idx.find_matches("water boils at 100 degrees")
        idx.remove_node(old_node_id)    # clean removal
    """

    # BM25 parameters (Okapi BM25 defaults)
    BM25_K1: float = 1.5
    BM25_B: float = 0.75

    def __init__(self) -> None:
        # word -> set of node_ids containing that word
        self._word_to_nodes: Dict[str, Set[str]] = {}
        # node_id -> set of words (for removal)
        self._node_to_words: Dict[str, Set[str]] = {}
        # BM25 statistics
        self._doc_lengths: Dict[str, int] = {}  # node_id -> content word count
        self._total_doc_length: int = 0
        self._corpus_size: int = 0

    @property
    def size(self) -> int:
        """Number of indexed nodes."""
        return len(self._node_to_words)

    def index_graph(self, graph) -> int:
        """
        Bulk-index all nodes in a graph. Clears any previous index.

        Args:
            graph: KnowledgeGraph instance with .nodes dict.

        Returns:
            Number of nodes indexed.
        """
        self._word_to_nodes.clear()
        self._node_to_words.clear()
        self._doc_lengths.clear()
        self._total_doc_length = 0
        self._corpus_size = 0

        for node in graph.nodes.values():
            self._index_node_internal(node.id, node.text)

        logger.debug("EntityIndex: indexed %d nodes", len(self._node_to_words))
        return len(self._node_to_words)

    def add_node(self, node) -> None:
        """
        Add or update a single node in the index.

        Args:
            node: KnowledgeNode with .id and .text attributes.
        """
        # Remove old entry if updating
        if node.id in self._node_to_words:
            self.remove_node(node.id)
        self._index_node_internal(node.id, node.text)

    def remove_node(self, node_id: str) -> None:
        """
        Remove a node from the index.

        Args:
            node_id: ID of the node to remove.
        """
        words = self._node_to_words.pop(node_id, None)
        if words is None:
            return
        dl = self._doc_lengths.pop(node_id, 0)
        self._total_doc_length -= dl
        self._corpus_size -= 1
        for word in words:
            node_set = self._word_to_nodes.get(word)
            if node_set is not None:
                node_set.discard(node_id)
                if not node_set:
                    del self._word_to_nodes[word]

    def find_matches(
        self,
        text: str,
        min_overlap: int = 2,
        exclude_ids: Optional[Set[str]] = None,
    ) -> List[Tuple[str, int]]:
        """
        Find nodes sharing at least ``min_overlap`` content words with text.

        Args:
            text: Query text to match against.
            min_overlap: Minimum number of shared content words.
            exclude_ids: Node IDs to skip.

        Returns:
            List of (node_id, overlap_count) tuples sorted by overlap
            descending.
        """
        query_words = extract_content_words(text)
        if len(query_words) < min_overlap:
            return []

        if exclude_ids is None:
            exclude_ids = set()

        # Count overlaps via inverted index — O(k * avg_posting_list_len)
        overlap_counts: Dict[str, int] = {}
        for word in query_words:
            node_ids = self._word_to_nodes.get(word)
            if node_ids is None:
                continue
            for nid in node_ids:
                if nid not in exclude_ids:
                    overlap_counts[nid] = overlap_counts.get(nid, 0) + 1

        # Filter by min_overlap and sort descending
        matches = [(nid, count) for nid, count in overlap_counts.items() if count >= min_overlap]
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def bm25_search(
        self,
        text: str,
        top_k: int = 10,
        exclude_ids: Optional[Set[str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        BM25-scored retrieval using the inverted index.

        Uses Okapi BM25 with binary term frequency (TF=1 when word
        is present, 0 otherwise). This is appropriate for short
        knowledge graph facts where word repetition is rare. The key
        improvement over raw overlap counts is IDF weighting (rare
        words like "Neptune" score higher than common words like
        "planet") and document-length normalization.

        Args:
            text: Query text.
            top_k: Max results to return.
            exclude_ids: Node IDs to skip.

        Returns:
            List of (node_id, bm25_score) tuples sorted by score
            descending.
        """
        query_words = extract_content_words(text)
        if not query_words or self._corpus_size == 0:
            return []

        if exclude_ids is None:
            exclude_ids = set()

        avgdl = self._total_doc_length / self._corpus_size
        N = self._corpus_size
        k1 = self.BM25_K1
        b = self.BM25_B

        scores: Dict[str, float] = {}
        for word in query_words:
            node_ids = self._word_to_nodes.get(word)
            if node_ids is None:
                continue
            df = len(node_ids)
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
            for nid in node_ids:
                if nid in exclude_ids:
                    continue
                dl = self._doc_lengths.get(nid, 0)
                # Binary TF (word present = 1)
                tf_component = (k1 + 1.0) / (1.0 + k1 * (1.0 - b + b * dl / avgdl))
                scores[nid] = scores.get(nid, 0.0) + idf * tf_component

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def _index_node_internal(self, node_id: str, text: str) -> None:
        """Index a single node's text into the inverted index."""
        words = extract_content_words(text)
        if not words:
            return
        self._node_to_words[node_id] = words
        self._doc_lengths[node_id] = len(words)
        self._total_doc_length += len(words)
        self._corpus_size += 1
        for word in words:
            if word not in self._word_to_nodes:
                self._word_to_nodes[word] = set()
            self._word_to_nodes[word].add(node_id)
