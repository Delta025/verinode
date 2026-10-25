"""
SQLite-backed knowledge graph storage.

Provides ``SqliteGraph``, a subclass of ``KnowledgeGraph`` that
automatically persists every mutation to a local SQLite database.
Uses Python's built-in ``sqlite3`` — no extra dependencies.

Design:
  SqliteGraph overrides the four mutation methods of KnowledgeGraph
  (add_node, remove_node, update_node_confidence, add_contradiction).
  Each override calls ``super()`` first to keep the in-memory state
  correct, then writes the change to SQLite.  This means every
  existing caller (Pipeline, DependencyDetector, ReasoningEngine, ...)
  works unchanged — SqliteGraph is a drop-in replacement.

Schema version: 1
"""

import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from .graph import KnowledgeGraph
from .node import (
    Derivation,
    KnowledgeNode,
    NodeStatus,
    NodeType,
    Source,
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _safe_json_loads(s):
    """Parse JSON string, returning empty dict on failure."""
    if not s:
        return {}
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return {}


# ------------------------------------------------------------------
# Schema
# ------------------------------------------------------------------

SCHEMA_VERSION = 2

_SCHEMA_SQL = """
-- Core node data
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    node_type TEXT NOT NULL,
    confidence REAL NOT NULL,
    confidence_lower REAL,
    confidence_upper REAL,
    domain TEXT NOT NULL DEFAULT 'general',
    status TEXT NOT NULL DEFAULT 'provisional',
    derivation_method TEXT DEFAULT 'asserted',
    derivation_depth INTEGER DEFAULT 0,
    derivation_type TEXT DEFAULT 'extracted',
    derivation_notes TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    last_validated TEXT NOT NULL,
    access_count INTEGER DEFAULT 0,
    metadata_json TEXT DEFAULT '{}'
);

-- Sources (one-to-many: node has many sources)
CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    source_type TEXT NOT NULL,
    title TEXT NOT NULL,
    author TEXT,
    date TEXT,
    url TEXT,
    quality_score REAL DEFAULT 0.5
);

-- Edges (predecessor → successor relationships)
CREATE TABLE IF NOT EXISTS edges (
    predecessor_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    successor_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    relation_type TEXT NOT NULL DEFAULT 'depends_on',
    PRIMARY KEY (predecessor_id, successor_id)
);

-- Conflicts (bidirectional contradiction markers)
CREATE TABLE IF NOT EXISTS conflicts (
    node_id1 TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    node_id2 TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    PRIMARY KEY (node_id1, node_id2)
);

-- Embeddings (stored separately to keep main table lean)
CREATE TABLE IF NOT EXISTS embeddings (
    node_id TEXT PRIMARY KEY REFERENCES nodes(id) ON DELETE CASCADE,
    vector BLOB NOT NULL
);

-- Version history / changelog
CREATE TABLE IF NOT EXISTS changelog (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    change_type TEXT NOT NULL,
    node_id TEXT,
    details TEXT
);

-- Schema version for future migrations
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);
"""

_INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_sources_node_id ON sources(node_id);
CREATE INDEX IF NOT EXISTS idx_edges_predecessor ON edges(predecessor_id);
CREATE INDEX IF NOT EXISTS idx_edges_successor ON edges(successor_id);
CREATE INDEX IF NOT EXISTS idx_conflicts_node1 ON conflicts(node_id1);
CREATE INDEX IF NOT EXISTS idx_conflicts_node2 ON conflicts(node_id2);
CREATE INDEX IF NOT EXISTS idx_nodes_domain ON nodes(domain);
CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_nodes_status ON nodes(status);
CREATE INDEX IF NOT EXISTS idx_changelog_type ON changelog(change_type);
"""


# ------------------------------------------------------------------
# SqliteGraph
# ------------------------------------------------------------------


class SqliteGraph(KnowledgeGraph):
    """
    KnowledgeGraph with automatic SQLite persistence.

    Every mutation (add_node, remove_node, update_node_confidence,
    add_contradiction) is immediately persisted to the underlying
    SQLite database.  The in-memory state (``self.nodes``, indexes)
    is always kept in sync via ``super()`` calls.

    Usage::

        # Create new database
        graph = SqliteGraph("my_graph.db")
        graph.add_node(node)  # persisted immediately

        # Reopen existing database
        graph = SqliteGraph("my_graph.db")
        # all previously-added nodes are already loaded

        # Convert from JSON
        graph = SqliteGraph.from_graph(old_graph, "new.db")
    """

    def __init__(self, db_path: str):
        super().__init__()
        self.db_path: str = db_path
        self._conn: Optional[sqlite3.Connection] = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._ensure_schema()
        self._migrate_schema()
        self._load_from_db()

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        """Create tables if they don't exist and stamp the schema version."""
        cur = self._conn.cursor()
        cur.executescript(_SCHEMA_SQL)
        cur.executescript(_INDEXES_SQL)

        # Stamp version if table is empty
        row = cur.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
        if row is None:
            cur.execute(
                "INSERT INTO schema_version (version) VALUES (?)",
                (SCHEMA_VERSION,),
            )
        self._conn.commit()

    def _migrate_schema(self) -> None:
        """Run schema migrations from current version to SCHEMA_VERSION."""
        cur = self._conn.cursor()
        row = cur.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
        current = row[0] if row else 1

        if current < 2:
            # Add relation_type column to edges table
            try:
                cur.execute("ALTER TABLE edges ADD COLUMN relation_type TEXT NOT NULL DEFAULT 'depends_on'")
            except sqlite3.OperationalError:
                pass  # Column already exists (idempotent)
            cur.execute("UPDATE schema_version SET version = 2")
            self._conn.commit()

    # ------------------------------------------------------------------
    # Loading from DB → in-memory state
    # ------------------------------------------------------------------

    def _load_from_db(self) -> None:
        """Read all rows from SQLite and populate the in-memory graph."""
        cur = self._conn.cursor()

        # 1. Load all nodes (without relationships — those come next)
        rows = cur.execute(
            """
            SELECT id, text, node_type, confidence, confidence_lower,
                   confidence_upper, domain, status, derivation_method,
                   derivation_depth, derivation_type, derivation_notes,
                   created_at, last_validated, access_count, metadata_json
            FROM nodes
            """
        ).fetchall()

        # Pre-fetch sources, edges, conflicts, embeddings in bulk
        sources_by_node: Dict[str, list] = {}
        for r in cur.execute(
            "SELECT node_id, source_type, title, author, date, url, quality_score FROM sources"
        ).fetchall():
            sources_by_node.setdefault(r[0], []).append(r[1:])

        predecessors_by_node: Dict[str, list] = {}
        edge_labels_by_node: Dict[str, Dict[str, str]] = {}
        try:
            edge_rows = cur.execute("SELECT successor_id, predecessor_id, relation_type FROM edges").fetchall()
        except sqlite3.OperationalError:
            # Pre-migration database without relation_type column
            edge_rows = [
                (r[0], r[1], "depends_on")
                for r in cur.execute("SELECT successor_id, predecessor_id FROM edges").fetchall()
            ]
        for r in edge_rows:
            predecessors_by_node.setdefault(r[0], []).append(r[1])
            edge_labels_by_node.setdefault(r[0], {})[r[1]] = r[2]

        conflicts_by_node: Dict[str, list] = {}
        for r in cur.execute("SELECT node_id1, node_id2 FROM conflicts").fetchall():
            conflicts_by_node.setdefault(r[0], []).append(r[1])
            conflicts_by_node.setdefault(r[1], []).append(r[0])

        embeddings_by_node: Dict[str, np.ndarray] = {}
        for r in cur.execute("SELECT node_id, vector FROM embeddings").fetchall():
            embeddings_by_node[r[0]] = np.frombuffer(r[1], dtype=np.float32).copy()

        # 2. Reconstruct KnowledgeNode objects and add to graph
        for row in rows:
            (
                node_id,
                text,
                node_type,
                confidence,
                conf_lower,
                conf_upper,
                domain,
                status,
                deriv_method,
                deriv_depth,
                deriv_type,
                deriv_notes,
                created_at,
                last_validated,
                access_count,
                metadata_json,
            ) = row

            # Sources
            node_sources = []
            for s in sources_by_node.get(node_id, []):
                src_type, title, author, date_str, url, quality = s
                node_sources.append(
                    Source(
                        source_type=src_type,
                        title=title,
                        author=author,
                        date=datetime.fromisoformat(date_str) if date_str else None,
                        url=url,
                        quality_score=quality,
                    )
                )

            # Derivation
            derivation = Derivation(
                method=deriv_method,
                inference_depth=deriv_depth,
                derivation_type=deriv_type,
                notes=deriv_notes,
            )

            # Embedding
            embedding = embeddings_by_node.get(node_id)

            # Predecessors
            preds = predecessors_by_node.get(node_id, [])

            node = KnowledgeNode(
                text=text,
                node_type=NodeType(node_type),
                confidence=confidence,
                sources=node_sources,
                predecessor_nodes=preds,
                domain=domain,
                embedding=embedding,
                derivation=derivation,
                node_id=node_id,
            )
            # Restore fields that aren't set by __init__
            node.confidence_bounds = (conf_lower, conf_upper)
            try:
                node.created_at = datetime.fromisoformat(created_at)
            except (ValueError, TypeError):
                pass  # keep default from __init__
            try:
                node.last_validated = datetime.fromisoformat(last_validated)
            except (ValueError, TypeError):
                pass  # keep default from __init__
            node.access_count = access_count
            node.status = NodeStatus(status)
            node.conflicts_with = conflicts_by_node.get(node_id, [])
            node.edge_labels = edge_labels_by_node.get(node_id, {})
            try:
                node.metadata = json.loads(metadata_json)
            except (json.JSONDecodeError, TypeError):
                node.metadata = {}

            # Use super().add_node() to build in-memory indexes
            # (without triggering SQLite persistence)
            super().add_node(node)

    # ------------------------------------------------------------------
    # Mutation overrides — call super(), then persist
    # ------------------------------------------------------------------

    def add_node(self, node: KnowledgeNode) -> None:
        """Add a node and persist it to SQLite."""
        super().add_node(node)
        self._persist_node(node)
        self._log_change("add_node", node.id, {"text": node.text[:100]})

    def remove_node(self, node_id: str) -> None:
        """Remove a node and delete it from SQLite (CASCADE)."""
        super().remove_node(node_id)
        self._delete_node(node_id)
        self._log_change("remove_node", node_id, {})

    def update_node_confidence(self, node_id: str, new_confidence: float) -> None:
        """Update confidence and persist the change."""
        old_conf = None
        if node_id in self.nodes:
            old_conf = self.nodes[node_id].confidence
        super().update_node_confidence(node_id, new_confidence)
        if node_id in self.nodes:
            self._update_node_fields(node_id)
            self._log_change(
                "update_confidence",
                node_id,
                {"old": old_conf, "new": new_confidence},
            )

    def add_contradiction(self, node_id1: str, node_id2: str) -> None:
        """Mark two nodes as contradicting and persist the conflict."""
        super().add_contradiction(node_id1, node_id2)
        if node_id1 in self.nodes and node_id2 in self.nodes:
            self._persist_conflict(node_id1, node_id2)
            # Also persist updated status for both nodes
            self._update_node_fields(node_id1)
            self._update_node_fields(node_id2)
            self._log_change("add_contradiction", node_id1, {"other": node_id2})

    # ------------------------------------------------------------------
    # Internal persistence helpers
    # ------------------------------------------------------------------

    def _persist_node(self, node: KnowledgeNode) -> None:
        """INSERT a node and all related rows into SQLite."""
        cur = self._conn.cursor()

        # Main node row
        cur.execute(
            """
            INSERT INTO nodes (
                id, text, node_type, confidence, confidence_lower,
                confidence_upper, domain, status, derivation_method,
                derivation_depth, derivation_type, derivation_notes,
                created_at, last_validated, access_count, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                node.id,
                node.text,
                node.node_type.value,
                node.confidence,
                node.confidence_bounds[0],
                node.confidence_bounds[1],
                node.domain,
                node.status.value,
                node.derivation.method,
                node.derivation.inference_depth,
                node.derivation.derivation_type,
                node.derivation.notes,
                node.created_at.isoformat(),
                node.last_validated.isoformat(),
                node.access_count,
                json.dumps(node.metadata),
            ),
        )

        # Sources
        for src in node.sources:
            cur.execute(
                """
                INSERT INTO sources (
                    node_id, source_type, title, author, date, url, quality_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    node.id,
                    src.source_type,
                    src.title,
                    src.author,
                    src.date.isoformat() if src.date else None,
                    src.url,
                    src.quality_score,
                ),
            )

        # Predecessor edges
        for pred_id in node.predecessor_nodes:
            rel_type = node.edge_labels.get(pred_id, "depends_on")
            cur.execute(
                "INSERT OR IGNORE INTO edges (predecessor_id, successor_id, relation_type) VALUES (?, ?, ?)",
                (pred_id, node.id, rel_type),
            )

        # Successor edges — only if the successor already exists in the DB.
        # (If it doesn't exist yet, the edge will be created when the
        # successor is added and lists us as its predecessor.)
        for succ_id in node.successor_nodes:
            exists = cur.execute("SELECT 1 FROM nodes WHERE id = ?", (succ_id,)).fetchone()
            if exists:
                succ_node = self.nodes.get(succ_id)
                rel_type = succ_node.edge_labels.get(node.id, "depends_on") if succ_node else "depends_on"
                cur.execute(
                    "INSERT OR IGNORE INTO edges (predecessor_id, successor_id, relation_type) VALUES (?, ?, ?)",
                    (node.id, succ_id, rel_type),
                )

        # Conflicts
        for conflict_id in node.conflicts_with:
            id1, id2 = sorted([node.id, conflict_id])
            cur.execute(
                "INSERT OR IGNORE INTO conflicts (node_id1, node_id2) VALUES (?, ?)",
                (id1, id2),
            )

        # Embedding
        if node.embedding is not None:
            blob = node.embedding.astype(np.float32).tobytes()
            cur.execute(
                "INSERT OR REPLACE INTO embeddings (node_id, vector) VALUES (?, ?)",
                (node.id, blob),
            )

        self._conn.commit()

    def _delete_node(self, node_id: str) -> None:
        """DELETE a node from SQLite (CASCADE handles related rows)."""
        cur = self._conn.cursor()
        cur.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
        # Also clean up conflict rows referencing this node
        cur.execute(
            "DELETE FROM conflicts WHERE node_id1 = ? OR node_id2 = ?",
            (node_id, node_id),
        )
        self._conn.commit()

    def _update_node_fields(self, node_id: str) -> None:
        """UPDATE mutable fields (confidence, status, last_validated) in SQLite."""
        node = self.nodes.get(node_id)
        if node is None:
            return
        cur = self._conn.cursor()
        cur.execute(
            """
            UPDATE nodes
            SET confidence = ?,
                confidence_lower = ?,
                confidence_upper = ?,
                status = ?,
                last_validated = ?,
                access_count = ?,
                metadata_json = ?
            WHERE id = ?
            """,
            (
                node.confidence,
                node.confidence_bounds[0],
                node.confidence_bounds[1],
                node.status.value,
                node.last_validated.isoformat(),
                node.access_count,
                json.dumps(node.metadata),
                node_id,
            ),
        )
        self._conn.commit()

    def _persist_conflict(self, node_id1: str, node_id2: str) -> None:
        """INSERT a conflict row (canonical ordering: sorted pair)."""
        id1, id2 = sorted([node_id1, node_id2])
        cur = self._conn.cursor()
        cur.execute(
            "INSERT OR IGNORE INTO conflicts (node_id1, node_id2) VALUES (?, ?)",
            (id1, id2),
        )
        self._conn.commit()

    def _log_change(
        self,
        change_type: str,
        node_id: Optional[str],
        details: dict,
    ) -> None:
        """Append a row to the changelog table."""
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO changelog (timestamp, change_type, node_id, details)
            VALUES (?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(),
                change_type,
                node_id,
                json.dumps(details),
            ),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_changelog(self, limit: int = 50) -> List[Dict]:
        """
        Return recent changelog entries.

        Returns:
            List of dicts with keys: id, timestamp, change_type, node_id, details.
        """
        cur = self._conn.cursor()
        rows = cur.execute(
            """
            SELECT id, timestamp, change_type, node_id, details
            FROM changelog
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [
            {
                "id": r[0],
                "timestamp": r[1],
                "change_type": r[2],
                "node_id": r[3],
                "details": _safe_json_loads(r[4]),
            }
            for r in rows
        ]

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __del__(self):
        """Ensure connection is closed on garbage collection."""
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Embedding persistence (public API)
    # ------------------------------------------------------------------

    def persist_embedding(self, node_id: str, embedding: np.ndarray) -> None:
        """
        Store or update a node's embedding vector in SQLite.

        Called by external code (e.g. EmbeddingEngine) after computing
        embeddings, so they survive across sessions.
        """
        if self._conn is None:
            return
        blob = embedding.astype(np.float32).tobytes()
        cur = self._conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO embeddings (node_id, vector) VALUES (?, ?)",
            (node_id, blob),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def open(cls, db_path: str) -> "SqliteGraph":
        """Open an existing SQLite graph database (alias for constructor)."""
        return cls(db_path)

    @classmethod
    def from_graph(cls, graph: KnowledgeGraph, db_path: str) -> "SqliteGraph":
        """
        Convert an in-memory ``KnowledgeGraph`` to a ``SqliteGraph``.

        Bulk-inserts all nodes from *graph* into a new SQLite database.
        """
        # Create a fresh instance without calling __init__'s _load_from_db
        sg = cls.__new__(cls)
        KnowledgeGraph.__init__(sg)
        sg.db_path = db_path
        sg._conn = sqlite3.connect(db_path)
        sg._conn.execute("PRAGMA journal_mode=WAL")
        sg._conn.execute("PRAGMA foreign_keys=ON")
        sg._ensure_schema()

        # Temporarily disable FK checks during bulk insert so that
        # edges referencing not-yet-inserted nodes don't fail.
        sg._conn.execute("PRAGMA foreign_keys=OFF")

        # Bulk insert all nodes (use _persist_node, not add_node,
        # to avoid double in-memory add)
        for node in graph.nodes.values():
            sg.nodes[node.id] = node
            sg._persist_node(node)

        # Re-enable FK checks
        sg._conn.execute("PRAGMA foreign_keys=ON")

        # Rebuild the in-memory indexes from scratch
        sg._rebuild_indexes()

        return sg

    def _rebuild_indexes(self) -> None:
        """Rebuild domain/type/status indexes and total_nodes counter."""
        self.domain_index.clear()
        for t in NodeType:
            self.type_index[t] = set()
        for s in NodeStatus:
            self.status_index[s] = set()

        for node_id, node in self.nodes.items():
            if node.domain not in self.domain_index:
                self.domain_index[node.domain] = set()
            self.domain_index[node.domain].add(node_id)
            self.type_index[node.node_type].add(node_id)
            self.status_index[node.status].add(node_id)

        self.total_nodes = len(self.nodes)


# ------------------------------------------------------------------
# Conversion functions
# ------------------------------------------------------------------


def json_to_sqlite(json_path: str, db_path: str) -> SqliteGraph:
    """
    Convert a JSON graph file to a SQLite database.

    Args:
        json_path: Path to the source JSON graph file.
        db_path: Path for the new SQLite database.

    Returns:
        The new SqliteGraph instance.
    """
    graph = KnowledgeGraph.load(json_path)
    return SqliteGraph.from_graph(graph, db_path)


def sqlite_to_json(db_path: str, json_path: str) -> None:
    """
    Export a SQLite graph to JSON format.

    Args:
        db_path: Path to the SQLite database.
        json_path: Destination path for the JSON file.
    """
    graph = SqliteGraph(db_path)
    graph.save(json_path)  # Inherited from KnowledgeGraph
    graph.close()
