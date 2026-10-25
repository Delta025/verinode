"""
Integration tests for SQLite-backed knowledge graph storage.

These tests require a running Ollama server with a pulled model.
Skipped by default; run with: pytest --run-integration

Each test exercises the SqliteGraph with real Pipeline ingestion,
querying, and persistence round-trips.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.graph import KnowledgeGraph
from src.core.storage import SqliteGraph, json_to_sqlite

pytestmark = pytest.mark.integration


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestSqliteGraphWithPipeline:
    """Test SqliteGraph as a drop-in replacement for KnowledgeGraph in the Pipeline."""

    def test_ingest_with_sqlite_graph(self, tmp_path):
        """Pipeline ingestion works with SqliteGraph as the graph backend."""
        from src.core.node import Source
        from src.core.pipeline import Pipeline

        db_path = str(tmp_path / "test.db")
        graph = SqliteGraph(db_path)
        try:
            pipeline = Pipeline(graph)
            result = pipeline.ingest(
                "Water boils at 100 degrees Celsius at sea level.",
                source=Source(source_type="textbook", title="Chem 101", quality_score=0.9),
                domain="chemistry",
            )
            assert result.nodes_added >= 1
            assert len(graph) >= 1
        finally:
            graph.close()

    def test_ingest_and_query_sqlite(self, tmp_path):
        """Full ingest → query workflow using SqliteGraph."""
        from src.core.node import Source
        from src.core.pipeline import Pipeline
        from src.core.reasoning import ReasoningEngine

        db_path = str(tmp_path / "test.db")
        graph = SqliteGraph(db_path)
        try:
            pipeline = Pipeline(graph)
            pipeline.ingest(
                "Water boils at 100 degrees Celsius. Ice melts at 0 degrees Celsius.",
                source=Source(source_type="textbook", title="Chem 101", quality_score=0.9),
                domain="chemistry",
            )

            # Rebuild embeddings for search
            pipeline.build_index()

            reasoner = ReasoningEngine()
            result = reasoner.query(
                graph,
                "What is the boiling point of water?",
                embedding_engine=pipeline.embedding_engine,
            )
            assert result is not None
            assert result.answer is not None
        finally:
            graph.close()


class TestSqlitePersistenceRoundTrip:
    """Test that data persists after close + reopen with live Ollama data."""

    def test_close_reopen_preserves_ingested_data(self, tmp_path):
        """Ingest data, close the DB, reopen it, verify data is intact."""
        from src.core.node import Source
        from src.core.pipeline import Pipeline

        db_path = str(tmp_path / "test.db")

        # Phase 1: Ingest
        graph1 = SqliteGraph(db_path)
        try:
            pipeline = Pipeline(graph1)
            pipeline.ingest(
                "Gold has the symbol Au and atomic number 79.",
                source=Source(source_type="textbook", title="Chem 101", quality_score=0.9),
                domain="chemistry",
            )
            count1 = len(graph1)
            assert count1 >= 1
        finally:
            graph1.close()

        # Phase 2: Reopen and verify
        graph2 = SqliteGraph(db_path)
        try:
            assert len(graph2) == count1
            # Verify a node with "Gold" or "Au" exists
            texts = [n.text for n in graph2.nodes.values()]
            assert any("gold" in t.lower() or "au" in t.lower() for t in texts)
        finally:
            graph2.close()


class TestJsonSqliteConversion:
    """Test JSON ↔ SQLite conversion with live ingested data."""

    def test_json_to_sqlite_with_live_data(self, tmp_path):
        """Ingest via JSON graph, convert to SQLite, verify data."""
        from src.core.node import Source
        from src.core.pipeline import Pipeline

        json_path = str(tmp_path / "test.json")
        db_path = str(tmp_path / "test.db")

        # Ingest into JSON graph
        graph = KnowledgeGraph()
        pipeline = Pipeline(graph)
        pipeline.ingest(
            "Silver has the symbol Ag.",
            source=Source(source_type="textbook", title="Chem 101", quality_score=0.9),
            domain="chemistry",
        )
        graph.save(json_path)
        count = len(graph)

        # Convert to SQLite
        sg = json_to_sqlite(json_path, db_path)
        try:
            assert len(sg) == count
        finally:
            sg.close()

    def test_sqlite_to_json_with_live_data(self, tmp_path):
        """Ingest via SQLite graph, export to JSON, verify data."""
        from src.core.node import Source
        from src.core.pipeline import Pipeline

        db_path = str(tmp_path / "test.db")
        json_path = str(tmp_path / "export.json")

        # Ingest into SQLite graph
        graph = SqliteGraph(db_path)
        try:
            pipeline = Pipeline(graph)
            pipeline.ingest(
                "Iron has the symbol Fe.",
                source=Source(source_type="textbook", title="Chem 101", quality_score=0.9),
                domain="chemistry",
            )
            count = len(graph)
            # Export to JSON
            graph.save(json_path)
        finally:
            graph.close()

        # Verify JSON
        loaded = KnowledgeGraph.load(json_path)
        assert len(loaded) == count


class TestSqliteGraphChangelog:
    """Test changelog with live Pipeline ingestion."""

    def test_changelog_records_pipeline_adds(self, tmp_path):
        """Verify that pipeline ingestion creates changelog entries."""
        from src.core.node import Source
        from src.core.pipeline import Pipeline

        db_path = str(tmp_path / "test.db")
        graph = SqliteGraph(db_path)
        try:
            pipeline = Pipeline(graph)
            pipeline.ingest(
                "Helium is a noble gas.",
                source=Source(source_type="textbook", title="Chem 101", quality_score=0.9),
                domain="chemistry",
            )

            entries = graph.get_changelog()
            add_entries = [e for e in entries if e["change_type"] == "add_node"]
            assert len(add_entries) >= 1
        finally:
            graph.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--run-integration"])
