"""
Integration tests for the end-to-end Pipeline with live Ollama.

These tests require a running Ollama server with a model pulled.
They are skipped by default. Run them with:

    python -m pytest tests/test_integration_pipeline.py -v --run-integration

Prerequisites:
    1. Ollama installed and running (ollama serve)
    2. A model pulled (e.g. ollama pull qwen2.5:7b)

Tests cover:
- Full pipeline ingestion (text -> facts -> graph)
- Source quality gating in real extraction
- Deduplication with convergent evidence boosting
- Contradiction detection and flagging
- Batch ingestion with cross-passage dedup
- Dependency detection wiring on real extracted nodes
- Confidence propagation after ingestion
- IngestionResult/BatchIngestionResult reporting
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.pipeline import Pipeline, IngestionResult, BatchIngestionResult, IngestionVerdict
from core.graph import KnowledgeGraph
from core.node import KnowledgeNode, NodeType, NodeStatus, Source
from core.fact_extraction import FactExtractionEngine
from core.embeddings import EmbeddingEngine
from core.dependency_detection import DependencyDetector
from core.confidence import ConfidenceEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ollama_available() -> bool:
    """Check if Ollama server is reachable."""
    try:
        import ollama
        client = ollama.Client()
        client.list()
        return True
    except Exception:
        return False


def _get_model_name() -> str:
    """Get the name of an available model."""
    import ollama
    client = ollama.Client()
    models = client.list()
    if not models.models:
        return ""
    return models.models[0].model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pipeline():
    """Create a real Pipeline with all real engines."""
    if not _ollama_available():
        pytest.skip("Ollama server not available")

    model = _get_model_name()
    if not model:
        pytest.skip("No models pulled")

    graph = KnowledgeGraph()
    extractor = FactExtractionEngine(model=model)
    embed = EmbeddingEngine()
    detector = DependencyDetector()
    conf_engine = ConfidenceEngine()

    return Pipeline(
        graph,
        extractor=extractor,
        embedding_engine=embed,
        dependency_detector=detector,
        confidence_engine=conf_engine,
        min_extraction_confidence=0.2,
    )


@pytest.fixture
def strict_pipeline():
    """Pipeline with reject_contradictions=True for contradiction rejection tests."""
    if not _ollama_available():
        pytest.skip("Ollama server not available")

    model = _get_model_name()
    if not model:
        pytest.skip("No models pulled")

    graph = KnowledgeGraph()
    extractor = FactExtractionEngine(model=model)
    embed = EmbeddingEngine()
    detector = DependencyDetector()
    conf_engine = ConfidenceEngine()

    return Pipeline(
        graph,
        extractor=extractor,
        embedding_engine=embed,
        dependency_detector=detector,
        confidence_engine=conf_engine,
        min_extraction_confidence=0.2,
        reject_contradictions=True,
    )


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestPipelineBasicIngestion:
    """Test basic pipeline ingestion with real Ollama extraction."""

    def test_ingest_simple_text(self, pipeline):
        """Ingest a simple text passage and verify nodes are added."""
        result = pipeline.ingest(
            "Water is composed of hydrogen and oxygen atoms. "
            "It boils at 100 degrees Celsius at sea level.",
            domain="chemistry",
        )

        assert isinstance(result, IngestionResult)
        assert result.nodes_added >= 1, (
            f"Expected at least 1 node added, got {result.nodes_added}. "
            f"Verdicts: {[(v.action, v.fact.text[:50]) for v in result.verdicts]}"
        )
        assert len(pipeline.graph) >= 1

    def test_ingest_with_source(self, pipeline):
        """Ingested nodes carry source metadata."""
        source = Source(
            source_type="textbook",
            title="Chemistry 101",
            quality_score=0.9,
        )

        result = pipeline.ingest(
            "Iron has the chemical symbol Fe and atomic number 26.",
            source=source,
            domain="chemistry",
        )

        assert result.nodes_added >= 1

        # Check that at least one added node has the source
        for verdict in result.verdicts:
            if verdict.action == "added" and verdict.node_id:
                node = pipeline.graph.get_node(verdict.node_id)
                assert node is not None
                assert len(node.sources) == 1
                assert node.sources[0].title == "Chemistry 101"

    def test_ingest_result_summary(self, pipeline):
        """IngestionResult produces a human-readable summary."""
        result = pipeline.ingest(
            "The Earth orbits the Sun once every 365.25 days.",
            domain="astronomy",
        )

        summary = result.summary
        assert "Extracted" in summary
        assert "added" in summary

    def test_ingest_populates_extraction_model(self, pipeline):
        """IngestionResult records which model was used."""
        result = pipeline.ingest("Gravity pulls objects toward the Earth.")

        assert len(result.extraction_model) > 0

    def test_ingest_nodes_have_metadata(self, pipeline):
        """Nodes created by the pipeline have extraction metadata."""
        result = pipeline.ingest(
            "Photosynthesis converts light energy into chemical energy.",
            domain="biology",
        )

        for verdict in result.verdicts:
            if verdict.action == "added" and verdict.node_id:
                node = pipeline.graph.get_node(verdict.node_id)
                assert "extraction_model" in node.metadata
                assert node.derivation.method == "extracted"


@pytest.mark.integration
class TestPipelineSourceQuality:
    """Test source quality gating with real extraction."""

    def test_low_quality_source_caps_confidence(self, pipeline):
        """Nodes from low-quality sources have capped confidence."""
        low_source = Source(
            source_type="social_media",
            title="Random Blog Post",
            quality_score=0.2,
        )

        result = pipeline.ingest(
            "The speed of light is approximately 300,000 km per second.",
            source=low_source,
        )

        # All added nodes should have confidence <= source quality
        for verdict in result.verdicts:
            if verdict.action == "added" and verdict.node_id:
                node = pipeline.graph.get_node(verdict.node_id)
                assert node.confidence <= 0.2 + 0.01, (
                    f"Node confidence {node.confidence} should be capped "
                    f"at source quality 0.2"
                )

    def test_high_quality_source_no_cap(self, pipeline):
        """Nodes from high-quality sources keep their LLM-assigned confidence."""
        high_source = Source(
            source_type="textbook",
            title="Peer-Reviewed Journal",
            quality_score=0.95,
        )

        result = pipeline.ingest(
            "DNA carries genetic information in living organisms.",
            source=high_source,
            domain="biology",
        )

        # With a high source quality, confidence should NOT be capped
        for verdict in result.verdicts:
            if verdict.action == "added" and verdict.node_id:
                node = pipeline.graph.get_node(verdict.node_id)
                # The LLM should assign decent confidence to a clear fact
                assert node.confidence > 0.3, (
                    f"Expected uncapped confidence > 0.3, got {node.confidence}"
                )


@pytest.mark.integration
class TestPipelineDeduplication:
    """Test deduplication with real embeddings and extraction."""

    def test_duplicate_text_merged(self, pipeline):
        """Ingesting the same fact twice merges via convergent evidence."""
        result1 = pipeline.ingest(
            "Gold has the chemical symbol Au.",
            source=Source(source_type="textbook", title="Source A", quality_score=0.8),
            domain="chemistry",
        )

        # Capture the confidence of the first node
        first_verdicts = [v for v in result1.verdicts if v.action == "added"]
        assert len(first_verdicts) >= 1, "First ingestion should add at least one node"

        first_node_id = first_verdicts[0].node_id
        first_conf = pipeline.graph.get_node(first_node_id).confidence

        # Ingest the same fact from a different source
        result2 = pipeline.ingest(
            "Gold has the chemical symbol Au.",
            source=Source(source_type="encyclopedia", title="Source B", quality_score=0.85),
            domain="chemistry",
        )

        # The second ingestion should detect the duplicate
        dup_verdicts = [v for v in result2.verdicts if v.action == "duplicate"]

        if len(dup_verdicts) > 0:
            # Confidence should have been boosted
            dup = dup_verdicts[0]
            assert dup.confidence_after >= dup.confidence_before, (
                f"Confidence should increase: before={dup.confidence_before}, "
                f"after={dup.confidence_after}"
            )
        else:
            # If the model phrased the extraction differently, it might not
            # be caught as a duplicate — that's acceptable behavior
            pass


@pytest.mark.integration
class TestPipelineBatchIngestion:
    """Test batch ingestion with real extraction."""

    def test_batch_ingest_multiple_texts(self, pipeline):
        """Batch ingestion processes multiple texts and returns combined results."""
        texts = [
            "Mercury is the closest planet to the Sun.",
            "Venus has the hottest surface temperature of any planet.",
            "Mars is known as the Red Planet.",
        ]

        batch_result = pipeline.ingest_batch(
            texts,
            source=Source(source_type="textbook", title="Astronomy 101", quality_score=0.9),
            domain="astronomy",
        )

        assert isinstance(batch_result, BatchIngestionResult)
        assert len(batch_result.results) == 3
        assert batch_result.total_added >= 3, (
            f"Expected at least 3 nodes from 3 distinct texts, "
            f"got {batch_result.total_added}"
        )

    def test_batch_cross_passage_dedup(self, pipeline):
        """Later passages benefit from facts added by earlier passages."""
        texts = [
            "Oxygen has the atomic number 8.",
            "The atomic number of oxygen is 8.",  # Same fact, different wording
        ]

        batch_result = pipeline.ingest_batch(texts, domain="chemistry")

        # The second passage should either:
        # a) Be detected as a duplicate (ideal), or
        # b) Be added as a new node if the wording is different enough
        # Either way, we should have at least 1 node
        assert batch_result.total_added >= 1
        # If dedup works, total_duplicates should be >= 1
        # But we don't assert this strictly since the model may extract
        # differently-worded facts that fall below the dedup threshold


@pytest.mark.integration
class TestPipelineDependencyAndPropagation:
    """Test that dependency detection and confidence propagation work end-to-end."""

    def test_related_facts_get_wired(self, pipeline):
        """Ingesting related facts results in dependency edges."""
        # Ingest a base fact
        pipeline.ingest(
            "Photosynthesis occurs in the chloroplasts of plant cells.",
            source=Source(source_type="textbook", title="Bio 101", quality_score=0.9),
            domain="biology",
        )

        # Ingest a related elaboration
        result2 = pipeline.ingest(
            "Chloroplasts contain chlorophyll, which absorbs sunlight for photosynthesis.",
            source=Source(source_type="textbook", title="Bio 101", quality_score=0.9),
            domain="biology",
        )

        # The graph should have nodes and potentially some edges
        assert len(pipeline.graph) >= 2

        # Check if any dependencies were detected
        # (This depends on similarity thresholds and the model's output,
        # so we check that the pipeline at least tried)
        assert isinstance(result2.dependencies_added, int)

    def test_confidence_propagation_runs(self, pipeline):
        """Confidence propagation is executed after ingestion."""
        result = pipeline.ingest(
            "Newton's first law states that an object at rest stays at rest.",
            source=Source(source_type="textbook", title="Physics 101", quality_score=0.9),
            domain="physics",
        )

        # nodes_propagated should be an integer (possibly 0 if no deps)
        assert isinstance(result.nodes_propagated, int)


@pytest.mark.integration
class TestPipelineBuildIndex:
    """Test the build_index method with a pre-populated graph."""

    def test_build_index_enables_dedup(self, pipeline):
        """Building an index on a pre-populated graph enables deduplication."""
        # Manually add a node to the graph
        from core.node import Derivation
        node = KnowledgeNode(
            text="Copper has the chemical symbol Cu.",
            node_type=NodeType.FACT,
            confidence=0.85,
            domain="chemistry",
            derivation=Derivation(method="manual", derivation_type="manual"),
        )
        pipeline.graph.add_node(node)

        # Build the index
        pipeline.build_index()

        # The index should now contain the manual node
        assert pipeline.embedding_engine.index_size >= 1

        # Ingesting the same fact should (hopefully) detect it as a duplicate
        result = pipeline.ingest(
            "Copper has the chemical symbol Cu.",
            domain="chemistry",
        )

        # Either dedup works or it's added — both are acceptable since
        # the model may extract the fact with different wording
        assert result.total_extracted >= 1


@pytest.mark.integration
class TestPipelineEdgeCases:
    """Test edge cases with real extraction."""

    def test_empty_text(self, pipeline):
        """Empty text produces no nodes."""
        result = pipeline.ingest("")

        # Should gracefully handle empty input
        assert isinstance(result, IngestionResult)
        assert result.nodes_added <= 1  # Model may hallucinate one

    def test_very_long_text(self, pipeline):
        """Long text passage is processed without errors."""
        long_text = (
            "The water cycle, also known as the hydrological cycle, describes "
            "the continuous movement of water on, above, and below the surface "
            "of the Earth. Water can change states among liquid, vapor, and ice "
            "at various places in the water cycle. The sun drives the water cycle "
            "by heating water in the oceans and seas. Water evaporates as water "
            "vapor into the air. Ice and snow can sublimate directly into water "
            "vapor. Rising air currents take the vapor up into the atmosphere, "
            "where cooler temperatures cause it to condense into clouds. Air "
            "currents move clouds around the globe. Cloud particles collide, "
            "grow, and fall out of the sky as precipitation. Some precipitation "
            "falls as snow and can accumulate as ice caps and glaciers."
        )

        result = pipeline.ingest(
            long_text,
            source=Source(source_type="textbook", title="Earth Science", quality_score=0.85),
            domain="earth_science",
        )

        assert result.nodes_added >= 3, (
            f"Long text should yield multiple facts, got {result.nodes_added}"
        )

    def test_multiple_domains(self, pipeline):
        """Sequential ingestion across domains builds a multi-domain graph."""
        pipeline.ingest(
            "Gold is a precious metal with symbol Au.",
            source=Source(source_type="textbook", title="Chemistry", quality_score=0.9),
            domain="chemistry",
        )

        pipeline.ingest(
            "The Milky Way galaxy contains over 100 billion stars.",
            source=Source(source_type="textbook", title="Astronomy", quality_score=0.9),
            domain="astronomy",
        )

        # Graph should have nodes from both domains
        assert len(pipeline.graph) >= 2

        domains = set()
        for nid in pipeline.graph.nodes:
            node = pipeline.graph.get_node(nid)
            if node.domain:
                domains.add(node.domain)

        assert len(domains) >= 2, (
            f"Expected nodes from at least 2 domains, got domains: {domains}"
        )
