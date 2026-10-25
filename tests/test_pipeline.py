"""
Tests for the end-to-end ingestion pipeline.

All tests mock the Ollama client and the embedding engine so they run
without a running Ollama server or downloaded sentence-transformer
models.

Tests cover:
- Basic ingestion (extract → add to graph)
- Quality filtering (low-confidence facts dropped)
- Source quality gating (untrusted → capped confidence)
- Deduplication via embedding similarity (convergent evidence boost)
- Contradiction flagging (new fact vs high-confidence existing)
- Contradiction rejection mode
- Batch ingestion
- Dependency detection and wiring
- Confidence propagation after ingestion
- IngestionResult / BatchIngestionResult summary properties

Run with:  python -m pytest tests/test_pipeline.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import json
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.node import KnowledgeNode, NodeType, NodeStatus, Source, SourceTier, Derivation
from core.graph import KnowledgeGraph
from core.confidence import ConfidenceEngine
from core.fact_extraction import ExtractedFact, ExtractionResult
from core.pipeline import (
    Pipeline,
    IngestionResult,
    IngestionVerdict,
    BatchIngestionResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_extraction(facts: list[ExtractedFact], model="test-model"):
    """Create a mock ExtractionResult."""
    return ExtractionResult(
        facts=facts,
        source_text="mock source text",
        model=model,
    )


def _make_pipeline(graph=None, **kwargs):
    """
    Create a Pipeline with all external dependencies mocked.

    Returns (pipeline, mock_extractor, mock_embedding_engine).
    """
    graph = graph or KnowledgeGraph()

    # Mock the extractor
    mock_extractor = MagicMock()

    # Mock the embedding engine with a working index_size property
    mock_embed = MagicMock()
    type(mock_embed).index_size = PropertyMock(return_value=0)
    mock_embed._id_to_position = {}

    # Default: disable LLM contradiction detection in unit tests
    # (tests that need it will override this)
    kwargs.setdefault("use_llm_contradiction", False)
    # Default: disable quote verification in existing tests
    kwargs.setdefault("verify_quotes", False)
    # Default: disable tiered ingestion in existing tests
    kwargs.setdefault("use_tiered_ingestion", False)
    # Default: disable content pre-filters in existing tests
    kwargs.setdefault("min_content_overlap", 0)
    kwargs.setdefault("use_unit_normalization", False)
    # Default: disable BM25 in existing tests (BM25 tests explicitly enable it)
    kwargs.setdefault("use_bm25", False)

    pipeline = Pipeline(
        graph=graph,
        extractor=mock_extractor,
        embedding_engine=mock_embed,
        **kwargs,
    )
    return pipeline, mock_extractor, mock_embed


def _source(quality=0.9, title="Test Source"):
    return Source(source_type="textbook", title=title, quality_score=quality)


# ---------------------------------------------------------------------------
# Tests: basic ingestion
# ---------------------------------------------------------------------------

def test_basic_ingest_adds_nodes():
    """Basic ingestion extracts facts and adds them to the graph."""
    pipeline, extractor, embed = _make_pipeline()
    extractor.extract.return_value = _mock_extraction([
        ExtractedFact(text="Water boils at 100C", confidence=0.95, domain="chemistry"),
        ExtractedFact(text="Water is H2O", confidence=0.9, domain="chemistry"),
    ])

    result = pipeline.ingest("Water boils at 100C. Water is H2O.", domain="chemistry")

    assert result.nodes_added == 2
    assert result.total_extracted == 2
    assert len(pipeline.graph) == 2
    assert all(v.action == "added" for v in result.verdicts)


def test_ingest_attaches_source():
    """Source metadata is attached to ingested nodes."""
    pipeline, extractor, embed = _make_pipeline()
    extractor.extract.return_value = _mock_extraction([
        ExtractedFact(text="Iron is a metal", confidence=0.9, domain="chemistry"),
    ])
    src = _source(quality=0.85, title="Chem 101")

    result = pipeline.ingest("Iron is a metal.", source=src)

    node = pipeline.graph.get_node(result.verdicts[0].node_id)
    assert len(node.sources) == 1
    assert node.sources[0].title == "Chem 101"


def test_ingest_sets_metadata():
    """Ingested nodes carry extraction metadata."""
    pipeline, extractor, embed = _make_pipeline()
    extractor.extract.return_value = _mock_extraction([
        ExtractedFact(text="Fact A", confidence=0.8, source_quote="quote A"),
    ], model="qwen2.5:7b")

    result = pipeline.ingest("Fact A")

    node = pipeline.graph.get_node(result.verdicts[0].node_id)
    assert node.metadata["extraction_model"] == "qwen2.5:7b"
    assert node.metadata["source_quote"] == "quote A"


def test_result_summary():
    """IngestionResult.summary produces a readable string."""
    pipeline, extractor, embed = _make_pipeline()
    extractor.extract.return_value = _mock_extraction([
        ExtractedFact(text="Fact", confidence=0.8),
    ])

    result = pipeline.ingest("Fact")
    assert "Extracted 1 fact(s)" in result.summary
    assert "added 1" in result.summary


# ---------------------------------------------------------------------------
# Tests: quality filtering
# ---------------------------------------------------------------------------

def test_low_confidence_facts_filtered():
    """Facts below min_extraction_confidence are silently dropped."""
    pipeline, extractor, embed = _make_pipeline(min_extraction_confidence=0.5)
    extractor.extract.return_value = _mock_extraction([
        ExtractedFact(text="Confident fact", confidence=0.8),
        ExtractedFact(text="Uncertain guess", confidence=0.2),
        ExtractedFact(text="Low confidence", confidence=0.49),
    ])

    result = pipeline.ingest("text")

    assert result.nodes_added == 1
    assert result.total_extracted == 1  # only 1 passed filter
    assert len(pipeline.graph) == 1


def test_all_facts_filtered():
    """When all facts are below threshold, nothing is added."""
    pipeline, extractor, embed = _make_pipeline(min_extraction_confidence=0.9)
    extractor.extract.return_value = _mock_extraction([
        ExtractedFact(text="Not confident", confidence=0.5),
    ])

    result = pipeline.ingest("text")

    assert result.nodes_added == 0
    assert len(pipeline.graph) == 0


# ---------------------------------------------------------------------------
# Tests: source quality gating (Layer 1)
# ---------------------------------------------------------------------------

def test_low_quality_source_caps_confidence():
    """Nodes from low-quality sources have confidence capped."""
    pipeline, extractor, embed = _make_pipeline(source_quality_cap=0.5)
    extractor.extract.return_value = _mock_extraction([
        ExtractedFact(text="Dubious claim", confidence=0.9),
    ])
    low_quality_src = _source(quality=0.3)

    result = pipeline.ingest("text", source=low_quality_src)

    node = pipeline.graph.get_node(result.verdicts[0].node_id)
    assert node.confidence == 0.3  # capped to source quality
    assert node.status in (NodeStatus.PROVISIONAL, NodeStatus.CONTESTED)


def test_high_quality_source_no_cap():
    """Nodes from high-quality sources keep their extraction confidence."""
    pipeline, extractor, embed = _make_pipeline(source_quality_cap=0.5)
    extractor.extract.return_value = _mock_extraction([
        ExtractedFact(text="Solid fact", confidence=0.9),
    ])
    high_quality_src = _source(quality=0.95)

    result = pipeline.ingest("text", source=high_quality_src)

    node = pipeline.graph.get_node(result.verdicts[0].node_id)
    assert node.confidence == 0.9  # not capped


def test_no_source_no_cap():
    """Without a source, confidence is not capped."""
    pipeline, extractor, embed = _make_pipeline()
    extractor.extract.return_value = _mock_extraction([
        ExtractedFact(text="Unsourced fact", confidence=0.8),
    ])

    result = pipeline.ingest("text")  # no source

    node = pipeline.graph.get_node(result.verdicts[0].node_id)
    assert node.confidence == 0.8


# ---------------------------------------------------------------------------
# Tests: deduplication (convergent evidence)
# ---------------------------------------------------------------------------

def test_duplicate_merged_with_confidence_boost():
    """Duplicate facts boost existing node confidence (noisy-OR)."""
    graph = KnowledgeGraph()
    existing = KnowledgeNode(
        text="Water boils at 100C",
        confidence=0.8,
        domain="chemistry",
    )
    graph.add_node(existing)

    pipeline, extractor, embed = _make_pipeline(graph=graph)

    # Make embedding engine find the existing node as a near-duplicate
    type(embed).index_size = PropertyMock(return_value=1)
    embed.search.return_value = [{"node_id": existing.id, "score": 0.95}]

    extractor.extract.return_value = _mock_extraction([
        ExtractedFact(text="Water boils at 100 degrees C", confidence=0.85),
    ])

    result = pipeline.ingest("Water boils at 100C.")

    assert result.duplicates_merged == 1
    assert result.nodes_added == 0
    assert len(graph) == 1  # no new node

    verdict = result.verdicts[0]
    assert verdict.action == "duplicate"
    assert verdict.confidence_before == 0.8
    # noisy-OR: 1 - (1-0.8)*(1-0.85) = 1 - 0.2*0.15 = 0.97
    assert abs(verdict.confidence_after - 0.97) < 0.01

    # The existing node's confidence was actually updated
    assert abs(existing.confidence - 0.97) < 0.01


def test_duplicate_adds_new_source():
    """When a duplicate is merged, the new source is added to existing node."""
    graph = KnowledgeGraph()
    existing = KnowledgeNode(text="Iron is a metal", confidence=0.8)
    graph.add_node(existing)

    pipeline, extractor, embed = _make_pipeline(graph=graph)
    type(embed).index_size = PropertyMock(return_value=1)
    embed.search.return_value = [{"node_id": existing.id, "score": 0.95}]

    extractor.extract.return_value = _mock_extraction([
        ExtractedFact(text="Iron is a metal", confidence=0.9),
    ])
    src = _source(title="New textbook")

    pipeline.ingest("Iron is a metal.", source=src)

    assert len(existing.sources) == 1
    assert existing.sources[0].title == "New textbook"


def test_below_duplicate_threshold_not_merged():
    """Similarity below duplicate_threshold does not trigger merge."""
    graph = KnowledgeGraph()
    existing = KnowledgeNode(text="Water boils at 100C", confidence=0.8)
    graph.add_node(existing)

    pipeline, extractor, embed = _make_pipeline(
        graph=graph, duplicate_threshold=0.95
    )
    type(embed).index_size = PropertyMock(return_value=1)
    # Similarity is 0.88 — below the 0.95 threshold
    embed.search.return_value = []  # nothing above threshold

    extractor.extract.return_value = _mock_extraction([
        ExtractedFact(text="Water boils at about 100 degrees", confidence=0.7),
    ])

    result = pipeline.ingest("text")

    assert result.nodes_added == 1
    assert result.duplicates_merged == 0
    assert len(graph) == 2


# ---------------------------------------------------------------------------
# Tests: contradiction detection and flagging (Layer 3)
# ---------------------------------------------------------------------------

def test_contradiction_flagged():
    """A fact contradicting a high-confidence node is flagged."""
    graph = KnowledgeGraph()
    existing = KnowledgeNode(
        text="Water freezes at 0 degrees Celsius",
        confidence=0.95,
        domain="chemistry",
    )
    graph.add_node(existing)

    pipeline, extractor, embed = _make_pipeline(graph=graph)
    pipeline.entity_retrieval = False  # test embedding-only retrieval
    type(embed).index_size = PropertyMock(return_value=1)

    # Contradiction check runs first now, then dedup
    embed.search.side_effect = [
        [{"node_id": existing.id, "score": 0.7}],  # contradiction search
        [],  # dedup search: no duplicate (not reached since flagged)
    ]

    extractor.extract.return_value = _mock_extraction([
        ExtractedFact(
            text="Water does not freeze at 0 degrees Celsius",
            confidence=0.8,
        ),
    ])

    result = pipeline.ingest("Water does not freeze at 0C.")

    assert result.facts_flagged == 1
    verdict = result.verdicts[0]
    assert verdict.action == "flagged"
    assert existing.id in verdict.contradiction_node_ids

    # The node was still added but with low confidence
    flagged_node = graph.get_node(verdict.node_id)
    assert flagged_node is not None
    assert flagged_node.confidence <= 0.3


def test_contradiction_rejected():
    """With reject_contradictions=True, contradictions are not added."""
    graph = KnowledgeGraph()
    existing = KnowledgeNode(
        text="Water freezes at 0 degrees Celsius",
        confidence=0.95,
    )
    graph.add_node(existing)

    pipeline, extractor, embed = _make_pipeline(
        graph=graph, reject_contradictions=True
    )
    pipeline.entity_retrieval = False
    type(embed).index_size = PropertyMock(return_value=1)
    # Contradiction runs first now
    embed.search.side_effect = [
        [{"node_id": existing.id, "score": 0.7}],  # contradiction search
    ]

    extractor.extract.return_value = _mock_extraction([
        ExtractedFact(
            text="Water does not freeze at 0 degrees Celsius",
            confidence=0.8,
        ),
    ])

    result = pipeline.ingest("text")

    assert result.facts_rejected == 1
    assert result.nodes_added == 0
    assert len(graph) == 1  # only the original


def test_contradiction_against_low_confidence_not_flagged():
    """Contradictions only matter against high-confidence nodes."""
    graph = KnowledgeGraph()
    existing = KnowledgeNode(
        text="Water freezes at 0 degrees Celsius",
        confidence=0.3,  # low confidence
    )
    graph.add_node(existing)

    pipeline, extractor, embed = _make_pipeline(
        graph=graph, contradiction_threshold=0.7
    )
    pipeline.entity_retrieval = False
    type(embed).index_size = PropertyMock(return_value=1)

    # Contradiction search runs first; node found but low confidence
    embed.search.side_effect = [
        [{"node_id": existing.id, "score": 0.7}],  # contradiction search
        [],  # dedup (reached since no contradiction flagged)
    ]

    extractor.extract.return_value = _mock_extraction([
        ExtractedFact(
            text="Water does not freeze at 0 degrees Celsius",
            confidence=0.8,
        ),
    ])

    result = pipeline.ingest("text")

    # Not flagged because existing node's confidence (0.3) is below threshold (0.7)
    assert result.facts_flagged == 0
    assert result.nodes_added == 1


# ---------------------------------------------------------------------------
# Tests: batch ingestion
# ---------------------------------------------------------------------------

def test_batch_ingestion():
    """Batch ingestion processes multiple texts sequentially."""
    pipeline, extractor, embed = _make_pipeline()
    extractor.extract.side_effect = [
        _mock_extraction([ExtractedFact(text="Fact A", confidence=0.8)]),
        _mock_extraction([ExtractedFact(text="Fact B", confidence=0.9)]),
        _mock_extraction([ExtractedFact(text="Fact C", confidence=0.7)]),
    ]

    batch = pipeline.ingest_batch(["text1", "text2", "text3"])

    assert len(batch.results) == 3
    assert batch.total_added == 3
    assert len(pipeline.graph) == 3


def test_batch_result_properties():
    """BatchIngestionResult aggregates counts correctly."""
    pipeline, extractor, embed = _make_pipeline()
    extractor.extract.side_effect = [
        _mock_extraction([
            ExtractedFact(text="Fact A", confidence=0.8),
            ExtractedFact(text="Fact B", confidence=0.1),  # below default 0.3
        ]),
        _mock_extraction([
            ExtractedFact(text="Fact C", confidence=0.9),
        ]),
    ]

    batch = pipeline.ingest_batch(["text1", "text2"])

    assert batch.total_added == 2
    assert batch.total_rejected == 0


# ---------------------------------------------------------------------------
# Tests: dependency detection and confidence propagation
# ---------------------------------------------------------------------------

def test_dependency_detection_called_for_new_nodes():
    """The detector is called for each newly added node."""
    pipeline, extractor, embed = _make_pipeline()
    embed._id_to_position = {}  # ensure _ensure_indexed works

    extractor.extract.return_value = _mock_extraction([
        ExtractedFact(text="Premise", confidence=0.9),
        ExtractedFact(text="Conclusion", confidence=0.8),
    ])

    # The detector returns no dependencies (simplest case)
    pipeline.detector = MagicMock()
    pipeline.detector.detect_for_node.return_value = []
    pipeline.detector.apply_dependencies.return_value = 0

    result = pipeline.ingest("text")

    assert result.nodes_added == 2
    # detect_for_node should have been called twice (once per new node)
    assert pipeline.detector.detect_for_node.call_count == 2


def test_confidence_propagation_runs():
    """Confidence propagation is triggered for new nodes."""
    pipeline, extractor, embed = _make_pipeline()
    extractor.extract.return_value = _mock_extraction([
        ExtractedFact(text="Fact", confidence=0.8),
    ])

    # Spy on the confidence engine
    pipeline.confidence_engine = MagicMock()
    pipeline.confidence_engine.propagate_confidence_update.return_value = set()

    result = pipeline.ingest("text")

    assert pipeline.confidence_engine.propagate_confidence_update.call_count == 1


# ---------------------------------------------------------------------------
# Tests: empty / edge cases
# ---------------------------------------------------------------------------

def test_empty_extraction():
    """When the LLM extracts no facts, result is empty."""
    pipeline, extractor, embed = _make_pipeline()
    extractor.extract.return_value = _mock_extraction([])

    result = pipeline.ingest("Nothing useful here.")

    assert result.total_extracted == 0
    assert result.nodes_added == 0
    assert len(pipeline.graph) == 0


def test_empty_text():
    """Empty text produces empty result."""
    pipeline, extractor, embed = _make_pipeline()
    extractor.extract.return_value = _mock_extraction([])

    result = pipeline.ingest("")
    assert result.total_extracted == 0


def test_build_index_on_empty_graph():
    """build_index on empty graph doesn't crash."""
    pipeline, extractor, embed = _make_pipeline()
    pipeline.build_index()  # should be a no-op
    embed.index_graph.assert_not_called()


def test_build_index_on_populated_graph():
    """build_index calls embedding_engine.index_graph."""
    graph = KnowledgeGraph()
    graph.add_node(KnowledgeNode(text="Existing fact", confidence=0.9))

    pipeline, extractor, embed = _make_pipeline(graph=graph)
    pipeline.build_index()

    embed.index_graph.assert_called_once_with(graph)


# ---------------------------------------------------------------------------
# Tests: LLM-enhanced contradiction detection in pipeline
# ---------------------------------------------------------------------------

def test_llm_enhanced_contradiction_flags():
    """LLM-enhanced pipeline catches numerical contradictions the heuristic misses."""
    from core.llm_contradiction import RelevanceResult

    graph = KnowledgeGraph()
    existing = KnowledgeNode(
        text="Water freezes at 0 degrees Celsius",
        confidence=0.95,
        domain="chemistry",
    )
    graph.add_node(existing)

    # Create a mock contradiction checker
    mock_checker = MagicMock()
    mock_checker.check_relevance_batch.return_value = [
        RelevanceResult(
            claim="Water freezes at -100 degrees Celsius",
            knowledge="Water freezes at 0 degrees Celsius",
            relationship="contradicts",
            confidence=0.92,
            explanation="Incompatible freezing point values: 0C vs -100C",
        ),
    ]

    pipeline, extractor, embed = _make_pipeline(
        graph=graph,
        contradiction_checker=mock_checker,
    )
    pipeline.entity_retrieval = False
    type(embed).index_size = PropertyMock(return_value=1)

    # Contradiction search runs first now
    embed.search.side_effect = [
        [{"node_id": existing.id, "score": 0.7}],  # contradiction search candidates
    ]

    extractor.extract.return_value = _mock_extraction([
        ExtractedFact(
            text="Water freezes at -100 degrees Celsius",
            confidence=0.85,
        ),
    ])

    result = pipeline.ingest("Water freezes at -100C.")

    assert result.facts_flagged == 1
    verdict = result.verdicts[0]
    assert verdict.action == "flagged"
    assert "contradiction" in verdict.reason.lower()
    assert "Incompatible" in verdict.reason


def test_llm_corrects_heuristic_false_positive():
    """LLM overrides heuristic false positive in pipeline contradiction check."""
    from core.llm_contradiction import RelevanceResult

    graph = KnowledgeGraph()
    existing = KnowledgeNode(
        text="Water boils at 100 degrees Celsius",
        confidence=0.95,
        domain="chemistry",
    )
    graph.add_node(existing)

    # LLM says: irrelevant (different properties)
    mock_checker = MagicMock()
    mock_checker.check_relevance_batch.return_value = [
        RelevanceResult(
            claim="Water does not freeze at 100 degrees Celsius",
            knowledge="Water boils at 100 degrees Celsius",
            relationship="irrelevant",
            confidence=0.9,
            explanation="Different properties: boiling vs freezing",
        ),
    ]

    pipeline, extractor, embed = _make_pipeline(
        graph=graph,
        contradiction_checker=mock_checker,
    )
    pipeline.entity_retrieval = False
    type(embed).index_size = PropertyMock(return_value=1)

    # Heuristic would flag this (has "not" + high word overlap)
    # Contradiction search runs first now
    embed.search.side_effect = [
        [{"node_id": existing.id, "score": 0.7}],  # contradiction candidates
        [],  # dedup (reached since LLM clears the false positive)
    ]

    extractor.extract.return_value = _mock_extraction([
        ExtractedFact(
            text="Water does not freeze at 100 degrees Celsius",
            confidence=0.8,
        ),
    ])

    result = pipeline.ingest("Water does not freeze at 100C.")

    # LLM should have overridden the heuristic → added normally, not flagged
    assert result.facts_flagged == 0
    assert result.nodes_added == 1


def test_pipeline_fallback_without_llm():
    """Without LLM checker, pipeline uses heuristic-only contradiction detection."""
    graph = KnowledgeGraph()
    existing = KnowledgeNode(
        text="Water freezes at 0 degrees Celsius",
        confidence=0.95,
    )
    graph.add_node(existing)

    pipeline, extractor, embed = _make_pipeline(
        graph=graph,
        use_llm_contradiction=False,
    )
    assert pipeline.contradiction_checker is None

    pipeline.entity_retrieval = False
    type(embed).index_size = PropertyMock(return_value=1)
    # Contradiction search runs first now
    embed.search.side_effect = [
        [{"node_id": existing.id, "score": 0.7}],  # contradiction candidates
    ]

    extractor.extract.return_value = _mock_extraction([
        ExtractedFact(
            text="Water does not freeze at 0 degrees Celsius",
            confidence=0.8,
        ),
    ])

    result = pipeline.ingest("text")

    # Heuristic catches this (negation mismatch)
    assert result.facts_flagged == 1
    assert "Negation mismatch" in result.verdicts[0].reason


def test_llm_verdict_includes_explanation():
    """LLM explanation is included in the verdict reason."""
    from core.llm_contradiction import RelevanceResult

    graph = KnowledgeGraph()
    existing = KnowledgeNode(
        text="The sky is blue",
        confidence=0.95,
    )
    graph.add_node(existing)

    mock_checker = MagicMock()
    mock_checker.check_relevance_batch.return_value = [
        RelevanceResult(
            claim="The sky is green",
            knowledge="The sky is blue",
            relationship="contradicts",
            confidence=0.88,
            explanation="Blue and green are mutually exclusive color attributes",
        ),
    ]

    pipeline, extractor, embed = _make_pipeline(
        graph=graph,
        contradiction_checker=mock_checker,
    )
    type(embed).index_size = PropertyMock(return_value=1)

    pipeline.entity_retrieval = False
    # Contradiction search runs first now
    embed.search.side_effect = [
        [{"node_id": existing.id, "score": 0.6}],  # contradiction candidates
    ]

    extractor.extract.return_value = _mock_extraction([
        ExtractedFact(text="The sky is green", confidence=0.7),
    ])

    result = pipeline.ingest("The sky is green.")

    assert result.facts_flagged == 1
    verdict = result.verdicts[0]
    assert "contradiction" in verdict.reason.lower()
    assert "mutually exclusive" in verdict.reason


def test_llm_checker_threaded_to_dependency_detector():
    """Contradiction checker is passed through to detect_for_node calls."""
    mock_checker = MagicMock()

    pipeline, extractor, embed = _make_pipeline(
        contradiction_checker=mock_checker,
    )
    embed._id_to_position = {}

    extractor.extract.return_value = _mock_extraction([
        ExtractedFact(text="A fact", confidence=0.9),
    ])

    # Mock the detector to inspect the call
    pipeline.detector = MagicMock()
    pipeline.detector.detect_for_node.return_value = []
    pipeline.detector.apply_dependencies.return_value = 0

    result = pipeline.ingest("text")

    assert result.nodes_added == 1
    # Verify contradiction_checker was passed
    call_kwargs = pipeline.detector.detect_for_node.call_args
    assert call_kwargs[1]["contradiction_checker"] is mock_checker


# ---------------------------------------------------------------------------
# Tests: tiered retrieval (neighbor expansion)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Tests: entity-based retrieval
# ---------------------------------------------------------------------------

def test_entity_retrieval_finds_matching_nodes():
    """Entity overlap retrieval finds nodes sharing content words."""
    graph = KnowledgeGraph()
    # "Tyrese Gibson is an American singer" has low embedding similarity to
    # "Tyrese Gibson is a cat" but shares the entity "Tyrese Gibson"
    existing = KnowledgeNode(
        text="Tyrese Gibson is an American singer and songwriter",
        confidence=0.95,
        domain="general",
    )
    graph.add_node(existing)

    pipeline, extractor, embed = _make_pipeline(graph=graph)

    results = pipeline._find_entity_matches(
        "Tyrese Gibson is a cat",
        exclude_ids=set(),
    )

    assert len(results) >= 1
    assert results[0]["node_id"] == existing.id
    assert results[0]["source"] == "entity_overlap"
    assert results[0]["overlap"] >= 2  # "tyrese", "gibson"


def test_entity_retrieval_requires_min_overlap():
    """Nodes with <2 shared content words are not returned."""
    graph = KnowledgeGraph()
    existing = KnowledgeNode(
        text="Water boils at 100 degrees Celsius",
        confidence=0.95,
    )
    graph.add_node(existing)

    pipeline, extractor, embed = _make_pipeline(graph=graph)

    # "cat" shares no content words with "water boils at 100..."
    results = pipeline._find_entity_matches(
        "A cat sat on the mat",
        exclude_ids=set(),
    )

    assert len(results) == 0


def test_entity_retrieval_excludes_known_ids():
    """Nodes already in the candidate set are excluded."""
    graph = KnowledgeGraph()
    existing = KnowledgeNode(
        text="Tyrese Gibson is a singer",
        confidence=0.95,
    )
    graph.add_node(existing)

    pipeline, extractor, embed = _make_pipeline(graph=graph)

    results = pipeline._find_entity_matches(
        "Tyrese Gibson is a cat",
        exclude_ids={existing.id},
    )

    assert len(results) == 0


def test_entity_retrieval_respects_confidence_threshold():
    """Low-confidence nodes are not returned by entity retrieval."""
    graph = KnowledgeGraph()
    existing = KnowledgeNode(
        text="Tyrese Gibson is an American singer",
        confidence=0.3,  # below default threshold
    )
    graph.add_node(existing)

    pipeline, extractor, embed = _make_pipeline(graph=graph, contradiction_threshold=0.5)

    results = pipeline._find_entity_matches(
        "Tyrese Gibson is a cat",
        exclude_ids=set(),
    )

    assert len(results) == 0


def test_entity_retrieval_in_pipeline_finds_contradiction():
    """Entity retrieval in the full pipeline catches cross-domain contradictions."""
    from core.llm_contradiction import RelevanceResult

    graph = KnowledgeGraph()
    existing = KnowledgeNode(
        text="Tyrese Gibson is an American singer and actor",
        confidence=0.95,
        domain="general",
    )
    graph.add_node(existing)

    # LLM correctly identifies "is a cat" vs "is a singer" as contradiction
    mock_checker = MagicMock()
    mock_checker.check_relevance_batch.return_value = [
        RelevanceResult(
            claim="Tyrese Gibson is a cat",
            knowledge="Tyrese Gibson is an American singer and actor",
            relationship="contradicts",
            confidence=0.95,
            explanation="A person cannot be a cat",
        ),
    ]

    pipeline, extractor, embed = _make_pipeline(
        graph=graph,
        contradiction_checker=mock_checker,
    )
    type(embed).index_size = PropertyMock(return_value=1)

    # Embedding search returns NOTHING (low cosine similarity)
    embed.search.return_value = []

    extractor.extract.return_value = _mock_extraction([
        ExtractedFact(
            text="Tyrese Gibson is a cat",
            confidence=0.85,
        ),
    ])

    result = pipeline.ingest("Tyrese Gibson is a cat.")

    # Entity retrieval should have found the existing node despite
    # embedding search returning nothing, and LLM confirms contradiction
    assert result.facts_flagged == 1
    verdict = result.verdicts[0]
    assert verdict.action == "flagged"
    assert "contradiction" in verdict.reason.lower()


def test_extract_content_words():
    """extract_content_words removes stopwords and short tokens."""
    from src.core.entity_index import extract_content_words
    words = extract_content_words("The water boils at 100 degrees Celsius")

    assert "the" not in words  # stopword
    assert "at" not in words   # stopword
    assert "water" in words
    assert "boils" in words
    assert "100" in words
    assert "degrees" in words
    assert "celsius" in words


def test_entity_retrieval_disabled():
    """With entity_retrieval=False, no entity matches are returned."""
    graph = KnowledgeGraph()
    existing = KnowledgeNode(
        text="Tyrese Gibson is an American singer",
        confidence=0.95,
    )
    graph.add_node(existing)

    pipeline, extractor, embed = _make_pipeline(graph=graph)
    pipeline.entity_retrieval = False

    type(embed).index_size = PropertyMock(return_value=1)
    # Embedding search returns nothing
    embed.search.return_value = []

    extractor.extract.return_value = _mock_extraction([
        ExtractedFact(text="Tyrese Gibson is a cat", confidence=0.85),
    ])

    result = pipeline.ingest("Tyrese Gibson is a cat.")

    # Without entity retrieval and no embedding match, the fact passes through
    assert result.nodes_added == 1
    assert result.facts_flagged == 0


# ---------------------------------------------------------------------------
# Tests: contradiction-before-dedup ordering
# ---------------------------------------------------------------------------

def test_contradiction_before_dedup_flags_correctly():
    """
    A fact that is both a near-duplicate AND a contradiction should be
    flagged as a contradiction, not silently merged as a duplicate.

    Example: "born on the 22nd" vs "born on the 21st" — high embedding
    similarity but contradictory dates.
    """
    from core.llm_contradiction import RelevanceResult

    graph = KnowledgeGraph()
    existing = KnowledgeNode(
        text="Chad Hayes was born on April 21st 1961",
        confidence=0.95,
    )
    graph.add_node(existing)

    mock_checker = MagicMock()
    mock_checker.check_relevance_batch.return_value = [
        RelevanceResult(
            claim="Chad Hayes was born on April 22nd 1961",
            knowledge="Chad Hayes was born on April 21st 1961",
            relationship="contradicts",
            confidence=0.92,
            explanation="Birth dates differ: April 21 vs April 22",
        ),
    ]

    pipeline, extractor, embed = _make_pipeline(
        graph=graph,
        contradiction_checker=mock_checker,
    )
    pipeline.entity_retrieval = False
    type(embed).index_size = PropertyMock(return_value=1)

    # This node would match both as duplicate (0.92 similarity) and as
    # contradiction candidate. Contradiction check runs first now.
    embed.search.return_value = [{"node_id": existing.id, "score": 0.92}]

    extractor.extract.return_value = _mock_extraction([
        ExtractedFact(
            text="Chad Hayes was born on April 22nd 1961",
            confidence=0.85,
        ),
    ])

    result = pipeline.ingest("Chad Hayes was born on April 22nd.")

    # Should be flagged as contradiction, NOT merged as duplicate
    assert result.facts_flagged == 1
    assert result.duplicates_merged == 0
    verdict = result.verdicts[0]
    assert verdict.action == "flagged"
    assert "contradiction" in verdict.reason.lower()


def test_true_duplicate_still_merges():
    """A genuine duplicate (no contradiction) is still merged correctly."""
    graph = KnowledgeGraph()
    existing = KnowledgeNode(
        text="Water boils at 100 degrees Celsius",
        confidence=0.8,
    )
    graph.add_node(existing)

    pipeline, extractor, embed = _make_pipeline(graph=graph)
    pipeline.entity_retrieval = False
    type(embed).index_size = PropertyMock(return_value=1)

    # Contradiction search finds the node but heuristic says no contradiction
    # Then dedup search also finds it (high similarity)
    embed.search.side_effect = [
        [{"node_id": existing.id, "score": 0.95}],  # contradiction search
        [{"node_id": existing.id, "score": 0.95}],  # dedup search
    ]

    extractor.extract.return_value = _mock_extraction([
        ExtractedFact(
            text="Water boils at 100 degrees C",
            confidence=0.85,
        ),
    ])

    result = pipeline.ingest("Water boils at 100C.")

    # Should merge as duplicate since no contradiction detected
    assert result.duplicates_merged == 1
    assert result.facts_flagged == 0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_tests():
    print("\nRunning pipeline tests...\n")
    test_basic_ingest_adds_nodes()
    test_ingest_attaches_source()
    test_ingest_sets_metadata()
    test_result_summary()
    test_low_confidence_facts_filtered()
    test_all_facts_filtered()
    test_low_quality_source_caps_confidence()
    test_high_quality_source_no_cap()
    test_no_source_no_cap()
    test_duplicate_merged_with_confidence_boost()
    test_duplicate_adds_new_source()
    test_below_duplicate_threshold_not_merged()
    test_contradiction_flagged()
    test_contradiction_rejected()
    test_contradiction_against_low_confidence_not_flagged()
    test_batch_ingestion()
    test_batch_result_properties()
    test_dependency_detection_called_for_new_nodes()
    test_confidence_propagation_runs()
    test_empty_extraction()
    test_empty_text()
    test_build_index_on_empty_graph()
    test_build_index_on_populated_graph()
    test_llm_enhanced_contradiction_flags()
    test_llm_corrects_heuristic_false_positive()
    test_pipeline_fallback_without_llm()
    test_llm_verdict_includes_explanation()
    test_llm_checker_threaded_to_dependency_detector()
    test_entity_retrieval_finds_matching_nodes()
    test_entity_retrieval_requires_min_overlap()
    test_entity_retrieval_excludes_known_ids()
    test_entity_retrieval_respects_confidence_threshold()
    test_entity_retrieval_in_pipeline_finds_contradiction()
    test_extract_content_words()
    test_entity_retrieval_disabled()
    test_contradiction_before_dedup_flags_correctly()
    test_true_duplicate_still_merges()
    print("\nAll pipeline tests passed!\n")


# ---------------------------------------------------------------------------
# Tests: SourceTier enum
# ---------------------------------------------------------------------------

def test_source_tier_from_quality_score():
    """SourceTier.from_quality_score correctly classifies scores."""
    assert SourceTier.from_quality_score(0.95) == SourceTier.AUTHORITATIVE
    assert SourceTier.from_quality_score(0.85) == SourceTier.AUTHORITATIVE
    assert SourceTier.from_quality_score(0.7) == SourceTier.STANDARD
    assert SourceTier.from_quality_score(0.5) == SourceTier.STANDARD
    assert SourceTier.from_quality_score(0.3) == SourceTier.UNVERIFIED
    assert SourceTier.from_quality_score(0.0) == SourceTier.UNVERIFIED


def test_source_tier_confidence_cap():
    """Each tier has the correct confidence cap."""
    assert SourceTier.AUTHORITATIVE.confidence_cap == 1.0
    assert SourceTier.STANDARD.confidence_cap == 0.8
    assert SourceTier.UNVERIFIED.confidence_cap == 0.5


def test_source_tier_property():
    """Source.tier derives correctly from quality_score."""
    s_high = Source(source_type="textbook", title="T", quality_score=0.9)
    s_mid = Source(source_type="paper", title="P", quality_score=0.7)
    s_low = Source(source_type="web", title="W", quality_score=0.3)
    assert s_high.tier == SourceTier.AUTHORITATIVE
    assert s_mid.tier == SourceTier.STANDARD
    assert s_low.tier == SourceTier.UNVERIFIED


# ---------------------------------------------------------------------------
# Tests: tiered ingestion
# ---------------------------------------------------------------------------

def test_tiered_ingestion_caps_unverified_source():
    """UNVERIFIED tier caps confidence at 0.5."""
    pipeline, extractor, embed = _make_pipeline(use_tiered_ingestion=True)
    extractor.extract.return_value = _mock_extraction([
        ExtractedFact(text="Dubious claim", confidence=0.9, domain="general"),
    ])
    src = Source(source_type="web", title="Random blog", quality_score=0.3)

    result = pipeline.ingest("Dubious claim", source=src)

    assert result.nodes_added == 1
    node = pipeline.graph.get_node(result.verdicts[0].node_id)
    # source_quality_cap=0.54 (default) -> caps to 0.3 (source quality)
    # tier cap = 0.5 -> already below, so source quality gate wins
    assert node.confidence <= 0.5


def test_tiered_ingestion_caps_standard_source():
    """STANDARD tier caps confidence at 0.8."""
    pipeline, extractor, embed = _make_pipeline(use_tiered_ingestion=True)
    extractor.extract.return_value = _mock_extraction([
        ExtractedFact(text="Good claim", confidence=0.95, domain="chemistry"),
    ])
    src = Source(source_type="paper", title="Research Paper", quality_score=0.7)

    result = pipeline.ingest("Good claim", source=src)

    assert result.nodes_added == 1
    node = pipeline.graph.get_node(result.verdicts[0].node_id)
    assert node.confidence <= 0.8


def test_tiered_ingestion_no_cap_authoritative():
    """AUTHORITATIVE tier does not cap confidence."""
    pipeline, extractor, embed = _make_pipeline(use_tiered_ingestion=True)
    extractor.extract.return_value = _mock_extraction([
        ExtractedFact(text="Solid fact", confidence=0.95, domain="chemistry"),
    ])
    src = Source(source_type="textbook", title="Chem 101", quality_score=0.95)

    result = pipeline.ingest("Solid fact", source=src)

    assert result.nodes_added == 1
    node = pipeline.graph.get_node(result.verdicts[0].node_id)
    assert node.confidence == 0.95


def test_tiered_batch_sorts_by_tier():
    """ingest_batch with tiered ingestion processes high-quality first."""
    pipeline, extractor, embed = _make_pipeline(use_tiered_ingestion=True)
    call_order = []
    original_ingest = pipeline.ingest

    def tracking_ingest(text, source=None, domain=None):
        call_order.append(source.quality_score if source else 0.0)
        return original_ingest(text, source=source, domain=domain)

    pipeline.ingest = tracking_ingest

    extractor.extract.return_value = _mock_extraction([
        ExtractedFact(text="F", confidence=0.8),
    ])

    texts = ["Low quality", "High quality", "Medium quality"]
    sources = [
        Source(source_type="web", title="Blog", quality_score=0.3),
        Source(source_type="textbook", title="Text", quality_score=0.95),
        Source(source_type="paper", title="Paper", quality_score=0.7),
    ]

    pipeline.ingest_batch(texts, sources=sources)

    assert call_order == [0.95, 0.7, 0.3]  # sorted descending by quality


def test_tiered_ingestion_disabled_by_default():
    """When use_tiered_ingestion=False (default), no tier cap applied."""
    pipeline, extractor, embed = _make_pipeline()
    extractor.extract.return_value = _mock_extraction([
        ExtractedFact(text="Fact", confidence=0.9, domain="general"),
    ])
    # source quality >= source_quality_cap (0.54), so no legacy cap
    src = Source(source_type="web", title="Decent source", quality_score=0.6)

    result = pipeline.ingest("Fact", source=src)

    assert result.nodes_added == 1
    node = pipeline.graph.get_node(result.verdicts[0].node_id)
    assert node.confidence == 0.9  # no tier cap applied


def test_tiered_batch_sources_length_mismatch():
    """sources length mismatch raises ValueError."""
    pipeline, extractor, embed = _make_pipeline(use_tiered_ingestion=True)

    with pytest.raises(ValueError, match="sources length"):
        pipeline.ingest_batch(
            ["text1", "text2"],
            sources=[Source(source_type="web", title="S", quality_score=0.5)],
        )


# ---------------------------------------------------------------------------
# Tests: quote verification
# ---------------------------------------------------------------------------

def test_quote_verification_exact_match():
    """Quote that is a substring of source text passes."""
    pipeline, extractor, embed = _make_pipeline(verify_quotes=True)
    extractor.extract.return_value = _mock_extraction([
        ExtractedFact(
            text="Water boils at 100C",
            confidence=0.9,
            source_quote="Water boils at 100C",
        ),
    ])

    result = pipeline.ingest("Water boils at 100C at sea level.")

    assert result.nodes_added == 1
    node = pipeline.graph.get_node(result.verdicts[0].node_id)
    assert node.metadata.get("quote_verified") is True


def test_quote_verification_fuzzy_match():
    """Quote with token overlap above threshold passes."""
    pipeline, extractor, embed = _make_pipeline(
        verify_quotes=True, min_quote_overlap=0.6,
    )
    extractor.extract.return_value = _mock_extraction([
        ExtractedFact(
            text="Water boils at 100C",
            confidence=0.9,
            # 4 of 5 quote tokens appear in source
            source_quote="Water boils at 100C quickly",
        ),
    ])

    result = pipeline.ingest("Water boils at 100C at sea level.")

    assert result.nodes_added == 1


def test_quote_verification_failure_penalizes():
    """Hallucinated quote gets confidence penalty."""
    pipeline, extractor, embed = _make_pipeline(
        verify_quotes=True, quote_penalty=0.7,
    )
    extractor.extract.return_value = _mock_extraction([
        ExtractedFact(
            text="Water boils at 100C",
            confidence=0.9,
            source_quote="Completely unrelated fabricated text here today",
        ),
    ])

    result = pipeline.ingest("Water boils at 100C at sea level.")

    assert result.nodes_added == 1
    node = pipeline.graph.get_node(result.verdicts[0].node_id)
    assert node.metadata.get("quote_verified") is False
    # 0.9 * 0.7 = 0.63
    assert node.confidence <= 0.63 + 0.01


def test_quote_verification_penalty_drops_below_threshold():
    """Penalized fact that falls below min_extraction_confidence is dropped."""
    pipeline, extractor, embed = _make_pipeline(
        verify_quotes=True,
        quote_penalty=0.5,
        min_extraction_confidence=0.5,
    )
    extractor.extract.return_value = _mock_extraction([
        ExtractedFact(
            text="Dubious fact",
            confidence=0.6,  # 0.6 * 0.5 = 0.3 < 0.5 threshold
            source_quote="Totally made up quote text not in source",
        ),
    ])

    result = pipeline.ingest("Some real text about chemistry and elements.")

    assert result.nodes_added == 0


def test_quote_verification_empty_quote_passes():
    """Facts with no source_quote pass verification."""
    pipeline, extractor, embed = _make_pipeline(verify_quotes=True)
    extractor.extract.return_value = _mock_extraction([
        ExtractedFact(text="Fact without quote", confidence=0.9, source_quote=""),
    ])

    result = pipeline.ingest("Some text.")

    assert result.nodes_added == 1


def test_quote_verification_disabled():
    """When verify_quotes=False, no verification happens."""
    pipeline, extractor, embed = _make_pipeline(verify_quotes=False)
    extractor.extract.return_value = _mock_extraction([
        ExtractedFact(
            text="Fact",
            confidence=0.9,
            source_quote="Fabricated quote not in source text at all here",
        ),
    ])

    result = pipeline.ingest("Actual source text.")

    assert result.nodes_added == 1
    node = pipeline.graph.get_node(result.verdicts[0].node_id)
    assert "quote_verified" not in node.metadata


# ---------------------------------------------------------------------------
# Tests: detect_dependencies flag
# ---------------------------------------------------------------------------

def test_detect_dependencies_false_skips_detection():
    """When detect_dependencies=False, dependency detection is skipped."""
    pipeline, extractor, embed = _make_pipeline(detect_dependencies=False)
    embed._id_to_position = {}

    extractor.extract.return_value = _mock_extraction([
        ExtractedFact(text="Fact A", confidence=0.9),
        ExtractedFact(text="Fact B", confidence=0.8),
    ])

    pipeline.detector = MagicMock()
    pipeline.detector.detect_for_node.return_value = []
    pipeline.detector.apply_dependencies.return_value = 0

    result = pipeline.ingest("text")

    assert result.nodes_added == 2
    assert pipeline.detector.detect_for_node.call_count == 0
    assert result.dependencies_added == 0


def test_detect_dependencies_default_true():
    """Default detect_dependencies=True runs dependency detection."""
    pipeline, extractor, embed = _make_pipeline()
    embed._id_to_position = {}

    extractor.extract.return_value = _mock_extraction([
        ExtractedFact(text="Fact", confidence=0.9),
    ])

    pipeline.detector = MagicMock()
    pipeline.detector.detect_for_node.return_value = []
    pipeline.detector.apply_dependencies.return_value = 0

    result = pipeline.ingest("text")

    assert result.nodes_added == 1
    assert pipeline.detector.detect_for_node.call_count == 1


# ---------------------------------------------------------------------------
# Tests: 3-way relevance fallback + rebuild_contradictions + threshold
# ---------------------------------------------------------------------------

def test_confidence_threshold_07():
    """Relevance results below 0.7 confidence are NOT flagged as contradictions."""
    from core.llm_contradiction import RelevanceResult

    graph = KnowledgeGraph()
    existing = KnowledgeNode(text="The sky is blue", confidence=0.95)
    graph.add_node(existing)

    mock_checker = MagicMock()
    mock_checker.check_relevance_batch.return_value = [
        RelevanceResult(
            claim="The sky is green",
            knowledge="The sky is blue",
            relationship="contradicts",
            confidence=0.65,  # Below 0.7 threshold
            explanation="Colors differ",
        ),
    ]

    pipeline, extractor, embed = _make_pipeline(
        graph=graph,
        contradiction_checker=mock_checker,
    )
    pipeline.entity_retrieval = False
    type(embed).index_size = PropertyMock(return_value=1)
    embed.search.side_effect = [
        [{"node_id": existing.id, "score": 0.7}],
        [],  # dedup search (no contradiction → continues to dedup)
    ]

    extractor.extract.return_value = _mock_extraction([
        ExtractedFact(text="The sky is green", confidence=0.7),
    ])

    result = pipeline.ingest("The sky is green.")

    # Below threshold → added normally, not flagged
    assert result.facts_flagged == 0
    assert result.nodes_added == 1


def test_rebuild_contradictions_clears_and_rechecks():
    """rebuild_contradictions clears existing conflicts and re-runs checks."""
    from core.llm_contradiction import RelevanceResult

    graph = KnowledgeGraph()
    node_a = KnowledgeNode(
        text="Water boils at 100C",
        confidence=0.30,  # Was penalized by conflict
        domain="chemistry",
    )
    node_b = KnowledgeNode(
        text="Water freezes at 0C",
        confidence=0.95,
        domain="chemistry",
    )
    graph.add_node(node_a)
    graph.add_node(node_b)

    # Set contested status AFTER add_node (to avoid status_index issues)
    node_a.status = NodeStatus.CONTESTED
    node_a.conflicts_with = ["fake_id"]

    # LLM says these are NOT contradictions (irrelevant — different properties)
    mock_checker = MagicMock()
    mock_checker.check_relevance_batch.return_value = [
        RelevanceResult(
            claim="", knowledge="",
            relationship="irrelevant",
            confidence=0.9,
            explanation="Different properties",
        ),
    ]

    pipeline, extractor, embed = _make_pipeline(
        graph=graph,
        contradiction_checker=mock_checker,
    )
    type(embed).index_size = PropertyMock(return_value=2)
    embed.search.return_value = [{"node_id": node_b.id, "score": 0.6}]

    stats = pipeline.rebuild_contradictions()

    # Conflicts should be cleared
    assert stats["conflicts_cleared"] > 0
    assert stats["nodes_checked"] == 2
    # Node A's confidence should be restored from 0.30 floor
    assert node_a.confidence > 0.30
    assert node_a.status == NodeStatus.PROVISIONAL
    assert node_a.conflicts_with == []


# ---------------------------------------------------------------------------
# Content pre-filter tests
# ---------------------------------------------------------------------------

def test_content_overlap_gate_filters_no_overlap():
    """Candidates with zero content-word overlap should be filtered."""
    graph = KnowledgeGraph()
    node_a = KnowledgeNode(
        node_id="a", text="Jupiter has 95 known moons",
        node_type=NodeType.FACT, confidence=0.9,
        sources=[_source()],
    )
    graph.add_node(node_a)

    pipeline, mock_ext, mock_embed = _make_pipeline(
        graph=graph,
        min_content_overlap=1,
    )

    # Candidate has no content-word overlap with "Mars has two moons"
    # (actually they share "moons" — use a truly different text)
    node_b = KnowledgeNode(
        node_id="b", text="The speed of light is constant",
        node_type=NodeType.FACT, confidence=0.9,
        sources=[_source()],
    )
    graph.add_node(node_b)

    candidates = [{"node_id": "b", "score": 0.8}]
    result = pipeline._apply_content_filters(
        "Jupiter has the Great Red Spot", candidates
    )
    # "speed", "light", "constant" vs "jupiter", "great", "red", "spot" → 0 overlap
    assert len(result) == 0


def test_content_overlap_gate_keeps_overlapping():
    """Candidates with sufficient content-word overlap should be kept."""
    graph = KnowledgeGraph()
    node_a = KnowledgeNode(
        node_id="a", text="Mars has two small moons",
        node_type=NodeType.FACT, confidence=0.9,
        sources=[_source()],
    )
    graph.add_node(node_a)

    pipeline, mock_ext, mock_embed = _make_pipeline(
        graph=graph,
        min_content_overlap=1,
    )

    candidates = [{"node_id": "a", "score": 0.8}]
    result = pipeline._apply_content_filters(
        "Mars has a thin atmosphere and two moons", candidates
    )
    # "mars", "moons" overlap → kept
    assert len(result) == 1


def test_content_filters_disabled_by_default_in_tests():
    """With default test settings (overlap=0), no filtering occurs."""
    graph = KnowledgeGraph()
    node_a = KnowledgeNode(
        node_id="a", text="Completely unrelated text about cooking pasta",
        node_type=NodeType.FACT, confidence=0.9,
        sources=[_source()],
    )
    graph.add_node(node_a)

    pipeline, _, _ = _make_pipeline(graph=graph)
    candidates = [{"node_id": "a", "score": 0.8}]
    result = pipeline._apply_content_filters("Jupiter is a gas giant", candidates)
    # Filters disabled → all candidates pass through
    assert len(result) == 1


def test_unit_normalization_applied_to_llm_pairs():
    """When unit normalization is enabled, normalized text should be sent to the LLM."""
    from core.llm_contradiction import RelevanceResult

    graph = KnowledgeGraph()
    node_a = KnowledgeNode(
        node_id="a", text="The distance is 100 miles",
        node_type=NodeType.FACT, confidence=0.9,
        sources=[_source()],
    )
    graph.add_node(node_a)

    mock_checker = MagicMock()
    mock_checker.check_relevance_batch.return_value = [
        RelevanceResult(
            claim="", knowledge="",
            relationship="irrelevant",
            confidence=0.1,
            explanation="Same measurement",
        )
    ]

    pipeline, mock_ext, mock_embed = _make_pipeline(
        graph=graph,
        use_llm_contradiction=False,
        contradiction_checker=mock_checker,
        use_unit_normalization=True,
        min_content_overlap=0,
    )

    type(mock_embed).index_size = PropertyMock(return_value=1)
    mock_embed.search.return_value = [{"node_id": "a", "score": 0.85}]

    pipeline._find_contradictions("The distance is 160.93 km")

    # The LLM should have received normalized pairs (both in km)
    call_args = mock_checker.check_relevance_batch.call_args
    pairs = call_args[0][0]
    # Both texts should contain "km" after normalization
    assert "km" in pairs[0][0]  # query text normalized
    assert "km" in pairs[0][1]  # candidate text normalized


# ------------------------------------------------------------------
# RRF and BM25 integration tests
# ------------------------------------------------------------------

def test_rrf_fusion_basic():
    """RRF merges two ranked lists by reciprocal rank scores."""
    from core.pipeline import reciprocal_rank_fusion

    list_a = [("n1", 0.9), ("n2", 0.8), ("n3", 0.7)]
    list_b = [("n3", 5.0), ("n1", 4.0), ("n4", 3.0)]

    results = reciprocal_rank_fusion([list_a, list_b], k=60, top_n=10)
    ids = [r["node_id"] for r in results]
    # n1 and n3 appear in both lists, should rank highest
    assert "n1" in ids[:3]
    assert "n3" in ids[:3]
    # n4 only in one list, should be present
    assert "n4" in ids
    # All scores positive
    for r in results:
        assert r["score"] > 0


def test_rrf_fusion_empty():
    """RRF with empty lists returns empty."""
    from core.pipeline import reciprocal_rank_fusion
    assert reciprocal_rank_fusion([], k=60, top_n=10) == []
    assert reciprocal_rank_fusion([[], []], k=60, top_n=10) == []


def test_rrf_fusion_with_dicts():
    """RRF accepts dict-based ranked lists."""
    from core.pipeline import reciprocal_rank_fusion

    list_a = [{"node_id": "n1", "score": 0.9}]
    list_b = [("n2", 0.8)]

    results = reciprocal_rank_fusion([list_a, list_b], k=60, top_n=10)
    ids = [r["node_id"] for r in results]
    assert "n1" in ids
    assert "n2" in ids


def test_rrf_top_n_cap():
    """RRF respects top_n limit."""
    from core.pipeline import reciprocal_rank_fusion

    long_list = [(f"n{i}", float(i)) for i in range(50)]
    results = reciprocal_rank_fusion([long_list], k=60, top_n=5)
    assert len(results) == 5


def test_pipeline_bm25_calls_both():
    """With use_bm25=True, _find_contradictions calls both embedding and BM25 search."""
    graph = KnowledgeGraph()
    node_a = KnowledgeNode(
        node_id="a", text="Earth is 149.6 million km from the Sun",
        node_type=NodeType.FACT, confidence=0.9,
        sources=[_source()],
    )
    graph.add_node(node_a)

    pipeline, mock_ext, mock_embed = _make_pipeline(
        graph=graph,
        use_llm_contradiction=False,
        use_bm25=True,
    )

    type(mock_embed).index_size = PropertyMock(return_value=1)
    mock_embed.search.return_value = [{"node_id": "a", "score": 0.85}]

    # Mock BM25 on the entity index
    pipeline.entity_index.bm25_search = MagicMock(return_value=[("a", 3.5)])

    pipeline._find_contradictions("Earth is 230 million km from the Sun")

    # Both should have been called
    mock_embed.search.assert_called_once()
    pipeline.entity_index.bm25_search.assert_called_once()


def test_pipeline_legacy_path_without_bm25():
    """With use_bm25=False, _find_contradictions uses legacy entity overlap."""
    graph = KnowledgeGraph()
    node_a = KnowledgeNode(
        node_id="a", text="Earth is 149.6 million km from the Sun",
        node_type=NodeType.FACT, confidence=0.9,
        sources=[_source()],
    )
    graph.add_node(node_a)

    pipeline, mock_ext, mock_embed = _make_pipeline(
        graph=graph,
        use_llm_contradiction=False,
        use_bm25=False,
    )

    type(mock_embed).index_size = PropertyMock(return_value=1)
    mock_embed.search.return_value = [{"node_id": "a", "score": 0.85}]
    mock_embed.embed_text.return_value = [0.0] * 384
    mock_embed.get_embedding.return_value = [0.0] * 384

    pipeline._find_contradictions("Earth is 230 million km from the Sun")

    # Embedding search was called (legacy path)
    mock_embed.search.assert_called_once()


if __name__ == "__main__":
    run_all_tests()
