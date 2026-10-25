"""
Fact extraction engine using Ollama or Gemini for LLM integration.

Extracts structured facts, relationships, and metadata from natural
language text and converts them into KnowledgeNode objects ready to
be inserted into the knowledge graph.

Supports two backends:
- **Ollama** (default): local LLM via ``FactExtractionEngine``
- **Gemini**: Google's Gemini API via ``GeminiFactExtractor``

Use ``create_fact_extractor()`` factory to select backend via the
``VERINODE_EXTRACTOR_BACKEND`` environment variable.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .node import Derivation, KnowledgeNode, NodeType, Source

logger = logging.getLogger(__name__)


def _safe_float(value, default: float = 0.5) -> float:
    """Convert value to float, returning default on failure."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


# Lazy imports
_ollama = None
_genai = None


def _get_ollama():
    global _ollama
    if _ollama is None:
        try:
            import ollama

            _ollama = ollama
        except ImportError:
            raise ImportError("ollama is required for fact extraction. Install with: pip install ollama")
    return _ollama


def _get_genai():
    """Lazy-load the google.genai module."""
    global _genai
    if _genai is None:
        try:
            from google import genai

            _genai = genai
        except ImportError:
            raise ImportError(
                "google-genai is required for Gemini fact extraction. Install with: pip install google-genai"
            )
    return _genai


# ------------------------------------------------------------------
# Data structures for extraction results
# ------------------------------------------------------------------


@dataclass
class ExtractedFact:
    """A single fact extracted from text."""

    text: str
    fact_type: str = "fact"  # maps to NodeType value
    confidence: float = 0.5
    domain: str = "general"
    related_facts: List[int] = field(default_factory=list)  # indices into same batch
    source_quote: str = ""  # relevant span from source text
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """Result of a fact extraction operation."""

    facts: List[ExtractedFact]
    source_text: str
    model: str
    domain: Optional[str] = None
    raw_response: str = ""


# ------------------------------------------------------------------
# Prompts
# ------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """\
You are a precise knowledge extraction system. Your job is to extract \
ALL discrete, atomic facts from the provided text. Extract EVERY distinct \
claim, not just the first one.

Rules:
1. Each fact must be ATOMIC: one single claim per fact. Do NOT combine \
   multiple properties or measurements into one fact. \
   BAD: "Mars has a diameter of 6,779 km and orbits at 228 million km." \
   GOOD: Two separate facts: "Mars has a diameter of 6,779 km." and \
   "Mars orbits at 228 million km from the Sun."
2. Prefer simple, declarative sentences.
3. Assign a fact_type from: fact, rule, observation, inference, definition.
4. Estimate a confidence score (0.0 to 1.0) based on how well-supported \
   the claim is in the source text. Use 0.9+ for clearly stated facts, \
   0.7-0.9 for well-supported claims, 0.5-0.7 for implied or uncertain \
   claims.
5. source_quote MUST be an exact, verbatim substring copied from the input \
   text. Do NOT paraphrase or abbreviate. If the relevant span is longer \
   than 100 characters, truncate it at a natural word boundary.
6. Only list a fact's index in related_facts if that fact is a direct \
   logical prerequisite or consequence. Do NOT link facts merely because \
   they mention the same subject.
7. Assign a domain from: astronomy, biology, chemistry, computer_science, \
   earth_science, economics, geography, geology, history, linguistics, \
   literature, mathematics, medicine, philosophy, physics, \
   political_science, psychology, sociology, technology, general. \
   Use "general" if no specific domain applies.
8. Only extract facts explicitly stated or directly implied by the text. \
   Do NOT add background knowledge, inferences, or context not present \
   in the source passage.

You MUST respond with a JSON object containing a "facts" key with an array \
of ALL extracted facts. Each element has these fields:
  text, fact_type, confidence, domain, related_facts, source_quote

Example format:
{"facts": [{"text": "...", "fact_type": "fact", "confidence": 0.9, \
"domain": "general", "related_facts": [], "source_quote": "..."}]}

IMPORTANT: Extract ALL facts from the text, not just one. Each sentence \
may contain one or more facts.\
"""

EXTRACTION_USER_TEMPLATE = """\
Extract all discrete facts from the following text:

---
{text}
---

{domain_hint}\
"""


# ------------------------------------------------------------------
# Engine
# ------------------------------------------------------------------


class FactExtractionEngine:
    """
    Extracts structured facts from text using a local Ollama LLM.

    Uses Ollama's JSON mode (format="json") for reliable structured
    output from local models.

    Usage:
        engine = FactExtractionEngine()
        result = engine.extract("Water boils at 100C at sea level.")
        for fact in result.facts:
            print(fact.text, fact.confidence)
    """

    DEFAULT_MODEL = "qwen3:14b"

    def __init__(
        self,
        model: Optional[str] = None,
        host: Optional[str] = None,
        temperature: float = 0.0,
        num_ctx: int = 4096,
    ):
        """
        Initialize extraction engine.

        Args:
            model: Ollama model name (e.g. "qwen3:14b", "llama3.1").
                   Defaults to DEFAULT_MODEL.
            host: Ollama server URL. Defaults to http://localhost:11434.
            temperature: Sampling temperature (0 = deterministic).
            num_ctx: Context window size in tokens.
        """
        self.model = model or self.DEFAULT_MODEL
        self.temperature = temperature
        self.num_ctx = num_ctx

        ollama_mod = _get_ollama()
        kwargs = {}
        if host:
            kwargs["host"] = host
        self._client = ollama_mod.Client(**kwargs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        text: str,
        domain: Optional[str] = None,
    ) -> ExtractionResult:
        """
        Extract facts from a text passage.

        Args:
            text: The natural-language text to analyse.
            domain: Optional domain hint (e.g. "chemistry").

        Returns:
            ExtractionResult with a list of ExtractedFact objects.
        """
        domain_hint = f"Domain hint: {domain}" if domain else ""
        user_msg = EXTRACTION_USER_TEMPLATE.format(text=text, domain_hint=domain_hint)

        response = self._client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            format="json",
            think=False,  # Disable Qwen3 thinking mode for structured output (2-5x speedup)
            options={
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
            },
        )

        raw_text = response.message.content
        facts = self._parse_response(raw_text, domain)

        return ExtractionResult(
            facts=facts,
            source_text=text,
            model=self.model,
            domain=domain,
            raw_response=raw_text,
        )

    def extract_to_nodes(
        self,
        text: str,
        domain: Optional[str] = None,
        source: Optional[Source] = None,
    ) -> List[KnowledgeNode]:
        """
        Extract facts and convert them directly to KnowledgeNode objects.

        Args:
            text: Text to extract from.
            domain: Optional domain hint.
            source: Optional Source to attach.

        Returns:
            List of KnowledgeNode instances (not yet added to any graph).
        """
        result = self.extract(text, domain=domain)
        return self.facts_to_nodes(result, source=source)

    def facts_to_nodes(
        self,
        result: ExtractionResult,
        source: Optional[Source] = None,
    ) -> List[KnowledgeNode]:
        """
        Convert an ExtractionResult into KnowledgeNode objects.

        Args:
            result: ExtractionResult from extract().
            source: Optional Source to attach to every node.

        Returns:
            List of KnowledgeNode instances.
        """
        nodes: List[KnowledgeNode] = []
        node_type_map = {v.value: v for v in NodeType}

        for fact in result.facts:
            nt = node_type_map.get(fact.fact_type, NodeType.FACT)
            node = KnowledgeNode(
                text=fact.text,
                node_type=nt,
                confidence=fact.confidence,
                domain=fact.domain,
                derivation=Derivation(
                    method="extracted",
                    derivation_type="extracted",
                    notes=f"LLM extraction via {result.model}",
                ),
            )
            if source:
                node.add_source(source)
            node.metadata["source_quote"] = fact.source_quote
            node.metadata["extraction_model"] = result.model
            nodes.append(node)

        # Wire up related_facts as predecessor relationships
        for i, fact in enumerate(result.facts):
            for rel_idx in fact.related_facts:
                if 0 <= rel_idx < len(nodes) and rel_idx != i:
                    nodes[i].add_predecessor(nodes[rel_idx].id)

        return nodes

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(
        raw: str,
        default_domain: Optional[str] = None,
    ) -> List[ExtractedFact]:
        """Parse the LLM JSON response into ExtractedFact objects."""
        # Strip markdown code fences if present
        cleaned = raw.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Failed to parse extraction response as JSON")
            return []

        # Ollama JSON mode may return {"facts": [...]} wrapper
        if isinstance(data, dict):
            # Look for a list value in the dict
            for key in ("facts", "results", "extracted_facts", "data"):
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break
            else:
                # Single fact object, wrap it
                data = [data]

        if not isinstance(data, list):
            data = [data]

        facts = []
        # Track mapping from original LLM index → filtered list index
        # so related_facts references stay correct after filtering.
        original_to_filtered: Dict[int, int] = {}

        for original_idx, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            text = item.get("text", "").strip()
            if not text:
                continue

            filtered_idx = len(facts)
            original_to_filtered[original_idx] = filtered_idx

            fact = ExtractedFact(
                text=text,
                fact_type=item.get("fact_type", "fact"),
                confidence=_safe_float(item.get("confidence", 0.5), 0.5),
                domain=item.get("domain", default_domain or "general"),
                related_facts=item.get("related_facts", []),
                source_quote=item.get("source_quote", ""),
                metadata=item.get("metadata", {}),
            )
            # Clamp confidence
            fact.confidence = max(0.0, min(1.0, fact.confidence))
            facts.append(fact)

        # Remap related_facts from original LLM indices to filtered indices
        for fact in facts:
            fact.related_facts = [
                original_to_filtered[idx]
                for idx in fact.related_facts
                if isinstance(idx, int) and idx in original_to_filtered
            ]

        return facts


# ------------------------------------------------------------------
# Gemini backend
# ------------------------------------------------------------------


class GeminiFactExtractor:
    """
    Fact extraction using Google's Gemini API.

    Drop-in replacement for :class:`FactExtractionEngine`.  Uses the
    same prompts and parsing logic but calls Gemini instead of Ollama.

    Accepts an optional :class:`~src.core.rate_limiter.GeminiRateLimiter`
    for shared rate limiting across extraction and contradiction checking.
    """

    DEFAULT_MODEL = "gemini-2.0-flash"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        rate_limiter=None,
    ):
        """
        Initialize the Gemini fact extractor.

        Args:
            api_key: Google Gemini API key.  Falls back to
                ``VERINODE_GEMINI_API_KEY`` environment variable.
            model: Gemini model name.  Defaults to DEFAULT_MODEL.
            temperature: Sampling temperature (0 = deterministic).
            rate_limiter: Optional :class:`GeminiRateLimiter` instance
                for shared rate limiting.
        """
        self.model = model or self.DEFAULT_MODEL
        self.temperature = temperature
        self.rate_limiter = rate_limiter

        resolved_key = api_key or os.environ.get("VERINODE_GEMINI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Gemini API key required. Set VERINODE_GEMINI_API_KEY "
                "environment variable or pass api_key= to constructor."
            )

        genai = _get_genai()
        self._client = genai.Client(api_key=resolved_key)

    # ------------------------------------------------------------------
    # Public API (mirrors FactExtractionEngine)
    # ------------------------------------------------------------------

    def extract(
        self,
        text: str,
        domain: Optional[str] = None,
    ) -> ExtractionResult:
        """
        Extract facts from a text passage using Gemini.

        Args:
            text: The natural-language text to analyse.
            domain: Optional domain hint (e.g. "chemistry").

        Returns:
            ExtractionResult with a list of ExtractedFact objects.
        """
        from .rate_limiter import DailyLimitReached

        domain_hint = f"Domain hint: {domain}" if domain else ""
        user_msg = EXTRACTION_USER_TEMPLATE.format(text=text, domain_hint=domain_hint)
        prompt_text = EXTRACTION_SYSTEM_PROMPT + "\n\n" + user_msg

        max_attempts = (self.rate_limiter.max_retries + 1) if self.rate_limiter else 1

        for attempt in range(max_attempts):
            if self.rate_limiter:
                if not self.rate_limiter.wait_for_slot():
                    raise DailyLimitReached("Gemini daily budget exhausted during extraction")

            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=[
                        {"role": "user", "parts": [{"text": prompt_text}]},
                    ],
                    config={
                        "temperature": self.temperature,
                        "response_mime_type": "application/json",
                    },
                )
                raw_text = response.text
                break
            except Exception as e:
                if "429" in str(e) and self.rate_limiter:
                    if self.rate_limiter.handle_429(attempt):
                        continue
                logger.warning("Gemini extraction failed: %s", e)
                return ExtractionResult(
                    facts=[],
                    source_text=text,
                    model=self.model,
                    domain=domain,
                    raw_response="",
                )

        facts = FactExtractionEngine._parse_response(raw_text, domain)

        return ExtractionResult(
            facts=facts,
            source_text=text,
            model=self.model,
            domain=domain,
            raw_response=raw_text,
        )

    def extract_to_nodes(
        self,
        text: str,
        domain: Optional[str] = None,
        source: Optional[Source] = None,
    ) -> List[KnowledgeNode]:
        """Extract facts and convert to KnowledgeNode objects."""
        result = self.extract(text, domain=domain)
        return self.facts_to_nodes(result, source=source)

    def facts_to_nodes(
        self,
        result: ExtractionResult,
        source: Optional[Source] = None,
    ) -> List[KnowledgeNode]:
        """Convert ExtractionResult into KnowledgeNode objects."""
        # Delegate to the shared implementation
        engine = FactExtractionEngine.__new__(FactExtractionEngine)
        return engine.facts_to_nodes(result, source=source)


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------


def create_fact_extractor(
    model: Optional[str] = None,
    host: Optional[str] = None,
    backend: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    rate_limiter=None,
):
    """
    Factory: create a fact extractor for the configured backend.

    Backend selection (in order of priority):
    1. ``backend`` parameter
    2. ``VERINODE_EXTRACTOR_BACKEND`` environment variable
    3. Falls back to ``"ollama"``

    Args:
        model: Model name override.
        host: Ollama host URL (Ollama backend only).
        backend: ``"ollama"`` or ``"gemini"``.
        gemini_api_key: Gemini API key (Gemini backend only).
        rate_limiter: Optional :class:`GeminiRateLimiter` (Gemini only).

    Returns:
        A fact extractor instance, or None if initialization fails.
    """
    resolved_backend = (backend or os.environ.get("VERINODE_EXTRACTOR_BACKEND", "ollama")).lower().strip()

    try:
        if resolved_backend == "gemini":
            return GeminiFactExtractor(
                api_key=gemini_api_key,
                model=model,
                rate_limiter=rate_limiter,
            )
        else:
            return FactExtractionEngine(model=model, host=host)
    except Exception as e:
        logger.warning("Failed to create %s fact extractor: %s", resolved_backend, e)
        return None
