"""
LLM-assisted contradiction detection using Ollama.

Provides semantic contradiction checking that goes beyond simple
negation-word heuristics. Uses a local LLM to determine whether
two factual claims contradict each other, including:
- Numerical value contradictions
- Semantic contradictions without negation words
- Logically invalid inferences

Falls back gracefully when Ollama is unavailable.

Uses Ollama's JSON mode for grammar-constrained structured output,
following the same patterns as fact_extraction.py.
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Lazy import (same pattern as fact_extraction.py)
_ollama = None
_genai = None


def _get_ollama():
    """Lazy-load the ollama module."""
    global _ollama
    if _ollama is None:
        try:
            import ollama

            _ollama = ollama
        except ImportError:
            raise ImportError("ollama is required for LLM contradiction detection. Install with: pip install ollama")
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
                "google-genai is required for Gemini contradiction detection. Install with: pip install google-genai"
            )
    return _genai


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------


@dataclass
class RelevanceResult:
    """Result of a 3-way relevance classification (contradicts/supports/irrelevant)."""

    claim: str
    knowledge: str
    relationship: str  # "contradicts", "supports", "irrelevant"
    confidence: float  # 0.0-1.0
    explanation: str
    raw_response: str = ""


# ------------------------------------------------------------------
# Prompts
# ------------------------------------------------------------------

RELEVANCE_SYSTEM_PROMPT = """\
You are a precise evidence relevance classifier. Given a CLAIM and a \
KNOWLEDGE statement, classify their relationship into exactly one of \
three categories.

Categories:
1. "contradicts" - The knowledge statement contradicts the claim. \
They CANNOT both be true. The knowledge addresses the SAME specific \
property/predicate as the claim but with an incompatible value. \
Example: Claim "Gold melts at 500C" vs Knowledge "Gold melts at 1064C" \
-> contradicts (same property: melting point, different values).
2. "supports" - The knowledge statement directly supports the claim. \
It addresses the SAME specific property/predicate AND is consistent. \
Example: Claim "Water boils at 100C" vs Knowledge "Water boils at \
100 degrees Celsius at sea level" -> supports.
3. "irrelevant" - The knowledge is about the same subject but \
addresses a DIFFERENT property, or is otherwise not directly relevant \
to the specific claim. Example: Claim "Gold melts at 500C" vs \
Knowledge "Gold is a chemical element with symbol Au" -> irrelevant \
(about gold, but not about melting point).

Key rules:
- "supports" requires the knowledge to address the SAME specific \
property/predicate as the claim AND be consistent with it.
- General background facts about the same subject are "irrelevant" \
unless they directly confirm or deny the claim's specific assertion.
- Numbers, dates, or measurements that differ for the SAME property \
are "contradicts".
- If the knowledge is about the same subject but a different property \
-> "irrelevant".

Examples to avoid false positives:
- Claim: "Mars has two moons" vs Knowledge: "Jupiter has 95 moons" \
-> "irrelevant" (different planets entirely).
- Claim: "The Sun's core temperature is 15 million C" \
vs Knowledge: "The Sun's surface temperature is 5,778 K" \
-> "irrelevant" (different properties: core vs surface).
- Claim: "Water boils at 100C" \
vs Knowledge: "Water boils at 100 degrees Celsius at standard pressure" \
-> "supports" (same property, consistent with more detail).
- Claim: "The Earth orbits the Sun at 150 million km" \
vs Knowledge: "The Earth orbits the Sun at 108 million km" \
-> "contradicts" (same property, incompatible values).

Temporal awareness:
- Statements about different time periods are NOT contradictions. \
"Pluto was the 9th planet" vs "The Solar System has 8 planets" \
-> "irrelevant" (reclassification over time, not a direct contradiction).

Scope-limited claims:
- A statement qualified with "at sea level" or "at standard pressure" \
is compatible with the unqualified version -> "supports".

Respond with a JSON object:
  relationship: "contradicts" | "supports" | "irrelevant"
  confidence: float (0.0 to 1.0) — use 0.85+ for clear cases, 0.7-0.85 \
  when fairly sure, 0.5-0.7 for borderline.
  explanation: string (brief reason)\
"""

RELEVANCE_USER_TEMPLATE = """\
Classify the relationship between this claim and knowledge statement. \
Remember: facts about DIFFERENT subjects or different properties of \
the same subject are "irrelevant", not contradictions.

Claim: {claim}
Knowledge: {knowledge}

Respond with JSON only.\
"""


# ------------------------------------------------------------------
# Engine
# ------------------------------------------------------------------


class ContradictionChecker:
    """
    LLM-assisted 3-way relevance classification using Ollama.

    Uses a local LLM to classify the relationship between a claim and
    a knowledge statement as "contradicts", "supports", or "irrelevant".

    Usage:
        checker = ContradictionChecker()
        result = checker.check_relevance(
            "Water freezes at 0 degrees Celsius",
            "Water freezes at -100 degrees Celsius"
        )
        if result.relationship == "contradicts":
            print(f"Contradiction: {result.explanation}")
    """

    DEFAULT_MODEL = "qwen3:14b"

    def __init__(
        self,
        model: Optional[str] = None,
        host: Optional[str] = None,
        temperature: float = 0.0,
        num_ctx: int = 1024,
    ):
        """
        Initialize the contradiction checker.

        Args:
            model: Ollama model name. Defaults to DEFAULT_MODEL.
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

        # Directional relevance cache keyed on (claim, knowledge) tuple.
        self._relevance_cache: Dict[tuple, RelevanceResult] = {}

    def clear_cache(self) -> None:
        """Clear the session-scoped relevance result cache."""
        self._relevance_cache.clear()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_relevance(
        self,
        claim: str,
        knowledge: str,
    ) -> RelevanceResult:
        """
        Classify the relationship between a claim and a knowledge statement
        as "contradicts", "supports", or "irrelevant".

        Results are cached by the directional (claim, knowledge) tuple.

        Args:
            claim: The factual claim being verified.
            knowledge: An existing knowledge statement from the graph.

        Returns:
            RelevanceResult with relationship classification.
        """
        cache_key = (claim, knowledge)
        cached = self._relevance_cache.get(cache_key)
        if cached is not None:
            logger.debug("check_relevance cache hit")
            return cached

        user_msg = RELEVANCE_USER_TEMPLATE.format(claim=claim, knowledge=knowledge)

        try:
            response = self._client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": RELEVANCE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                format="json",
                think=False,
                options={
                    "temperature": self.temperature,
                    "num_ctx": self.num_ctx,
                },
            )
            raw_text = response.message.content
        except Exception as e:
            logger.warning("LLM relevance check failed: %s", e)
            return RelevanceResult(
                claim=claim,
                knowledge=knowledge,
                relationship="irrelevant",
                confidence=0.0,
                explanation=f"LLM call failed: {e}",
            )

        parsed = self._parse_relevance_response(raw_text)

        result = RelevanceResult(
            claim=claim,
            knowledge=knowledge,
            relationship=parsed["relationship"],
            confidence=parsed["confidence"],
            explanation=parsed["explanation"],
            raw_response=raw_text,
        )
        self._relevance_cache[cache_key] = result
        return result

    def check_relevance_batch(
        self,
        pairs: List[Tuple[str, str]],
        max_workers: int = 2,
    ) -> List[RelevanceResult]:
        """
        Classify multiple (claim, knowledge) pairs with optional parallelism.

        Args:
            pairs: List of (claim, knowledge) tuples.
            max_workers: Number of concurrent threads.

        Returns:
            List of RelevanceResult, one per pair (order preserved).
        """
        if max_workers <= 1 or len(pairs) <= 1:
            return [self.check_relevance(c, k) for c, k in pairs]

        from concurrent.futures import ThreadPoolExecutor, as_completed

        results: List[Optional[RelevanceResult]] = [None] * len(pairs)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(self.check_relevance, c, k): i for i, (c, k) in enumerate(pairs)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.warning("Batch relevance pair %d failed: %s", idx, e)
                    c, k = pairs[idx]
                    results[idx] = RelevanceResult(
                        claim=c,
                        knowledge=k,
                        relationship="irrelevant",
                        confidence=0.0,
                        explanation=f"Batch check failed: {e}",
                    )

        # Fill any remaining None slots
        for i, r in enumerate(results):
            if r is None:
                c, k = pairs[i]
                results[i] = RelevanceResult(
                    claim=c,
                    knowledge=k,
                    relationship="irrelevant",
                    confidence=0.0,
                    explanation="Result was unexpectedly None",
                )

        return results  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_relevance_response(raw: str) -> dict:
        """
        Parse the LLM relevance JSON response.

        Returns a dict with keys: relationship, confidence, explanation.
        Defaults unrecognized relationship values to "irrelevant".
        """
        cleaned = raw.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        defaults = {
            "relationship": "irrelevant",
            "confidence": 0.0,
            "explanation": "",
        }

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Failed to parse relevance response as JSON")
            defaults["explanation"] = "Failed to parse LLM response"
            return defaults

        if not isinstance(data, dict):
            defaults["explanation"] = "LLM response was not a JSON object"
            return defaults

        valid_relationships = {"contradicts", "supports", "irrelevant"}
        relationship = data.get("relationship", "irrelevant")
        if isinstance(relationship, str):
            relationship = relationship.lower().strip()
        if relationship not in valid_relationships:
            relationship = "irrelevant"

        confidence = float(data.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))

        explanation = str(data.get("explanation", ""))

        return {
            "relationship": relationship,
            "confidence": confidence,
            "explanation": explanation,
        }


# ------------------------------------------------------------------
# Gemini backend
# ------------------------------------------------------------------


class GeminiContradictionChecker:
    """
    LLM-assisted 3-way relevance classification using Google's Gemini API.

    Drop-in replacement for ContradictionChecker with the same public
    interface (check_relevance, check_relevance_batch).
    Uses the same prompts and JSON parsing.

    Requires:
        pip install google-genai
        Set VERINODE_GEMINI_API_KEY environment variable.

    Usage:
        checker = GeminiContradictionChecker(api_key="...")
        result = checker.check_relevance("Claim A", "Knowledge B")
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
        Initialize the Gemini contradiction checker.

        Args:
            api_key: Google Gemini API key.  Falls back to
                VERINODE_GEMINI_API_KEY environment variable.
            model: Gemini model name.  Defaults to DEFAULT_MODEL.
            temperature: Sampling temperature (0 = deterministic).
            rate_limiter: Optional :class:`GeminiRateLimiter` instance
                for shared rate limiting.
        """
        import os

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

        # Directional relevance cache keyed on (claim, knowledge) tuple.
        self._relevance_cache: Dict[tuple, RelevanceResult] = {}

    def clear_cache(self) -> None:
        """Clear the session-scoped relevance result cache."""
        self._relevance_cache.clear()

    # ------------------------------------------------------------------
    # Public API (mirrors ContradictionChecker)
    # ------------------------------------------------------------------

    def check_relevance(self, claim: str, knowledge: str) -> RelevanceResult:
        """Classify the relationship between a claim and knowledge statement."""
        from .rate_limiter import DailyLimitReached

        cache_key = (claim, knowledge)
        cached = self._relevance_cache.get(cache_key)
        if cached is not None:
            return cached

        user_msg = RELEVANCE_USER_TEMPLATE.format(claim=claim, knowledge=knowledge)
        prompt_text = RELEVANCE_SYSTEM_PROMPT + "\n\n" + user_msg

        max_attempts = (self.rate_limiter.max_retries + 1) if self.rate_limiter else 1
        raw_text = ""

        for attempt in range(max_attempts):
            if self.rate_limiter:
                if not self.rate_limiter.wait_for_slot():
                    raise DailyLimitReached("Gemini daily budget exhausted during relevance check")

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
            except DailyLimitReached:
                raise
            except Exception as e:
                if "429" in str(e) and self.rate_limiter:
                    if self.rate_limiter.handle_429(attempt):
                        continue
                logger.warning("Gemini relevance check failed: %s", e)
                return RelevanceResult(
                    claim=claim,
                    knowledge=knowledge,
                    relationship="irrelevant",
                    confidence=0.0,
                    explanation=f"Gemini call failed: {e}",
                )

        parsed = ContradictionChecker._parse_relevance_response(raw_text)

        result = RelevanceResult(
            claim=claim,
            knowledge=knowledge,
            relationship=parsed["relationship"],
            confidence=parsed["confidence"],
            explanation=parsed["explanation"],
            raw_response=raw_text,
        )
        self._relevance_cache[(claim, knowledge)] = result
        return result

    def check_relevance_batch(
        self,
        pairs: List[Tuple[str, str]],
        max_workers: int = 1,
    ) -> List[RelevanceResult]:
        """Classify multiple (claim, knowledge) pairs sequentially.

        Each pair is checked individually via :meth:`check_relevance`,
        which handles caching and rate limiting per call.
        """
        if not pairs:
            return []

        results: List[RelevanceResult] = []
        for c, k in pairs:
            results.append(self.check_relevance(c, k))

        return results

        return results  # type: ignore[return-value]


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------


def create_contradiction_checker(
    model: Optional[str] = None,
    host: Optional[str] = None,
    backend: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    rate_limiter=None,
) -> Optional[ContradictionChecker]:
    """
    Factory function that returns a contradiction checker or None.

    Returns None (instead of raising) if the chosen backend is
    unavailable.  This is the recommended way to create a checker
    for optional use.

    Args:
        model: Model name (Ollama model or Gemini model).
        host: Ollama server URL (Ollama backend only).
        backend: "ollama" (default) or "gemini".  Falls back to
            VERINODE_CHECKER_BACKEND environment variable.
        gemini_api_key: Gemini API key (overrides VERINODE_GEMINI_API_KEY
            environment variable).
        rate_limiter: Optional :class:`GeminiRateLimiter` instance
            for shared rate limiting (Gemini backend only).

    Returns:
        ContradictionChecker or GeminiContradictionChecker instance,
        or None if the chosen backend is unavailable.
    """
    import os

    resolved_backend = (backend or os.environ.get("VERINODE_CHECKER_BACKEND", "ollama")).lower().strip()

    if resolved_backend == "gemini":
        try:
            return GeminiContradictionChecker(
                api_key=gemini_api_key,
                model=model,
                rate_limiter=rate_limiter,
            )
        except Exception as e:
            logger.info("Gemini not available; LLM contradiction detection disabled: %s", e)
            return None
    else:
        try:
            return ContradictionChecker(model=model, host=host)
        except Exception:
            logger.info("Ollama not available; LLM contradiction detection disabled")
            return None
