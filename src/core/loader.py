"""
Web content loader and text chunker for knowledge graph ingestion.

Fetches URLs, extracts article text using trafilatura, and splits
into pipeline-ready chunks that fit within the LLM context window.

Dependencies (in requirements.txt):
    trafilatura>=1.6.0
    requests>=2.28.0
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# Lazy imports
_trafilatura = None
_requests = None


def _get_trafilatura():
    global _trafilatura
    if _trafilatura is None:
        try:
            import trafilatura
            _trafilatura = trafilatura
        except ImportError:
            raise ImportError(
                "trafilatura is required for web content loading. "
                "Install with: pip install trafilatura"
            )
    return _trafilatura


def _get_requests():
    global _requests
    if _requests is None:
        try:
            import requests
            _requests = requests
        except ImportError:
            raise ImportError(
                "requests is required for web content loading. "
                "Install with: pip install requests"
            )
    return _requests


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class LoadedContent:
    """Result of fetching and extracting text from a URL."""
    url: str
    title: str = ""
    author: str = ""
    date: str = ""
    text: str = ""
    chunks: List[str] = field(default_factory=list)
    success: bool = True
    error: str = ""


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def fetch_and_extract(url: str, timeout: int = 30) -> LoadedContent:
    """
    Fetch a URL and extract clean article text.

    Uses ``trafilatura`` for article extraction (handles boilerplate
    removal, navigation stripping, comment filtering).

    Args:
        url: The URL to fetch.
        timeout: HTTP timeout in seconds.

    Returns:
        LoadedContent with extracted text and metadata.
    """
    traf = _get_trafilatura()
    requests = _get_requests()

    try:
        resp = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (compatible; EngramAI/1.0)"},
        )
        resp.raise_for_status()
        html = resp.text
    except Exception as e:
        return LoadedContent(url=url, success=False, error=f"Fetch failed: {e}")

    # Extract article text
    text = traf.extract(
        html,
        include_comments=False,
        include_tables=True,
        favor_precision=True,
    )

    if not text:
        return LoadedContent(url=url, success=False, error="No text extracted")

    # Extract metadata
    title = ""
    author = ""
    date = ""
    try:
        metadata = traf.extract_metadata(html)
        if metadata:
            title = metadata.title or ""
            author = metadata.author or ""
            date = metadata.date or ""
    except Exception:
        pass  # metadata is best-effort

    return LoadedContent(
        url=url,
        title=title,
        author=author,
        date=date,
        text=text,
        success=True,
    )


def chunk_text(
    text: str,
    max_tokens: int = 1500,
    overlap_sentences: int = 1,
) -> List[str]:
    """
    Split text into chunks suitable for the 4096-token context window.

    Strategy: split on paragraph boundaries (``\\n\\n``), then merge
    paragraphs into chunks up to ``max_tokens``.  If a single paragraph
    exceeds the limit, split on sentence boundaries with overlap.

    Uses a conservative token estimate: 1 token ~= 4 characters.

    Args:
        text: The full extracted text.
        max_tokens: Maximum tokens per chunk (default 1500, well within
            the 4096 context window after accounting for the system
            prompt and extraction instructions).
        overlap_sentences: Number of sentences to overlap between
            chunks for continuity.

    Returns:
        List of text chunks ready for ``pipeline.ingest_batch()``.
    """
    max_chars = max_tokens * 4  # conservative estimate

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: List[str] = []
    current_chunk: List[str] = []
    current_size = 0

    for para in paragraphs:
        para_size = len(para)

        if para_size > max_chars:
            # Flush current chunk first
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_size = 0

            # Split oversized paragraph on sentences
            sentences = _split_sentences(para)
            sent_chunk: List[str] = []
            sent_size = 0
            for sent in sentences:
                if sent_size + len(sent) > max_chars and sent_chunk:
                    chunks.append(" ".join(sent_chunk))
                    # Overlap: keep last N sentences for continuity
                    sent_chunk = sent_chunk[-overlap_sentences:]
                    sent_size = sum(len(s) for s in sent_chunk)
                sent_chunk.append(sent)
                sent_size += len(sent)
            if sent_chunk:
                chunks.append(" ".join(sent_chunk))

        elif current_size + para_size > max_chars:
            # Flush current chunk, start new one
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_size = para_size
        else:
            current_chunk.append(para)
            current_size += para_size

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return [c for c in chunks if c.strip()]


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _split_sentences(text: str) -> List[str]:
    """Simple sentence splitter using regex."""
    # Split on period/question/exclamation followed by space and uppercase
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [p.strip() for p in parts if p.strip()]
