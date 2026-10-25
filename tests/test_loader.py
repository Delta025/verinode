"""
Tests for the web content loader and text chunker.

All tests mock ``requests`` and ``trafilatura`` so they run without
network access or external dependencies.

Covers:
- LoadedContent dataclass construction and defaults
- fetch_and_extract: success, HTTP errors, timeouts, extraction failures,
  metadata fallback
- chunk_text: empty input, short text, multi-paragraph merging, oversized
  paragraph splitting, sentence overlap
- _split_sentences: basic sentence boundary detection

Run with:  python -m pytest tests/test_loader.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.loader import LoadedContent, fetch_and_extract, chunk_text, _split_sentences


# ---------------------------------------------------------------------------
# LoadedContent dataclass
# ---------------------------------------------------------------------------

def test_loaded_content_defaults():
    """Minimal construction with just the url fills in sane defaults."""
    lc = LoadedContent(url="https://example.com")
    assert lc.url == "https://example.com"
    assert lc.title == ""
    assert lc.author == ""
    assert lc.date == ""
    assert lc.text == ""
    assert lc.chunks == []
    assert lc.success is True
    assert lc.error == ""


def test_loaded_content_full_construction():
    """All fields can be set explicitly."""
    lc = LoadedContent(
        url="https://example.com/article",
        title="Test Title",
        author="Jane Doe",
        date="2025-01-15",
        text="Some text",
        chunks=["chunk1", "chunk2"],
        success=True,
        error="",
    )
    assert lc.title == "Test Title"
    assert lc.author == "Jane Doe"
    assert lc.date == "2025-01-15"
    assert lc.text == "Some text"
    assert lc.chunks == ["chunk1", "chunk2"]


# ---------------------------------------------------------------------------
# fetch_and_extract  (mock requests + trafilatura)
# ---------------------------------------------------------------------------

def _patch_loader():
    """Return patches for the lazy-imported requests and trafilatura modules."""
    mock_requests = MagicMock()
    mock_traf = MagicMock()

    return (
        patch("core.loader._get_requests", return_value=mock_requests),
        patch("core.loader._get_trafilatura", return_value=mock_traf),
        mock_requests,
        mock_traf,
    )


def test_fetch_and_extract_success():
    """Successful fetch returns text and metadata."""
    p_req, p_traf, mock_requests, mock_traf = _patch_loader()

    with p_req, p_traf:
        # Simulate successful HTTP response
        mock_resp = MagicMock()
        mock_resp.text = "<html><body>Article HTML</body></html>"
        mock_resp.raise_for_status = MagicMock()
        mock_requests.get.return_value = mock_resp

        # Simulate trafilatura extraction
        mock_traf.extract.return_value = "Extracted article text."

        # Simulate metadata
        mock_meta = MagicMock()
        mock_meta.title = "Article Title"
        mock_meta.author = "John Smith"
        mock_meta.date = "2025-06-01"
        mock_traf.extract_metadata.return_value = mock_meta

        result = fetch_and_extract("https://example.com/article")

    assert result.success is True
    assert result.url == "https://example.com/article"
    assert result.text == "Extracted article text."
    assert result.title == "Article Title"
    assert result.author == "John Smith"
    assert result.date == "2025-06-01"
    assert result.error == ""


def test_fetch_and_extract_http_error():
    """HTTP 404 returns a failed LoadedContent."""
    p_req, p_traf, mock_requests, mock_traf = _patch_loader()

    with p_req, p_traf:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("404 Client Error")
        mock_requests.get.return_value = mock_resp

        result = fetch_and_extract("https://example.com/missing")

    assert result.success is False
    assert "Fetch failed" in result.error
    assert "404" in result.error
    assert result.text == ""


def test_fetch_and_extract_connection_timeout():
    """Connection timeout returns a failed LoadedContent."""
    p_req, p_traf, mock_requests, mock_traf = _patch_loader()

    with p_req, p_traf:
        mock_requests.get.side_effect = Exception("Connection timed out")

        result = fetch_and_extract("https://example.com/slow", timeout=5)

    assert result.success is False
    assert "Fetch failed" in result.error
    assert "timed out" in result.error


def test_fetch_and_extract_no_text_extracted():
    """Trafilatura returning None results in a failed LoadedContent."""
    p_req, p_traf, mock_requests, mock_traf = _patch_loader()

    with p_req, p_traf:
        mock_resp = MagicMock()
        mock_resp.text = "<html><body>No article here</body></html>"
        mock_resp.raise_for_status = MagicMock()
        mock_requests.get.return_value = mock_resp

        mock_traf.extract.return_value = None

        result = fetch_and_extract("https://example.com/empty")

    assert result.success is False
    assert result.error == "No text extracted"
    assert result.text == ""


def test_fetch_and_extract_metadata_failure():
    """Metadata extraction failure does not break the overall extraction."""
    p_req, p_traf, mock_requests, mock_traf = _patch_loader()

    with p_req, p_traf:
        mock_resp = MagicMock()
        mock_resp.text = "<html><body>Good article</body></html>"
        mock_resp.raise_for_status = MagicMock()
        mock_requests.get.return_value = mock_resp

        mock_traf.extract.return_value = "Good article text here."
        mock_traf.extract_metadata.side_effect = Exception("metadata parse error")

        result = fetch_and_extract("https://example.com/no-meta")

    assert result.success is True
    assert result.text == "Good article text here."
    # Metadata falls back to empty strings
    assert result.title == ""
    assert result.author == ""
    assert result.date == ""


def test_fetch_and_extract_metadata_returns_none():
    """extract_metadata returning None is handled gracefully."""
    p_req, p_traf, mock_requests, mock_traf = _patch_loader()

    with p_req, p_traf:
        mock_resp = MagicMock()
        mock_resp.text = "<html>content</html>"
        mock_resp.raise_for_status = MagicMock()
        mock_requests.get.return_value = mock_resp

        mock_traf.extract.return_value = "Some content."
        mock_traf.extract_metadata.return_value = None

        result = fetch_and_extract("https://example.com/null-meta")

    assert result.success is True
    assert result.title == ""
    assert result.author == ""


def test_fetch_and_extract_passes_timeout_and_headers():
    """Verify requests.get is called with the expected arguments."""
    p_req, p_traf, mock_requests, mock_traf = _patch_loader()

    with p_req, p_traf:
        mock_resp = MagicMock()
        mock_resp.text = "<html></html>"
        mock_resp.raise_for_status = MagicMock()
        mock_requests.get.return_value = mock_resp
        mock_traf.extract.return_value = "text"
        mock_traf.extract_metadata.return_value = None

        fetch_and_extract("https://example.com", timeout=15)

    mock_requests.get.assert_called_once_with(
        "https://example.com",
        timeout=15,
        headers={"User-Agent": "Mozilla/5.0 (compatible; EngramAI/1.0)"},
    )


# ---------------------------------------------------------------------------
# chunk_text  (pure function, no mocks needed)
# ---------------------------------------------------------------------------

def test_chunk_text_empty_string():
    """Empty string returns no chunks."""
    assert chunk_text("") == []


def test_chunk_text_whitespace_only():
    """Whitespace-only input returns no chunks."""
    assert chunk_text("   \n\n   ") == []


def test_chunk_text_short_text_single_chunk():
    """Short text well under the limit produces a single chunk."""
    text = "The Earth orbits the Sun at an average distance of 149.6 million km."
    chunks = chunk_text(text, max_tokens=1500)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_multiple_paragraphs_fit_one_chunk():
    """Multiple short paragraphs merged into a single chunk."""
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
    chunks = chunk_text(text, max_tokens=1500)
    assert len(chunks) == 1
    assert "Paragraph one." in chunks[0]
    assert "Paragraph three." in chunks[0]


def test_chunk_text_paragraphs_split_into_multiple_chunks():
    """Paragraphs exceeding max_tokens trigger new chunks."""
    # Each paragraph is ~50 chars. max_tokens=20 → max_chars=80.
    # So each chunk fits ~1-2 paragraphs.
    para = "A" * 50
    text = f"{para}\n\n{para}\n\n{para}\n\n{para}"
    chunks = chunk_text(text, max_tokens=20)
    assert len(chunks) >= 2
    # Each chunk should contain text
    for c in chunks:
        assert len(c) > 0


def test_chunk_text_oversized_paragraph_sentence_split():
    """A single paragraph exceeding max_chars is split on sentences."""
    # Build a paragraph with several sentences totaling > max_chars
    sentences = [f"Sentence number {i} has some content here." for i in range(20)]
    big_para = " ".join(sentences)  # all in one paragraph (no \n\n)

    # max_tokens=25 → max_chars=100
    chunks = chunk_text(big_para, max_tokens=25)
    assert len(chunks) >= 2
    # All original content should be represented
    full_text = " ".join(chunks)
    assert "Sentence number 0" in full_text
    assert "Sentence number 19" in full_text


def test_chunk_text_overlap_sentences():
    """Overlapping sentences appear in consecutive chunks."""
    # Build a paragraph large enough to require splitting
    sentences = [f"Fact {i} is important." for i in range(20)]
    big_para = " ".join(sentences)

    # Small limit to force multiple sentence-level chunks, overlap=1
    chunks = chunk_text(big_para, max_tokens=20, overlap_sentences=1)
    assert len(chunks) >= 2

    # The last sentence of one chunk should appear at the start of the next
    for i in range(len(chunks) - 1):
        # Extract last sentence of chunk i
        sents_i = _split_sentences(chunks[i])
        last_sent = sents_i[-1] if sents_i else ""
        # It should appear in the next chunk
        if last_sent:
            assert last_sent in chunks[i + 1], (
                f"Overlap sentence not found in next chunk: {last_sent!r}"
            )


def test_chunk_text_mixed_short_and_oversized():
    """Mix of normal paragraphs and one oversized paragraph."""
    short = "Short paragraph here."
    big = " ".join([f"Long sentence number {i} for testing." for i in range(30)])
    text = f"{short}\n\n{big}\n\n{short}"

    # Small max to force splitting of the big paragraph
    chunks = chunk_text(text, max_tokens=30)
    assert len(chunks) >= 3
    # The short paragraphs should appear
    assert any(short in c for c in chunks)


def test_chunk_text_no_empty_chunks():
    """No chunk in the output should be empty or whitespace-only."""
    text = "Hello.\n\n\n\n\n\nWorld."
    chunks = chunk_text(text, max_tokens=1500)
    for c in chunks:
        assert c.strip() != ""


# ---------------------------------------------------------------------------
# _split_sentences
# ---------------------------------------------------------------------------

def test_split_sentences_basic():
    """Splits on sentence-ending punctuation followed by space + uppercase."""
    text = "The cat sat. The dog ran. A bird flew?"
    result = _split_sentences(text)
    assert result == ["The cat sat.", "The dog ran.", "A bird flew?"]


def test_split_sentences_single_sentence():
    """Single sentence returns a one-element list."""
    result = _split_sentences("Just one sentence.")
    assert result == ["Just one sentence."]


def test_split_sentences_exclamation_and_question():
    """Handles ! and ? as sentence terminators."""
    text = "Wow! Is that true? Yes it is."
    result = _split_sentences(text)
    assert result == ["Wow!", "Is that true?", "Yes it is."]


def test_split_sentences_no_split_on_abbreviations():
    """Does not split on lowercase after period (e.g. abbreviations)."""
    text = "The temp. was 20 degrees. It was warm."
    result = _split_sentences(text)
    # "temp. was" should NOT cause a split (lowercase 'w' after period)
    assert len(result) == 2
    assert "temp. was" in result[0]


def test_split_sentences_empty():
    """Empty string returns empty list."""
    assert _split_sentences("") == []
