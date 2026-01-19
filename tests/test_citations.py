"""Tests for the citation extraction and validation module."""

from src.rag.citations import (
    extract_citations,
    split_paragraphs,
    validate_citations_detailed,
)


class TestExtractCitations:
    """Test citation extraction from text."""

    def test_simple_cid_format(self):
        """Should extract [cid:N] format citations."""
        text = "This is a claim [cid:42] and another [cid:7]."
        citations = extract_citations(text)
        assert citations == {42, 7}

    def test_source_format(self):
        """Should extract [Source: filename | cid:N] format citations."""
        text = "Statement [Source: doc.pdf | cid:123] here."
        citations = extract_citations(text)
        assert citations == {123}

    def test_mixed_formats(self):
        """Should extract both citation formats from same text."""
        text = "Fact [cid:1]. Another fact [Source: paper.pdf | cid:2]."
        citations = extract_citations(text)
        assert citations == {1, 2}

    def test_no_citations(self):
        """Should return empty set when no citations present."""
        text = "This text has no citations at all."
        citations = extract_citations(text)
        assert citations == set()

    def test_duplicate_citations(self):
        """Should deduplicate citations."""
        text = "[cid:5] and again [cid:5] and [Source: x | cid:5]"
        citations = extract_citations(text)
        assert citations == {5}

    def test_citations_in_multiline_text(self):
        """Should extract citations across multiple lines."""
        text = """
        First paragraph [cid:1].

        Second paragraph [cid:2].

        Third [Source: doc.txt | cid:3].
        """
        citations = extract_citations(text)
        assert citations == {1, 2, 3}


class TestSplitParagraphs:
    """Test paragraph splitting."""

    def test_single_paragraph(self):
        """Single paragraph should return list of one."""
        text = "Just one paragraph here."
        paragraphs = split_paragraphs(text)
        assert len(paragraphs) == 1
        assert paragraphs[0] == "Just one paragraph here."

    def test_multiple_paragraphs(self):
        """Should split on blank lines."""
        text = "First para.\n\nSecond para.\n\nThird para."
        paragraphs = split_paragraphs(text)
        assert len(paragraphs) == 3

    def test_multiple_blank_lines(self):
        """Multiple blank lines should still split correctly."""
        text = "Para one.\n\n\n\nPara two."
        paragraphs = split_paragraphs(text)
        assert len(paragraphs) == 2

    def test_strips_whitespace(self):
        """Should strip leading/trailing whitespace from paragraphs."""
        text = "  First  \n\n  Second  "
        paragraphs = split_paragraphs(text)
        assert paragraphs == ["First", "Second"]

    def test_empty_text(self):
        """Empty text should return empty list."""
        paragraphs = split_paragraphs("")
        assert paragraphs == []

    def test_whitespace_only(self):
        """Whitespace-only text should return empty list."""
        paragraphs = split_paragraphs("   \n\n   ")
        assert paragraphs == []


class TestValidateCitationsDetailed:
    """Test detailed citation validation."""

    def test_valid_citations(self):
        """Should pass when all requirements met."""
        text = "Claim one [cid:1].\n\nClaim two [cid:2]."
        ok, report = validate_citations_detailed(
            answer_text=text,
            allowed_chunk_ids=[1, 2, 3],
            min_unique_citations=1,
            require_citation_per_paragraph=True,
        )

        assert ok is True
        assert report["reason"] == "ok"

    def test_missing_citations_fails(self):
        """Should fail when not enough citations."""
        text = "No citations here."
        ok, report = validate_citations_detailed(
            answer_text=text,
            allowed_chunk_ids=[1, 2],
            min_unique_citations=1,
        )

        assert ok is False
        assert report["reason"] == "not enough unique citations"

    def test_invalid_citation_ids_fails(self):
        """Should fail when citation IDs not in allowed list."""
        text = "Citing unknown source [cid:999]."
        ok, report = validate_citations_detailed(
            answer_text=text,
            allowed_chunk_ids=[1, 2, 3],
        )

        assert ok is False
        assert report["reason"] == "contains invalid citation ids"
        assert 999 in report["invalid_ids"]

    def test_missing_paragraph_citations_fails(self):
        """Should fail when paragraph missing citation."""
        text = "Has citation [cid:1].\n\nNo citation here."
        ok, report = validate_citations_detailed(
            answer_text=text,
            allowed_chunk_ids=[1],
            require_citation_per_paragraph=True,
        )

        assert ok is False
        assert report["reason"] == "some paragraphs are missing citations"
        assert 1 in report["missing_paragraphs"]

    def test_paragraph_check_disabled(self):
        """Should pass without per-paragraph check when disabled."""
        text = "Has citation [cid:1].\n\nNo citation here."
        ok, report = validate_citations_detailed(
            answer_text=text,
            allowed_chunk_ids=[1],
            require_citation_per_paragraph=False,
        )

        assert ok is True

    def test_report_contains_diagnostics(self):
        """Report should contain detailed diagnostic information."""
        text = "Para 1 [cid:1].\n\nPara 2 [cid:2]."
        _, report = validate_citations_detailed(
            answer_text=text,
            allowed_chunk_ids=[1, 2],
        )

        assert "paragraph_count" in report
        assert "found_citations" in report
        assert "unique_citations_count" in report
        assert "per_paragraph_citations" in report
        assert report["paragraph_count"] == 2
        assert report["unique_citations_count"] == 2

    def test_min_unique_citations_threshold(self):
        """Should enforce minimum unique citations threshold."""
        text = "Citation [cid:1] and same [cid:1]."
        ok, report = validate_citations_detailed(
            answer_text=text,
            allowed_chunk_ids=[1, 2],
            min_unique_citations=2,
            require_citation_per_paragraph=False,
        )

        assert ok is False
        assert report["unique_citations_count"] == 1
