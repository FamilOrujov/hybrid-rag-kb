from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

# Old format
_CID_SIMPLE = re.compile(r"\[cid:(\d+)\]")

# New human readable format.
# It matches any bracket that contains "cid:<number>".
_CID_SOURCE = re.compile(r"\[Source:[^\]]*?\bcid:(\d+)\b[^\]]*\]")


def extract_citations(text: str) -> set[int]:
    """
    Extract unique citation ids from an answer.

    Supported formats:
    - [cid:123]
    - [Source: filename | cid:123]
    """
    out: set[int] = set()
    out.update(int(m.group(1)) for m in _CID_SIMPLE.finditer(text))
    out.update(int(m.group(1)) for m in _CID_SOURCE.finditer(text))
    return out


def split_paragraphs(text: str) -> list[str]:
    """
    Split into paragraphs using blank lines.
    """
    parts = re.split(r"\n\s*\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def extract_citations_from_paragraph(paragraph: str) -> set[int]:
    out: set[int] = set()
    out.update(int(m.group(1)) for m in _CID_SIMPLE.finditer(paragraph))
    out.update(int(m.group(1)) for m in _CID_SOURCE.finditer(paragraph))
    return out


def validate_citations_detailed(
    *,
    answer_text: str,
    allowed_chunk_ids: Iterable[int],
    min_unique_citations: int = 1,
    require_citation_per_paragraph: bool = True,
) -> tuple[bool, dict[str, Any]]:
    """
    Validate citations with detailed diagnostics.

    Checks:
    1) at least min_unique_citations unique citations exist
    2) all cited ids are within allowed_chunk_ids
    3) optionally: every paragraph has at least one citation
    """
    allowed = {int(x) for x in allowed_chunk_ids}

    paragraphs = split_paragraphs(answer_text)
    per_paragraph: list[list[int]] = []
    missing_paragraphs: list[int] = []

    for i, p in enumerate(paragraphs):
        cids = sorted(extract_citations_from_paragraph(p))
        per_paragraph.append(cids)
        if require_citation_per_paragraph and not cids:
            missing_paragraphs.append(i)

    found = sorted({cid for cids in per_paragraph for cid in cids})
    invalid = sorted([cid for cid in found if cid not in allowed])

    report: dict[str, Any] = {
        "paragraph_count": len(paragraphs),
        "found_citations": found,
        "unique_citations_count": len(found),
        "min_unique_citations_required": min_unique_citations,
        "invalid_ids": invalid,
        "require_citation_per_paragraph": require_citation_per_paragraph,
        "missing_paragraphs": missing_paragraphs,
        "per_paragraph_citations": per_paragraph,
    }

    if len(found) < min_unique_citations:
        report["reason"] = "not enough unique citations"
        return False, report

    if invalid:
        report["reason"] = "contains invalid citation ids"
        return False, report

    if require_citation_per_paragraph and missing_paragraphs:
        report["reason"] = "some paragraphs are missing citations"
        return False, report

    report["reason"] = "ok"
    return True, report
