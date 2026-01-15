from __future__ import annotations

from pathlib import Path
from typing import Tuple

from pypdf import PdfReader


def load_text_from_path(path: str) -> Tuple[str, str]:
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(str(p))
        pages = []
        for i, page in enumerate(reader.pages):
            pages.append(page.extract_text() or "")
        text = "\n".join(pages)
        meta = {"type": "pdf", "pages": len(reader.pages)}
        return text, meta

    
    # text-like fallback
    data = p.read_bytes()
    text = data.decode("utf-8", errors="ignore")
    meta = {"type": "text"}
    return text, meta
