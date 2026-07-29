import logging
from pathlib import Path
from typing import Any

import fitz

logger = logging.getLogger(__name__)


def extract_text_with_pages(pdf_path: Path) -> list[dict[str, Any]]:
    """Extract text per page from a PDF with page numbers.

    Each returned dict contains the shared metadata schema:
        - source_file: str
        - page_number: int
        - chunk_type: 'text'
        - section: str | None  (populated downstream)
        - fiscal_year: int | None  (populated downstream)
        - text: str
    """
    pages: list[dict[str, Any]] = []

    try:
        with fitz.open(pdf_path) as document:
            for page_number in range(len(document)):
                page = document.load_page(page_number)
                text = page.get_text()
                pages.append({
                    "page_number": page_number + 1,
                    "text": text,
                    "source_file": str(pdf_path.name),
                    "chunk_type": "text",
                    "section": None,
                    "fiscal_year": None,
                })
    except Exception as exc:
        logger.error(
            "Failed to open or read PDF '%s': %s", pdf_path, exc
        )
        raise

    return pages

