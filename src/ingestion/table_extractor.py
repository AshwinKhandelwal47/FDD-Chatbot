import logging
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import pdfplumber

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ghostscript availability check (needed by camelot)
# ---------------------------------------------------------------------------
_GHOSTSCRIPT_AVAILABLE: bool = shutil.which("gswin64c") is not None or \
                                shutil.which("gswin32c") is not None or \
                                shutil.which("gs") is not None

_CAMELOT_AVAILABLE: bool = False
try:
    import camelot  # noqa: F401
    _CAMELOT_AVAILABLE = True
except ImportError:
    pass

if _CAMELOT_AVAILABLE and not _GHOSTSCRIPT_AVAILABLE:
    logger.warning(
        "camelot-py is installed but Ghostscript was not found on PATH. "
        "The camelot fallback for table extraction will be disabled. "
        "Install Ghostscript: https://ghostscript.com/releases/gsdnld.html"
    )


def _camelot_extract_page(pdf_path: Path, page_number: int) -> list[pd.DataFrame]:
    """Try camelot lattice, then stream, for a single page. Returns list of DataFrames."""
    if not (_CAMELOT_AVAILABLE and _GHOSTSCRIPT_AVAILABLE):
        return []

    import camelot

    for flavor in ("lattice", "stream"):
        try:
            result = camelot.read_pdf(
                str(pdf_path), pages=str(page_number), flavor=flavor
            )
            if result.n > 0:
                dfs = [t.df for t in result]
                logger.info(
                    "camelot (%s) found %d table(s) on page %d of %s",
                    flavor, len(dfs), page_number, pdf_path.name,
                )
                return dfs
        except Exception as exc:
            logger.debug(
                "camelot (%s) failed on page %d of %s: %s",
                flavor, page_number, pdf_path.name, exc,
            )
    return []


def _df_to_serializable(df: pd.DataFrame) -> tuple[list[dict], list[str]]:
    """Convert a DataFrame to a JSON-serializable (records, headers) pair."""
    headers = df.columns.tolist()
    records = df.to_dict(orient="records")
    return records, headers


def _needs_camelot_fallback(pdfplumber_tables: list[list[list]]) -> bool:
    """Return True if pdfplumber found nothing useful on the page."""
    if not pdfplumber_tables:
        return True
    # Also fall back if every table has fewer than 2 columns
    for table in pdfplumber_tables:
        if table and len(table[0]) >= 2:
            return False
    return True


def extract_tables_from_pdf(pdf_path: Path) -> list[dict[str, Any]]:
    """Extract tables from PDF pages and return them as serializable dicts.

    Uses pdfplumber as the primary method.  Falls back to camelot-py
    (lattice → stream) when pdfplumber finds no tables or only tables
    with fewer than 2 columns on a page.

    Each returned dict contains the shared metadata schema:
        - data: list[dict]   (DataFrame.to_dict(orient='records'))
        - headers: list[str] (column names in order)
        - page_number: int
        - table_index: int
        - source_file: str
        - chunk_type: 'table'
        - section: str | None  (populated downstream)
        - fiscal_year: int | None  (populated downstream)
        - extraction_method: str  ('pdfplumber' | 'camelot')
    """
    tables: list[dict[str, Any]] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            raw_tables = page.extract_tables()

            if _needs_camelot_fallback(raw_tables):
                # --- camelot fallback path ---
                camelot_dfs = _camelot_extract_page(pdf_path, page_number)
                for table_index, df in enumerate(camelot_dfs, start=1):
                    records, headers = _df_to_serializable(df)
                    tables.append({
                        "page_number": page_number,
                        "table_index": table_index,
                        "data": records,
                        "headers": headers,
                        "source_file": str(pdf_path.name),
                        "chunk_type": "table",
                        "section": None,
                        "fiscal_year": None,
                        "extraction_method": "camelot",
                    })
                if not camelot_dfs:
                    logger.debug(
                        "No tables found on page %d of %s by any method",
                        page_number, pdf_path.name,
                    )
            else:
                # --- pdfplumber primary path ---
                for table_index, table in enumerate(raw_tables, start=1):
                    if not table or len(table) < 2:
                        continue
                    df = pd.DataFrame(table[1:], columns=table[0])
                    records, headers = _df_to_serializable(df)
                    tables.append({
                        "page_number": page_number,
                        "table_index": table_index,
                        "data": records,
                        "headers": headers,
                        "source_file": str(pdf_path.name),
                        "chunk_type": "table",
                        "section": None,
                        "fiscal_year": None,
                        "extraction_method": "pdfplumber",
                    })
                    logger.info(
                        "pdfplumber found table %d on page %d of %s",
                        table_index, page_number, pdf_path.name,
                    )

    return tables
