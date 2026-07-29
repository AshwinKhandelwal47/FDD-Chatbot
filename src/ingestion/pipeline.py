import json
import logging
from pathlib import Path
from typing import Any

from src.ingestion.pdf_parser import extract_text_with_pages
from src.ingestion.table_extractor import extract_tables_from_pdf
from src.ingestion.excel_parser import extract_tables_from_excel
from src.ingestion.chunker import (
    create_text_chunks,
    create_table_chunks,
    merge_consecutive_tables,
)
from src.rag.embedder import get_embedder
from src.rag.vector_store import load_chroma_store, sanitize_metadata

logger = logging.getLogger(__name__)


def _vector_store_dir(path: Path) -> Path:
    store_dir = path.parent.parent / "vector_store"
    if path.parent.name != "raw":
        store_dir = path.parent / "vector_store"
    return store_dir


def _embed_chunks(path: Path, chunks: list[dict[str, Any]]) -> int:
    if not chunks:
        return 0

    collection = load_chroma_store(_vector_store_dir(path))
    embedder = get_embedder()

    ids = [f"{path.name}::{i}" for i in range(len(chunks))]
    documents = [chunk["content"] for chunk in chunks]
    metadatas = [sanitize_metadata(chunk) for chunk in chunks]
    embeddings = embedder.encode(documents)

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents,
    )
    return len(chunks)


def ingest_document(file_path: str) -> dict[str, Any]:
    """Ingest a single document (PDF or Excel/CSV), extract and chunk contents, 
    and save the combined chunks to disk.
    
    Returns a summary dict with the total counts and output path.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Ensure processed directory exists
    processed_dir = path.parent.parent / "processed"
    if path.parent.name != "raw":
        # Fallback if path is not in data/raw
        processed_dir = path.parent / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output JSON path
    output_path = processed_dir / f"{path.name}.json"
    
    chunks: list[dict[str, Any]] = []
    text_chunk_count = 0
    table_chunk_count = 0

    suffix = path.suffix.lower()
    
    if suffix == ".pdf":
        logger.info("Starting PDF ingestion for: %s", path.name)
        # 1. Extract text and create text chunks
        pages = extract_text_with_pages(path)
        for page in pages:
            page_chunks = create_text_chunks(
                text=page["text"],
                source_file=page["source_file"],
                page_number=page["page_number"],
                section=page["section"],
                fiscal_year=page["fiscal_year"],
            )
            chunks.extend(page_chunks)
            text_chunk_count += len(page_chunks)
        
        # 2. Extract tables and create table chunks
        raw_tables = extract_tables_from_pdf(path)
        merged_tables = merge_consecutive_tables(raw_tables)
        
        for table_idx, table in enumerate(merged_tables, start=1):
            t_chunks = create_table_chunks(
                table_data=table["data"],
                headers=table["headers"],
                source_file=table["source_file"],
                page_number=table["page_number"],
                section=table["section"],
                fiscal_year=table["fiscal_year"],
                table_index=table_idx,
            )
            chunks.extend(t_chunks)
            table_chunk_count += len(t_chunks)

    elif suffix in (".xlsx", ".xls", ".csv"):
        logger.info("Starting Excel/CSV ingestion for: %s", path.name)
        raw_tables = extract_tables_from_excel(path)
        for table_idx, table in enumerate(raw_tables, start=1):
            t_chunks = create_table_chunks(
                table_data=table["data"],
                headers=table["headers"],
                source_file=table["source_file"],
                page_number=table["page_number"],
                section=table["section"],
                fiscal_year=table["fiscal_year"],
                table_index=table_idx,
            )
            chunks.extend(t_chunks)
            table_chunk_count += len(t_chunks)
            
    else:
        raise ValueError(f"Unsupported file extension: {suffix}")

    # Write chunks to disk
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    embedded_chunks = _embed_chunks(path, chunks)
        
    logger.info("Ingestion complete. Saved %d chunks to %s", len(chunks), output_path)

    return {
        "total_chunks": len(chunks),
        "text_chunks": text_chunk_count,
        "table_chunks": table_chunk_count,
        "output_path": str(output_path),
        "embedded_chunks": embedded_chunks,
    }
