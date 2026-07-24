"""Assemble retrieval context for the chat system prompt."""

import json
from typing import Any, Callable

MAX_CONTEXT_CHARS = 6000

_NO_DOCS_CONTEXT = (
    "CONTEXT:\n\n"
    "No relevant documents were found for this query. "
    "Do not invent information — tell the user no documents are available."
)


def format_table_json_to_markdown(raw: str) -> str:
    """Convert a JSON table string (``{headers, rows}``) to a markdown table.

    Falls back to the raw string if parsing fails.
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw

    headers: list[str] = data.get("headers", [])
    rows: list[dict] = data.get("rows", [])
    if not headers:
        return raw

    # Header row + separator
    header_line = "| " + " | ".join(str(h) for h in headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    lines = [header_line, separator]

    for row in rows:
        cells = [str(row.get(h, "")) for h in headers]
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def format_chunk_for_context(chunk: dict[str, Any]) -> str:
    """Return human-readable content from a retrieval result dict.

    Table chunks are reformatted from JSON into markdown tables.
    Text chunks pass through unchanged.
    """
    content: str = chunk.get("content", "")
    metadata: dict = chunk.get("metadata", {})
    chunk_type = metadata.get("chunk_type", "text")

    if chunk_type == "table":
        return format_table_json_to_markdown(content)
    return content


def build_context(
    query: str,
    retriever: Callable[[str], list[dict[str, Any]]],
    k: int | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Retrieve relevant chunks for *query* and assemble a numbered context block.

    Returns
    -------
    context_string : str
        Ready to prepend to the system prompt.
    sources_list : list[dict]
        Per-source metadata for the UI citation display.
    """
    results: list[dict[str, Any]] = retriever(query)

    if not results:
        return _NO_DOCS_CONTEXT, []

    # Build numbered blocks, truncating chunks if they exceed remaining budget
    blocks: list[str] = []
    sources: list[dict[str, Any]] = []
    total_chars = 0

    for i, result in enumerate(results, start=1):
        metadata = result.get("metadata", {})
        formatted = format_chunk_for_context(result)

        source_file = metadata.get("source_file", "unknown")
        page_number = metadata.get("page_number", "")
        chunk_type = metadata.get("chunk_type", "text")
        table_index = metadata.get("table_index", "")
        row_range = metadata.get("row_range", "")

        header = f"[{i}] (Source: {source_file}, Page: {page_number}, Type: {chunk_type})"

        # Calculate remaining budget for this block
        needed_overhead = len(header) + 1
        if blocks:
            needed_overhead += 2  # separator \n\n

        avail = MAX_CONTEXT_CHARS - total_chars - needed_overhead

        if avail <= 0:
            # No space left even for the header of this chunk
            break

        # Check if content needs to be truncated to fit
        if len(formatted) > avail:
            truncation_suffix = " ... [content truncated]"
            if avail <= len(truncation_suffix):
                # If we cannot even fit the suffix, stop here
                break
            formatted = formatted[:avail - len(truncation_suffix)] + truncation_suffix

        block = f"{header}\n{formatted}"
        blocks.append(block)

        sources.append({
            "index": i,
            "source_file": source_file,
            "page_number": page_number,
            "chunk_type": chunk_type,
            "table_index": table_index,
            "row_range": row_range,
        })

        if len(blocks) == 1:
            total_chars += len(block)
        else:
            total_chars += len(block) + 2

    if not blocks:
        return _NO_DOCS_CONTEXT, []

    context_string = "CONTEXT:\n\n" + "\n\n".join(blocks)
    return context_string, sources
