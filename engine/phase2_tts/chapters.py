"""Chapter detection from Phase 1 source_location fields.

Derives chapter boundaries from page ranges (PDF) or section/file IDs (EPUB).
If no clear structure exists, treats the entire book as one section.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chapter:
    id: str
    block_start: int  # inclusive index in the filtered body blocks list
    block_end: int    # exclusive


def detect_chapters(body_blocks: list[dict]) -> list[Chapter]:
    """Detect chapter boundaries from source_location metadata.

    Strategy:
    - PDF: group by page number; each distinct page is a "chapter".
    - EPUB: group by section name; each distinct section file is a "chapter".
    - If source_location has neither page nor section: one chapter for the whole book.
    - Chapter IDs are human-readable ("page_1", "section_ch01.xhtml", "chapter_1").
    """
    if not body_blocks:
        return []

    # Determine grouping key
    first_loc = body_blocks[0].get("source_location", {})
    has_page = "page" in first_loc
    has_section = "section" in first_loc

    if not has_page and not has_section:
        # No structure — whole book as one section
        return [Chapter(id="full_book", block_start=0, block_end=len(body_blocks))]

    # Group consecutive blocks by key value
    chapters: list[Chapter] = []
    current_key = None
    start_idx = 0

    for i, block in enumerate(body_blocks):
        loc = block.get("source_location", {})
        if has_page:
            key = str(loc.get("page", "unknown"))
            label = f"page_{key}"
        else:
            key = loc.get("section", "unknown")
            label = f"section_{key}"

        if key != current_key:
            if current_key is not None:
                chapters.append(Chapter(id=label, block_start=start_idx, block_end=i))
            current_key = key
            start_idx = i

    # Final chapter
    if current_key is not None:
        if has_page:
            label = f"page_{current_key}"
        else:
            label = f"section_{current_key}"
        chapters.append(Chapter(id=label, block_start=start_idx, block_end=len(body_blocks)))

    return chapters
