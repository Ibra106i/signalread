#!/usr/bin/env python3
"""
Layout Classifier — Phase 0 validation script.

Classifies document blocks from PDF and EPUB files into categories:
Title, NarrativeText, Footnote, Header, Footer, PageNumber,
UncategorizedText, or ambiguous.

Usage:
    python layout_classifier.py <folder_path> [--output-dir <dir>]
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from collections import Counter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Category mapping from unstructured element types → canonical categories
# ---------------------------------------------------------------------------
UNSTRUCTURED_CATEGORY_MAP = {
    "Title": "Title",
    "NarrativeText": "NarrativeText",
    "UncategorizedText": "UncategorizedText",
    "Footnote": "Footnote",
    "Footnote-Reference": "Footnote",
    "Header": "Header",
    "Page-header": "Header",
    "Section-header": "Header",
    "Subheadline": "Header",
    "Headline": "Header",
    "Footer": "Footer",
    "Page-footer": "Footer",
    "PageNumber": "PageNumber",
    # Intentionally not mapped → will become "ambiguous"
}


def _map_unstructured_type(element_type: str) -> str:
    """Map an unstructured ElementType string to our canonical category."""
    return UNSTRUCTURED_CATEGORY_MAP.get(element_type, "ambiguous")


# ---------------------------------------------------------------------------
# PDF processing
# ---------------------------------------------------------------------------
def process_pdf(filepath: Path) -> dict:
    """Partition a PDF with unstructured and classify each block."""
    from unstructured.partition.pdf import partition_pdf

    # ---- scanned-PDF detection: fast strategy extracts raw text ----
    try:
        elements_fast = partition_pdf(filename=str(filepath), strategy="fast")
    except Exception as exc:
        raise RuntimeError(f"PDF fast-partition failed: {exc}") from exc

    total_chars = sum(len(el.text or "") for el in elements_fast)
    if total_chars < 50:
        return {
            "file": filepath.name,
            "file_type": "pdf",
            "status": "needs_ocr",
            "blocks": [],
            "error": None,
        }

    # ---- full partition with auto strategy ----
    try:
        elements = partition_pdf(filename=str(filepath), strategy="auto")
    except Exception as exc:
        raise RuntimeError(f"PDF auto-partition failed: {exc}") from exc

    blocks: list[dict] = []
    for el in elements:
        text = (el.text or "").strip()
        if not text:
            continue

        category = _map_unstructured_type(el.category)

        source_location: dict = {}
        meta = getattr(el, "metadata", None)
        if meta is not None:
            page_no = getattr(meta, "page_number", None)
            if page_no is not None:
                source_location["page"] = page_no
            coords = getattr(meta, "coordinates", None)
            if coords is not None:
                source_location["has_coordinates"] = True

        blocks.append({
            "text": text,
            "category": category,
            "source_location": source_location,
        })

    # ---- two-column sanity check ----
    page_counts = Counter(
        b["source_location"].get("page") for b in blocks if b["source_location"].get("page")
    )
    if page_counts:
        avg = len(blocks) / max(len(page_counts), 1)
        if avg > 80:
            logger.warning(
                "%s: high blocks-per-page ratio (%.1f) — possible two-column layout issue",
                filepath.name,
                avg,
            )

    return {
        "file": filepath.name,
        "file_type": "pdf",
        "status": "ok",
        "blocks": blocks,
        "error": None,
    }


# ---------------------------------------------------------------------------
# EPUB processing
# ---------------------------------------------------------------------------
_BLOCK_TAGS = frozenset({
    "p", "div", "h1", "h2", "h3", "h4", "h5", "h6",
    "li", "blockquote", "pre", "table", "section", "aside",
    "article", "nav", "figcaption",
})


def _ancestor_epub_type(element) -> str:
    """Walk up ancestors to find an epub:type attribute (EPUB3 semantic markup)."""
    node = element.parent
    while node and hasattr(node, "name") and node.name not in (None, "body", "html"):
        et = node.get("epub:type", "")
        if et:
            return et
        # Also check for aside elements (typically footnotes in EPUBs)
        if node.name == "aside":
            return "footnote"
        node = node.parent
    return ""


def _classify_epub_element(element) -> str:
    """Classify a single EPUB HTML element into a canonical category."""
    epub_type = element.get("epub:type", "")
    tag = element.name or ""
    classes = " ".join(element.get("class", [])).lower()

    # --- Footnotes / endnotes (EPUB3 epub:type on element or ancestor) ---
    if epub_type in ("footnote", "endnote", "footnotes", "endnotes", "annotations"):
        return "Footnote"
    if tag == "aside":
        return "Footnote"
    # Check ancestors for epub:type (e.g. <p> inside <aside epub:type="footnote">)
    ancestor_type = _ancestor_epub_type(element)
    if ancestor_type in ("footnote", "endnote", "footnotes", "endnotes", "annotations"):
        return "Footnote"

    # --- Class-name heuristics (EPUB2 fallback) ---
    if any(kw in classes for kw in ("footnote", "endnote", "fnote", "note-ref")):
        return "Footnote"
    # Also check ancestor class names
    ancestor_node = element.parent
    while ancestor_node and hasattr(ancestor_node, "name") and ancestor_node.name not in (None, "body", "html"):
        anc_classes = " ".join(ancestor_node.get("class", [])).lower()
        if any(kw in anc_classes for kw in ("footnote", "endnote", "fnote")):
            return "Footnote"
        ancestor_node = ancestor_node.parent

    # --- Headers ---
    if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
        return "Header"
    if epub_type in ("title", "subtitle", "heading", "headline", "subheading"):
        return "Header"
    if any(kw in classes for kw in ("heading", "title", "subtitle")):
        return "Header"

    # --- Page numbers ---
    if any(kw in classes for kw in ("page-number", "pagenum", "page_num")):
        return "PageNumber"

    # --- Footer ---
    if epub_type in ("footer", "page-footer"):
        return "Footer"
    if any(kw in classes for kw in ("footer", "page-footer")):
        return "Footer"

    # --- Body text: plain paragraphs without special semantics ---
    if tag == "p" and not epub_type and not classes:
        return "NarrativeText"

    # --- Everything else → ambiguous (never guess) ---
    return "ambiguous"


def process_epub(filepath: Path) -> dict:
    """Parse an EPUB with ebooklib + BeautifulSoup and classify blocks."""
    import warnings
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

    try:
        book = epub.read_epub(str(filepath), options={"ignore_ncx": True})
    except Exception as exc:
        raise RuntimeError(f"EPUB read failed: {exc}") from exc

    blocks: list[dict] = []

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        content = item.get_content()
        # EPUB XHTML content is well-formed XML; use xml parser
        try:
            soup = BeautifulSoup(content, "lxml-xml")
        except Exception:
            soup = BeautifulSoup(content, "lxml")
        body = soup.body
        if body is None:
            continue

        for element in body.descendants:
            if not hasattr(element, "name") or element.name is None:
                continue
            if element.name in ("script", "style", "head", "html"):
                continue

            # Only process block-level or inline elements that hold text
            text = element.get_text(strip=True) if hasattr(element, "get_text") else ""
            if not text or len(text) < 3:
                continue

            # Skip elements that have block children — we process those instead
            has_block_children = any(
                hasattr(c, "name") and c.name in _BLOCK_TAGS
                for c in element.children
            )
            if has_block_children:
                continue

            category = _classify_epub_element(element)

            source_location: dict = {"section": item.get_name()}
            eid = element.get("id")
            if eid:
                source_location["element_id"] = eid

            blocks.append({
                "text": text,
                "category": category,
                "source_location": source_location,
            })

    return {
        "file": filepath.name,
        "file_type": "epub",
        "status": "ok",
        "blocks": blocks,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Folder scanning
# ---------------------------------------------------------------------------
def scan_folder(folder: Path) -> list[Path]:
    """Return sorted list of PDF/EPUB files in *folder*."""
    extensions = {".pdf", ".epub"}
    return sorted(
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    )


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------
def generate_summary(results: list[dict]) -> str:
    """Build a human-readable per-file summary."""
    lines = [
        "=" * 60,
        "LAYOUT CLASSIFICATION SUMMARY",
        "=" * 60,
        "",
    ]

    ok = sum(1 for r in results if r["status"] == "ok")
    ocr = sum(1 for r in results if r["status"] == "needs_ocr")
    err = sum(1 for r in results if r["status"] == "error")

    lines.append(f"Total files processed: {len(results)}")
    lines.append(f"  Successful : {ok}")
    lines.append(f"  Needs OCR  : {ocr}")
    lines.append(f"  Errors     : {err}")
    lines.append("")

    for r in results:
        lines.append(f"--- {r['file']} ({r['file_type'].upper()}) ---")
        if r["status"] == "needs_ocr":
            lines.append("  Status: NEEDS OCR (no extractable text layer)")
        elif r["status"] == "error":
            lines.append(f"  Status: ERROR — {r['error']}")
        else:
            counts = Counter(b["category"] for b in r["blocks"])
            lines.append(f"  Total blocks: {len(r['blocks'])}")
            for cat, count in sorted(counts.items(), key=lambda x: -x[1]):
                lines.append(f"    {cat}: {count}")
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify document blocks from PDF and EPUB files.",
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Path to folder containing PDF/EPUB files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for JSON files (default: same as input folder)",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=None,
        help="Write summary report to this file (in addition to stdout)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug-level logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if not args.folder.is_dir():
        logger.error("Input path is not a directory: %s", args.folder)
        sys.exit(1)

    output_dir = args.output_dir or args.folder
    output_dir.mkdir(parents=True, exist_ok=True)

    files = scan_folder(args.folder)
    if not files:
        logger.warning("No PDF or EPUB files found in %s", args.folder)
        sys.exit(0)

    logger.info("Found %d file(s) to process", len(files))

    results: list[dict] = []
    for filepath in files:
        logger.info("Processing: %s", filepath.name)
        try:
            if filepath.suffix.lower() == ".pdf":
                result = process_pdf(filepath)
            elif filepath.suffix.lower() == ".epub":
                result = process_epub(filepath)
            else:
                continue

            results.append(result)

            json_path = output_dir / f"{filepath.stem}_layout.json"
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(result, fh, indent=2, ensure_ascii=False)
            logger.info(
                "  -> %s (%d blocks)", json_path.name, len(result.get("blocks", []))
            )

        except Exception as exc:
            logger.error("  FAILED %s: %s", filepath.name, exc)
            results.append({
                "file": filepath.name,
                "file_type": filepath.suffix.lower().lstrip("."),
                "status": "error",
                "blocks": [],
                "error": str(exc),
            })

    summary = generate_summary(results)
    print(summary)

    if args.summary_file:
        args.summary_file.parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_file, "w", encoding="utf-8") as fh:
            fh.write(summary)
        logger.info("Summary written to %s", args.summary_file)


if __name__ == "__main__":
    main()
