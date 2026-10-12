
import re
from pathlib import Path

import pymupdf


def normalize_text(text):
    """Normalize whitespace while preserving useful line breaks."""
    if not text:
        return ""

    text = text.replace("\xa0", " ").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]*\n[ \t]*\n+", "\n\n", text)

    return text.strip()


def boxes_overlap(a, b):
    """Return True if two rectangles overlap."""
    return not (
        a[2] <= b[0]
        or b[2] <= a[0]
        or a[3] <= b[1]
        or b[3] <= a[1]
    )


def _get_table_edges(table):
    """Get sorted table boundaries from detected cell geometry."""
    try:
        rows = table.rows
        cells = [
            cell
            for row in rows
            for cell in row.cells
            if cell is not None
        ]
    except Exception:
        return None, None

    if not cells:
        return None, None

    x_edges = sorted({
        round(edge, 2)
        for cell in cells
        for edge in (cell[0], cell[2])
    })

    y_edges = sorted({
        round(edge, 2)
        for cell in cells
        for edge in (cell[1], cell[3])
    })

    if len(x_edges) < 2 or len(y_edges) < 2:
        return None, None

    return x_edges, y_edges


def _assign_word_to_column(word, x_edges):
    """
    Assign a word to the column with the greatest horizontal overlap.

    Returns:
        (column_index, is_ambiguous)

    A word is flagged if it crosses an internal column boundary
    or if its position does not clearly identify one column.
    """
    wx0, _, wx1, _, *_ = word
    word_width = wx1 - wx0

    if word_width <= 0:
        return None, True

    overlaps = []

    for index, (left, right) in enumerate(
        zip(x_edges, x_edges[1:])
    ):
        overlap = max(
            0.0,
            min(wx1, right) - max(wx0, left)
        )
        overlaps.append((index, overlap))

    if not overlaps:
        return None, True

    best_column, best_overlap = max(
        overlaps,
        key=lambda item: item[1]
    )

    if best_overlap <= 0:
        return None, True

    # Detect words that cross internal column boundaries.
    crossed_boundaries = [
        edge
        for edge in x_edges[1:-1]
        if wx0 < edge < wx1
    ]

    if crossed_boundaries:
        return best_column, True

    # Flag words mostly outside the assigned column.
    if best_overlap / word_width < 0.60:
        return best_column, True

    return best_column, False


def extract_table_by_geometry(page, table):
    """
    Extract table rows using detected cell boundaries and word positions.

    Ambiguous words are flagged for manual verification. This method
    cannot reliably split text that is already merged in the PDF layer.
    """
    x_edges, y_edges = _get_table_edges(table)

    if x_edges is None or y_edges is None:
        return None

    words = page.get_text("words")
    table_bbox = tuple(table.bbox)

    # Select words that intersect the table region.
    table_words = [
        word
        for word in words
        if word[2] > table_bbox[0]
        and word[0] < table_bbox[2]
        and word[3] > table_bbox[1]
        and word[1] < table_bbox[3]
    ]

    output = [
        "TABLE CONTENT (geometry-based; verify uncertain cells):"
    ]

    for row_index, (y0, y1) in enumerate(
        zip(y_edges, y_edges[1:]),
        start=1
    ):
        row_words = [
            word
            for word in table_words
            if y0 <= (word[1] + word[3]) / 2 < y1
        ]

        cells = [[] for _ in range(len(x_edges) - 1)]
        ambiguous_words = []

        for word in row_words:
            column, is_ambiguous = _assign_word_to_column(
                word, x_edges
            )

            if column is not None:
                cells[column].append(word)

            if is_ambiguous:
                ambiguous_words.append(word[4])

        cell_texts = []

        for cell_words in cells:
            cell_words.sort(
                key=lambda word: (word[1], word[0])
            )

            text = normalize_text(
                " ".join(word[4] for word in cell_words)
            )
            cell_texts.append(text)

        if any(cell_texts):
            output.append(
                f"Row {row_index}: "
                + " | ".join(cell_texts)
            )

            if ambiguous_words:
                # Keep the warning separate from the actual cell data.
                output.append(
                    "  [CHECK AMBIGUOUS WORDS: "
                    + ", ".join(ambiguous_words)
                    + "]"
                )

    return "\n".join(output) if len(output) > 1 else None


def extract_page_content(page):
    """Extract table content and preserve non-table text blocks."""
    elements = []
    table_regions = []

    try:
        tables = page.find_tables().tables

        for table in tables:
            bbox = tuple(table.bbox)
            table_text = extract_table_by_geometry(page, table)

            if table_text:
                elements.append({
                    "y": bbox[1],
                    "x": bbox[0],
                    "text": table_text,
                })

                table_regions.append(bbox)

    except Exception as error:
        print(
            f"Table detection warning on page "
            f"{page.number + 1}: {error}"
        )

    # Extract ordinary text outside detected table regions.
    for block in page.get_text("blocks", sort=True):
        if len(block) < 5:
            continue

        x0, y0, x1, y1, block_text = block[:5]

        if not isinstance(block_text, str):
            continue

        block_text = normalize_text(block_text)

        if not block_text:
            continue

        block_bbox = (x0, y0, x1, y1)

        if any(
            boxes_overlap(block_bbox, table_bbox)
            for table_bbox in table_regions
        ):
            continue

        elements.append({
            "y": y0,
            "x": x0,
            "text": block_text,
        })

    # Restore approximate reading order.
    elements.sort(
        key=lambda item: (item["y"], item["x"])
    )

    result = "\n\n".join(
        item["text"] for item in elements
    )

    # Fallback if no content was extracted.
    if not result.strip():
        result = page.get_text("text", sort=True)

    return normalize_text(result)


def extract_text_from_pdf(pdf_path):
    """Return extracted content as one dictionary per page."""
    pages = []

    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(
            document, start=1
        ):
            pages.append({
                "page_number": page_number,
                "text": extract_page_content(page),
            })

    return pages


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    pdf_path = (
        project_root
        / "data"
        / "raw"
        / "sample_hld.pdf"
    )

    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        raise SystemExit(1)

    pages = extract_text_from_pdf(str(pdf_path))

    print(f"Total pages: {len(pages)}")

    for page in pages:
        print("\n" + "=" * 70)
        print(f"PAGE {page['page_number']}")
        print("=" * 70)
        print(page["text"][:6000])
