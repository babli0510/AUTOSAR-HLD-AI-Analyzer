
import re
from pathlib import Path

import pymupdf


def normalize_text(text):
    if not text:
        return ""
    text = text.replace("\xa0", " ").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]*\n[ \t]*\n+", "\n\n", text)
    return text.strip()


def boxes_overlap(a, b):
    return not (
        a[2] <= b[0] or b[2] <= a[0]
        or a[3] <= b[1] or b[3] <= a[1]
    )


def extract_table_by_geometry(page, table):
    """
    Reconstruct a bordered table from its detected cell geometry.
    Return None if usable cell geometry is unavailable.
    """
    try:
        rows = table.rows
        if not rows or not rows[0].cells:
            return None

        first_row = rows[0].cells
        if any(cell is None for cell in first_row):
            return None

        x_edges = sorted(set(
            [round(cell[0], 1) for cell in first_row]
            + [round(cell[2], 1) for cell in first_row]
        ))

        if len(x_edges) < 3:
            return None

        y_edges = sorted(set(
            round(edge, 1)
            for row in rows
            for cell in row.cells if cell is not None
            for edge in (cell[1], cell[3])
        ))

        if len(y_edges) < 3:
            return None

        words = page.get_text("words")
        output = ["TABLE CONTENT (geometry-based; verify uncertain cells):"]

        for row_index, (y0, y1) in enumerate(zip(y_edges, y_edges[1:])):
            cells_text = []

            for x0, x1 in zip(x_edges, x_edges[1:]):
                cell_words = [
                    word for word in words
                    if y0 <= (word[1] + word[3]) / 2 < y1
                    and x0 <= (word[0] + word[2]) / 2 < x1
                ]
                cell_words.sort(key=lambda word: (word[1], word[0]))
                cells_text.append(
                    normalize_text(" ".join(word[4] for word in cell_words))
                )

            if any(cells_text):
                output.append(
                    f"Row {row_index + 1}: "
                    + " | ".join(cells_text)
                )

        return "\n".join(output) if len(output) > 1 else None

    except Exception:
        return None


def extract_page_content(page):
    elements = []
    table_regions = []

    try:
        for table in page.find_tables().tables:
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
        print(f"Table detection warning on page {page.number + 1}: {error}")

    # Preserve ordinary text blocks outside successfully extracted tables.
    for block in page.get_text("blocks", sort=True):
        if len(block) < 5:
            continue

        x0, y0, x1, y1, block_text = block[:5]
        if not isinstance(block_text, str):
            continue

        block_text = normalize_text(block_text)
        if not block_text:
            continue

        bbox = (x0, y0, x1, y1)

        if any(boxes_overlap(bbox, table_bbox) for table_bbox in table_regions):
            continue

        elements.append({"y": y0, "x": x0, "text": block_text})

    elements.sort(key=lambda item: (item["y"], item["x"]))
    result = "\n\n".join(item["text"] for item in elements)

    # Never silently discard page text if table reconstruction failed.
    if not result.strip():
        result = page.get_text("text", sort=True)

    return normalize_text(result)


def extract_text_from_pdf(pdf_path):
    pages = []
    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            pages.append({
                "page_number": page_number,
                "text": extract_page_content(page),
            })
    return pages


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    pdf_path = project_root / "data" / "raw" / "sample_hld.pdf"

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
