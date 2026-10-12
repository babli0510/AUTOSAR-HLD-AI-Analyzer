import re
from pathlib import Path

import pymupdf


def normalize_text(text: str) -> str:
    """Normalize whitespace without removing meaningful line breaks."""
    if not text:
        return ""
    text = text.replace("\xa0", " ").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]*\n[ \t]*\n+", "\n\n", text)
    return text.strip()


def boxes_overlap(box_a, box_b) -> bool:
    """Return True when two rectangles overlap."""
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    return not (
        ax1 <= bx0 or bx1 <= ax0 or
        ay1 <= by0 or by1 <= ay0
    )


def _table_to_text(table):
    """
    Convert PyMuPDF's extracted table matrix to readable text.
    This deliberately does not guess where overlapping words belong.
    """
    try:
        matrix = table.extract()
    except Exception:
        return ""

    if not matrix:
        return ""

    cleaned_rows = []
    for row in matrix:
        if not row:
            continue
        cells = []
        for cell in row:
            value = normalize_text(str(cell or ""))
            # Keep each cell on one line for consistent serialization.
            value = re.sub(r"\s*\n\s*", " ", value)
            cells.append(value)

        if any(cells):
            cleaned_rows.append(cells)

    if not cleaned_rows:
        return ""

    # Use first row as column headers when available. Do not assume
    # any AUTOSAR-specific column names or document schema.
    headers = cleaned_rows[0]
    output = ["TABLE CONTENT:"]

    for row_number, row in enumerate(cleaned_rows[1:], start=1):
        fields = []
        for index, value in enumerate(row):
            if not value:
                continue
            header = headers[index].strip() if index < len(headers) else ""
            if not header:
                header = f"Column {index + 1}"
            fields.append(f"{header}: {value}")

        if fields:
            output.append(f"Row {row_number}: " + "; ".join(fields))

    # A detected table may contain only one row or no useful header.
    if len(output) == 1:
        for row_number, row in enumerate(cleaned_rows, start=1):
            values = [value for value in row if value]
            if values:
                output.append(f"Row {row_number}: " + " | ".join(values))

    return "\n".join(output)


def _raw_table_region_text(page, bbox):
    """
    Conservative fallback: preserve text from the table region in reading
    order when table-cell reconstruction is empty or clearly unreliable.
    This is not a claim that the fallback has recovered the exact columns.
    """
    try:
        text = page.get_text("text", clip=bbox, sort=True)
    except Exception:
        text = ""
    return normalize_text(text)


def _table_has_cross_column_spans(page, table):
    """
    Detect text spans that cross an internal table-column boundary.
    This is a layout warning, not proof that a table is wrong: long text
    can legitimately extend beyond a cell in poorly formatted PDFs.
    """
    try:
        first_row = table.rows[0].cells
    except Exception:
        return False

    valid_cells = [cell for cell in first_row if cell is not None]
    if len(valid_cells) < 2:
        return False

    boundaries = sorted(set([valid_cells[0][0]] + [cell[2] for cell in valid_cells]))
    if len(boundaries) < 3:
        return False

    try:
        raw = page.get_text("dict")
    except Exception:
        return False

    table_top, table_bottom = table.bbox[1], table.bbox[3]
    for block in raw.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                x0, y0, x1, y1 = span.get("bbox", (0, 0, 0, 0))
                if y1 < table_top or y0 > table_bottom:
                    continue
                for boundary in boundaries[1:-1]:
                    # A small tolerance avoids flagging mere contact with a border.
                    if x0 < boundary - 2 and x1 > boundary + 2:
                        return True
    return False


def _table_text_quality(table_text: str) -> bool:
    """
    Check that the table matrix produced a minimally structured result.
    Layout overlap is checked separately and can force the conservative
    raw-region fallback.
    """
    if not table_text:
        return False

    rows = [line for line in table_text.splitlines() if line.strip().startswith("Row ")]
    if len(rows) < 2:
        return False

    rows_with_fields = sum((";" in row or "|" in row) for row in rows)
    return rows_with_fields >= max(1, len(rows) // 2)


def extract_page_content(page):
    """
    Extract page text and detected tables.

    Tables are parsed through PyMuPDF's table matrix first. If the result
    appears structurally weak, use the raw text clipped to the table region
    rather than assigning overlapping words to columns based on their
    bounding-box centers. This fallback preserves evidence but may not
    preserve exact table columns.
    """
    elements = []
    table_regions = []

    try:
        finder = page.find_tables()
        for table in finder.tables:
            bbox = tuple(table.bbox)
            table_text = _table_to_text(table)

            has_cross_column_spans = _table_has_cross_column_spans(page, table)
            if _table_text_quality(table_text) and not has_cross_column_spans:
                rendered_text = table_text
            else:
                raw_region = _raw_table_region_text(page, bbox)
                if raw_region:
                    rendered_text = "TABLE REGION TEXT (column mapping uncertain):\n" + raw_region
                else:
                    rendered_text = table_text

            if rendered_text:
                table_regions.append(bbox)
                elements.append({
                    "y": bbox[1],
                    "x": bbox[0],
                    "text": rendered_text,
                })
    except Exception as error:
        print(f"Warning: Table detection failed on page {page.number + 1}: {error}")

    # Extract text blocks that do not overlap detected table regions.
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

        elements.append({
            "y": y0,
            "x": x0,
            "text": block_text,
        })

    elements.sort(key=lambda item: (item["y"], item["x"]))
    page_text = "\n\n".join(item["text"] for item in elements)

    # Final fallback for pages where layout filtering removed all text.
    if not page_text.strip():
        page_text = page.get_text("text", sort=True)

    return normalize_text(page_text)


def extract_text_from_pdf(pdf_path):
    """
    Extract PDF content page by page.

    Returns:
        list[dict]: Each item contains page_number and text.
    """
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
