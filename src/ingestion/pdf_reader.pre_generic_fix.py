
import re
import pymupdf


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """Normalize whitespace while preserving meaningful lines."""

    if not text:
        return ""

    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================
# WORD AND TABLE HELPERS
# ============================================================

def split_token_at_boundary(word, x0, x1, boundaries):
    """
    Split a token when its bounding box crosses a table column
    boundary. Prefer a CamelCase boundary close to the geometric
    split point. Otherwise retain the original token.
    """

    pieces = [(word, x0, x1)]

    for boundary in boundaries:
        updated = []

        for text, left, right in pieces:
            if not (left < boundary < right) or len(text) < 4:
                updated.append((text, left, right))
                continue

            ratio = (boundary - left) / (right - left)
            estimated_index = round(len(text) * ratio)

            # Candidate split points before uppercase characters.
            candidates = [
                index
                for index in range(1, len(text))
                if text[index].isupper()
                and text[index - 1].islower()
            ]

            if not candidates:
                updated.append((text, left, right))
                continue

            split_index = min(
                candidates,
                key=lambda index: abs(index - estimated_index)
            )

            # Avoid splitting too close to either end.
            if split_index < 2 or split_index > len(text) - 2:
                updated.append((text, left, right))
                continue

            split_x = left + (
                (right - left) * split_index / len(text)
            )

            updated.append((text[:split_index], left, split_x))
            updated.append((text[split_index:], split_x, right))

        pieces = updated

    return pieces


def get_table_column_boundaries(table):
    """Read column boundaries from the table's header cells."""

    if not table.rows:
        return []

    header_cells = table.rows[0].cells

    valid_cells = [
        cell for cell in header_cells
        if cell is not None
    ]

    if not valid_cells:
        return []

    boundaries = [valid_cells[0][0]]

    for cell in valid_cells:
        boundaries.append(cell[2])

    return sorted(set(boundaries))


def get_table_row_ranges(table):
    """Return vertical bounds for each detected table row."""

    ranges = []

    for row in table.rows:
        valid_cells = [
            cell for cell in row.cells
            if cell is not None
        ]

        if not valid_cells:
            continue

        top = min(cell[1] for cell in valid_cells)
        bottom = max(cell[3] for cell in valid_cells)

        ranges.append((top, bottom))

    return ranges


def extract_table_text(page, table):
    """
    Reconstruct a table from page word coordinates.

    Words are assigned to rows by vertical position and columns
    by horizontal position, rather than using table.extract(),
    whose cell text was incorrectly assembled for this PDF.
    """

    boundaries = get_table_column_boundaries(table)
    row_ranges = get_table_row_ranges(table)

    if len(boundaries) < 2 or not row_ranges:
        return ""

    column_count = len(boundaries) - 1

    # Header names are extracted from the actual header cells.
    headers = []

    for column_index in range(column_count):
        cell = table.rows[0].cells[column_index]

        if cell is None:
            headers.append(f"Column {column_index + 1}")
            continue

        x0, y0, x1, y1 = cell

        header_words = []

        for word in page.get_text("words", sort=True):
            wx0, wy0, wx1, wy1, token = word[:5]
            center_x = (wx0 + wx1) / 2
            center_y = (wy0 + wy1) / 2

            if (
                x0 <= center_x <= x1
                and y0 <= center_y <= y1
            ):
                header_words.append((wx0, token))

        header_words.sort(key=lambda item: item[0])

        header = " ".join(token for _, token in header_words)
        headers.append(header or f"Column {column_index + 1}")

    words = page.get_text("words", sort=True)
    output = ["TABLE CONTENT:"]

    for row_index, (row_top, row_bottom) in enumerate(
        row_ranges[1:],
        start=1
    ):
        cells = [[] for _ in range(column_count)]

        for word in words:
            wx0, wy0, wx1, wy1, token = word[:5]
            center_y = (wy0 + wy1) / 2

            if not (row_top <= center_y <= row_bottom):
                continue

            # Ignore words outside the table's horizontal extent.
            if wx1 <= boundaries[0] or wx0 >= boundaries[-1]:
                continue

            fragments = split_token_at_boundary(
                token,
                wx0,
                wx1,
                boundaries[1:-1]
            )

            for fragment, frag_x0, frag_x1 in fragments:
                center_x = (frag_x0 + frag_x1) / 2

                for column_index in range(column_count):
                    left = boundaries[column_index]
                    right = boundaries[column_index + 1]

                    if (
                        left <= center_x < right
                        or (
                            column_index == column_count - 1
                            and center_x <= right
                        )
                    ):
                        cells[column_index].append(
                            (frag_x0, fragment)
                        )
                        break

        fields = []

        for column_index, cell_words in enumerate(cells):
            cell_words.sort(key=lambda item: item[0])

            value = " ".join(
                token for _, token in cell_words
            ).strip()

            if value:
                fields.append(f"{headers[column_index]}: {value}")

        if fields:
            output.append(
                f"Row {row_index}: " + "; ".join(fields)
            )

    return "\n".join(output)


# ============================================================
# PAGE EXTRACTION
# ============================================================

def boxes_overlap(box_a, box_b):
    """Check whether two rectangular regions overlap."""

    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b

    return not (
        ax1 <= bx0
        or bx1 <= ax0
        or ay1 <= by0
        or by1 <= ay0
    )


def extract_page_content(page):
    """Extract normal text and reconstruct detected tables."""

    table_regions = []
    elements = []

    # Detect and reconstruct tables.
    try:
        finder = page.find_tables()

        for table in finder.tables:
            table_text = extract_table_text(page, table)

            if table_text:
                table_regions.append(tuple(table.bbox))

                elements.append({
                    "y": table.bbox[1],
                    "x": table.bbox[0],
                    "text": table_text
                })

    except Exception as error:
        print(
            f"Warning: Table detection failed on page "
            f"{page.number + 1}: {error}"
        )

    # Extract ordinary text blocks outside table regions.
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
            "text": block_text
        })

    # Preserve reading order.
    elements.sort(key=lambda item: (item["y"], item["x"]))

    page_text = "\n\n".join(
        element["text"] for element in elements
    )

    # Fallback if no content was extracted.
    if not page_text.strip():
        page_text = page.get_text("text", sort=True)

    return normalize_text(page_text)


# ============================================================
# PDF READER
# ============================================================

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
                "text": extract_page_content(page)
            })

    return pages


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    from pathlib import Path

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
        print(page["text"][:4000])
