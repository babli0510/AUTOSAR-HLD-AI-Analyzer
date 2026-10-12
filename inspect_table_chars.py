
import pymupdf

PDF_PATH = "data/raw/sample_hld.pdf"
PAGE_INDEX = 1

with pymupdf.open(PDF_PATH) as document:
    page = document[PAGE_INDEX]
    table = page.find_tables().tables[0]

    print("Table bounds:", table.bbox)

    raw = page.get_text("rawdict")
    char_count = 0

    for block in raw["blocks"]:
        if "lines" not in block:
            continue

        for line in block["lines"]:
            for span in line["spans"]:
                for char in span.get("chars", []):
                    x0, y0, x1, y1 = char["bbox"]

                    if 218 <= y0 <= 261:
                        print(
                            f"x={x0:7.2f}-{x1:7.2f} "
                            f"y={y0:7.2f} "
                            f"char={char['c']!r}"
                        )
                        char_count += 1

    print("\nCharacters found in selected rows:", char_count)
