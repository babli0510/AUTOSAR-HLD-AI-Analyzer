
import pymupdf

PDF_PATH = "data/raw/sample_hld.pdf"
PAGE_INDEX = 1

with pymupdf.open(PDF_PATH) as document:
    page = document[PAGE_INDEX]
    raw = page.get_text("dict")

    print("TEXT SPANS IN TABLE ROWS")
    print("-" * 70)

    for block in raw["blocks"]:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                x0, y0, x1, y1 = span["bbox"]

                if 218 <= y0 <= 261:
                    print(
                        f"y={y0:.2f} "
                        f"x={x0:.2f}-{x1:.2f} "
                        f"text={span['text']!r}"
                    )
