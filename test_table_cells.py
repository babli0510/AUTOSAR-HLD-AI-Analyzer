
import pymupdf

pdf_path = "data/raw/sample_hld.pdf"

doc = pymupdf.open(pdf_path)
page = doc[1]

x_edges = [62.4, 150.2, 283.5, 405.4, 532.9]
y_edges = [
    203.4, 218.4, 233.4, 248.4, 263.4,
    278.4, 293.4, 308.4, 323.4
]

words = page.get_text("words")

print("\n--- CELL EXTRACTION TEST V2 ---")

for row in range(len(y_edges) - 1):
    print(f"\nROW {row + 1}")

    y0, y1 = y_edges[row], y_edges[row + 1]

    row_words = [
        w for w in words
        if y0 <= (w[1] + w[3]) / 2 < y1
    ]

    cells = [[] for _ in range(len(x_edges) - 1)]

    for word in row_words:
        wx0, _, wx1, _, text = word[:5]

        overlaps = []
        for col in range(len(x_edges) - 1):
            left = max(wx0, x_edges[col])
            right = min(wx1, x_edges[col + 1])
            overlap = max(0, right - left)
            overlaps.append(overlap)

        best_col = overlaps.index(max(overlaps))
        cells[best_col].append((wx0, text))

    for col, cell_words in enumerate(cells):
        cell_words.sort(key=lambda item: item[0])
        text = " ".join(item[1] for item in cell_words)
        print(f"Column {col + 1}: {text}")
