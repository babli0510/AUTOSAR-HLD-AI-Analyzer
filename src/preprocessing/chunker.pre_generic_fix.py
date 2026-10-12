import re
import sys
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

CHUNK_SIZE = 1400
CHUNK_OVERLAP = 200


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Clean extracted PDF text while preserving
    useful AUTOSAR HLD information.
    """

    if not text:
        return ""

    # Normalize spaces and tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Remove synthetic footer/header
    text = re.sub(
        r"VCC-HLD-001 \| Synthetic Pilot Document",
        "",
        text
    )

    # Remove page number lines
    text = re.sub(
        r"\nPage\s+\d+\n",
        "\n",
        text,
        flags=re.IGNORECASE
    )

    # Normalize excessive blank lines
    text = re.sub(
        r"\n\s*\n\s*\n+",
        "\n\n",
        text
    )

    return text.strip()


# ============================================================
# SECTION HEADING DETECTION
# ============================================================

def is_section_heading(line):
    """
    Detect genuine HLD section headings.

    Valid examples:
        1. System Overview
        1.1 Primary Functions
        1.2 Architectural Components
        2. Interfaces and Ports
        2.1 Port Catalogue
        3. Signals
        4. Functional Flow
        4.1 Dependency Summary
        5. Architecture Notes
        6. Revision History

    Important:
    Numbered functional-flow sentences such as:

        1. SensorManager reads...
        2. SensorManager validates...

    must NOT be treated as section headings.
    """

    line = line.strip()

    if not line:
        return False

    # --------------------------------------------------------
    # Reject numbered sentences
    # --------------------------------------------------------

    numbered_sentence = re.match(
        r"^\d+\.\s+.+\b(reads|validates|publishes|filters|"
        r"calculates|evaluates|converts|monitors|"
        r"provides|determines|controls|communicates|"
        r"acquires|manages|exchanges)\b",
        line,
        flags=re.IGNORECASE
    )

    if numbered_sentence:
        return False

    # --------------------------------------------------------
    # Reject long numbered sentences
    # --------------------------------------------------------

    if re.match(r"^\d+\.\s+", line):

        # Real headings are normally short.
        if len(line) > 80:
            return False

    # --------------------------------------------------------
    # Main section
    #
    # Example:
    # 1. System Overview
    # --------------------------------------------------------

    if re.match(
        r"^\d+\.\s+[A-Za-z][A-Za-z0-9 /&_-]*$",
        line
    ):
        return True

    # --------------------------------------------------------
    # Subsection
    #
    # Example:
    # 1.1 Primary Functions
    # --------------------------------------------------------

    if re.match(
        r"^\d+\.\d+\s+[A-Za-z][A-Za-z0-9 /&_-]*$",
        line
    ):
        return True

    return False


# ============================================================
# SECTION DETECTION
# ============================================================

def detect_sections(text):
    """
    Split text into logical sections.
    """

    lines = text.splitlines()

    sections = []

    current_heading = "General"
    current_lines = []

    for line in lines:

        stripped = line.strip()

        if is_section_heading(stripped):

            # Save previous section
            if current_lines:

                section_text = "\n".join(
                    current_lines
                ).strip()

                if section_text:

                    sections.append({
                        "section": current_heading,
                        "text": section_text
                    })

            # Start new section
            current_heading = stripped
            current_lines = []

        else:

            current_lines.append(line)

    # Save final section
    if current_lines:

        section_text = "\n".join(
            current_lines
        ).strip()

        if section_text:

            sections.append({
                "section": current_heading,
                "text": section_text
            })

    return sections


# ============================================================
# STRUCTURED CONTENT DETECTION
# ============================================================

def looks_like_structured_content(text):
    """
    Detect AUTOSAR-style structured/table content.
    """

    keywords = [
        "Component",
        "Interface",
        "Provider",
        "Consumer",
        "Signals",
        "Port",
        "Direction",
        "Responsibility",
        "Main Inputs",
        "Main Outputs",
        "Type",
        "Data Type",
        "Source",
        "Destination",
        "Range / Values"
    ]

    matches = 0

    lower_text = text.lower()

    for keyword in keywords:

        if keyword.lower() in lower_text:
            matches += 1

    return matches >= 2


# ============================================================
# NATURAL BREAK DETECTION
# ============================================================

def find_natural_break(
    text,
    start,
    end
):
    """
    Find a natural point for splitting text.
    """

    possible_breaks = [

        text.rfind(
            "\n\n",
            start,
            end
        ),

        text.rfind(
            "\n",
            start,
            end
        ),

        text.rfind(
            ". ",
            start,
            end
        ),

        text.rfind(
            "; ",
            start,
            end
        )
    ]

    valid_breaks = [
        position
        for position in possible_breaks
        if position > start
    ]

    if not valid_breaks:
        return end

    best_break = max(
        valid_breaks
    )

    # Don't create extremely small chunks
    if best_break - start < (
        (end - start) * 0.5
    ):
        return end

    return best_break


# ============================================================
# SPLIT LARGE SECTION
# ============================================================

def split_large_section(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):
    """
    Split large sections using natural boundaries.
    """

    if not text:
        return []

    if overlap >= chunk_size:

        raise ValueError(
            "CHUNK_OVERLAP must be smaller "
            "than CHUNK_SIZE."
        )

    # Small section
    if len(text) <= chunk_size:

        return [
            text.strip()
        ]

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = min(
            start + chunk_size,
            text_length
        )

        # Final chunk
        if end >= text_length:

            chunk = text[
                start:end
            ].strip()

            if chunk:
                chunks.append(chunk)

            break

        # Find natural boundary
        end = find_natural_break(
            text,
            start,
            end
        )

        chunk = text[
            start:end
        ].strip()

        if chunk:
            chunks.append(chunk)

        # Overlap
        next_start = end - overlap

        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


# ============================================================
# CREATE SECTION CHUNKS
# ============================================================

def create_section_chunks(
    section_name,
    section_text
):
    """
    Create chunks for one section.
    """

    if not section_text:
        return []

    structured = (
        looks_like_structured_content(
            section_text
        )
    )

    if structured:

        return split_large_section(
            section_text,
            chunk_size=1800,
            overlap=250
        )

    return split_large_section(
        section_text,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP
    )


# ============================================================
# CREATE DOCUMENT CHUNKS
# ============================================================

def create_document_chunks(pages):
    """
    Create structure-aware chunks.

    Each chunk contains:

        chunk_id
        page_number
        section
        chunk_number
        content_type
        text
        character_count
    """

    all_chunks = []

    for page in pages:

        page_number = page[
            "page_number"
        ]

        raw_text = page[
            "text"
        ]

        cleaned_text = clean_text(
            raw_text
        )

        if not cleaned_text:
            continue

        # Detect logical sections
        sections = detect_sections(
            cleaned_text
        )

        # Fallback
        if not sections:

            sections = [
                {
                    "section": "General",
                    "text": cleaned_text
                }
            ]

        page_chunk_number = 1

        for section in sections:

            section_name = section[
                "section"
            ]

            section_text = section[
                "text"
            ]

            if not section_text:
                continue

            # Determine content type
            if looks_like_structured_content(
                section_text
            ):

                content_type = "structured"

            else:

                content_type = "text"

            # Split section
            section_chunks = (
                create_section_chunks(
                    section_name,
                    section_text
                )
            )

            for chunk_text in section_chunks:

                final_text = (
                    f"Section: "
                    f"{section_name}\n\n"
                    f"{chunk_text}"
                )

                chunk_data = {

                    "chunk_id": (
                        f"page_{page_number}"
                        f"_chunk_{page_chunk_number}"
                    ),

                    "page_number":
                        page_number,

                    "section":
                        section_name,

                    "chunk_number":
                        page_chunk_number,

                    "content_type":
                        content_type,

                    "text":
                        final_text,

                    "character_count":
                        len(final_text)
                }

                all_chunks.append(
                    chunk_data
                )

                page_chunk_number += 1

    return all_chunks


# ============================================================
# VALIDATE CHUNKS
# ============================================================

def validate_chunks(chunks):
    """
    Perform basic quality checks.
    """

    warnings = []

    if not chunks:

        warnings.append(
            "No chunks were generated."
        )

        return warnings

    for chunk in chunks:

        chunk_id = chunk[
            "chunk_id"
        ]

        text = chunk[
            "text"
        ]

        # Empty chunk
        if not text.strip():

            warnings.append(
                f"Empty chunk: {chunk_id}"
            )

        # Very small chunk
        if len(text) < 50:

            warnings.append(
                f"Very small chunk: "
                f"{chunk_id} "
                f"({len(text)} characters)"
            )

        # Missing section context
        if "Section:" not in text:

            warnings.append(
                f"Missing section context: "
                f"{chunk_id}"
            )

    return warnings


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_chunk_summary(chunks):

    print("\n" + "=" * 70)
    print("CHUNKING SUMMARY")
    print("=" * 70)

    print(
        f"Total chunks: {len(chunks)}"
    )

    if not chunks:
        return

    total_characters = sum(
        chunk["character_count"]
        for chunk in chunks
    )

    average_size = (
        total_characters
        / len(chunks)
    )

    structured_count = sum(
        1
        for chunk in chunks
        if chunk["content_type"]
        == "structured"
    )

    text_count = sum(
        1
        for chunk in chunks
        if chunk["content_type"]
        == "text"
    )

    print(
        f"Average chunk size: "
        f"{average_size:.2f} characters"
    )

    print(
        f"Structured chunks: "
        f"{structured_count}"
    )

    print(
        f"Text chunks: "
        f"{text_count}"
    )

    print("\nGenerated chunks:")

    for chunk in chunks:

        print(
            f"- {chunk['chunk_id']} | "
            f"Page: {chunk['page_number']} | "
            f"Section: {chunk['section']} | "
            f"Type: {chunk['content_type']} | "
            f"Size: {chunk['character_count']}"
        )


# ============================================================
# DISPLAY CHUNKS
# ============================================================

def display_chunks(
    chunks,
    number=8
):

    count = min(
        number,
        len(chunks)
    )

    print("\n" + "=" * 70)

    print(
        f"FIRST {count} CHUNKS"
    )

    print("=" * 70)

    for chunk in chunks[:count]:

        print("\n" + "-" * 70)

        print(
            f"Chunk ID      : "
            f"{chunk['chunk_id']}"
        )

        print(
            f"Page Number   : "
            f"{chunk['page_number']}"
        )

        print(
            f"Section       : "
            f"{chunk['section']}"
        )

        print(
            f"Chunk Number  : "
            f"{chunk['chunk_number']}"
        )

        print(
            f"Content Type  : "
            f"{chunk['content_type']}"
        )

        print(
            f"Character Size: "
            f"{chunk['character_count']}"
        )

        print("-" * 70)

        print(
            chunk["text"]
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Find project root
    # --------------------------------------------------------

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    # --------------------------------------------------------
    # Add project root to Python path
    # --------------------------------------------------------

    if str(project_root) not in sys.path:

        sys.path.insert(
            0,
            str(project_root)
        )

    # --------------------------------------------------------
    # Import PDF reader
    # --------------------------------------------------------

    from src.ingestion.pdf_reader import (
        extract_text_from_pdf
    )

    # --------------------------------------------------------
    # PDF path
    # --------------------------------------------------------

    pdf_path = (
        project_root
        / "data"
        / "raw"
        / "sample_hld.pdf"
    )

    # --------------------------------------------------------
    # Check PDF
    # --------------------------------------------------------

    if not pdf_path.exists():

        print(
            "\nERROR: PDF file not found."
        )

        print(
            f"Expected location:\n"
            f"{pdf_path}"
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Start
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "AUTOSAR HLD "
        "STRUCTURE-AWARE CHUNKING"
    )

    print("=" * 70)

    print(
        f"\nPDF: {pdf_path}"
    )

    # --------------------------------------------------------
    # Extract pages
    # --------------------------------------------------------

    pages = extract_text_from_pdf(
        str(pdf_path)
    )

    print(
        f"Total pages extracted: "
        f"{len(pages)}"
    )

    # --------------------------------------------------------
    # Create chunks
    # --------------------------------------------------------

    chunks = create_document_chunks(
        pages
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    warnings = validate_chunks(
        chunks
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_chunk_summary(
        chunks
    )

    # --------------------------------------------------------
    # Validation output
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CHUNK VALIDATION")
    print("=" * 70)

    if warnings:

        print(
            f"Warnings found: "
            f"{len(warnings)}"
        )

        for warning in warnings:

            print(
                f"WARNING: {warning}"
            )

    else:

        print(
            "No chunk quality warnings found."
        )

    # --------------------------------------------------------
    # Display chunks
    # --------------------------------------------------------

    display_chunks(
        chunks,
        number=8
    )

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "STRUCTURE-AWARE CHUNKING "
        "COMPLETED SUCCESSFULLY"
    )

    print("=" * 70)