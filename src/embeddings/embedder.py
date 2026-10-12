import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# EMBEDDING MODEL
# ============================================================

class EmbeddingModel:
    """
    Wrapper around the Sentence Transformers embedding model.
    """

    def __init__(self, model_name=MODEL_NAME):
        """
        Load the embedding model.

        Args:
            model_name (str): Hugging Face/Sentence Transformers
                              model name.
        """

        print("\n" + "=" * 70)
        print("LOADING EMBEDDING MODEL")
        print("=" * 70)

        print(f"Model: {model_name}")

        self.model = SentenceTransformer(
            model_name
        )

        print(
            "Embedding model loaded successfully."
        )

        print(
            f"Embedding dimension: "
            f"{self.model.get_sentence_embedding_dimension()}"
        )

    def encode_text(self, text):
        """
        Generate an embedding for a single text.

        Args:
            text (str): Input text.

        Returns:
            numpy.ndarray: Embedding vector.
        """

        if not text or not text.strip():
            raise ValueError(
                "Cannot generate embedding "
                "for empty text."
            )

        embedding = self.model.encode(
            text,
            normalize_embeddings=True
        )

        return embedding

    def encode_chunks(self, chunks):
        """
        Generate embeddings for document chunks.

        Args:
            chunks (list): List of chunk dictionaries.

        Returns:
            list: Chunks with embedding vectors.
        """

        if not chunks:
            return []

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        print("\nGenerating embeddings...")

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        embedded_chunks = []

        for chunk, embedding in zip(
            chunks,
            embeddings
        ):

            chunk_with_embedding = {
                **chunk,
                "embedding": embedding
            }

            embedded_chunks.append(
                chunk_with_embedding
            )

        return embedded_chunks


# ============================================================
# TEST WITH SAMPLE HLD
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
    # Import chunker
    # --------------------------------------------------------

    from src.preprocessing.chunker import (
        create_document_chunks
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
    # Extract PDF
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("AUTOSAR HLD EMBEDDING TEST")
    print("=" * 70)

    print(
        f"\nPDF: {pdf_path}"
    )

    pages = extract_text_from_pdf(
        str(pdf_path)
    )

    print(
        f"Pages extracted: "
        f"{len(pages)}"
    )

    # --------------------------------------------------------
    # Create chunks
    # --------------------------------------------------------

    chunks = create_document_chunks(
        pages
    )

    print(
        f"Chunks created: "
        f"{len(chunks)}"
    )

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    embedding_model = EmbeddingModel()

    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    embedded_chunks = (
        embedding_model.encode_chunks(
            chunks
        )
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("EMBEDDING RESULTS")
    print("=" * 70)

    print(
        f"Total embedded chunks: "
        f"{len(embedded_chunks)}"
    )

    if embedded_chunks:

        first_chunk = (
            embedded_chunks[0]
        )

        embedding = (
            first_chunk["embedding"]
        )

        print(
            f"\nFirst chunk ID: "
            f"{first_chunk['chunk_id']}"
        )

        print(
            f"Page: "
            f"{first_chunk['page_number']}"
        )

        print(
            f"Section: "
            f"{first_chunk['section']}"
        )

        print(
            f"Embedding dimension: "
            f"{len(embedding)}"
        )

        print(
            "\nFirst 10 embedding values:"
        )

        print(
            embedding[:10]
        )

    # --------------------------------------------------------
    # Completion
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "EMBEDDING GENERATION "
        "COMPLETED SUCCESSFULLY"
    )
    print("=" * 70)