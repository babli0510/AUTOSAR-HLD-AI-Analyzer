
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VECTOR_DB_PATH = PROJECT_ROOT / "data" / "vectorstore"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

COLLECTION_NAME = "autosar_hld_chunks"


# ============================================================
# VECTOR STORE
# ============================================================

class VectorStore:
    """Manage document embeddings and retrieval using ChromaDB."""

    def __init__(self):
        print("\nInitializing ChromaDB...")

        self.embedding_model = SentenceTransformer(MODEL_NAME)

        self.client = chromadb.PersistentClient(
            path=str(VECTOR_DB_PATH)
        )

        # Reuse the existing collection; do not delete old data.
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME
        )

        print("ChromaDB initialized successfully.")
        print(f"Collection: {COLLECTION_NAME}")
        print(f"Database path: {VECTOR_DB_PATH}")

    # ========================================================
    # EMBEDDINGS
    # ========================================================

    def generate_embeddings(self, texts):
        """Generate normalized embeddings for text documents."""

        if not texts:
            return []

        embeddings = self.embedding_model.encode(
            texts,
            normalize_embeddings=True
        )

        return embeddings.tolist()

    # ========================================================
    # ADD DOCUMENTS
    # ========================================================

    def add_documents(self, chunks, document_id="legacy"):
        """
        Index chunks belonging to a specific document.

        Args:
            chunks: Output from create_document_chunks().
            document_id: Unique identifier for the source PDF.

        Returns:
            Number of chunks indexed.
        """

        if not chunks:
            print("No chunks provided.")
            return 0

        document_id = str(document_id).strip()

        if not document_id:
            raise ValueError("document_id cannot be empty.")

        documents = []
        ids = []
        metadatas = []

        for chunk in chunks:
            original_chunk_id = str(chunk["chunk_id"])

            # Different PDFs cannot overwrite one another's chunks.
            unique_chunk_id = (
                f"{document_id}_{original_chunk_id}"
            )

            text = str(chunk.get("text", "")).strip()

            if not text:
                continue

            documents.append(text)
            ids.append(unique_chunk_id)

            metadatas.append({
                "document_id": document_id,
                "original_chunk_id": original_chunk_id,
                "page_number": int(chunk["page_number"]),
                "section": str(chunk["section"]),
                "chunk_number": int(chunk["chunk_number"]),
                "content_type": str(chunk["content_type"]),
            })

        if not documents:
            print("No valid text chunks to index.")
            return 0

        embeddings = self.generate_embeddings(documents)

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )

        print(
            f"Indexed {len(documents)} chunks "
            f"for document '{document_id}'."
        )

        return len(documents)

    # ========================================================
    # SEARCH
    # ========================================================

    def search(self, query, top_k=3, document_id=None):
        """
        Search for relevant chunks.

        Args:
            query: User's question.
            top_k: Maximum number of results.
            document_id: Optional document namespace filter.

        Returns:
            List of retrieved chunks with metadata and distances.
        """

        if not isinstance(query, str) or not query.strip():
            return []

        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")

        if self.collection.count() == 0:
            return []

        query_embedding = self.embedding_model.encode(
            query.strip(),
            normalize_embeddings=True
        ).tolist()

        search_args = {
            "query_embeddings": [query_embedding],
            "n_results": min(
                top_k,
                self.collection.count()
            ),
            "include": [
                "documents",
                "metadatas",
                "distances"
            ]
        }

        # Only return chunks from the selected PDF.
        if document_id is not None:
            document_id = str(document_id).strip()

            if not document_id:
                return []

            search_args["where"] = {
                "document_id": document_id
            }

        results = self.collection.query(**search_args)

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        formatted_results = []

        for index, chunk_id in enumerate(ids):
            formatted_results.append({
                "id": chunk_id,
                "document": documents[index],
                "metadata": metadatas[index] or {},
                "distance": (
                    distances[index]
                    if index < len(distances)
                    else None
                )
            })

        return formatted_results

    # ========================================================
    # DOCUMENT UTILITIES
    # ========================================================

    def count(self):
        """Return the total number of chunks in the collection."""

        return self.collection.count()

    def count_document(self, document_id):
        """Return the number of chunks belonging to one document."""

        document_id = str(document_id).strip()

        if not document_id:
            return 0

        results = self.collection.get(
            where={"document_id": document_id},
            include=["metadatas"]
        )

        return len(results.get("ids", []))


# ============================================================
# BASIC TEST
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("AUTOSAR HLD VECTOR STORE TEST")
    print("=" * 70)

    vector_store = VectorStore()

    print(
        "\nTotal indexed chunks:",
        vector_store.count()
    )

    test_query = (
        "What are the main components and their responsibilities?"
    )

    results = vector_store.search(
        query=test_query,
        top_k=3
    )

    if not results:
        print(
            "\nNo indexed chunks found for this query."
        )
    else:
        for index, result in enumerate(results, start=1):
            metadata = result["metadata"]

            print(f"\nResult {index}")
            print("-" * 50)
            print("Chunk ID:", result["id"])
            print("Document ID:", metadata.get("document_id", "legacy"))
            print("Page:", metadata.get("page_number", "N/A"))
            print("Section:", metadata.get("section", "N/A"))
            print("Distance:", result["distance"])
            print("Content:", result["document"][:500])

    print("\nVector store test completed.")
