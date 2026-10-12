
import hashlib
import sys
import tempfile
from pathlib import Path

import streamlit as st


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = str(PROJECT_ROOT / "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from ingestion.pdf_reader import extract_text_from_pdf
from preprocessing.chunker import create_document_chunks
from vectorstore.vector_store import VectorStore
from rag.answer_generator import RAGAnswerGenerator


MAX_PDF_SIZE_MB = 30
MAX_PDF_SIZE_BYTES = MAX_PDF_SIZE_MB * 1024 * 1024


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AUTOSAR HLD AI Analyzer",
    page_icon="📘",
    layout="wide",
)


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 36px;
        font-weight: 750;
        margin-bottom: 4px;
    }

    .subtitle {
        color: #64748b;
        font-size: 16px;
        margin-bottom: 24px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {
    "active_document_id": None,
    "document_name": None,
    "document_hash": None,
    "document_pages": 0,
    "document_chunks": 0,
    "rag": None,
    "vector_store": None,
    "last_result": None,
    "last_question": None,
}

for key, default_value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📘 AUTOSAR HLD AI Analyzer</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Evidence-grounded engineering document analysis'
    '</div>',
    unsafe_allow_html=True,
)

st.info(
    "Upload an engineering HLD PDF, index its content, "
    "and ask questions grounded in the document."
)


# ============================================================
# SIDEBAR: PDF UPLOAD
# ============================================================

with st.sidebar:
    st.header("Document Workspace")

    uploaded_file = st.file_uploader(
        "Upload an HLD PDF",
        type=["pdf"],
        help=f"Maximum upload size: {MAX_PDF_SIZE_MB} MB.",
    )

    st.caption(
        "Text-based PDFs are supported. "
        "Scanned PDFs may require OCR."
    )

    index_button = st.button(
        "Process and Index PDF",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None,
    )

    st.divider()

    st.subheader("Pipeline")

    st.markdown(
        """
        1. PDF validation
        2. Text extraction
        3. Section-aware chunking
        4. Embedding and indexing
        5. Document-specific retrieval
        6. Evidence-grounded answer
        """
    )


# ============================================================
# PDF INGESTION
# ============================================================

if index_button and uploaded_file is not None:
    pdf_bytes = uploaded_file.getvalue()

    if not pdf_bytes:
        st.error("The uploaded file is empty.")

    elif len(pdf_bytes) > MAX_PDF_SIZE_BYTES:
        st.error(
            f"The PDF exceeds the {MAX_PDF_SIZE_MB} MB limit."
        )

    elif not pdf_bytes.startswith(b"%PDF-"):
        st.error(
            "The uploaded file does not have a valid PDF signature."
        )

    else:
        document_hash = hashlib.sha256(pdf_bytes).hexdigest()
        document_id = f"doc_{document_hash[:24]}"

        if st.session_state.document_hash == document_hash:
            st.info(
                "This document is already indexed in this session."
            )

        else:
            temporary_pdf = None

            try:
                with st.spinner(
                    "Extracting, chunking, and indexing the PDF..."
                ):
                    with tempfile.NamedTemporaryFile(
                        mode="wb",
                        suffix=".pdf",
                        delete=False,
                    ) as temporary_file:
                        temporary_file.write(pdf_bytes)
                        temporary_pdf = Path(temporary_file.name)

                    pages = extract_text_from_pdf(
                        str(temporary_pdf)
                    )

                    if not pages:
                        raise ValueError(
                            "No pages could be extracted from the PDF."
                        )

                    extracted_text = "\n".join(
                        page.get("text", "")
                        for page in pages
                    ).strip()

                    if not extracted_text:
                        raise ValueError(
                            "No selectable text was found. "
                            "The PDF may be scanned and require OCR."
                        )

                    chunks = create_document_chunks(pages)

                    if not chunks:
                        raise ValueError(
                            "The PDF was read, but no chunks were created."
                        )

                    if st.session_state.vector_store is None:
                        st.session_state.vector_store = VectorStore()

                    indexed_count = (
                        st.session_state.vector_store.add_documents(
                            chunks,
                            document_id=document_id,
                        )
                    )

                    if not indexed_count:
                        raise ValueError(
                            "No chunks were indexed successfully."
                        )

                    # Activate the document only after indexing.
                    st.session_state.active_document_id = document_id
                    st.session_state.document_name = uploaded_file.name
                    st.session_state.document_hash = document_hash
                    st.session_state.document_pages = len(pages)
                    st.session_state.document_chunks = indexed_count

                    # Use the current document ID for retrieval.
                    st.session_state.rag = RAGAnswerGenerator(
                        top_k=3,
                        document_id=document_id,
                    )

                    st.session_state.last_result = None
                    st.session_state.last_question = None

                st.success(
                    f"Successfully indexed {indexed_count} chunks."
                )

            except Exception as error:
                st.error(f"PDF processing failed: {error}")

            finally:
                if temporary_pdf is not None:
                    try:
                        temporary_pdf.unlink(missing_ok=True)
                    except OSError:
                        pass


# ============================================================
# DOCUMENT STATUS
# ============================================================

st.header("Document Status")

if st.session_state.active_document_id:
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "PDF Pages",
        st.session_state.document_pages,
    )

    col2.metric(
        "Indexed Chunks",
        st.session_state.document_chunks,
    )

    col3.metric(
        "Search Scope",
        "Active PDF",
    )

    st.write(
        f"**Active document:** "
        f"{st.session_state.document_name}"
    )

    st.caption(
        f"Document ID: {st.session_state.active_document_id}"
    )

else:
    st.warning(
        "No PDF is active in this session. "
        "You can connect to the existing ChromaDB collection below."
    )

    if st.session_state.rag is None:
        if st.button(
            "Connect to Existing ChromaDB",
            use_container_width=True,
        ):
            try:
                with st.spinner(
                    "Connecting to the existing knowledge base..."
                ):
                    # No document_id means search the configured
                    # existing collection using the current retriever.
                    st.session_state.rag = RAGAnswerGenerator(
                        top_k=3
                    )

                st.success(
                    "Connected to the existing ChromaDB knowledge base."
                )

                st.rerun()

            except Exception as error:
                st.error(
                    f"Could not initialize the analysis engine: {error}"
                )


# ============================================================
# QUESTION INPUT
# ============================================================

st.header("Ask the Document")

question = st.text_area(
    "Your question",
    placeholder=(
        "Example: What are the three aspects covered in the scope "
        "of the AUTOSAR Software-Component Template?"
    ),
    height=110,
)

analyze_button = st.button(
    "Analyze Document",
    type="primary",
    use_container_width=True,
    disabled=st.session_state.rag is None,
)


# ============================================================
# ANSWER GENERATION AND EVIDENCE MAPPING
# ============================================================

if analyze_button:
    if not question.strip():
        st.warning("Enter a question before analyzing.")

    elif st.session_state.rag is None:
        st.error(
            "The analysis engine is not initialized. "
            "Index a PDF or connect to ChromaDB first."
        )

    else:
        try:
            with st.spinner(
                "Retrieving evidence and generating an answer..."
            ):
                rag = st.session_state.rag
                result = rag.generate_answer(question.strip())

                if not isinstance(result, dict):
                    raise TypeError(
                        "Answer generator returned an unexpected result."
                    )

                # Build a map of actual retrieved text by chunk ID.
                # The current retriever returns chunk_id/content fields.
                evidence_by_id = {}

                evidence_records = rag.retriever.retrieve(
                    query=question.strip(),
                    top_k=rag.top_k,
                    document_id=rag.document_id,
                )

                for record in evidence_records or []:
                    record_id = (
                        record.get("chunk_id")
                        or record.get("id")
                    )

                    record_text = (
                        record.get("content")
                        or record.get("text")
                        or record.get("evidence")
                        or record.get("document")
                        or ""
                    )

                    if record_id and record_text:
                        evidence_by_id[str(record_id)] = str(
                            record_text
                        )

                fixed_sources = []

                for source in result.get("sources", []) or []:
                    source_copy = dict(source)

                    chunk_id = str(
                        source_copy.get("chunk_id")
                        or source_copy.get("id")
                        or ""
                    )

                    # Prefer the actual source content already returned.
                    evidence = (
                        source_copy.get("content")
                        or source_copy.get("evidence")
                        or source_copy.get("document")
                        or ""
                    )

                    # If missing, match by chunk ID.
                    if not evidence and chunk_id:
                        evidence = evidence_by_id.get(
                            chunk_id,
                            "",
                        )

                    source_copy["evidence"] = evidence
                    fixed_sources.append(source_copy)

                result["sources"] = fixed_sources

                st.session_state.last_result = result
                st.session_state.last_question = question.strip()

        except Exception as error:
            st.error(f"Analysis failed: {error}")


# ============================================================
# DISPLAY ANSWER AND SUPPORTING EVIDENCE
# ============================================================

result = st.session_state.last_result

if result:
    st.header("Analysis Result")

    st.markdown("### Answer")

    answer_text = result.get(
        "answer",
        "No answer generated.",
    )

    st.markdown(
        answer_text
        if answer_text
        else "No answer was returned by the model."
    )

    sources = result.get("sources", []) or []

    st.markdown("### Supporting Evidence")

    if not sources:
        st.warning(
            "No supporting evidence was retrieved. "
            "Do not treat the answer as verified."
        )

    else:
        st.caption(
            f"{len(sources)} retrieved source chunk(s). "
            "Retrieval alone does not guarantee answer correctness."
        )

        for index, source in enumerate(sources, start=1):
            with st.expander(
                f"Source {index}",
                expanded=(index == 1),
            ):
                col1, col2, col3 = st.columns([1, 1, 2])

                col1.write(
                    f"**Page:** {source.get('page', 'Unknown')}"
                )

                col2.write(
                    f"**Section:** {source.get('section', 'Unknown')}"
                )

                col3.write(
                    f"**Chunk:** {source.get('chunk_id', 'Unknown')}"
                )

                evidence = (
                    source.get("evidence")
                    or source.get("content")
                    or source.get("document")
                    or source.get("text")
                    or ""
                )

                st.markdown("**Retrieved text**")

                if evidence and str(evidence).strip():
                    st.text(str(evidence).strip())

                else:
                    st.warning(
                        "Evidence text could not be matched to this "
                        "source. The source reference is shown, but "
                        "its text is not verified in this panel."
                    )

                distance = source.get("distance")

                if isinstance(distance, (int, float)):
                    st.caption(
                        f"Vector distance: {distance:.4f} "
                        "(not a probability or confidence percentage)"
                    )

    if result.get("error"):
        st.error(str(result["error"]))


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AUTOSAR HLD AI Analyzer | "
    "ChromaDB | Sentence Transformers | Ollama"
)
