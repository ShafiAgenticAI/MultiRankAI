from pathlib import Path
import tempfile

from src.ingestion.parser import parse_document
from src.ingestion.vision import enrich_image_elements
from src.ingestion.chunker import create_chunks
from src.retrieval.vector_store import add_documents


def ingest_pdf(
    file_bytes: bytes,
    file_name: str,
) -> dict:
    """
    Complete PDF ingestion pipeline.

    Flow:

    PDF bytes
        ↓
    Temporary PDF file
        ↓
    Docling parser
        ↓
    Vision enrichment
        ↓
    Chunking
        ↓
    OpenAI embeddings
        ↓
    PostgreSQL + pgvector
    """

    if not file_bytes:
        raise ValueError("Uploaded file is empty.")

    if not file_name:
        raise ValueError("File name is required.")

    if not file_name.lower().endswith(".pdf"):
        raise ValueError(
            "Only PDF files are supported."
        )

    temporary_path: Path | None = None

    try:

        # --------------------------------------------------
        # Create temporary PDF
        # --------------------------------------------------

        with tempfile.NamedTemporaryFile(
            suffix=".pdf",
            delete=False,
        ) as temp_file:

            temp_file.write(file_bytes)

            temporary_path = Path(
                temp_file.name
            )

        print()
        print("=" * 60)
        print("INGESTION PIPELINE STARTED")
        print("=" * 60)
        print(f"File: {file_name}")

        # --------------------------------------------------
        # Step 1: Parse PDF
        # --------------------------------------------------

        parsed_elements = parse_document(
            temporary_path
        )

        print(
            f"Parsed elements: {len(parsed_elements)}"
        )

        # --------------------------------------------------
        # Step 2: Vision enrichment
        # --------------------------------------------------

        enriched_elements = enrich_image_elements(
            parsed_elements
        )

        print(
            f"Enriched elements: "
            f"{len(enriched_elements)}"
        )

        # --------------------------------------------------
        # Step 3: Create LangChain Documents
        # --------------------------------------------------

        documents = create_chunks(
            enriched_elements,
            document_name=file_name,
        )

        print(
            f"Documents to embed: "
            f"{len(documents)}"
        )

        if not documents:
            raise ValueError(
                "No searchable documents were created "
                "from the PDF."
            )

        # --------------------------------------------------
        # Step 4: Store in PostgreSQL / PGVector
        # --------------------------------------------------

        add_documents(
            documents
        )

        print()
        print("=" * 60)
        print("DOCUMENTS STORED SUCCESSFULLY")
        print("=" * 60)

        return {
            "file_name": file_name,
            "parsed_elements": len(
                parsed_elements
            ),
            "documents_created": len(
                documents
            ),
            "status": "success",
        }

    finally:

        # --------------------------------------------------
        # Remove temporary PDF
        # --------------------------------------------------

        if (
            temporary_path
            and temporary_path.exists()
        ):
            temporary_path.unlink()