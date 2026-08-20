from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


CHUNK_SIZE = 512
CHUNK_OVERLAP = 100


def create_text_splitter():
    """
    Create the text splitter.

    Chunk size: 512
    Overlap: 100
    """

    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=lambda text: len(text.split()),
    )


def build_metadata(
    element: dict,
    content_type: str,
    document_name: str,
) -> dict:
    """
    Convert parser metadata into retrieval metadata.

    The original uploaded filename is explicitly passed
    so temporary parser filenames are never exposed
    in citations.
    """

    source_metadata = element.get(
        "metadata",
        {},
    )

    return {
        "chunk_type": (
            "image_caption"
            if content_type == "image"
            else content_type
        ),
        "source_page": source_metadata.get(
            "page_number"
        ),
        "document_name": document_name,
        "section": source_metadata.get(
            "section"
        ),
        "element_type": source_metadata.get(
            "element_type"
        ),
        "image_ref": source_metadata.get(
            "image_base64"
        ),
        "position": source_metadata.get(
            "position"
        ),
    }


def create_chunks(
    parsed_elements: list[dict],
    document_name: str,
) -> list[Document]:
    """
    Convert parsed elements into LangChain Documents.

    Text:
        Group text belonging to the same section and
        split using 512 size with 100 overlap.

    Tables:
        Keep as separate chunks.

    Images:
        Keep as separate chunks.
    """

    splitter = create_text_splitter()

    final_chunks: list[Document] = []

    text_buffer: list[str] = []
    text_metadata: dict | None = None

    def flush_text_buffer():

        nonlocal text_buffer
        nonlocal text_metadata

        if not text_buffer:
            return

        combined_text = "\n\n".join(
            text_buffer
        ).strip()

        if not combined_text:
            text_buffer = []
            text_metadata = None
            return

        documents = splitter.create_documents(
            [combined_text],
            metadatas=[
                text_metadata or {}
            ],
        )

        final_chunks.extend(
            documents
        )

        text_buffer = []
        text_metadata = None

    for element in parsed_elements:

        content = element.get(
            "content",
            "",
        ).strip()

        if not content:
            continue

        content_type = element.get(
            "content_type",
            "text",
        )

        metadata = build_metadata(
            element,
            content_type,
            document_name,
        )

        # -----------------------------------------------------
        # TEXT
        # -----------------------------------------------------

        if content_type == "text":

            if not text_buffer:

                text_buffer.append(
                    content
                )

                text_metadata = metadata

                continue

            existing_section = (
                text_metadata.get("section")
                if text_metadata
                else None
            )

            current_section = metadata.get(
                "section"
            )

            if (
                existing_section
                == current_section
            ):

                text_buffer.append(
                    content
                )

            else:

                flush_text_buffer()

                text_buffer.append(
                    content
                )

                text_metadata = metadata

        # -----------------------------------------------------
        # TABLE
        # -----------------------------------------------------

        elif content_type == "table":

            flush_text_buffer()

            final_chunks.append(
                Document(
                    page_content=content,
                    metadata=metadata,
                )
            )

        # -----------------------------------------------------
        # IMAGE
        # -----------------------------------------------------

        elif content_type == "image":

            flush_text_buffer()

            final_chunks.append(
                Document(
                    page_content=content,
                    metadata=metadata,
                )
            )

    # Flush remaining text.
    flush_text_buffer()

    return final_chunks