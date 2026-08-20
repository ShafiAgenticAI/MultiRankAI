from langchain_core.documents import Document

from src.retrieval.vector_store import get_vector_store
from src.retrieval.query_router import get_product_section


def _get_product_prefix(
    product_section: str | None,
) -> str | None:
    """
    Convert:

        SECTION 1: HOME LOAN PRODUCTS

    into:

        1.
    """

    if not product_section:
        return None

    section_upper = (
        product_section.upper().strip()
    )

    if not section_upper.startswith(
        "SECTION "
    ):
        return None

    try:

        section_number = (
            section_upper
            .replace(
                "SECTION ",
                "",
                1,
            )
            .split(
                ":",
                1,
            )[0]
            .strip()
        )

        return f"{section_number}."

    except Exception:

        return None


def _is_product_match(
    document: Document,
    product_section: str | None,
) -> bool:
    """
    Check whether a document belongs to the
    product identified from the query.
    """

    if not product_section:

        return True

    document_section = str(
        document.metadata.get(
            "section",
            "",
        )
    ).strip()

    document_section_upper = (
        document_section.upper()
    )

    product_section_upper = (
        product_section.upper().strip()
    )

    # --------------------------------------------------------
    # Exact section match
    # --------------------------------------------------------

    if (
        document_section_upper
        == product_section_upper
    ):

        return True

    # --------------------------------------------------------
    # Subsection match
    # --------------------------------------------------------

    product_prefix = _get_product_prefix(
        product_section
    )

    if (
        product_prefix
        and document_section_upper.startswith(
            product_prefix
        )
    ):

        return True

    return False


def _is_section_header(
    document: Document,
) -> bool:
    """
    Identify section-header-only chunks.

    We use the actual metadata structure coming from the
    indexed PDF.
    """

    return (
        document.metadata.get(
            "element_type"
        )
        == "section_header"
    )


def _document_key(
    document: Document,
) -> str:
    """
    Create a stable deduplication key.
    """

    return (
        document.page_content
        .strip()
        .lower()
    )


def search_vector(
    query: str,
    k: int = 5,
    candidate_k: int = 20,
) -> list[Document]:
    """
    Product-aware semantic vector search.

    Flow:

        1. Retrieve a large semantic candidate pool.
        2. Detect product.
        3. Keep product-matching documents.
        4. Remove section-header-only chunks.
        5. Remove duplicate content.
        6. Return the best actual content chunks.

    Section headers are NOT used as fallback results because
    they do not contain enough information to answer the user.
    """

    print(
        "[VECTOR SEARCH] called"
    )

    if not query or not query.strip():

        return []

    vector_store = get_vector_store()

    product_section = get_product_section(
        query
    )

    # --------------------------------------------------------
    # Retrieve a larger candidate pool.
    #
    # The previous implementation could stop too early and
    # return section headers before reaching useful content.
    # --------------------------------------------------------

    raw_candidate_k = max(
        candidate_k * 5,
        50,
    )

    candidates = (
        vector_store.similarity_search(
            query=query.strip(),
            k=raw_candidate_k,
        )
    )

    if not candidates:

        return []

    # --------------------------------------------------------
    # Product filtering
    # --------------------------------------------------------

    if product_section:

        product_results = [
            document
            for document in candidates
            if _is_product_match(
                document,
                product_section,
            )
        ]

    else:

        product_results = candidates

    # --------------------------------------------------------
    # Remove section headers
    # --------------------------------------------------------

    content_results = [
        document
        for document in product_results
        if not _is_section_header(
            document
        )
    ]

    # --------------------------------------------------------
    # Deduplicate actual content
    # --------------------------------------------------------

    unique_results = []

    seen = set()

    for document in content_results:

        key = _document_key(
            document
        )

        if not key:
            continue

        if key in seen:
            continue

        seen.add(key)

        unique_results.append(
            document
        )

        if len(unique_results) >= k:
            break

    # --------------------------------------------------------
    # If product filtering produced nothing useful,
    # try semantic candidates without product filtering.
    #
    # This prevents an overly strict product filter from
    # hiding the actual answer.
    # --------------------------------------------------------

    if not unique_results and product_section:

        fallback_candidates = [
            document
            for document in candidates
            if not _is_section_header(
                document
            )
        ]

        for document in fallback_candidates:

            key = _document_key(
                document
            )

            if not key:
                continue

            if key in seen:
                continue

            seen.add(key)

            unique_results.append(
                document
            )

            if len(unique_results) >= k:
                break

    return unique_results[:k]