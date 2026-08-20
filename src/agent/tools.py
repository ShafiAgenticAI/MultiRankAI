from langchain_core.tools import tool

from src.retrieval.vector_search import search_vector
from src.retrieval.fts_search import search_fts
from src.retrieval.hybrid_search import search_hybrid


def _build_search_query(
    query: str,
    product: str | None = None,
    section: str | None = None,
) -> str:
    """
    Build a retrieval query that preserves product context.
    """

    parts = [
        query.strip()
    ]

    if product:
        parts.append(
            f"Product: {product.replace('_', ' ')}"
        )

    if section:
        parts.append(
            f"Section: {section}"
        )

    return "\n".join(parts)


def _format_documents(
    documents,
) -> str:
    """
    Convert retrieved Documents into text
    that the final LLM can understand.
    """

    formatted = []

    for index, document in enumerate(
        documents,
        start=1,
    ):

        metadata = document.metadata

        formatted.append(
            f"""
--- Document {index} ---

Content:
{document.page_content}

Section:
{metadata.get("section")}

Page:
{metadata.get("source_page")}

Document:
{metadata.get("document_name")}

Content Type:
{metadata.get("chunk_type")}
""".strip()
        )

    return "\n\n".join(
        formatted
    )


# ============================================================
# VECTOR SEARCH
# ============================================================

@tool
def vector_search_tool(
    query: str,
    product: str | None = None,
    section: str | None = None,
) -> str:
    """
    Search the banking knowledge base using
    semantic vector similarity.
    """

    print()
    print("=" * 70)
    print("[RAG TOOL] VECTOR SEARCH TOOL CALLED")
    print("=" * 70)
    print(f"[RAG TOOL] Original query : {query}")
    print(f"[RAG TOOL] Product        : {product}")
    print(f"[RAG TOOL] Section        : {section}")

    search_query = _build_search_query(
        query,
        product,
        section,
    )

    print(
        f"[RAG TOOL] Search query   : {search_query}"
    )

    documents = search_vector(
        query=search_query,
        k=5,
    )

    print(
        f"[RAG TOOL] Documents returned: {len(documents)}"
    )

    print("=" * 70)
    print()

    if not documents:
        return "No relevant documents were found."

    return _format_documents(
        documents
    )


# ============================================================
# FTS SEARCH
# ============================================================

@tool
def fts_search_tool(
    query: str,
    product: str | None = None,
    section: str | None = None,
) -> str:
    """
    Search the banking knowledge base using
    PostgreSQL Full-Text Search.
    """

    print()
    print("=" * 70)
    print("[RAG TOOL] FTS SEARCH TOOL CALLED")
    print("=" * 70)
    print(f"[RAG TOOL] Original query : {query}")
    print(f"[RAG TOOL] Product        : {product}")
    print(f"[RAG TOOL] Section        : {section}")

    search_query = _build_search_query(
        query,
        product,
        section,
    )

    print(
        f"[RAG TOOL] Search query   : {search_query}"
    )

    documents = search_fts(
        query=search_query,
        k=5,
    )

    print(
        f"[RAG TOOL] Documents returned: {len(documents)}"
    )

    print("=" * 70)
    print()

    if not documents:
        return "No relevant documents were found."

    return _format_documents(
        documents
    )


# ============================================================
# HYBRID SEARCH
# ============================================================

@tool
def hybrid_search_tool(
    query: str,
    product: str | None = None,
    section: str | None = None,
) -> str:
    """
    Search the banking knowledge base using
    vector similarity and PostgreSQL FTS
    combined with RRF.
    """

    print()
    print("=" * 70)
    print("[RAG TOOL] HYBRID SEARCH TOOL CALLED")
    print("=" * 70)
    print(f"[RAG TOOL] Original query : {query}")
    print(f"[RAG TOOL] Product        : {product}")
    print(f"[RAG TOOL] Section        : {section}")

    search_query = _build_search_query(
        query,
        product,
        section,
    )

    print(
        f"[RAG TOOL] Search query   : {search_query}"
    )

    documents = search_hybrid(
        query=search_query,
        k=5,
        candidate_k=10,
    )

    print(
        f"[RAG TOOL] Documents returned: {len(documents)}"
    )

    print("=" * 70)
    print()

    if not documents:
        return "No relevant documents were found."

    return _format_documents(
        documents
    )


# ============================================================
# TOOLS
# ============================================================

TOOLS = [
    vector_search_tool,
    fts_search_tool,
    hybrid_search_tool,
]