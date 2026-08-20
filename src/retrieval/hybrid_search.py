from langchain_core.documents import Document

from src.retrieval.vector_search import search_vector
from src.retrieval.fts_search import search_fts
from src.retrieval.reranker import rerank_documents


def _document_key(
    document: Document,
) -> str:
    """
    Create a stable key so the same document returned by
    vector search and FTS is treated as one candidate.
    """

    return document.page_content.strip()


def _is_section_header(
    document: Document,
) -> bool:
    """
    Identify chunks that contain only a section header.
    """

    return (
        document.metadata.get(
            "element_type"
        )
        == "section_header"
    )


def search_hybrid(
    query: str,
    k: int = 5,
    candidate_k: int = 10,
) -> list[Document]:
    """
    Hybrid retrieval using:

    1. Vector similarity search
    2. PostgreSQL Full-Text Search
    3. Reciprocal Rank Fusion (RRF)
    4. Cohere reranking

    Flow:

        Vector + FTS
              ↓
             RRF
              ↓
        Cohere Reranker
              ↓
             Top-K
    """

    print("[HYBRID SEARCH] called")

    if not query or not query.strip():
        return []


    vector_results = search_vector(
        query=query,
        k=candidate_k,
        candidate_k=max(
            candidate_k * 2,
            20,
        ),
    )

    fts_results = search_fts(
        query=query,
        k=candidate_k,
    )

    candidates: dict[str, dict] = {}


    for rank, document in enumerate(
        vector_results,
        start=1,
    ):

        key = _document_key(
            document
        )

        if key not in candidates:

            candidates[key] = {
                "document": document,
                "rrf_score": 0.0,
                "vector_rank": None,
                "fts_rank": None,
            }

        candidates[key][
            "vector_rank"
        ] = rank

        candidates[key][
            "rrf_score"
        ] += (
            1.0 / (60 + rank)
        )

    for rank, document in enumerate(
        fts_results,
        start=1,
    ):

        key = _document_key(
            document
        )

        if key not in candidates:

            candidates[key] = {
                "document": document,
                "rrf_score": 0.0,
                "vector_rank": None,
                "fts_rank": None,
            }

        candidates[key][
            "fts_rank"
        ] = rank

        candidates[key][
            "rrf_score"
        ] += (
            1.0 / (60 + rank)
        )


    for item in candidates.values():

        document = item["document"]

        if _is_section_header(
            document
        ):

            item[
                "rrf_score"
            ] *= 0.50


    ranked_candidates = sorted(
        candidates.values(),
        key=lambda item: (
            item["rrf_score"]
        ),
        reverse=True,
    )


    rerank_candidates: list[Document] = []

    for item in ranked_candidates[
        :candidate_k
    ]:

        document = item["document"]

        metadata = dict(
            document.metadata
        )

        metadata[
            "retrieval_type"
        ] = "hybrid"

        metadata[
            "rrf_score"
        ] = item["rrf_score"]

        metadata[
            "vector_rank"
        ] = item["vector_rank"]

        metadata[
            "fts_rank"
        ] = item["fts_rank"]

        rerank_candidates.append(
            Document(
                page_content=(
                    document.page_content
                ),
                metadata=metadata,
            )
        )

    if not rerank_candidates:
        return []


    reranked_documents = rerank_documents(
        query=query,
        documents=rerank_candidates,
        top_n=k,
    )


    final_results: list[Document] = []

    for document in reranked_documents:

        metadata = dict(
            document.metadata
        )

        metadata[
            "retrieval_type"
        ] = "hybrid_reranked"

        final_results.append(
            Document(
                page_content=(
                    document.page_content
                ),
                metadata=metadata,
            )
        )

    return final_results[:k]