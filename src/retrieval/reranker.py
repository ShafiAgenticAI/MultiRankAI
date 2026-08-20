import os

import cohere
from langchain_core.documents import Document



COHERE_API_KEY = os.getenv(
    "COHERE_API_KEY"
)

COHERE_RERANK_MODEL = os.getenv(
    "COHERE_RERANK_MODEL",
    "rerank-v4.0-fast",
)

_client = (
    cohere.ClientV2(
        api_key=COHERE_API_KEY
    )
    if COHERE_API_KEY
    else None
)


def rerank_documents(
    query: str,
    documents: list[Document],
    top_n: int = 5,
) -> list[Document]:
    """
    Rerank retrieved documents using Cohere.

    The original retrieval order is used as a safe fallback
    if Cohere is unavailable or reranking fails.
    """

    if not documents:
        return []

    if not query or not query.strip():
        return documents[:top_n]

    if _client is None:

        print(
            "[RERANKER] skipped - COHERE_API_KEY missing"
        )

        return documents[:top_n]

    print("[RERANKER] called")

    try:

        texts = [
            document.page_content
            for document in documents
        ]

        response = _client.rerank(
            model=COHERE_RERANK_MODEL,
            query=query.strip(),
            documents=texts,
            top_n=min(
                top_n,
                len(texts),
            ),
        )

        reranked_documents = []

        for result in response.results:

            original_document = documents[
                result.index
            ]

            metadata = dict(
                original_document.metadata
            )

            metadata[
                "rerank_score"
            ] = float(
                result.relevance_score
            )

            metadata[
                "reranked"
            ] = True

            reranked_documents.append(
                Document(
                    page_content=(
                        original_document.page_content
                    ),
                    metadata=metadata,
                )
            )

        return reranked_documents

    except Exception as exc:

        print(
            f"[RERANKER ERROR] {exc}"
        )

        return documents[:top_n]