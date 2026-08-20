import os
import re

from dotenv import load_dotenv
from langchain_core.documents import Document
from sqlalchemy import create_engine, text


load_dotenv()


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://postgres:Pass%40123@localhost:5433/smart_banking",
)


STOP_WORDS = {
    "what",
    "is",
    "are",
    "the",
    "a",
    "an",
    "for",
    "of",
    "to",
    "in",
    "on",
    "with",
    "and",
    "or",
    "how",
    "much",
    "many",
    "can",
    "could",
    "would",
    "should",
    "please",
    "tell",
    "me",
    "do",
    "does",
    "about",
    "give",
    "explain",
    "which",
    "who",
    "when",
    "where",
    "why",
}


def _extract_search_terms(
    query: str,
) -> list[str]:
    """
    Extract meaningful search terms from the query.
    """

    words = re.findall(
        r"[A-Za-z0-9]+",
        query.lower(),
    )

    meaningful_words = [
        word
        for word in words
        if word not in STOP_WORDS
        and len(word) > 1
    ]

    return meaningful_words


def _build_or_tsquery(
    terms: list[str],
) -> str:

    return " | ".join(
        f"{term}:*"
        for term in terms
    )


def search_fts(
    query: str,
    k: int = 5,
) -> list[Document]:
    """
    PostgreSQL Full-Text Search.
    """

    print("[FTS SEARCH] called")

    if not query or not query.strip():
        return []

    engine = create_engine(
        DATABASE_URL
    )

    documents_by_id = {}

    search_terms = _extract_search_terms(
        query
    )

    if not search_terms:
        engine.dispose()
        return []

    tsquery = _build_or_tsquery(
        search_terms
    )

    search_sql = text(
        """
        SELECT
            id,
            document,
            cmetadata,

            ts_rank(
                to_tsvector(
                    'english',
                    document
                ),
                to_tsquery(
                    'english',
                    :tsquery
                )
            ) AS fts_rank,

            (
                SELECT COUNT(*)
                FROM unnest(
                    string_to_array(
                        lower(:search_terms),
                        ' '
                    )
                ) AS term
                WHERE lower(document)
                    LIKE '%' || term || '%'
            ) AS matched_terms

        FROM langchain_pg_embedding

        WHERE
            to_tsvector(
                'english',
                document
            ) @@ to_tsquery(
                'english',
                :tsquery
            )

        ORDER BY
            matched_terms DESC,
            fts_rank DESC

        LIMIT :limit
        """
    )

    try:

        with engine.connect() as connection:

            rows = connection.execute(
                search_sql,
                {
                    "tsquery": tsquery,
                    "search_terms": " ".join(
                        search_terms
                    ),
                    "limit": max(
                        k * 5,
                        25,
                    ),
                },
            )

            for row in rows:

                metadata = (
                    dict(row.cmetadata)
                    if row.cmetadata
                    else {}
                )

                metadata["fts_score"] = float(
                    row.fts_rank
                )

                metadata["fts_matched_terms"] = int(
                    row.matched_terms
                )

                metadata[
                    "retrieval_type"
                ] = "fts"

                document = Document(
                    page_content=row.document,
                    metadata=metadata,
                )

                documents_by_id[
                    str(row.id)
                ] = document

    except Exception as exc:

        print(
            f"[FTS ERROR] {exc}"
        )

        engine.dispose()

        return []

    engine.dispose()

    results = list(
        documents_by_id.values()
    )

    results.sort(
        key=lambda document: (
            document.metadata.get(
                "fts_matched_terms",
                0,
            ),
            document.metadata.get(
                "fts_score",
                0.0,
            ),
        ),
        reverse=True,
    )

    def is_section_heading(
        document: Document,
    ) -> bool:

        content = (
            document.page_content
            .strip()
        )

        chunk_type = document.metadata.get(
            "chunk_type"
        )

        element_type = document.metadata.get(
            "element_type"
        )

        if chunk_type == "text" and (
            element_type == "section_header"
            or len(content) < 60
        ):
            return True

        return False

    useful_results = [
        document
        for document in results
        if not is_section_heading(
            document
        )
    ]

    heading_results = [
        document
        for document in results
        if is_section_heading(
            document
        )
    ]

    final_results = (
        useful_results
        + heading_results
    )

    return final_results[:k]