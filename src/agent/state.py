from typing import Any, TypedDict

from langchain_core.messages import BaseMessage


class RAGState(TypedDict, total=False):

    query: str

    retrieval_query: str

    session_id: str
    account_id: str | None

    messages: list[BaseMessage]


    documents: list[Any]
    answer: str
    citations: list[dict]

    route: str

    sql_query: str | None
    sql_result: list[dict]
    sql_status: str | None
    sql_error: str | None

    retry_count: int
    max_retries: int

    rag_retry_count: int
    rag_max_retries: int