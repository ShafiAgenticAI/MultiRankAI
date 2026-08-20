from fastapi import APIRouter
from pydantic import BaseModel, Field

from src.agent.rag_agent import ask_question
from src.agent.session_memory import (
    build_contextual_question,
    get_session,
    save_session,
)

from src.retrieval.query_router import (
    detect_product,
    get_product_section,
)


router = APIRouter(
    tags=["Query"]
)


# ============================================================
# REQUEST MODEL
# ============================================================

class QueryRequest(BaseModel):

    query: str = Field(
        min_length=1
    )

    session_id: str = Field(
        default="default-session",
        min_length=1,
    )

    account_id: str | None = Field(
        default=None,
        min_length=1,
    )


# ============================================================
# QUERY ENDPOINT
# ============================================================

@router.post("/query")
async def query(
    request: QueryRequest,
):

    question = request.query.strip()

    # --------------------------------------------------------
    # Load existing session
    # --------------------------------------------------------

    session = get_session(
        request.session_id
    )

    previous_product = session.get(
        "product"
    )

    previous_section = session.get(
        "section"
    )

    previous_account_id = session.get(
        "account_id"
    )

    # --------------------------------------------------------
    # Account context
    #
    # Explicit request account_id takes priority.
    # Otherwise reuse the session account_id.
    # --------------------------------------------------------

    account_id = (
        request.account_id
        or previous_account_id
    )

    # --------------------------------------------------------
    # Detect product from current question
    # --------------------------------------------------------

    product = detect_product(
        question
    )

    section = get_product_section(
        question
    )

    # --------------------------------------------------------
    # Reuse session product context for follow-up questions
    # --------------------------------------------------------

    if not product and previous_product:

        product = previous_product

    if not section and previous_section:

        section = previous_section

    # --------------------------------------------------------
    # Build context-aware question
    # --------------------------------------------------------

    contextual_question = (
        build_contextual_question(
            question,
            session,
        )
    )

    # --------------------------------------------------------
    # LangGraph RAG / SQL
    # --------------------------------------------------------

    rag_result = await ask_question(
        question=contextual_question,
        session_id=request.session_id,
        account_id=account_id,
    )

    # --------------------------------------------------------
    # Extract result
    # --------------------------------------------------------

    answer = rag_result.get(
        "answer",
        "",
    )

    citations = rag_result.get(
        "citations",
        [],
    )

    query_path = rag_result.get(
        "query_path",
        "langgraph_rag",
    )

    sql_query = rag_result.get(
        "sql_query",
        None,
    )

    sql_result = rag_result.get(
        "sql_result",
        [],
    )

    retry_count = rag_result.get(
        "retry_count",
        0,
    )

    # --------------------------------------------------------
    # Save session after successful processing
    # --------------------------------------------------------

    save_session(
        session_id=request.session_id,
        product=product,
        section=section,
        account_id=account_id,
        question=question,
        answer=answer,
    )

    # --------------------------------------------------------
    # API response
    # --------------------------------------------------------

    return {

        "query": question,

        "answer": answer,

        "query_path": query_path,

        "product": product,

        "section": section,

        "citations": citations,

        "sql_query": sql_query,

        "sql_result": sql_result,

        "retry_count": retry_count,

        "confidence_score": None,

        "session_id": request.session_id,

        "langsmith_trace_id": None,

        "status": "success",
    }