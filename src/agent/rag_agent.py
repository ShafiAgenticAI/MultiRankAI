import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    ToolMessage,
)

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from src.agent.state import RAGState
from src.agent.tools import hybrid_search_tool

from src.retrieval.vector_search import search_vector
from src.retrieval.fts_search import search_fts
from src.retrieval.hybrid_search import search_hybrid

from src.retrieval.sql_generator import generate_sql
from src.retrieval.sql_executor import execute_sql


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# LLM
# ============================================================

llm = ChatOpenAI(
    model=os.getenv(
        "OPENAI_CHAT_MODEL",
        "gpt-4.1-mini",
    ),
    api_key=os.getenv(
        "OPENAI_API_KEY",
    ),
    temperature=0,
)


# ============================================================
# RETRIEVAL CLASSIFIER
# ============================================================

class RetrievalDecision(
    BaseModel
):
    """
    Structured decision for normal RAG retrieval.

    Exactly one method is selected:
        VECTOR
        FTS
        HYBRID
    """

    retrieval_type: Literal[
        "VECTOR",
        "FTS",
        "HYBRID",
    ] = Field(
        description=(
            "Choose exactly one retrieval method: "
            "VECTOR, FTS, or HYBRID."
        )
    )


retrieval_classifier = (
    llm.with_structured_output(
        RetrievalDecision
    )
)


# ============================================================
# MAIN ROUTER PROMPT
# ============================================================

ROUTER_PROMPT = """
You are the main routing component of a Smart Banking Assistant.

Decide which path should answer the user's question.

There are exactly three possible paths:

1. RAG
2. SQL
3. HYBRID


------------------------------------------------------------
RAG
------------------------------------------------------------

Use RAG when the question asks about information contained
in the NorthStar Bank knowledge base.

Examples:

- Home loan features
- Interest rates
- Processing fees
- Eligibility
- Fixed deposit rules
- Credit card features
- Product terms and conditions
- General banking policies
- Fees and charges described in bank documents


------------------------------------------------------------
SQL
------------------------------------------------------------

Use SQL when the question asks for actual structured
customer/account data stored in PostgreSQL.

Examples:

- Show my loan outstanding.
- What is my EMI?
- Show my active fixed deposits.
- Show transactions above 50000.
- What is my credit card outstanding?
- Show my credit card transactions.


------------------------------------------------------------
HYBRID
------------------------------------------------------------

Use HYBRID when the question requires BOTH:

1. Knowledge-base information from bank documents
AND
2. Structured banking/customer information from PostgreSQL.

Examples:

- What are the foreclosure charges for my home loan?
- Tell me my current home loan outstanding and the applicable
  foreclosure charges.
- Based on my loan details, explain the applicable bank policy.
- What charges apply to my current home loan?


------------------------------------------------------------
IMPORTANT
------------------------------------------------------------

Return exactly one word:

RAG

or

SQL

or

HYBRID

Do not answer the user's question.
"""


# ============================================================
# FINAL ANSWER PROMPT
# ============================================================

FINAL_ANSWER_PROMPT = """
You are a Smart Banking Assistant.

Answer the user's question using ONLY the information supplied
by the application.

Rules:

1. Do not invent information.
2. Do not use general knowledge.
3. If the source information is sufficient, answer directly.
4. Keep the answer concise and clear.
5. Do not mention internal implementation details.
6. Do not mention LangGraph.
7. Do not mention vector search, FTS, hybrid search, SQL
   generation, SQL retries, or internal tools unless explicitly
   asked.

For RAG results:

- Answer only from the retrieved knowledge-base content.

For SQL results:

- Explain the returned database rows in natural language.
- Do not invent values.
- Preserve important numeric values.
- Only mention fields that are actually present in the result.
- If no rows are returned, clearly say that no matching
  records were found.

For HYBRID results:

- Use BOTH the retrieved knowledge-base content and the
  structured database result when available.
- Clearly distinguish customer/account-specific facts from
  general banking-policy information.
- Do not claim that a policy applies to the customer unless
  the supplied information supports that conclusion.

If the supplied information does not contain enough information,
say:

"The available banking data does not provide enough information
to answer this question."
"""


# ============================================================
# SQL REPAIR PROMPT
# ============================================================

SQL_REPAIR_PROMPT = """
You are a SQL repair component for a Smart Banking Assistant.

The application generated a PostgreSQL SELECT query, but
execution failed.

Your job is to generate a corrected SELECT query for the
original user request.

Rules:

1. Return ONLY the SQL query.
2. The query MUST be read-only.
3. Only SELECT statements are allowed.
4. Do not use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE,
   TRUNCATE, GRANT, REVOKE, EXECUTE, CALL, or MERGE.
5. Do not use PostgreSQL system catalogs.
6. Preserve the user's original intent.
7. Use the actual database column names when possible.
8. If the query is customer-specific and the original query
   requires :account_id, preserve :account_id.
9. Do not invent an account_id.
10. Add LIMIT 100 unless the query already has a LIMIT.

The database error is provided only to help repair the query.
Do not mention the error in the generated SQL.
"""


# ============================================================
# RAG RETRY PROMPT
# ============================================================

RAG_RETRY_PROMPT = """
You are a retrieval query rewriting component for a banking
knowledge-base system.

The original retrieval query returned no relevant documents.

Generate ONE alternate search query that preserves the user's
original intent but uses different wording.

Rules:

1. Return only the alternate query.
2. Do not answer the user.
3. Do not add explanations.
4. Preserve important product names, fees, rates, charges,
   eligibility terms, or banking concepts.
5. Use natural search wording.
6. Do not invent banking facts.
"""


# ============================================================
# MAIN ROUTER NODE
# ============================================================

def route_node(
    state: RAGState,
):
    """
    Decide whether the question should use
    RAG, SQL, or HYBRID.
    """

    query = state["query"]

    response = llm.invoke(
        [
            SystemMessage(
                content=ROUTER_PROMPT
            ),
            HumanMessage(
                content=query
            ),
        ]
    )

    decision = str(
        response.content
    ).strip().upper()

    if decision.startswith(
        "HYBRID"
    ):

        route = "hybrid"

    elif decision.startswith(
        "SQL"
    ):

        route = "sql"

    else:

        route = "rag"

    print(
        f"[MAIN ROUTER] {route.upper()}"
    )

    return {
        "route": route,
    }


# ============================================================
# DOCUMENT FORMATTING
# ============================================================

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
# NORMAL RAG RETRIEVAL CLASSIFIER
# ============================================================

def retrieve_node(
    state: RAGState,
):
    """
    Classify the RAG query into exactly one retrieval type
    and execute only that retrieval method.

    VECTOR -> search_vector()
    FTS    -> search_fts()
    HYBRID -> search_hybrid()
    """

    query = state.get(
        "retrieval_query",
        state["query"],
    )

    retrieval_prompt = """
You are the retrieval classifier for a banking knowledge base.

Choose EXACTLY ONE retrieval method.

------------------------------------------------------------
VECTOR
------------------------------------------------------------

Use VECTOR when the question is primarily semantic,
conceptual, or meaning-based.Use VECTOR when the question asks for conceptual or descriptive
information where the exact wording is less important.

Examples:

- Who is eligible for a home loan?
- What are the benefits of a home loan?
- Who can apply for a fixed deposit?
- Explain the main features of a home loan.
- What requirements must a borrower satisfy?
- What are the advantages of this product?

------------------------------------------------------------
FTS
------------------------------------------------------------

Use FTS when the question primarily depends on exact
banking terms, names, amounts, rates, fees, charges,
or specific terminology.Use FTS when the question contains specific product terminology,
named sections, exact fees, rates, charges, variants, annual fees,
or other phrases likely to appear literally in the knowledge base.


Examples:

- What is the processing fee for a home loan?
- What is the minimum processing fee?
- What is the maximum processing fee?
- What is the foreclosure charge?
- What is the interest rate?
- What is the annual fee for the credit card?
- What are the different credit card products and their annual fees?
- What is the processing fee for a home loan?
- What is the maximum processing fee?
- What are the international transaction fees?
- What are the card variants and annual fees?

------------------------------------------------------------
HYBRID
------------------------------------------------------------

Use HYBRID ONLY when the question genuinely benefits from
both semantic understanding AND exact keyword matching.

Examples:

- What are the fees and eligibility conditions for a home loan?
- Explain the home loan charges and the conditions under
  which they apply.
- Compare home loan charges with the eligibility conditions.
- Explain the different charges applicable to different
  home loan conditions.

------------------------------------------------------------
IMPORTANT
------------------------------------------------------------

The presence of a product name such as "home loan",
"fixed deposit", or "credit card" does NOT automatically
mean HYBRID.

A simple eligibility question should normally be VECTOR.

A simple fee/rate/charge question should normally be FTS.

Use HYBRID only when BOTH semantic and exact-term retrieval
are genuinely useful.

Return exactly one classification:
VECTOR
FTS
or
HYBRID
"""

    decision = retrieval_classifier.invoke(
        [
            SystemMessage(
                content=retrieval_prompt
            ),
            HumanMessage(
                content=query
            ),
        ]
    )

    retrieval_type = (
        decision.retrieval_type.upper()
    )

    print(
        f"[RAG ROUTER] {retrieval_type}"
    )

    # --------------------------------------------------------
    # VECTOR
    # --------------------------------------------------------

    if retrieval_type == "VECTOR":

        documents = search_vector(
            query=query,
            k=5,
        )

    # --------------------------------------------------------
    # FTS
    # --------------------------------------------------------

    elif retrieval_type == "FTS":

        documents = search_fts(
            query=query,
            k=5,
        )

    # --------------------------------------------------------
    # HYBRID
    # --------------------------------------------------------

    else:

        documents = search_hybrid(
            query=query,
            k=5,
            candidate_k=10,
        )

    # --------------------------------------------------------
    # No documents
    # --------------------------------------------------------

    if not documents:

        return {
            "messages": [
                ToolMessage(
                    content=(
                        "No relevant documents were found."
                    ),
                    tool_call_id=(
                        f"retrieval_"
                        f"{retrieval_type.lower()}"
                    ),
                )
            ]
        }

    # --------------------------------------------------------
    # Format documents
    # --------------------------------------------------------

    formatted_context = _format_documents(
        documents
    )

    return {
        "messages": [
            ToolMessage(
                content=formatted_context,
                tool_call_id=(
                    f"retrieval_"
                    f"{retrieval_type.lower()}"
                ),
            )
        ]
    }


# ============================================================
# DIRECT TOP-LEVEL HYBRID RETRIEVAL
# ============================================================

def hybrid_retrieve_node(
    state: RAGState,
):
    """
    Directly execute the Hybrid retrieval tool.

    Top-level HYBRID means:

        Vector + FTS + RRF + Cohere Reranker
    """

    query = state["query"]

    result = hybrid_search_tool.invoke(
        {
            "query": query,
        }
    )

    return {
        "messages": [
            ToolMessage(
                content=str(result),
                tool_call_id="hybrid_direct",
            )
        ]
    }


# ============================================================
# CITATION EXTRACTION
# ============================================================

def _extract_citations(
    tool_messages: list[ToolMessage],
) -> list[dict]:

    citations = []

    seen = set()

    for message in tool_messages:

        content = message.content

        if not content:
            continue

        lines = content.splitlines()

        document_name = None
        section = None
        page = None

        for index, line in enumerate(
            lines
        ):

            line = line.strip()

            if (
                line == "Document:"
                and index + 1 < len(lines)
            ):

                document_name = (
                    lines[index + 1].strip()
                )

            elif (
                line == "Section:"
                and index + 1 < len(lines)
            ):

                section = (
                    lines[index + 1].strip()
                )

            elif (
                line == "Page:"
                and index + 1 < len(lines)
            ):

                page = (
                    lines[index + 1].strip()
                )

        if (
            not document_name
            and not section
            and not page
        ):

            continue

        citation_key = (
            document_name,
            section,
            page,
        )

        if citation_key in seen:

            continue

        seen.add(
            citation_key
        )

        citations.append(
            {
                "document": document_name,
                "section": section,
                "page": (
                    int(page)
                    if page and page.isdigit()
                    else page
                ),
            }
        )

    return citations


# ============================================================
# TOOL MESSAGE EXTRACTION
# ============================================================

def _get_tool_messages(
    state: RAGState,
) -> list[ToolMessage]:

    messages = state.get(
        "messages",
        [],
    )

    return [
        message
        for message in messages
        if isinstance(
            message,
            ToolMessage,
        )
    ]


# ============================================================
# RAG RETRIEVAL FAILURE CHECK
# ============================================================

def _rag_retrieval_failed(
    state: RAGState,
) -> bool:
    """
    Return True when all current RAG tool results
    indicate that no relevant documents were found.
    """

    tool_messages = _get_tool_messages(
        state
    )

    if not tool_messages:

        return True

    usable_messages = [
        message
        for message in tool_messages
        if message.content
    ]

    if not usable_messages:

        return True

    for message in usable_messages:

        content = str(
            message.content
        ).strip().lower()

        if (
            content
            and "no relevant documents were found"
            not in content
        ):

            return False

    return True


# ============================================================
# RAG RETRY QUERY NODE
# ============================================================

def rag_retry_query_node(
    state: RAGState,
):
    """
    Generate an alternate retrieval query.
    """

    original_query = state["query"]

    current_query = state.get(
        "retrieval_query",
        original_query,
    )

    rag_retry_count = state.get(
        "rag_retry_count",
        0,
    )

    response = llm.invoke(
        [
            SystemMessage(
                content=RAG_RETRY_PROMPT
            ),
            HumanMessage(
                content=(
                    f"ORIGINAL USER QUESTION:\n"
                    f"{original_query}\n\n"
                    f"PREVIOUS RETRIEVAL QUERY:\n"
                    f"{current_query}\n\n"
                    f"Generate one alternate search query."
                )
            ),
        ]
    )

    alternate_query = str(
        response.content
    ).strip()

    if (
        not alternate_query
        or alternate_query.lower()
        == current_query.lower()
    ):

        alternate_query = (
            original_query
        )

    next_retry_count = (
        rag_retry_count + 1
    )

    print(
        f"[RAG RETRY] attempt {next_retry_count}"
    )

    return {
        "retrieval_query": alternate_query,
        "rag_retry_count": next_retry_count,
        "messages": [],
    }


# ============================================================
# RAG RETRY DECISION
# ============================================================

def route_after_rag(
    state: RAGState,
):
    """
    Retry RAG retrieval only when no relevant documents
    were returned.
    """

    if not _rag_retrieval_failed(
        state
    ):

        return "rag_answer"

    retry_count = state.get(
        "rag_retry_count",
        0,
    )

    max_retries = state.get(
        "rag_max_retries",
        2,
    )

    if retry_count < max_retries:

        return "rag_retry"

    return "rag_answer"


# ============================================================
# RAG ANSWER NODE
# ============================================================

def rag_answer_node(
    state: RAGState,
):
    """
    Generate a final answer using only RAG results.
    """

    query = state["query"]

    tool_messages = _get_tool_messages(
        state
    )

    if _rag_retrieval_failed(
        state
    ):

        return {
            "answer": (
                "The knowledge base does not provide "
                "enough information to answer this question."
            ),
            "citations": [],
        }

    retrieved_context = "\n\n".join(
        message.content
        for message in tool_messages
        if message.content
        and "no relevant documents were found"
        not in str(
            message.content
        ).lower()
    )

    if not retrieved_context:

        return {
            "answer": (
                "The knowledge base does not provide "
                "enough information to answer this question."
            ),
            "citations": [],
        }

    response = llm.invoke(
        [
            SystemMessage(
                content=FINAL_ANSWER_PROMPT
            ),
            HumanMessage(
                content=(
                    f"USER QUESTION:\n"
                    f"{query}\n\n"
                    f"RETRIEVED KNOWLEDGE BASE:\n"
                    f"{retrieved_context}"
                )
            ),
        ]
    )

    content = response.content

    if isinstance(
        content,
        list,
    ):

        text_parts = []

        for item in content:

            if isinstance(
                item,
                dict,
            ):

                text_parts.append(
                    item.get(
                        "text",
                        "",
                    )
                )

        content = " ".join(
            text_parts
        )

    return {
        "answer": str(
            content
        ).strip(),
        "citations": _extract_citations(
            tool_messages
        ),
    }


# ============================================================
# SQL NODE
# ============================================================

async def sql_node(
    state: RAGState,
):
    """
    Generate and execute a safe SQL query.
    """
    print("=========== SQL called =============")

    query = state["query"]

    account_id = state.get(
        "account_id"
    )

    sql_query = None

    try:

        sql_query = generate_sql(
            query
        )

        if sql_query == "UNSUPPORTED":

            return {
                "sql_query": None,
                "sql_result": [],
                "sql_status": "unsupported",
                "sql_error": None,
            }

        sql_result = await execute_sql(
            sql_query,
            account_id=account_id,
        )

        return {
            "sql_query": sql_query,
            "sql_result": sql_result,
            "sql_status": "success",
            "sql_error": None,
        }

    except Exception as exc:

        return {
            "sql_query": sql_query,
            "sql_result": [],
            "sql_status": "error",
            "sql_error": str(
                exc
            ),
        }


# ============================================================
# SQL RETRY / REPAIR NODE
# ============================================================

async def sql_retry_node(
    state: RAGState,
):
    """
    Repair a failed SQL query and execute it again.
    """

    query = state["query"]

    account_id = state.get(
        "account_id"
    )

    retry_count = state.get(
        "retry_count",
        0,
    )

    max_retries = state.get(
        "max_retries",
        1,
    )

    current_sql = state.get(
        "sql_query"
    )

    sql_error = state.get(
        "sql_error"
    )

    if retry_count >= max_retries:

        return {
            "sql_status": "error",
            "answer": (
                "The banking database could not be "
                "queried for this request."
            ),
        }

    try:

        repair_response = llm.invoke(
            [
                SystemMessage(
                    content=SQL_REPAIR_PROMPT
                ),
                HumanMessage(
                    content=(
                        f"USER QUESTION:\n"
                        f"{query}\n\n"
                        f"FAILED SQL QUERY:\n"
                        f"{current_sql}\n\n"
                        f"DATABASE ERROR:\n"
                        f"{sql_error}\n\n"
                        f"Generate the corrected SQL query."
                    )
                ),
            ]
        )

        repaired_sql = str(
            repair_response.content
        ).strip()

        if repaired_sql.startswith(
            "```"
        ):

            repaired_sql = (
                repaired_sql
                .replace(
                    "```sql",
                    "",
                )
                .replace(
                    "```SQL",
                    "",
                )
                .replace(
                    "```",
                    "",
                )
                .strip()
            )

        if repaired_sql == "UNSUPPORTED":

            return {
                "retry_count": (
                    retry_count + 1
                ),
                "sql_status": "unsupported",
                "sql_query": None,
                "sql_result": [],
                "sql_error": None,
                "answer": (
                    "This question cannot be answered "
                    "using the available banking database."
                ),
            }

        repaired_result = await execute_sql(
            repaired_sql,
            account_id=account_id,
        )

        return {
            "retry_count": (
                retry_count + 1
            ),
            "sql_query": repaired_sql,
            "sql_result": repaired_result,
            "sql_status": "success",
            "sql_error": None,
        }

    except Exception as exc:

        return {
            "retry_count": (
                retry_count + 1
            ),
            "sql_status": "error",
            "sql_error": str(
                exc
            ),
            "sql_result": [],
            "answer": (
                "The banking database could not be "
                "queried for this request."
            ),
        }


# ============================================================
# SQL RETRY DECISION
# ============================================================

def route_after_sql(
    state: RAGState,
):
    """
    Decide whether SQL execution succeeded or needs retry.
    """

    sql_status = state.get(
        "sql_status"
    )

    retry_count = state.get(
        "retry_count",
        0,
    )

    max_retries = state.get(
        "max_retries",
        1,
    )

    if sql_status == "success":

        return "sql_answer"

    if sql_status == "unsupported":

        return "sql_answer"

    if (
        sql_status == "error"
        and retry_count < max_retries
    ):

        return "sql_retry"

    return "sql_answer"


# ============================================================
# SQL RETRY DECISION
# ============================================================

def route_after_sql_retry(
    state: RAGState,
):

    return "sql_answer"


# ============================================================
# SQL ANSWER NODE
# ============================================================

def sql_answer_node(
    state: RAGState,
):
    """
    Convert SQL results into a natural-language answer.
    """

    query = state["query"]

    sql_query = state.get(
        "sql_query"
    )

    sql_result = state.get(
        "sql_result",
        [],
    )

    existing_answer = state.get(
        "answer",
        "",
    )

    sql_status = state.get(
        "sql_status"
    )

    if (
        existing_answer
        and sql_status in {
            "error",
            "unsupported",
        }
    ):

        return {
            "answer": existing_answer,
        }

    if sql_status == "error":

        return {
            "answer": (
                "The banking database could not be "
                "queried for this request."
            )
        }

    if sql_status == "unsupported":

        return {
            "answer": (
                "This question cannot be answered "
                "using the available banking database."
            )
        }

    if not sql_query:

        return {
            "answer": (
                "The available banking data does not "
                "provide enough information to answer "
                "this question."
            )
        }

    if not sql_result:

        return {
            "answer": (
                "No matching banking records were found."
            )
        }

    response = llm.invoke(
        [
            SystemMessage(
                content=FINAL_ANSWER_PROMPT
            ),
            HumanMessage(
                content=(
                    f"USER QUESTION:\n"
                    f"{query}\n\n"
                    f"SQL QUERY:\n"
                    f"{sql_query}\n\n"
                    f"DATABASE RESULT:\n"
                    f"{sql_result}"
                )
            ),
        ]
    )

    content = response.content

    if isinstance(
        content,
        list,
    ):

        text_parts = []

        for item in content:

            if isinstance(
                item,
                dict,
            ):

                text_parts.append(
                    item.get(
                        "text",
                        "",
                    )
                )

        content = " ".join(
            text_parts
        )

    return {
        "answer": str(
            content
        ).strip(),
    }


# ============================================================
# HYBRID ANSWER NODE
# ============================================================

def hybrid_answer_node(
    state: RAGState,
):
    """
    Generate a grounded answer using both:

    1. Knowledge-base retrieval
    2. Structured SQL data
    """

    query = state["query"]

    tool_messages = _get_tool_messages(
        state
    )

    sql_query = state.get(
        "sql_query"
    )

    sql_result = state.get(
        "sql_result",
        [],
    )

    sql_status = state.get(
        "sql_status"
    )

    retrieved_context = "\n\n".join(
        message.content
        for message in tool_messages
        if message.content
    )

    if not retrieved_context:

        retrieved_context = (
            "No knowledge-base information was retrieved."
        )

    if sql_query:

        sql_context = (
            f"SQL QUERY:\n"
            f"{sql_query}\n\n"
            f"DATABASE RESULT:\n"
            f"{sql_result}"
        )

    else:

        sql_context = (
            f"SQL STATUS: {sql_status}\n"
            f"DATABASE RESULT:\n"
            f"{sql_result}"
        )

    response = llm.invoke(
        [
            SystemMessage(
                content=FINAL_ANSWER_PROMPT
            ),
            HumanMessage(
                content=(
                    f"USER QUESTION:\n"
                    f"{query}\n\n"
                    f"KNOWLEDGE BASE INFORMATION:\n"
                    f"{retrieved_context}\n\n"
                    f"STRUCTURED BANKING DATA:\n"
                    f"{sql_context}"
                )
            ),
        ]
    )

    content = response.content

    if isinstance(
        content,
        list,
    ):

        text_parts = []

        for item in content:

            if isinstance(
                item,
                dict,
            ):

                text_parts.append(
                    item.get(
                        "text",
                        "",
                    )
                )

        content = " ".join(
            text_parts
        )

    return {
        "answer": str(
            content
        ).strip(),
        "citations": _extract_citations(
            tool_messages
        ),
    }


# ============================================================
# MAIN ROUTING FUNCTION
# ============================================================

def route_after_router(
    state: RAGState,
):

    route = state.get(
        "route",
        "rag",
    )

    if route == "sql":

        return "sql"

    if route == "hybrid":

        return "hybrid_retrieve"

    return "rag"


# ============================================================
# BUILD GRAPH
# ============================================================

builder = StateGraph(
    RAGState
)


# ============================================================
# NODES
# ============================================================

builder.add_node(
    "router",
    route_node,
)

builder.add_node(
    "retrieve",
    retrieve_node,
)

builder.add_node(
    "rag_retry",
    rag_retry_query_node,
)

builder.add_node(
    "rag_answer",
    rag_answer_node,
)

builder.add_node(
    "sql",
    sql_node,
)

builder.add_node(
    "sql_retry",
    sql_retry_node,
)

builder.add_node(
    "sql_answer",
    sql_answer_node,
)

builder.add_node(
    "hybrid_retrieve",
    hybrid_retrieve_node,
)

builder.add_node(
    "hybrid_sql",
    sql_node,
)

builder.add_node(
    "hybrid_answer",
    hybrid_answer_node,
)


# ============================================================
# START → ROUTER
# ============================================================

builder.add_edge(
    START,
    "router",
)


# ============================================================
# ROUTER → RAG / SQL / HYBRID
# ============================================================

builder.add_conditional_edges(
    "router",
    route_after_router,
    {
        "rag": "retrieve",
        "sql": "sql",
        "hybrid_retrieve": "hybrid_retrieve",
    },
)


# ============================================================
# NORMAL RAG FLOW
# ============================================================

builder.add_conditional_edges(
    "retrieve",
    route_after_rag,
    {
        "rag_retry": "rag_retry",
        "rag_answer": "rag_answer",
    },
)

builder.add_edge(
    "rag_retry",
    "retrieve",
)

builder.add_edge(
    "rag_answer",
    END,
)


# ============================================================
# NORMAL SQL FLOW
# ============================================================

builder.add_conditional_edges(
    "sql",
    route_after_sql,
    {
        "sql_retry": "sql_retry",
        "sql_answer": "sql_answer",
    },
)

builder.add_conditional_edges(
    "sql_retry",
    route_after_sql_retry,
    {
        "sql_answer": "sql_answer",
    },
)

builder.add_edge(
    "sql_answer",
    END,
)


# ============================================================
# HYBRID FLOW
# ============================================================

builder.add_edge(
    "hybrid_retrieve",
    "hybrid_sql",
)

builder.add_edge(
    "hybrid_sql",
    "hybrid_answer",
)

builder.add_edge(
    "hybrid_answer",
    END,
)


# ============================================================
# COMPILE
# ============================================================

rag_graph = builder.compile()


# ============================================================
# PUBLIC FUNCTION
# ============================================================

async def ask_question(
    question: str,
    session_id: str = "default-session",
    account_id: str | None = None,
) -> dict:
    """
    Ask a question using:

    - RAG
    - SQL
    - HYBRID
    """

    if not question or not question.strip():

        return {
            "answer": "Please provide a question.",
            "citations": [],
            "query_path": "langgraph_rag",
            "sql_query": None,
            "sql_result": [],
            "retry_count": 0,
        }

    query = question.strip()

    result = await rag_graph.ainvoke(
        {
            "messages": [],
            "query": query,
            "retrieval_query": query,
            "session_id": session_id,
            "account_id": account_id,
            "documents": [],
            "answer": "",
            "citations": [],
            "route": "",
            "sql_query": None,
            "sql_result": [],
            "sql_status": None,
            "sql_error": None,
            "retry_count": 0,
            "max_retries": 1,
            "rag_retry_count": 0,
            "rag_max_retries": 2,
        }
    )

    answer = result.get(
        "answer",
        "",
    )

    citations = result.get(
        "citations",
        [],
    )

    route = result.get(
        "route",
        "rag",
    )

    sql_query = result.get(
        "sql_query"
    )

    sql_result = result.get(
        "sql_result",
        [],
    )

    retry_count = result.get(
        "retry_count",
        0,
    )

    if not answer:

        answer = (
            "The available banking data does not "
            "provide enough information to answer "
            "this question."
        )

    if route == "sql":

        query_path = "sql"

    elif route == "hybrid":

        query_path = "hybrid"

    else:

        query_path = "langgraph_rag"

    return {
        "answer": str(
            answer
        ).strip(),
        "citations": citations,
        "query_path": query_path,
        "sql_query": sql_query,
        "sql_result": sql_result,
        "retry_count": retry_count,
    }