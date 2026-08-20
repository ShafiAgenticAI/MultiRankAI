from typing import Any



_SESSIONS: dict[str, dict[str, Any]] = {}


def _default_session() -> dict[str, Any]:
    return {
        "product": None,
        "section": None,
        "account_id": None,
        "messages": [],
    }

def get_session(
    session_id: str,
) -> dict[str, Any]:
    """
    Return the session context.

    A new empty session is returned when the session_id
    does not yet exist.
    """

    if not session_id:
        return _default_session()

    session = _SESSIONS.get(
        session_id
    )

    if session is None:
        return _default_session()

    return session

def save_session(
    session_id: str,
    product: str | None,
    section: str | None,
    account_id: str | None,
    question: str,
    answer: str,
) -> None:
    """
    Save the latest conversation turn and context.
    """

    if not session_id:
        return

    session = _SESSIONS.setdefault(
        session_id,
        _default_session(),
    )


    if product:
        session["product"] = product


    if section:
        session["section"] = section

    if account_id:
        session["account_id"] = account_id

    session["messages"].append(
        {
            "question": question,
            "answer": answer,
        }
    )

  

    session["messages"] = (
        session["messages"][-5:]
    )



def clear_session(
    session_id: str,
) -> None:
    """
    Clear one session.
    """

    if session_id:
        _SESSIONS.pop(
            session_id,
            None,
        )


def build_contextual_question(
    question: str,
    session: dict[str, Any],
) -> str:
    """
    Build a context-aware question for the agent.

    This helps the agent understand follow-up questions such as:

        User: What is the processing fee for a home loan?
        User: What is the minimum amount?

    The second question receives the previous product context.
    """

    question = question.strip()

    if not question:
        return question

    product = session.get(
        "product"
    )

    section = session.get(
        "section"
    )

    messages = session.get(
        "messages",
        [],
    )


    if (
        not product
        and not section
        and not messages
    ):
        return question

    contextual_parts = []

    if product:

        contextual_parts.append(
            f"Current product context: {product}"
        )

    if section:

        contextual_parts.append(
            f"Current section context: {section}"
        )


    if messages:

        contextual_parts.append(
            "Recent conversation:"
        )

        for message in messages[-3:]:

            previous_question = (
                message.get(
                    "question",
                    "",
                )
            )

            previous_answer = (
                message.get(
                    "answer",
                    "",
                )
            )

            contextual_parts.append(
                f"User: {previous_question}"
            )

            contextual_parts.append(
                f"Assistant: {previous_answer}"
            )

    contextual_parts.append(
        f"Current user question: {question}"
    )

    return "\n".join(
        contextual_parts
    )