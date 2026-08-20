import os

from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from dotenv import load_dotenv

load_dotenv()


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://postgres:Pass@123@localhost:5433/smart_banking",
)

COLLECTION_NAME = "smart_banking_documents"


def get_embeddings() -> OpenAIEmbeddings:
    """
    Create the OpenAI embedding model.
    """

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not configured."
        )

    return OpenAIEmbeddings(
        model=os.getenv(
            "OPENAI_EMBEDDING_MODEL",
            "text-embedding-3-small",
        ),
        api_key=api_key,
    )


def get_vector_store() -> PGVector:
    """
    Create the PostgreSQL PGVector store.
    """

    return PGVector(
        embeddings=get_embeddings(),
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
    )


def add_documents(documents):
    """
    Add LangChain Documents to PostgreSQL/PGVector.
    """

    vector_store = get_vector_store()

    vector_store.add_documents(
        documents
    )

    return vector_store


def similarity_search(
    query: str,
    k: int = 5,
):
    """
    Perform vector similarity search.
    """

    vector_store = get_vector_store()

    return vector_store.similarity_search(
        query,
        k=k,
    )