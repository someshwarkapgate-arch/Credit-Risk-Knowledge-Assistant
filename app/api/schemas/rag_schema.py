from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------
# Query Request
# ---------------------------------------------------------

class QueryRequest(BaseModel):
    """
    Request body for asking a question to the RAG system.
    """

    question: str = Field(
        ...,
        min_length=1,
        description="Question to ask the Credit Risk Knowledge Assistant."
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of relevant document chunks to retrieve."
    )

    source_file: Optional[str] = Field(
        default=None,
        description="Optional source document filter."
    )


# ---------------------------------------------------------
# Source Response
# ---------------------------------------------------------

class SourceResponse(BaseModel):
    """
    Metadata for one source used by the RAG system.
    """

    source_file: str

    page_number: int | str

    chunk_id: Optional[int] = None


# ---------------------------------------------------------
# Query Response
# ---------------------------------------------------------

class QueryResponse(BaseModel):
    """
    Structured response returned by the RAG API.
    """

    question: str

    answer: str

    sources: list[SourceResponse]

    retrieved_chunks: int