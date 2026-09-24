from fastapi import APIRouter, HTTPException

from app.api.schemas.rag_schema import (
    QueryRequest,
    QueryResponse
)

from app.services.rag_service import (
    answer_question
)


# ---------------------------------------------------------
# Router
# ---------------------------------------------------------

router = APIRouter(
    prefix="/api",
    tags=["RAG"]
)


# ---------------------------------------------------------
# Query Endpoint
# ---------------------------------------------------------

@router.post(
    "/query",
    response_model=QueryResponse
)
def query_rag(
    request: QueryRequest
):
    """
    Ask a question to the Credit Risk Knowledge Assistant.
    """

    try:

        # -------------------------------------------------
        # Optional metadata filtering
        # -------------------------------------------------

        filter_metadata = None

        if request.source_file:

            filter_metadata = {
                "source_file":
                    request.source_file
            }


        # -------------------------------------------------
        # Call RAG Service
        # -------------------------------------------------

        result = answer_question(
            question=request.question,
            top_k=request.top_k,
            filter_metadata=filter_metadata
        )


        return result


    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )


    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "An internal error occurred "
                "while processing the question."
            )
        ) from error