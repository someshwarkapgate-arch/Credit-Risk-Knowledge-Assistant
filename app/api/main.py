from fastapi import FastAPI

from app.api.routes.rag_routes import (
    router as rag_router
)


# ---------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------

app = FastAPI(
    title="Credit Risk Knowledge Assistant API",
    description=(
        "RAG-powered API for querying "
        "credit risk and regulatory documents."
    ),
    version="1.0.0"
)


# ---------------------------------------------------------
# Register Routers
# ---------------------------------------------------------

app.include_router(
    rag_router
)


# ---------------------------------------------------------
# Root Endpoint
# ---------------------------------------------------------

@app.get("/")
def root():

    return {
        "message":
            "Credit Risk Knowledge Assistant API"
    }


# ---------------------------------------------------------
# Health Endpoint
# ---------------------------------------------------------

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "service":
            "credit-risk-knowledge-assistant"
    }