from pathlib import Path

from langchain_chroma import Chroma

from app.services.embedding_service import (
    get_embedding_model
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHROMA_DIR = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = "credit_risk_knowledge"


def get_vector_store():

    embedding_model = get_embedding_model()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(CHROMA_DIR)
    )

    return vector_store