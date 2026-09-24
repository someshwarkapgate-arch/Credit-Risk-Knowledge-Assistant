import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq


# ---------------------------------------------------------
# 1. Find the project root folder
# llm_service.py is inside:
# project_root/app/services/llm_service.py
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Stores the exact path of the .env file
ENV_PATH = PROJECT_ROOT / ".env"


# ---------------------------------------------------------
# 2. Load variables from .env
# ---------------------------------------------------------
load_dotenv(dotenv_path=ENV_PATH)


# Default model used by our application
DEFAULT_LLM_MODEL = "openai/gpt-oss-20b"


def get_llm(
    model_name=DEFAULT_LLM_MODEL,
    temperature=0,
):
    """
    Create and return the LLM used by the application.
    """

    # Read GROQ_API_KEY from the .env file
    groq_api_key = os.getenv("GROQ_API_KEY")

    # Stop immediately if the API key was not loaded
    if not groq_api_key:
        raise ValueError(
            f"GROQ_API_KEY not found. Checked .env file at: {ENV_PATH}"
        )

    # Create the Groq LLM
    llm = ChatGroq(
        model=model_name,
        temperature=temperature,
        max_retries=2,
        reasoning_format="hidden",
        api_key=groq_api_key
    )

    return llm