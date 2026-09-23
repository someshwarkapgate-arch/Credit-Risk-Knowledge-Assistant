from langchain_groq import ChatGroq


DEFAULT_LLM_MODEL = "openai/gpt-oss-20b"


def get_llm(
    model_name=DEFAULT_LLM_MODEL,
    temperature=0,
):
    """
    Create and return the LLM used by the application.
    """

    llm = ChatGroq(
        model=model_name,
        temperature=temperature,
        max_retries=2,
        reasoning_format="hidden"
    )

    return llm