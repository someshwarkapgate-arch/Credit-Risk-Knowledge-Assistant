from langchain_core.prompts import (
    ChatPromptTemplate
)


RAG_SYSTEM_PROMPT = """
You are a Credit Risk Knowledge Assistant.

Answer the user's question using ONLY the supplied context.

Rules:

1. Use only information contained in the context.
2. Do not use outside knowledge to answer the question.
3. If the context does not contain enough information, say:
   "I could not find enough information in the provided documents."
4. Do not invent regulatory requirements, thresholds, dates,
   classifications, definitions, or numerical values.
5. Give a clear and concise answer.
6. Cite the source filename and page number when the context
   provides them.
7. If multiple sources are relevant, distinguish them clearly.
"""


def get_rag_prompt():
    """
    Return the prompt template used for grounded RAG answers.
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                RAG_SYSTEM_PROMPT
            ),
            (
                "human",
                """
Context:
{context}

Question:
{question}

Answer:
"""
            ),
        ]
    )

    return prompt


def format_context(documents):
    """
    Convert retrieved LangChain Documents into
    formatted context for the LLM.
    """

    if not documents:
        return "No relevant context was retrieved."

    formatted_chunks = []

    for document in documents:

        source = document.metadata.get(
            "source_file",
            "Unknown"
        )

        page = document.metadata.get(
            "page_number",
            "Unknown"
        )

        content = document.page_content.strip()

        formatted_chunk = (
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content:\n{content}"
        )

        formatted_chunks.append(
            formatted_chunk
        )

    return "\n\n---\n\n".join(
        formatted_chunks
    )

