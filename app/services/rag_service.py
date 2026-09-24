from app.services.retriever_service import (
    retrieve_documents
)

from app.services.prompt_service import (
    get_rag_prompt,
    format_context
)

from app.services.llm_service import (
    get_llm
)


# ---------------------------------------------------------
# Extract Sources
# ---------------------------------------------------------

def extract_sources(documents):
    """
    Extract unique source/page information from
    retrieved LangChain Document objects.

    Parameters
    ----------
    documents : list
        Retrieved LangChain Document objects.

    Returns
    -------
    list
        Structured source information.
    """

    sources = []

    seen_sources = set()

    for document in documents:

        source_file = document.metadata.get(
            "source_file",
            "Unknown"
        )

        page_number = document.metadata.get(
            "page_number",
            "Unknown"
        )

        chunk_id = document.metadata.get(
            "chunk_id"
        )

        source_key = (
            source_file,
            page_number
        )

        # Avoid duplicate source/page entries
        if source_key not in seen_sources:

            sources.append(
                {
                    "source_file": source_file,
                    "page_number": page_number,
                    "chunk_id": chunk_id
                }
            )

            seen_sources.add(
                source_key
            )

    return sources


# ---------------------------------------------------------
# Main RAG Function
# ---------------------------------------------------------

def answer_question(
    question,
    top_k=5,
    filter_metadata=None,
):
    """
    Run the complete RAG pipeline.

    Pipeline
    --------
    Question
        ↓
    Retriever
        ↓
    Relevant Documents
        ↓
    Context Formatting
        ↓
    Prompt
        ↓
    LLM
        ↓
    Answer + Sources

    Parameters
    ----------
    question : str
        User question.

    top_k : int
        Number of retrieved chunks.

    filter_metadata : dict or None
        Optional vector-store metadata filter.

    Returns
    -------
    dict
        Structured RAG response containing:
        - question
        - answer
        - sources
        - retrieved_chunks
    """

    # -----------------------------------------------------
    # Input Validation
    # -----------------------------------------------------

    if not question:

        raise ValueError(
            "Question cannot be empty."
        )

    question = question.strip()

    if not question:

        raise ValueError(
            "Question cannot be empty."
        )


    # -----------------------------------------------------
    # Step 1: Retrieve Documents
    # -----------------------------------------------------
    print("STEP 1: Starting retrieval")

    documents = retrieve_documents(
        question=question,
        search_type="similarity",
        top_k=top_k,
        filter_metadata=filter_metadata,
    )


    # -----------------------------------------------------
    # Step 2: Format Context
    # -----------------------------------------------------

    print("STEP 2: Documents retrieved:", len(documents))

    context = format_context(
        documents
    )

    print("STEP 3: Context created")
    # -----------------------------------------------------
    # Step 3: Build Prompt
    # -----------------------------------------------------

    prompt = get_rag_prompt()

    print("STEP 4: Prompt created")

    messages = prompt.invoke(
        {
            "context": context,
            "question": question,
        }
    )


    # -----------------------------------------------------
    # Step 4: Generate Answer
    # -----------------------------------------------------
    print("STEP 5A: About to create LLM")

    llm = get_llm()

    print("STEP 5B: LLM created successfully")

    response = llm.invoke(
        messages
    )
    print("STEP 6: LLM response received")
    

    # -----------------------------------------------------
    # Step 5: Extract Sources
    # -----------------------------------------------------

    sources = extract_sources(
        documents
    )


    # -----------------------------------------------------
    # Step 6: Structured Response
    # -----------------------------------------------------

    return {
        "question": question,
        "answer": response.content,
        "sources": sources,
        "retrieved_chunks": len(documents),
    }


# ---------------------------------------------------------
# Local Test
# ---------------------------------------------------------

if __name__ == "__main__":

    question = (
        "What objective is stated in "
        "Chapter 1 of Ind AS 109 "
        "Financial Instruments?"
    )

    result = answer_question(
        question=question,
        top_k=5,
    )

    print("\n" + "=" * 80)

    print("QUESTION:")
    print(
        result["question"]
    )

    print("\nANSWER:")
    print(
        result["answer"]
    )

    print("\nSOURCES:")

    for source in result["sources"]:

        print(
            f"- {source['source_file']} "
            f"| Page "
            f"{source['page_number']}"
        )

    print(
        "\nRetrieved chunks:",
        result["retrieved_chunks"]
    )