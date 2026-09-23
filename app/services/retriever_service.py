from app.services.vector_store import get_vector_store


# ---------------------------------------------------------
# Default Retriever Configuration
# ---------------------------------------------------------

DEFAULT_TOP_K = 5

SUPPORTED_SEARCH_TYPES = {
    "similarity",
    "mmr",
    "similarity_score_threshold",}


# ---------------------------------------------------------
# Create Retriever
# ---------------------------------------------------------

def get_retriever(
    search_type="similarity",
    top_k=DEFAULT_TOP_K,
    score_threshold=None,
    filter_metadata=None,
    fetch_k=20,
    lambda_mult=0.5,
):
    """
    Create and return a LangChain retriever using the
    persistent Chroma vector store.

    Parameters
    ----------
    search_type : str
        Retrieval strategy.

        Supported:
        - similarity
        - mmr
        - similarity_score_threshold

    top_k : int
        Number of chunks to return.

    score_threshold : float or None
        Minimum relevance threshold when using
        similarity_score_threshold.

    filter_metadata : dict or None
        Optional metadata filter.

        Example:
        {
            "source_file": "INDAS109.pdf"
        }

    fetch_k : int
        Number of candidate chunks initially fetched
        when using MMR.

    lambda_mult : float
        Controls relevance vs diversity for MMR.

        1.0 = more relevance
        0.0 = more diversity

    Returns
    -------
    VectorStoreRetriever
        Configured LangChain retriever.
    """

    # -----------------------------------------------------
    # Validate top_k
    # -----------------------------------------------------

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than 0."
        )

    # -----------------------------------------------------
    # Validate retrieval strategy
    # -----------------------------------------------------

    if search_type not in SUPPORTED_SEARCH_TYPES:
        raise ValueError(
            f"Unsupported search_type: {search_type}. "
            f"Supported values are: "
            f"{SUPPORTED_SEARCH_TYPES}"
        )

    # -----------------------------------------------------
    # Load persistent vector store
    # -----------------------------------------------------

    vector_store = get_vector_store()

    # Common search configuration
    search_kwargs = {
        "k": top_k
    }

    # -----------------------------------------------------
    # Optional metadata filtering
    # -----------------------------------------------------

    if filter_metadata is not None:
        search_kwargs["filter"] = filter_metadata

    # -----------------------------------------------------
    # Score-threshold configuration
    # -----------------------------------------------------

    if search_type == "similarity_score_threshold":

        if score_threshold is None:
            raise ValueError(
                "score_threshold must be provided "
                "when using similarity_score_threshold."
            )

        if not 0 <= score_threshold <= 1:
            raise ValueError(
                "score_threshold must be between 0 and 1."
            )

        search_kwargs["score_threshold"] = (
            score_threshold
        )

    # -----------------------------------------------------
    # MMR configuration
    # -----------------------------------------------------

    if search_type == "mmr":

        if fetch_k < top_k:
            raise ValueError(
                "fetch_k must be greater than "
                "or equal to top_k."
            )

        if not 0 <= lambda_mult <= 1:
            raise ValueError(
                "lambda_mult must be between 0 and 1."
            )

        search_kwargs["fetch_k"] = fetch_k
        search_kwargs["lambda_mult"] = lambda_mult

    # -----------------------------------------------------
    # Build Retriever
    # -----------------------------------------------------

    retriever = vector_store.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )

    return retriever


# ---------------------------------------------------------
# Convenience Retrieval Function
# ---------------------------------------------------------

def retrieve_documents(
    question,
    search_type="similarity",
    top_k=DEFAULT_TOP_K,
    score_threshold=None,
    filter_metadata=None,
    fetch_k=20,
    lambda_mult=0.5,
):
    """
    Retrieve relevant document chunks for a question.
    """

    if not question:
        raise ValueError(
            "Question cannot be empty."
        )

    retriever = get_retriever(
        search_type=search_type,
        top_k=top_k,
        score_threshold=score_threshold,
        filter_metadata=filter_metadata,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult,
    )

    documents = retriever.invoke(question)

    return documents


# ---------------------------------------------------------
# Local Test
# ---------------------------------------------------------

if __name__ == "__main__":

    question = (
        "What is the objective of Ind AS 109?"
    )

    # Our Phase 6 baseline:
    # similarity retrieval with top_k = 5
    documents = retrieve_documents(
        question=question,
        search_type="similarity",
        top_k=5,
    )

    print(
        f"\nQuestion: {question}"
    )

    print(
        f"Retrieved documents: "
        f"{len(documents)}"
    )

    for rank, document in enumerate(
        documents,
        start=1,
    ):

        print("\n" + "=" * 80)

        print(
            f"Rank: {rank}"
        )

        print(
            "Source:",
            document.metadata.get(
                "source_file"
            ),
        )

        print(
            "Page:",
            document.metadata.get(
                "page_number"
            ),
        )

        print(
            "Chunk:",
            document.metadata.get(
                "chunk_id"
            ),
        )

        print("\nContent:")

        print(
            document.page_content[:500]
        )