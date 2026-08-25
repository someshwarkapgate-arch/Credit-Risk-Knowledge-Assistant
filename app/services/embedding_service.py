from langchain_huggingface import (
    HuggingFaceEmbeddings
)


DEFAULT_EMBEDDING_MODEL = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)


def get_embedding_model(
    model_name=DEFAULT_EMBEDDING_MODEL,
    device="cpu"
):
    """
    Create and return the embedding model.

    Parameters
    ----------
    model_name : str
        Hugging Face embedding model name.

    device : str
        Device used for inference.
        Example: 'cpu' or 'cuda'.

    Returns
    -------
    HuggingFaceEmbeddings
        Configured LangChain embedding model.
    """

    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={
            "device": device
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    return embedding_model


## Temporary testing code to verify embedding model functionality

if __name__ == "__main__":

    embedding_model = (
        get_embedding_model()
    )

    test_text = (
        "Expected credit loss "
        "under Ind AS 109"
    )

    vector = (
        embedding_model.embed_query(
            test_text
        )
    )

    print(
        "Embedding dimension:",
        len(vector)
    )

    print(
        "First 10 values:"
    )

    print(
        vector[:10]
    )

    