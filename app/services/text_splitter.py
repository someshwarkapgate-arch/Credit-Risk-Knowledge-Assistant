from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents,
    chunk_size=1000,
    chunk_overlap=200
):
    """
    Split LangChain documents into smaller chunks.

    Parameters
    ----------
    documents : list
        List of LangChain Document objects.

    chunk_size : int
        Maximum target size of each chunk.

    chunk_overlap : int
        Amount of text shared between neighboring chunks.

    Returns
    -------
    list
        Chunked LangChain Document objects.
    """

    if not documents:
        raise ValueError("No documents provided for chunking.")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = text_splitter.split_documents(documents)

    # Add additional metadata
    for index, chunk in enumerate(chunks):

        chunk.metadata["chunk_id"] = index

        chunk.metadata["chunk_length"] = len(
            chunk.page_content
        )

    return chunks



#Temporary test

if __name__ == "__main__":

    from app.services.document_loader import (
        DOCUMENT_DIR,
        load_all_pdfs
    )

    # -----------------------------------------
    # Load documents
    # -----------------------------------------

    documents = load_all_pdfs(
        DOCUMENT_DIR
    )

    print("\n" + "=" * 60)

    print(
        f"Documents before chunking: "
        f"{len(documents)}"
    )


    # -----------------------------------------
    # Split documents
    # -----------------------------------------

    chunks = split_documents(
        documents=documents,
        chunk_size=1000,
        chunk_overlap=200
    )

    print(
        f"Chunks after splitting: "
        f"{len(chunks)}"
    )


    # -----------------------------------------
    # Inspect first chunk
    # -----------------------------------------

    print("\n" + "=" * 60)

    print("FIRST CHUNK")

    print("\nMetadata:")
    print(
        chunks[0].metadata
    )

    print("\nChunk length:")
    print(
        len(chunks[0].page_content)
    )

    print("\nContent:")
    print(
        chunks[0].page_content
    )