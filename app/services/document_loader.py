from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

#----------------------------------------------
# Project Paths
#----------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DOCUMENT_DIR = PROJECT_ROOT / "documents"

# ---------------------------------------
# PDF Loader
# ---------------------------------------

def load_pdf(file_path):

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    if file_path.suffix.lower() != ".pdf":
        raise ValueError(
            "Only PDF files are supported."
        )

    loader = PyPDFLoader(
        str(file_path),
        mode="page"
    )

    documents = loader.load()

    for document in documents:

        document.metadata["source_file"] = file_path.name

        if "page" in document.metadata:
            document.metadata["page_number"] = (
                document.metadata["page"] + 1
            )

    return documents




## Loading all pdf files at once


def load_all_pdfs(directory_path):
    """
    Load all PDF files from a directory.

    Parameters
    ----------
    directory_path : str or Path
        Folder containing PDF files.

    Returns
    -------
    list
        Combined list of LangChain Document objects
        from all PDF files.
    """

    directory_path = Path(directory_path)

    # 1. Check whether directory exists
    if not directory_path.exists():
        raise FileNotFoundError(
            f"Directory not found: {directory_path}"
        )

    # 2. Check whether path is actually a directory
    if not directory_path.is_dir():
        raise ValueError(
            f"Expected a directory, received: {directory_path}"
        )

    # 3. Find all PDF files
    pdf_files = list(directory_path.glob("*.pdf"))

    # 4. Check whether any PDFs were found
    if not pdf_files:
        raise ValueError(
            f"No PDF files found in: {directory_path}"
        )

    # 5. Create empty list for all documents
    all_documents = []

    print(f"\nFound {len(pdf_files)} PDF file(s).")

    # 6. Loop through every PDF
    for pdf_file in pdf_files:

        print(f"\nLoading: {pdf_file.name}")

        try:
            # Use our existing load_pdf() function
            documents = load_pdf(pdf_file)

            print(
                f"Loaded {len(documents)} page(s)"
            )

            # Add documents to main list
            all_documents.extend(documents)

        except Exception as error:

            print(
                f"Error loading {pdf_file.name}: {error}"
            )

    print(
        f"\nTotal pages/documents loaded: "
        f"{len(all_documents)}"
    )

    return all_documents


# ---------------------------------------
# Temporary test
# ---------------------------------------

if __name__ == "__main__":

    documents = load_all_pdfs(DOCUMENT_DIR)

    print("\n" + "=" * 60)

    print("INGESTION COMPLETED")

    print(f"Total documents/pages: {len(documents)}")

    print("\nFirst document metadata:")
    print(documents[0].metadata)

    print("\nFirst 1500 characters:")
    print(documents[0].page_content[:1500])

