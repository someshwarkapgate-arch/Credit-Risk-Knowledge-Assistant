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

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


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

# ---------------------------------------
# Temporary test
# ---------------------------------------

if __name__ == "__main__":

    pdf_path = DOCUMENT_DIR / "Master Circular - Prudential norms on Income Recognition, Asset Classification and.pdf"

    documents = load_pdf(pdf_path)

    print("\nPDF loaded successfully")

    print(
        f"Total pages loaded: {len(documents)}"
    )

    print("\nFirst page metadata:")
    print(documents[0].metadata)

    print("\nFirst 500 characters:")
    print(
        documents[0].page_content[:500]
    )

for i, document in enumerate(documents[:3]):

    print("\n" + "=" * 60)

    print(f"PAGE {i + 1}")

    print("\nMetadata:")
    print(document.metadata)

    print("\nContent:")
    print(document.page_content[:500])