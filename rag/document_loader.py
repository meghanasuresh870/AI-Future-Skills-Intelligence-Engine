from pathlib import Path
from pypdf import PdfReader


def load_text_file(file_path):
    """Load text from a TXT file."""
    path = Path(file_path)

    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def load_pdf_file(file_path):
    """Extract text from a PDF file."""
    reader = PdfReader(file_path)

    text = []

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text.append(page_text)

    return "\n".join(text)


def load_documents(folder_path):
    """Load TXT and PDF documents from the knowledge base."""

    folder = Path(folder_path)
    documents = []

    for file_path in folder.iterdir():

        if file_path.suffix.lower() == ".txt":
            text = load_text_file(file_path)

        elif file_path.suffix.lower() == ".pdf":
            text = load_pdf_file(file_path)

        else:
            continue

        if text.strip():
            documents.append(
                {
                    "source": file_path.name,
                    "text": text,
                }
            )

    return documents