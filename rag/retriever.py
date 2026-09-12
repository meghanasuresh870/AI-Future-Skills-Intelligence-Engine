from pathlib import Path

from rag.document_loader import load_documents
from rag.embeddings import create_embeddings
from rag.vector_store import VectorStore


class RAGRetriever:

    def __init__(self, folder_path=None):

        if folder_path is None:

            project_root = Path(__file__).resolve().parent.parent

            folder_path = (
                project_root
                / "knowledge_base"
                / "documents"
            )

        self.documents = load_documents(folder_path)

        self.embedding_model, self.vectors = create_embeddings(
            self.documents
        )

        self.vector_store = VectorStore(
            self.documents,
            self.vectors
        )

    def retrieve(self, question, top_k=2):

        query_vector = self.embedding_model.transform(
            [question]
        )

        return self.vector_store.search(
            query_vector,
            top_k
        )


def build_context(results):

    context_parts = []

    for result in results:

        context_parts.append(
            f"Source: {result['source']}\n"
            f"{result['text']}"
        )

    return "\n\n".join(context_parts)