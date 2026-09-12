import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class VectorStore:

    def __init__(self, documents, vectors):
        self.documents = documents
        self.vectors = vectors

    def search(self, query_vector, top_k=2):
        scores = cosine_similarity(
            query_vector,
            self.vectors
        )[0]

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for index in top_indices:
            results.append({
                "source": self.documents[index]["source"],
                "text": self.documents[index]["text"],
                "score": float(scores[index])
            })

        return results