import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class EmbeddingModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english"
        )

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts).toarray()

    def transform(self, texts):
        return self.vectorizer.transform(texts).toarray()


def create_embeddings(documents):
    texts = [doc["text"] for doc in documents]

    model = EmbeddingModel()
    vectors = model.fit_transform(texts)

    return model, vectors