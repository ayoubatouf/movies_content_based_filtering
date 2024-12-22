from sklearn.feature_extraction.text import TfidfVectorizer
from utils import combine_columns, load_embeddings, save_embeddings
from embedding_generator import EmbeddingGenerator
import pandas as pd
import numpy as np
from typing import List


class TfidfEmbeddingGenerator(EmbeddingGenerator):
    def __init__(
        self, data: pd.DataFrame, columns_to_include: List[str], embeddings_dir: str
    ):
        super().__init__(
            data, columns_to_include, embeddings_dir, "tfidf_embeddings.npy"
        )
        self.tfidf_vectorizer = TfidfVectorizer(stop_words="english")

    def generate_embeddings(self) -> np.ndarray:
        embeddings = load_embeddings(self.file_path)
        if embeddings is None:
            print("generating new TF-IDF embeddings...")
            combined_data = combine_columns(self.data, self.columns_to_include)
            embeddings = self.tfidf_vectorizer.fit_transform(combined_data).toarray()
            save_embeddings(embeddings, self.file_path)
        return embeddings
