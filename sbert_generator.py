from sentence_transformers import SentenceTransformer
from utils import combine_columns, load_embeddings, save_embeddings
from embedding_generator import EmbeddingGenerator
import pandas as pd
import numpy as np
from typing import List


class SentenceBertEmbeddingGenerator(EmbeddingGenerator):
    def __init__(
        self,
        data: pd.DataFrame,
        columns_to_include: List[str],
        embeddings_dir: str,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        super().__init__(
            data, columns_to_include, embeddings_dir, "sentence_bert_embeddings.npy"
        )
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self) -> np.ndarray:
        embeddings = load_embeddings(self.file_path)
        if embeddings is None:
            print("generating new sentence-BERT embeddings...")
            combined_data = combine_columns(self.data, self.columns_to_include)
            embeddings = self.model.encode(combined_data, convert_to_numpy=True)
            save_embeddings(embeddings, self.file_path)
        return embeddings
