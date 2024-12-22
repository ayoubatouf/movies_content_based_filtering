import os
import pandas as pd
from typing import List


class EmbeddingGenerator:
    def __init__(
        self,
        data: pd.DataFrame,
        columns_to_include: List[str],
        embeddings_dir: str,
        file_name: str,
    ):
        if not os.path.isdir(embeddings_dir):
            raise ValueError(f"embeddings directory {embeddings_dir} does not exist.")
        self.data = data
        self.columns_to_include = columns_to_include
        self.embeddings_dir = embeddings_dir
        self.file_path = os.path.join(embeddings_dir, file_name)

    def generate_embeddings(self):
        raise NotImplementedError("this method should be overridden in subclasses.")
