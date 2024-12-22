import numpy as np
import faiss
import os
import pandas as pd
from typing import List, Union


def combine_columns(data: pd.DataFrame, columns: List[str]) -> pd.Series:

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if not all(col in data.columns for col in columns):
        raise ValueError("one or more specified columns do not exist in the DataFrame.")
    return data[columns].fillna("").agg(" ".join, axis=1)


def save_embeddings(embeddings: np.ndarray, file_path: str) -> None:

    try:
        np.save(file_path, embeddings)
        print(f"saved embeddings to {file_path}")
    except Exception as e:
        raise IOError(f"failed to save embeddings to {file_path}: {e}")


def load_embeddings(file_path: str) -> Union[np.ndarray, None]:

    if not os.path.exists(file_path):
        print(f"file {file_path} does not exist.")
        return None
    try:
        print(f"loading embeddings from {file_path}...")
        return np.load(file_path, allow_pickle=True)
    except Exception as e:
        raise IOError(f"failed to load embeddings from {file_path}: {e}")


def compute_similarity(
    movie_embedding: np.ndarray, embeddings: np.ndarray, top_n: int
) -> np.ndarray:
    """
    compute similarity using FAISS and return indices of top similar items
    """
    try:
        embeddings = np.array(embeddings, dtype=np.float32)
        movie_embedding = np.array([movie_embedding], dtype=np.float32)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        _, indices = index.search(movie_embedding, top_n + 1)
        return indices[0][1:]  # exclude the query itself
    except Exception as e:
        raise ValueError(f"failed to compute similarity: {e}")
