import pandas as pd
from similarity import SimilarityRecommender
from config import EMBEDDINGS_DIR
from sbert_generator import SentenceBertEmbeddingGenerator
from tfidf_generator import TfidfEmbeddingGenerator
from typing import List


def get_recommender(
    data: pd.DataFrame, model_type: str, columns: List[str]
) -> SimilarityRecommender:
    if model_type == "tfidf":
        generator = TfidfEmbeddingGenerator(data, columns, EMBEDDINGS_DIR)
    elif model_type == "sentence":
        generator = SentenceBertEmbeddingGenerator(data, columns, EMBEDDINGS_DIR)
    else:
        raise ValueError("invalid model_type. choose 'tfidf' or 'sentence'.")
    return SimilarityRecommender(data, generator)


if __name__ == "__main__":
    file_path = "netflix_titles.csv"
    columns_to_include = ["description", "rating", "listed_in"] # you can reduce/add other variables
    try:
        data = pd.read_csv(file_path)
        recommender = get_recommender(data, "sentence", columns_to_include) # change model type
        recommender.get_similar_movies(movie_index=0) # change movie_index
    except FileNotFoundError:
        print(f"file {file_path} not found. Please provide a valid file path.")
    except Exception as e:
        print(f"an error occurred: {e}")
