from abc import ABC, abstractmethod
import pandas as pd
from typing import List


class MovieRecommender(ABC):
    def __init__(
        self, data: pd.DataFrame, embedding_generator, num_top_similar: int = 5
    ):
        if num_top_similar <= 0:
            raise ValueError("num_top_similar must be greater than 0.")
        self.data = data
        self.embedding_generator = embedding_generator
        self.embeddings = embedding_generator.generate_embeddings()
        self.num_top_similar = num_top_similar

    @abstractmethod
    def get_similar_movies(self, movie_index: int) -> None:
        pass

    def _print_movie_info(self, movie_index: int, top_indices: List[int]) -> None:
        try:
            print(f"original Movie ({self.data['title'].iloc[movie_index]}):")
            print(f"description: {self.data['description'].iloc[movie_index]}")
            print(f"rating: {self.data['rating'].iloc[movie_index]}")
            print(f"genres: {self.data['listed_in'].iloc[movie_index]}")
            print("\nsimilar Movies:")

            for idx in top_indices:
                print(f"title: {self.data['title'].iloc[idx]}")
                print(f"description: {self.data['description'].iloc[idx]}")
                print(f"rating: {self.data['rating'].iloc[idx]}")
                print(f"listed In: {self.data['listed_in'].iloc[idx]}")
                print()
        except Exception as e:
            print(f"error while printing movie info: {e}")
