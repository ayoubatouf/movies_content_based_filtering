from recommender import MovieRecommender
from utils import compute_similarity


class SimilarityRecommender(MovieRecommender):
    def get_similar_movies(self, movie_index: int) -> None:
        if movie_index < 0 or movie_index >= len(self.embeddings):
            print(f"error: index {movie_index} out of range.")
            return
        try:
            movie_embedding = self.embeddings[movie_index]
            top_indices = compute_similarity(
                movie_embedding, self.embeddings, self.num_top_similar
            )
            self._print_movie_info(movie_index, top_indices)
        except Exception as e:
            print(f"error while retrieving similar movies: {e}")
