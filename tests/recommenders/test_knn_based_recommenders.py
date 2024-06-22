"""
Tests for knn based recommenders from recommenders module.
"""

import numpy as np
import pytest

# from unittest.mock import patch, Mock

# from recommenders.PAkNN import (
#     merge_biclusters,
#     calculate_weighted_rating,
#     get_k_top_neighbors,
#     PAkNN,
# )
# from pattern_mining.formal_concept_analysis import Concept, create_concept

# from surprise import PredictionImpossible

# # pylint: disable=missing-function-docstring,missing-class-docstring


# class TestMergeBiclusters:
#     def test_invalid_args(self):
#         with pytest.raises(AssertionError):
#             merge_biclusters("not a list")

#         with pytest.raises(AssertionError):
#             merge_biclusters([1, 2, 3])

#         with pytest.raises(AssertionError):
#             merge_biclusters([])

#         with pytest.raises(AssertionError):
#             merge_biclusters([Concept(np.array([1, 2, 3]), np.array([4, 5, 6])), 2])

#         with pytest.raises(AssertionError):
#             merge_biclusters([Concept(np.array([1, 2, 3]), np.array([4, 5, 6])), "not a concept"])

#         with pytest.raises(AssertionError):
#             merge_biclusters([Concept(np.array([]), np.array([4, 5, 6]))])

#         with pytest.raises(AssertionError):
#             merge_biclusters([Concept(np.array([1]), np.array([]))])

#     def test_merge_single_bicluster(self):
#         extent = np.array([1, 2, 3])
#         intent = np.array([4, 5, 6])
#         bicluster = create_concept(extent, intent)
#         biclusters = [bicluster]
#         merged_bicluster = merge_biclusters(biclusters)
#         assert np.array_equal(merged_bicluster.extent, extent)
#         assert np.array_equal(merged_bicluster.intent, intent)

#     def test_merge_multiple_biclusters(self):
#         bicluster1 = create_concept(np.array([1, 2, 3]), np.array([4, 5, 6]))
#         bicluster2 = create_concept(np.array([3, 4, 5]), np.array([6, 7, 8]))
#         bicluster3 = create_concept(np.array([5, 6, 7]), np.array([8, 9, 10]))

#         biclusters = [bicluster1, bicluster2, bicluster3]

#         merged_bicluster = merge_biclusters(biclusters)

#         assert np.array_equal(merged_bicluster.extent, np.array([1, 2, 3, 4, 5, 6, 7]))
#         assert np.array_equal(merged_bicluster.intent, np.array([4, 5, 6, 7, 8, 9, 10]))

#         bicluster1 = create_concept(np.array([1, 2, 3, 8]), np.array([4, 5, 6, 10, 11]))
#         bicluster2 = create_concept(np.array([3, 4, 5, 8]), np.array([6, 7, 8]))
#         bicluster3 = create_concept(np.array([5, 6, 7, 8, 9]), np.array([8, 9, 10]))

#         biclusters = [bicluster1, bicluster2, bicluster3]

#         merged_bicluster = merge_biclusters(biclusters)

#         assert np.array_equal(merged_bicluster.extent, np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))
#         assert np.array_equal(merged_bicluster.intent, np.array([4, 5, 6, 7, 8, 9, 10, 11]))

#         bicluster1 = create_concept(np.array([1]), np.array([4]))
#         bicluster2 = create_concept(np.array([3]), np.array([6]))
#         bicluster3 = create_concept(np.array([5]), np.array([8]))

#         biclusters = [bicluster1, bicluster2, bicluster3]

#         merged_bicluster = merge_biclusters(biclusters)

#         assert np.array_equal(merged_bicluster.extent, np.array([1, 3, 5]))
#         assert np.array_equal(merged_bicluster.intent, np.array([4, 6, 8]))

#         bicluster1 = create_concept(np.array([1, 2, 3, 4, 5]), np.array([4, 5, 6, 7, 8, 9]))
#         bicluster2 = create_concept(np.array([3]), np.array([6]))

#         biclusters = [bicluster1, bicluster2, bicluster3]

#         merged_bicluster = merge_biclusters(biclusters)

#         assert np.array_equal(merged_bicluster.extent, np.array([1, 2, 3, 4, 5]))
#         assert np.array_equal(merged_bicluster.intent, np.array([4, 5, 6, 7, 8, 9]))


# class TestCalculateWeightedRating:
#     def test_invalid_args(self):
#         with pytest.raises(AssertionError):
#             calculate_weighted_rating(
#                 "3.0",
#                 np.array([1, 2, 3], dtype=np.float64),
#                 np.array([1, 1, 1], dtype=np.float64),
#                 np.array([1, 2, 3], dtype=np.float64),
#             )

#         with pytest.raises(AssertionError):
#             calculate_weighted_rating(
#                 1,
#                 np.array([1, 2, 3], dtype=np.float64),
#                 np.array([1, 1, 1], dtype=np.float64),
#                 np.array([1, 2, 3], dtype=np.float64),
#             )

#         with pytest.raises(AssertionError):
#             calculate_weighted_rating(
#                 1.1,
#                 np.array([1, 2, 3], dtype=np.int64),
#                 np.array([1, 1, 1], dtype=np.float64),
#                 np.array([1, 2, 3], dtype=np.float64),
#             )

#         with pytest.raises(AssertionError):
#             calculate_weighted_rating(
#                 1.1,
#                 np.array([1, 2, 3], dtype=np.float64),
#                 np.array([1, 1, 1], dtype=np.int64),
#                 np.array([1, 2, 3], dtype=np.float64),
#             )

#         with pytest.raises(AssertionError):
#             calculate_weighted_rating(
#                 1.1,
#                 np.array([1, 2, 3], dtype=np.float64),
#                 np.array([1, 1, 1], dtype=np.float64),
#                 np.array([1, 2, 3], dtype=np.int64),
#             )

#         with pytest.raises(AssertionError):
#             calculate_weighted_rating(
#                 1.1,
#                 np.array([1, 2, 3], dtype=np.float64),
#                 np.array([0, 1, 1], dtype=np.float64),
#                 np.array([1, 2, 3], dtype=np.float64),
#             )

#         with pytest.raises(AssertionError):
#             calculate_weighted_rating(
#                 1.1,
#                 np.array([1, 2, 3], dtype=np.float64),
#                 np.array([1, 1], dtype=np.float64),
#                 np.array([1, 2, 3], dtype=np.float64),
#             )

#         with pytest.raises(AssertionError):
#             calculate_weighted_rating(
#                 1.1,
#                 np.array([], dtype=np.float64),
#                 np.array([], dtype=np.float64),
#                 np.array([], dtype=np.float64),
#             )

#         with pytest.raises(AssertionError):
#             calculate_weighted_rating(
#                 1.1,
#                 np.array([1, 2, 3], dtype=np.float64),
#                 [1, 1, 1],
#                 np.array([1, 2, 3], dtype=np.float64),
#             )

#     def test_success_1(self):
#         target_mean = 3.0
#         neighbors_ratings = np.array([4.0, 2.0, 5.0], dtype=np.float64)
#         neighbors_similarities = np.array([0.1, 0.3, 0.6], dtype=np.float64)
#         neighbors_means = np.array([3.5, 2.5, 4.0], dtype=np.float64)

#         result = calculate_weighted_rating(
#             target_mean, neighbors_ratings, neighbors_similarities, neighbors_means
#         )
#         expected_result = 3.0 + (
#             (0.1 * (4.0 - 3.5) + 0.3 * (2.0 - 2.5) + 0.6 * (5.0 - 4.0)) / (0.1 + 0.3 + 0.6)
#         )
#         assert np.isclose(result, expected_result)

#     def test_success_2(self):
#         target_mean = 4.2
#         neighbors_ratings = np.array([4.0], dtype=np.float64)
#         neighbors_similarities = np.array([0.1], dtype=np.float64)
#         neighbors_means = np.array([3.5], dtype=np.float64)

#         result = calculate_weighted_rating(
#             target_mean, neighbors_ratings, neighbors_similarities, neighbors_means
#         )
#         expected_result = 4.2 + ((0.1 * (4.0 - 3.5)) / (0.1))
#         assert np.isclose(result, expected_result)

#     def test_success_3(self):
#         target_mean = 1.3
#         neighbors_ratings = np.array([4.0, 1.1, 3.2, 2.7], dtype=np.float64)
#         neighbors_similarities = np.array([0.1, 0.1, 0.2, 0.3], dtype=np.float64)
#         neighbors_means = np.array([3.5, 1.2, 4.3, 1.7], dtype=np.float64)

#         result = calculate_weighted_rating(
#             target_mean, neighbors_ratings, neighbors_similarities, neighbors_means
#         )
#         expected_result = 1.3 + (
#             (0.1 * (4.0 - 3.5) + 0.1 * (1.1 - 1.2) + 0.2 * (3.2 - 4.3) + 0.3 * (2.7 - 1.7))
#             / (0.1 + 0.1 + 0.2 + 0.3)
#         )
#         assert np.isclose(result, expected_result)


# class TestGetKTopNeighbors:
#     # fmt: off
#     square_dataset = np.array([[ 1,  2,  3,  4],
#                                [ 5,  6,  7,  8],
#                                [ 9, 10, 11, 12],
#                                [13, 14, 15, 16]],
#                                 dtype=np.float64)

#     similarity_matrix = np.array([[1.0, 0.5, 1.0, 0.1],
#                                   [0.5, 1.0, 0.7, 0.3],
#                                   [1.0, 0.7, 1.0, 0.0],
#                                   [0.1, 0.3, 0.0, 1.0]])
    
#     means = np.array([1.0, 2.0, 3.0, 4.0])
#     # fmt: on

#     @staticmethod
#     def assert_result_get_k_top_neighbors(
#         x,
#         y,
#         dataset,
#         users_neighborhood,
#         similarity_matrix,
#         means,
#         knn_k,
#         expected_ratings,
#         expected_similarity,
#         expected_means,
#     ):
#         k_top_ratings, k_top_similarity, k_top_means = get_k_top_neighbors(
#             x, y, dataset, users_neighborhood, similarity_matrix, means, knn_k
#         )

#         assert np.array_equal(k_top_ratings, expected_ratings)
#         assert np.array_equal(k_top_similarity, expected_similarity)
#         assert np.array_equal(k_top_means, expected_means)

#     def test_invalid_args(self):
#         dataset = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
#         users_neighborhood = np.array([0, 1], dtype=np.int64)
#         similarity_matrix = np.array(
#             [[1, 0.9, 0.8], [0.9, 1, 0.7], [0.8, 0.7, 1]], dtype=np.float64
#         )
#         means = np.array([2, 5, 8], dtype=np.float64)
#         knn_k = 2

#         with pytest.raises(AssertionError):
#             get_k_top_neighbors(
#                 "0", 1, dataset, users_neighborhood, similarity_matrix, means, knn_k
#             )

#         with pytest.raises(AssertionError):
#             get_k_top_neighbors(
#                 0, "1", dataset, users_neighborhood, similarity_matrix, means, knn_k
#             )

#         with pytest.raises(AssertionError):
#             get_k_top_neighbors(
#                 0, 1, dataset.tolist(), users_neighborhood, similarity_matrix, means, knn_k
#             )

#         with pytest.raises(AssertionError):
#             get_k_top_neighbors(
#                 0, 1, dataset, users_neighborhood.tolist(), similarity_matrix, means, knn_k
#             )

#         with pytest.raises(AssertionError):
#             get_k_top_neighbors(
#                 0, 1, dataset, users_neighborhood, similarity_matrix.tolist(), means, knn_k
#             )

#         with pytest.raises(AssertionError):
#             get_k_top_neighbors(
#                 0, 1, dataset, users_neighborhood, similarity_matrix, means.tolist(), knn_k
#             )

#         with pytest.raises(AssertionError):
#             get_k_top_neighbors(0, 1, dataset, users_neighborhood, similarity_matrix, means, "2")

#     def test_no_similarity_square_dataset(self):
#         x = 0
#         y = 0

#         # fmt: off
#         square_dataset = np.array([[ 1,  2,  3,  4],
#                                    [ 5,  6,  7,  8],
#                                    [ 9, 10, 11, 12],
#                                    [13, 14, 15, 16]],
#                                    dtype=np.float64)
        
#         similarity_matrix = np.array([[1.0, 0.0, 0.0, 0.0],
#                                       [0.0, 1.0, 0.0, 0.0],
#                                       [0.0, 0.0, 1.0, 0.0],
#                                       [0.0, 0.0, 0.0, 1.0]])
#         # fmt: on

#         empty_array = np.array([])

#         users_neighborhood = np.array([1, 2])
#         means = np.array([2.0, 5.0, 8.0, 1.0])
#         knn_k = 4

#         TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#             x=x,
#             y=y,
#             dataset=square_dataset,
#             users_neighborhood=users_neighborhood,
#             similarity_matrix=similarity_matrix,
#             means=means,
#             knn_k=knn_k,
#             expected_ratings=empty_array,
#             expected_similarity=empty_array,
#             expected_means=empty_array,
#         )

#         knn_k = 3
#         TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#             x=x,
#             y=y,
#             dataset=square_dataset,
#             users_neighborhood=users_neighborhood,
#             similarity_matrix=similarity_matrix,
#             means=means,
#             knn_k=knn_k,
#             expected_ratings=empty_array,
#             expected_similarity=empty_array,
#             expected_means=empty_array,
#         )

#         knn_k = 2
#         TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#             x=x,
#             y=y,
#             dataset=square_dataset,
#             users_neighborhood=users_neighborhood,
#             similarity_matrix=similarity_matrix,
#             means=means,
#             knn_k=knn_k,
#             expected_ratings=empty_array,
#             expected_similarity=empty_array,
#             expected_means=empty_array,
#         )

#         knn_k = 1
#         TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#             x=x,
#             y=y,
#             dataset=square_dataset,
#             users_neighborhood=users_neighborhood,
#             similarity_matrix=similarity_matrix,
#             means=means,
#             knn_k=knn_k,
#             expected_ratings=empty_array,
#             expected_similarity=empty_array,
#             expected_means=empty_array,
#         )

#         knn_k = 2
#         y = 1
#         TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#             x=x,
#             y=y,
#             dataset=square_dataset,
#             users_neighborhood=users_neighborhood,
#             similarity_matrix=similarity_matrix,
#             means=means,
#             knn_k=knn_k,
#             expected_ratings=empty_array,
#             expected_similarity=empty_array,
#             expected_means=empty_array,
#         )

#         y = 2
#         TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#             x=x,
#             y=y,
#             dataset=square_dataset,
#             users_neighborhood=users_neighborhood,
#             similarity_matrix=similarity_matrix,
#             means=means,
#             knn_k=knn_k,
#             expected_ratings=empty_array,
#             expected_similarity=empty_array,
#             expected_means=empty_array,
#         )

#         y = 3
#         TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#             x=x,
#             y=y,
#             dataset=square_dataset,
#             users_neighborhood=users_neighborhood,
#             similarity_matrix=similarity_matrix,
#             means=means,
#             knn_k=knn_k,
#             expected_ratings=empty_array,
#             expected_similarity=empty_array,
#             expected_means=empty_array,
#         )

#         y = 2
#         x = 1
#         users_neighborhood = np.array([0, 2])
#         TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#             x=x,
#             y=y,
#             dataset=square_dataset,
#             users_neighborhood=users_neighborhood,
#             similarity_matrix=similarity_matrix,
#             means=means,
#             knn_k=2,
#             expected_ratings=empty_array,
#             expected_similarity=empty_array,
#             expected_means=empty_array,
#         )

#         x = 2
#         y = 1
#         users_neighborhood = np.array([0, 3])
#         TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#             x=x,
#             y=y,
#             dataset=square_dataset,
#             users_neighborhood=users_neighborhood,
#             similarity_matrix=similarity_matrix,
#             means=means,
#             knn_k=2,
#             expected_ratings=empty_array,
#             expected_similarity=empty_array,
#             expected_means=empty_array,
#         )

#         x = 2
#         y = 2
#         users_neighborhood = np.array([0, 1])
#         TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#             x=x,
#             y=y,
#             dataset=square_dataset,
#             users_neighborhood=users_neighborhood,
#             similarity_matrix=similarity_matrix,
#             means=means,
#             knn_k=2,
#             expected_ratings=empty_array,
#             expected_similarity=empty_array,
#             expected_means=empty_array,
#         )

#         x = 3
#         y = 2
#         users_neighborhood = np.array([0, 1, 2])
#         TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#             x=x,
#             y=y,
#             dataset=square_dataset,
#             users_neighborhood=users_neighborhood,
#             similarity_matrix=similarity_matrix,
#             means=means,
#             knn_k=2,
#             expected_ratings=empty_array,
#             expected_similarity=empty_array,
#             expected_means=empty_array,
#         )

#     def test_no_similarity_non_square_dataset(self):
#         # fmt: off
#         non_square_dataset = np.array([[ 1,  2,  3,  4],
#                                        [ 5,  6,  7,  8],
#                                        [ 9, 10, 11, 12]],
#                                        dtype=np.float64)
        
#         similarity_matrix = np.array([[1.0, 0.0, 0.0],
#                                       [0.0, 1.0, 0.0],
#                                       [0.0, 0.0, 1.0]],
#                                       dtype=np.float64)
#         # fmt: on

#         empty_array = np.array([])

#         users_neighborhood = np.array([1, 2])
#         means = np.array([2.0, 5.0, 8.0])
#         knn_k = 3

#         x = 0
#         y = 0
#         TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#             x=x,
#             y=y,
#             dataset=non_square_dataset,
#             users_neighborhood=users_neighborhood,
#             similarity_matrix=similarity_matrix,
#             means=means,
#             knn_k=knn_k,
#             expected_ratings=empty_array,
#             expected_similarity=empty_array,
#             expected_means=empty_array,
#         )

#         x = 0
#         y = 1
#         TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#             x=x,
#             y=y,
#             dataset=non_square_dataset,
#             users_neighborhood=users_neighborhood,
#             similarity_matrix=similarity_matrix,
#             means=means,
#             knn_k=knn_k,
#             expected_ratings=empty_array,
#             expected_similarity=empty_array,
#             expected_means=empty_array,
#         )

#         x = 0
#         y = 2
#         TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#             x=x,
#             y=y,
#             dataset=non_square_dataset,
#             users_neighborhood=users_neighborhood,
#             similarity_matrix=similarity_matrix,
#             means=means,
#             knn_k=knn_k,
#             expected_ratings=empty_array,
#             expected_similarity=empty_array,
#             expected_means=empty_array,
#         )

#         x = 1
#         y = 0
#         users_neighborhood = np.array([0, 2])
#         TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#             x=x,
#             y=y,
#             dataset=non_square_dataset,
#             users_neighborhood=users_neighborhood,
#             similarity_matrix=similarity_matrix,
#             means=means,
#             knn_k=knn_k,
#             expected_ratings=empty_array,
#             expected_similarity=empty_array,
#             expected_means=empty_array,
#         )

#         x = 1
#         y = 1
#         TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#             x=x,
#             y=y,
#             dataset=non_square_dataset,
#             users_neighborhood=users_neighborhood,
#             similarity_matrix=similarity_matrix,
#             means=means,
#             knn_k=knn_k,
#             expected_ratings=empty_array,
#             expected_similarity=empty_array,
#             expected_means=empty_array,
#         )

#         x = 1
#         y = 2
#         users_neighborhood = np.array([2])
#         TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#             x=x,
#             y=y,
#             dataset=non_square_dataset,
#             users_neighborhood=users_neighborhood,
#             similarity_matrix=similarity_matrix,
#             means=means,
#             knn_k=1,
#             expected_ratings=empty_array,
#             expected_similarity=empty_array,
#             expected_means=empty_array,
#         )

#         # fmt: off
#         non_square_dataset = np.array([[ 1,  2,  3],
#                                        [ 4 , 5,  6],
#                                        [ 7,  8,  9],
#                                        [10, 11, 12]],
#                                        dtype=np.float64)
        
#         similarity_matrix = np.array([[1.0, 0.0, 0.0, 0.0],
#                                       [0.0, 1.0, 0.0, 0.0],
#                                       [0.0, 0.0, 1.0, 0.0],
#                                       [0.0, 0.0, 0.0, 1.0]])
        
#         means = np.array([2.0, 5.0, 8.0, 1.0])
#         # fmt: on

#         x = 0
#         y = 0
#         users_neighborhood = np.array([1, 2, 3])
#         TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#             x=x,
#             y=y,
#             dataset=non_square_dataset,
#             users_neighborhood=users_neighborhood,
#             similarity_matrix=similarity_matrix,
#             means=means,
#             knn_k=knn_k,
#             expected_ratings=empty_array,
#             expected_similarity=empty_array,
#             expected_means=empty_array,
#         )

#         x = 0
#         y = 1
#         TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#             x=x,
#             y=y,
#             dataset=non_square_dataset,
#             users_neighborhood=users_neighborhood,
#             similarity_matrix=similarity_matrix,
#             means=means,
#             knn_k=knn_k,
#             expected_ratings=empty_array,
#             expected_similarity=empty_array,
#             expected_means=empty_array,
#         )

#         knn_k = 1
#         TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#             x=x,
#             y=y,
#             dataset=non_square_dataset,
#             users_neighborhood=users_neighborhood,
#             similarity_matrix=similarity_matrix,
#             means=means,
#             knn_k=knn_k,
#             expected_ratings=empty_array,
#             expected_similarity=empty_array,
#             expected_means=empty_array,
#         )

#         knn_k = 3
#         x = 0
#         y = 2
#         users_neighborhood = np.array([1, 3])
#         TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#             x=x,
#             y=y,
#             dataset=non_square_dataset,
#             users_neighborhood=users_neighborhood,
#             similarity_matrix=similarity_matrix,
#             means=means,
#             knn_k=3,
#             expected_ratings=empty_array,
#             expected_similarity=empty_array,
#             expected_means=empty_array,
#         )

#         x = 0
#         y = 2
#         users_neighborhood = np.array([1, 2, 3])
#         TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#             x=x,
#             y=y,
#             dataset=non_square_dataset,
#             users_neighborhood=users_neighborhood,
#             similarity_matrix=similarity_matrix,
#             means=means,
#             knn_k=3,
#             expected_ratings=empty_array,
#             expected_similarity=empty_array,
#             expected_means=empty_array,
#         )

#         x = 0
#         y = 2
#         users_neighborhood = np.array([1, 2, 3])
#         knn_k = 2
#         TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#             x=x,
#             y=y,
#             dataset=non_square_dataset,
#             users_neighborhood=users_neighborhood,
#             similarity_matrix=similarity_matrix,
#             means=means,
#             knn_k=2,
#             expected_ratings=empty_array,
#             expected_similarity=empty_array,
#             expected_means=empty_array,
#         )

#     def test_no_valid_neighbors(self):
#         # fmt: off
#         dataset = np.array([[ 1, np.nan,  3],
#                             [ 4, np.nan,  6],
#                             [ 7, np.nan,  9]],
#                             dtype=np.float64)
        
#         similarity_matrix = np.array([[1.0, 0.5, 0.2],
#                                       [0.5, 1.0, 0.8],
#                                       [0.2, 0.8, 1.0]],
#                                       dtype=np.float64)
#         # fmt: on

#         empty_array = np.array([])

#         users_neighborhood = np.array([1, 2])
#         means = np.array([2.0, 5.0, 8.0])
#         knn_k = 3

#         x = 0
#         y = 1
#         TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#             x=x,
#             y=y,
#             dataset=dataset,
#             users_neighborhood=users_neighborhood,
#             similarity_matrix=similarity_matrix,
#             means=means,
#             knn_k=knn_k,
#             expected_ratings=empty_array,
#             expected_similarity=empty_array,
#             expected_means=empty_array,
#         )

#         users_neighborhood = np.array([2])
#         TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#             x=x,
#             y=y,
#             dataset=dataset,
#             users_neighborhood=users_neighborhood,
#             similarity_matrix=similarity_matrix,
#             means=means,
#             knn_k=knn_k,
#             expected_ratings=empty_array,
#             expected_similarity=empty_array,
#             expected_means=empty_array,
#         )

#     class TestUserScenario:
#         class TestNoNaNs:
#             class TestNoZeroSimilarity:
#                 def test_1(self):
#                     x = 0
#                     y = 0

#                     users_neighborhood = np.array([1, 2])
#                     knn_k = 3

#                     expected_ratings = np.array([9.0, 5.0])
#                     expected_similarity = np.array([1.0, 0.5])
#                     expected_means = np.array([3.0, 2.0])

#                     TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                         x=x,
#                         y=y,
#                         dataset=TestGetKTopNeighbors.square_dataset,
#                         users_neighborhood=users_neighborhood,
#                         similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
#                         means=TestGetKTopNeighbors.means,
#                         knn_k=knn_k,
#                         expected_ratings=expected_ratings,
#                         expected_similarity=expected_similarity,
#                         expected_means=expected_means,
#                     )

#                 def test_2(self):
#                     x = 0
#                     y = 0

#                     users_neighborhood = np.array([1, 2, 3])
#                     knn_k = 3

#                     expected_ratings = np.array([9.0, 5.0, 13.0])
#                     expected_similarity = np.array([1.0, 0.5, 0.1])
#                     expected_means = np.array([3.0, 2.0, 4.0])

#                     TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                         x=x,
#                         y=y,
#                         dataset=TestGetKTopNeighbors.square_dataset,
#                         users_neighborhood=users_neighborhood,
#                         similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
#                         means=TestGetKTopNeighbors.means,
#                         knn_k=knn_k,
#                         expected_ratings=expected_ratings,
#                         expected_similarity=expected_similarity,
#                         expected_means=expected_means,
#                     )

#                 def test_3(self):
#                     x = 0
#                     y = 0

#                     users_neighborhood = np.array([1, 2, 3])
#                     knn_k = 1

#                     expected_ratings = np.array([9.0])
#                     expected_similarity = np.array([1.0])
#                     expected_means = np.array([3.0])

#                     TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                         x=x,
#                         y=y,
#                         dataset=TestGetKTopNeighbors.square_dataset,
#                         users_neighborhood=users_neighborhood,
#                         similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
#                         means=TestGetKTopNeighbors.means,
#                         knn_k=knn_k,
#                         expected_ratings=expected_ratings,
#                         expected_similarity=expected_similarity,
#                         expected_means=expected_means,
#                     )

#                 def test_4(self):
#                     x = 1
#                     y = 2

#                     users_neighborhood = np.array([0, 2, 3])
#                     knn_k = 1

#                     expected_ratings = np.array([11.0])
#                     expected_similarity = np.array([0.7])
#                     expected_means = np.array([3.0])

#                     TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                         x=x,
#                         y=y,
#                         dataset=TestGetKTopNeighbors.square_dataset,
#                         users_neighborhood=users_neighborhood,
#                         similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
#                         means=TestGetKTopNeighbors.means,
#                         knn_k=knn_k,
#                         expected_ratings=expected_ratings,
#                         expected_similarity=expected_similarity,
#                         expected_means=expected_means,
#                     )

#                 def test_5(self):
#                     x = 1
#                     y = 3

#                     users_neighborhood = np.array([0, 2, 3])
#                     knn_k = 1

#                     expected_ratings = np.array([12.0])
#                     expected_similarity = np.array([0.7])
#                     expected_means = np.array([3.0])

#                     TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                         x=x,
#                         y=y,
#                         dataset=TestGetKTopNeighbors.square_dataset,
#                         users_neighborhood=users_neighborhood,
#                         similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
#                         means=TestGetKTopNeighbors.means,
#                         knn_k=knn_k,
#                         expected_ratings=expected_ratings,
#                         expected_similarity=expected_similarity,
#                         expected_means=expected_means,
#                     )

#                 def test_6(self):
#                     x = 1
#                     y = 3

#                     users_neighborhood = np.array([0, 2])
#                     knn_k = 1

#                     expected_ratings = np.array([12.0])
#                     expected_similarity = np.array([0.7])
#                     expected_means = np.array([3.0])

#                     TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                         x=x,
#                         y=y,
#                         dataset=TestGetKTopNeighbors.square_dataset,
#                         users_neighborhood=users_neighborhood,
#                         similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
#                         means=TestGetKTopNeighbors.means,
#                         knn_k=knn_k,
#                         expected_ratings=expected_ratings,
#                         expected_similarity=expected_similarity,
#                         expected_means=expected_means,
#                     )

#                 def test_7(self):
#                     x = 1
#                     y = 3

#                     users_neighborhood = np.array([0])
#                     knn_k = 2

#                     expected_ratings = np.array([4.0])
#                     expected_similarity = np.array([0.5])
#                     expected_means = np.array([1.0])

#                     TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                         x=x,
#                         y=y,
#                         dataset=TestGetKTopNeighbors.square_dataset,
#                         users_neighborhood=users_neighborhood,
#                         similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
#                         means=TestGetKTopNeighbors.means,
#                         knn_k=knn_k,
#                         expected_ratings=expected_ratings,
#                         expected_similarity=expected_similarity,
#                         expected_means=expected_means,
#                     )

#                 def test_8(self):
#                     x = 0
#                     y = 1

#                     users_neighborhood = np.array([1, 2, 3])
#                     knn_k = 1

#                     expected_ratings = np.array([10.0])
#                     expected_similarity = np.array([1.0])
#                     expected_means = np.array([3.0])

#                     TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                         x=x,
#                         y=y,
#                         dataset=TestGetKTopNeighbors.square_dataset,
#                         users_neighborhood=users_neighborhood,
#                         similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
#                         means=TestGetKTopNeighbors.means,
#                         knn_k=knn_k,
#                         expected_ratings=expected_ratings,
#                         expected_similarity=expected_similarity,
#                         expected_means=expected_means,
#                     )

#             class TestWithZeroSimilarity:
#                 def test_1(self):
#                     x = 2
#                     y = 1

#                     users_neighborhood = np.array([0, 1, 3])
#                     knn_k = 1

#                     expected_ratings = np.array([2.0])
#                     expected_similarity = np.array([1.0])
#                     expected_means = np.array([1.0])

#                     TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                         x=x,
#                         y=y,
#                         dataset=TestGetKTopNeighbors.square_dataset,
#                         users_neighborhood=users_neighborhood,
#                         similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
#                         means=TestGetKTopNeighbors.means,
#                         knn_k=knn_k,
#                         expected_ratings=expected_ratings,
#                         expected_similarity=expected_similarity,
#                         expected_means=expected_means,
#                     )

#                 def test_2(self):
#                     x = 2
#                     y = 2

#                     users_neighborhood = np.array([0, 1, 3])
#                     knn_k = 2

#                     expected_ratings = np.array([3.0, 7.0])
#                     expected_similarity = np.array([1.0, 0.7])
#                     expected_means = np.array([1.0, 2.0])

#                     TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                         x=x,
#                         y=y,
#                         dataset=TestGetKTopNeighbors.square_dataset,
#                         users_neighborhood=users_neighborhood,
#                         similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
#                         means=TestGetKTopNeighbors.means,
#                         knn_k=knn_k,
#                         expected_ratings=expected_ratings,
#                         expected_similarity=expected_similarity,
#                         expected_means=expected_means,
#                     )

#                 def test_3(self):
#                     x = 2
#                     y = 2

#                     users_neighborhood = np.array([0, 1, 3])
#                     knn_k = 3

#                     expected_ratings = np.array([3.0, 7.0])
#                     expected_similarity = np.array([1.0, 0.7])
#                     expected_means = np.array([1.0, 2.0])

#                     TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                         x=x,
#                         y=y,
#                         dataset=TestGetKTopNeighbors.square_dataset,
#                         users_neighborhood=users_neighborhood,
#                         similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
#                         means=TestGetKTopNeighbors.means,
#                         knn_k=knn_k,
#                         expected_ratings=expected_ratings,
#                         expected_similarity=expected_similarity,
#                         expected_means=expected_means,
#                     )

#                 def test_4(self):
#                     x = 2
#                     y = 3

#                     users_neighborhood = np.array([0, 1, 3])
#                     knn_k = 3

#                     expected_ratings = np.array([4.0, 8.0])
#                     expected_similarity = np.array([1.0, 0.7])
#                     expected_means = np.array([1.0, 2.0])

#                     TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                         x=x,
#                         y=y,
#                         dataset=TestGetKTopNeighbors.square_dataset,
#                         users_neighborhood=users_neighborhood,
#                         similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
#                         means=TestGetKTopNeighbors.means,
#                         knn_k=knn_k,
#                         expected_ratings=expected_ratings,
#                         expected_similarity=expected_similarity,
#                         expected_means=expected_means,
#                     )

#                 def test_5(self):
#                     x = 2
#                     y = 3

#                     users_neighborhood = np.array([3])
#                     knn_k = 3

#                     expected_ratings = np.array([])
#                     expected_similarity = np.array([])
#                     expected_means = np.array([])

#                     TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                         x=x,
#                         y=y,
#                         dataset=TestGetKTopNeighbors.square_dataset,
#                         users_neighborhood=users_neighborhood,
#                         similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
#                         means=TestGetKTopNeighbors.means,
#                         knn_k=knn_k,
#                         expected_ratings=expected_ratings,
#                         expected_similarity=expected_similarity,
#                         expected_means=expected_means,
#                     )

#         class TestWithNaNsOnDataset:
#             class TestNoZeroSimilarity:
#                 # fmt: off
#                 square_dataset = np.array([[ 1,  2,  3,  np.NAN],
#                                            [ 5,  6,  np.NAN,  8],
#                                            [ np.NAN, 10, 11, 12],
#                                            [13, np.NAN, 15, 16]],
#                                            dtype=np.float64)
#                 # fmt: on

#                 def test_1(self):
#                     x = 1
#                     y = 1

#                     users_neighborhood = np.array([0, 2, 3])
#                     knn_k = 2

#                     expected_ratings = np.array([10.0, 2.0])
#                     expected_similarity = np.array([0.7, 0.5])
#                     expected_means = np.array([3.0, 1.0])

#                     TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                         x=x,
#                         y=y,
#                         dataset=self.square_dataset,
#                         users_neighborhood=users_neighborhood,
#                         similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
#                         means=TestGetKTopNeighbors.means,
#                         knn_k=knn_k,
#                         expected_ratings=expected_ratings,
#                         expected_similarity=expected_similarity,
#                         expected_means=expected_means,
#                     )

#                 def test_2(self):
#                     x = 1
#                     y = 0

#                     users_neighborhood = np.array([0, 2, 3])
#                     knn_k = 2

#                     expected_ratings = np.array([1.0, 13.0])
#                     expected_similarity = np.array([0.5, 0.3])
#                     expected_means = np.array([1.0, 4.0])

#                     TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                         x=x,
#                         y=y,
#                         dataset=self.square_dataset,
#                         users_neighborhood=users_neighborhood,
#                         similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
#                         means=TestGetKTopNeighbors.means,
#                         knn_k=knn_k,
#                         expected_ratings=expected_ratings,
#                         expected_similarity=expected_similarity,
#                         expected_means=expected_means,
#                     )

#                 def test_3(self):
#                     x = 1
#                     y = 0

#                     users_neighborhood = np.array([0, 3])
#                     knn_k = 2

#                     expected_ratings = np.array([1.0, 13.0])
#                     expected_similarity = np.array([0.5, 0.3])
#                     expected_means = np.array([1.0, 4.0])

#                     TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                         x=x,
#                         y=y,
#                         dataset=self.square_dataset,
#                         users_neighborhood=users_neighborhood,
#                         similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
#                         means=TestGetKTopNeighbors.means,
#                         knn_k=knn_k,
#                         expected_ratings=expected_ratings,
#                         expected_similarity=expected_similarity,
#                         expected_means=expected_means,
#                     )

#             class TestWithZeroSimilarity:
#                 # fmt: off
#                 square_dataset = np.array([[ 1,  2,  3,  np.NAN],
#                                            [ 5,  6,  np.NAN,  8],
#                                            [ np.NAN, 10, 11, 12],
#                                            [13, np.NAN, 15, 16]],
#                                            dtype=np.float64)
#                 # fmt: on
#                 def test_1(self):
#                     x = 3
#                     y = 3

#                     users_neighborhood = np.array([0, 1, 2])
#                     knn_k = 3

#                     expected_ratings = np.array([8.0])
#                     expected_similarity = np.array([0.3])
#                     expected_means = np.array([2.0])

#                     TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                         x=x,
#                         y=y,
#                         dataset=self.square_dataset,
#                         users_neighborhood=users_neighborhood,
#                         similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
#                         means=TestGetKTopNeighbors.means,
#                         knn_k=knn_k,
#                         expected_ratings=expected_ratings,
#                         expected_similarity=expected_similarity,
#                         expected_means=expected_means,
#                     )

#                 def test_2(self):
#                     x = 3
#                     y = 3

#                     users_neighborhood = np.array([0, 1, 2])
#                     knn_k = 2

#                     expected_ratings = np.array([8.0])
#                     expected_similarity = np.array([0.3])
#                     expected_means = np.array([2.0])

#                     TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                         x=x,
#                         y=y,
#                         dataset=self.square_dataset,
#                         users_neighborhood=users_neighborhood,
#                         similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
#                         means=TestGetKTopNeighbors.means,
#                         knn_k=knn_k,
#                         expected_ratings=expected_ratings,
#                         expected_similarity=expected_similarity,
#                         expected_means=expected_means,
#                     )

#                 def test_3(self):
#                     x = 3
#                     y = 3

#                     users_neighborhood = np.array([0, 1, 2])
#                     knn_k = 1

#                     expected_ratings = np.array([8.0])
#                     expected_similarity = np.array([0.3])
#                     expected_means = np.array([2.0])

#                     TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                         x=x,
#                         y=y,
#                         dataset=self.square_dataset,
#                         users_neighborhood=users_neighborhood,
#                         similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
#                         means=TestGetKTopNeighbors.means,
#                         knn_k=knn_k,
#                         expected_ratings=expected_ratings,
#                         expected_similarity=expected_similarity,
#                         expected_means=expected_means,
#                     )

#                 def test_5(self):
#                     x = 3
#                     y = 3

#                     users_neighborhood = np.array([1])
#                     knn_k = 2

#                     expected_ratings = np.array([8.0])
#                     expected_similarity = np.array([0.3])
#                     expected_means = np.array([2.0])

#                     TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                         x=x,
#                         y=y,
#                         dataset=self.square_dataset,
#                         users_neighborhood=users_neighborhood,
#                         similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
#                         means=TestGetKTopNeighbors.means,
#                         knn_k=knn_k,
#                         expected_ratings=expected_ratings,
#                         expected_similarity=expected_similarity,
#                         expected_means=expected_means,
#                     )

#                 def test_6(self):
#                     x = 2
#                     y = 3

#                     users_neighborhood = np.array([0, 1, 3])
#                     knn_k = 2

#                     expected_ratings = np.array([8.0])
#                     expected_similarity = np.array([0.7])
#                     expected_means = np.array([2.0])

#                     TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                         x=x,
#                         y=y,
#                         dataset=self.square_dataset,
#                         users_neighborhood=users_neighborhood,
#                         similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
#                         means=TestGetKTopNeighbors.means,
#                         knn_k=knn_k,
#                         expected_ratings=expected_ratings,
#                         expected_similarity=expected_similarity,
#                         expected_means=expected_means,
#                     )

#                 def test_7(self):
#                     x = 2
#                     y = 2

#                     users_neighborhood = np.array([0, 1, 3])
#                     knn_k = 2

#                     expected_ratings = np.array([3.0])
#                     expected_similarity = np.array([1.0])
#                     expected_means = np.array([1.0])

#                     TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                         x=x,
#                         y=y,
#                         dataset=self.square_dataset,
#                         users_neighborhood=users_neighborhood,
#                         similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
#                         means=TestGetKTopNeighbors.means,
#                         knn_k=knn_k,
#                         expected_ratings=expected_ratings,
#                         expected_similarity=expected_similarity,
#                         expected_means=expected_means,
#                     )

#         class TestWithNaNsOnSimilarity:
#             class TestNoZeroSimilarity:
#                 # fmt: off
#                 similarity_matrix = np.array([[1.0, np.NAN, 1.0, 0.1],
#                                               [0.5, 1.0, np.NAN, 0.3],
#                                               [1.0, 0.7, np.NAN, 0.0],
#                                               [np.NAN, 0.3, 0.0, 1.0]],
#                                               dtype=np.float64)
#                 # fmt: on

#                 def test_1(self):
#                     x = 0
#                     y = 0

#                     users_neighborhood = np.array([1, 2, 3])
#                     knn_k = 2

#                     expected_ratings = np.array([9.0, 13.0])
#                     expected_similarity = np.array([1.0, 0.1])
#                     expected_means = np.array([3.0, 4.0])

#                     TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                         x=x,
#                         y=y,
#                         dataset=TestGetKTopNeighbors.square_dataset,
#                         users_neighborhood=users_neighborhood,
#                         similarity_matrix=self.similarity_matrix,
#                         means=TestGetKTopNeighbors.means,
#                         knn_k=knn_k,
#                         expected_ratings=expected_ratings,
#                         expected_similarity=expected_similarity,
#                         expected_means=expected_means,
#                     )

#                 def test_2(self):
#                     x = 0
#                     y = 0

#                     users_neighborhood = np.array([1, 2, 3])
#                     knn_k = 1

#                     expected_ratings = np.array([9.0])
#                     expected_similarity = np.array([1.0])
#                     expected_means = np.array([3.0])

#                     TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                         x=x,
#                         y=y,
#                         dataset=TestGetKTopNeighbors.square_dataset,
#                         users_neighborhood=users_neighborhood,
#                         similarity_matrix=self.similarity_matrix,
#                         means=TestGetKTopNeighbors.means,
#                         knn_k=knn_k,
#                         expected_ratings=expected_ratings,
#                         expected_similarity=expected_similarity,
#                         expected_means=expected_means,
#                     )

#                 def test_3(self):
#                     x = 0
#                     y = 0

#                     users_neighborhood = np.array([1, 2, 3])
#                     knn_k = 3

#                     expected_ratings = np.array([9.0, 13.0])
#                     expected_similarity = np.array([1.0, 0.1])
#                     expected_means = np.array([3.0, 4.0])

#                     TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                         x=x,
#                         y=y,
#                         dataset=TestGetKTopNeighbors.square_dataset,
#                         users_neighborhood=users_neighborhood,
#                         similarity_matrix=self.similarity_matrix,
#                         means=TestGetKTopNeighbors.means,
#                         knn_k=knn_k,
#                         expected_ratings=expected_ratings,
#                         expected_similarity=expected_similarity,
#                         expected_means=expected_means,
#                     )

#                 def test_4(self):
#                     x = 0
#                     y = 1

#                     users_neighborhood = np.array([1, 2, 3])
#                     knn_k = 2

#                     expected_ratings = np.array([10.0, 14.0])
#                     expected_similarity = np.array([1.0, 0.1])
#                     expected_means = np.array([3.0, 4.0])

#                     TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                         x=x,
#                         y=y,
#                         dataset=TestGetKTopNeighbors.square_dataset,
#                         users_neighborhood=users_neighborhood,
#                         similarity_matrix=self.similarity_matrix,
#                         means=TestGetKTopNeighbors.means,
#                         knn_k=knn_k,
#                         expected_ratings=expected_ratings,
#                         expected_similarity=expected_similarity,
#                         expected_means=expected_means,
#                     )

#             class TestWithZeroSimilarity:
#                 # fmt: off
#                 similarity_matrix = np.array([[1.0, np.NAN, 1.0, 0.1],
#                                               [0.5, 1.0, np.NAN, 0.3],
#                                               [1.0, 0.7, np.NAN, 0.0],
#                                               [np.NAN, 0.3, 0.0, 1.0]],
#                                               dtype=np.float64)
#                 # fmt: on
#                 def test_1(self):
#                     x = 3
#                     y = 3

#                     users_neighborhood = np.array([0, 1, 2])
#                     knn_k = 2

#                     expected_ratings = np.array([8.0])
#                     expected_similarity = np.array([0.3])
#                     expected_means = np.array([2.0])

#                     TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                         x=x,
#                         y=y,
#                         dataset=TestGetKTopNeighbors.square_dataset,
#                         similarity_matrix=self.similarity_matrix,
#                         users_neighborhood=users_neighborhood,
#                         means=TestGetKTopNeighbors.means,
#                         knn_k=knn_k,
#                         expected_ratings=expected_ratings,
#                         expected_similarity=expected_similarity,
#                         expected_means=expected_means,
#                     )

#                 def test_2(self):
#                     x = 3
#                     y = 3

#                     users_neighborhood = np.array([0, 1, 2])
#                     knn_k = 1

#                     expected_ratings = np.array([8.0])
#                     expected_similarity = np.array([0.3])
#                     expected_means = np.array([2.0])

#                     TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                         x=x,
#                         y=y,
#                         dataset=TestGetKTopNeighbors.square_dataset,
#                         similarity_matrix=self.similarity_matrix,
#                         users_neighborhood=users_neighborhood,
#                         means=TestGetKTopNeighbors.means,
#                         knn_k=knn_k,
#                         expected_ratings=expected_ratings,
#                         expected_similarity=expected_similarity,
#                         expected_means=expected_means,
#                     )

#                 def test_3(self):
#                     x = 3
#                     y = 1

#                     users_neighborhood = np.array([0, 1, 2])
#                     knn_k = 3

#                     expected_ratings = np.array([6.0])
#                     expected_similarity = np.array([0.3])
#                     expected_means = np.array([2.0])

#                     TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                         x=x,
#                         y=y,
#                         dataset=TestGetKTopNeighbors.square_dataset,
#                         similarity_matrix=self.similarity_matrix,
#                         users_neighborhood=users_neighborhood,
#                         means=TestGetKTopNeighbors.means,
#                         knn_k=knn_k,
#                         expected_ratings=expected_ratings,
#                         expected_similarity=expected_similarity,
#                         expected_means=expected_means,
#                     )

#         class TestWithNaNsOnDatasetAndSimilarity:
#             # fmt: off
#             square_dataset = np.array([[ 1,  2,  3,  np.NAN],
#                                        [ 5,  6,  np.NAN,  8],
#                                        [ np.NAN, 10, 11, 12],
#                                        [13, np.NAN, 15, 16]],
#                                        dtype=np.float64)
#             similarity_matrix = np.array([[1.0, np.NAN, 1.0, 0.1],
#                                           [0.5, 1.0, np.NAN, 0.3],
#                                           [1.0, 0.7, np.NAN, 0.0],
#                                           [np.NAN, 0.3, 0.0, 1.0]],
#                                           dtype=np.float64)
#             # fmt: on

#             def test_1(self):
#                 x = 1
#                 y = 1

#                 users_neighborhood = np.array([0, 2, 3])
#                 knn_k = 2

#                 expected_ratings = np.array([2.0])
#                 expected_similarity = np.array([0.5])
#                 expected_means = np.array([1.0])

#                 TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                     x=x,
#                     y=y,
#                     dataset=self.square_dataset,
#                     similarity_matrix=self.similarity_matrix,
#                     users_neighborhood=users_neighborhood,
#                     means=TestGetKTopNeighbors.means,
#                     knn_k=knn_k,
#                     expected_ratings=expected_ratings,
#                     expected_similarity=expected_similarity,
#                     expected_means=expected_means,
#                 )

#             def test_2(self):
#                 x = 1
#                 y = 1

#                 users_neighborhood = np.array([2, 3])
#                 knn_k = 2

#                 expected_ratings = np.array([])
#                 expected_similarity = np.array([])
#                 expected_means = np.array([])

#                 TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                     x=x,
#                     y=y,
#                     dataset=self.square_dataset,
#                     similarity_matrix=self.similarity_matrix,
#                     users_neighborhood=users_neighborhood,
#                     means=TestGetKTopNeighbors.means,
#                     knn_k=knn_k,
#                     expected_ratings=expected_ratings,
#                     expected_similarity=expected_similarity,
#                     expected_means=expected_means,
#                 )

#             def test_3(self):
#                 x = 1
#                 y = 3

#                 users_neighborhood = np.array([2, 3])
#                 knn_k = 2

#                 expected_ratings = np.array([16.0])
#                 expected_similarity = np.array([0.3])
#                 expected_means = np.array([4.0])

#                 TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                     x=x,
#                     y=y,
#                     dataset=self.square_dataset,
#                     similarity_matrix=self.similarity_matrix,
#                     users_neighborhood=users_neighborhood,
#                     means=TestGetKTopNeighbors.means,
#                     knn_k=knn_k,
#                     expected_ratings=expected_ratings,
#                     expected_similarity=expected_similarity,
#                     expected_means=expected_means,
#                 )

#             def test_4(self):
#                 x = 1
#                 y = 3

#                 users_neighborhood = np.array([3])
#                 knn_k = 2

#                 expected_ratings = np.array([16.0])
#                 expected_similarity = np.array([0.3])
#                 expected_means = np.array([4.0])

#                 TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                     x=x,
#                     y=y,
#                     dataset=self.square_dataset,
#                     similarity_matrix=self.similarity_matrix,
#                     users_neighborhood=users_neighborhood,
#                     means=TestGetKTopNeighbors.means,
#                     knn_k=knn_k,
#                     expected_ratings=expected_ratings,
#                     expected_similarity=expected_similarity,
#                     expected_means=expected_means,
#                 )

#             def test_5(self):
#                 x = 1
#                 y = 3

#                 users_neighborhood = np.array([2])
#                 knn_k = 2

#                 expected_ratings = np.array([])
#                 expected_similarity = np.array([])
#                 expected_means = np.array([])

#                 TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                     x=x,
#                     y=y,
#                     dataset=self.square_dataset,
#                     similarity_matrix=self.similarity_matrix,
#                     users_neighborhood=users_neighborhood,
#                     means=TestGetKTopNeighbors.means,
#                     knn_k=knn_k,
#                     expected_ratings=expected_ratings,
#                     expected_similarity=expected_similarity,
#                     expected_means=expected_means,
#                 )

#             def test_6(self):
#                 x = 2
#                 y = 1

#                 users_neighborhood = np.array([0, 1, 3])
#                 knn_k = 2

#                 expected_ratings = np.array([2.0, 6.0])
#                 expected_similarity = np.array([1.0, 0.7])
#                 expected_means = np.array([1.0, 2.0])

#                 TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                     x=x,
#                     y=y,
#                     dataset=self.square_dataset,
#                     similarity_matrix=self.similarity_matrix,
#                     users_neighborhood=users_neighborhood,
#                     means=TestGetKTopNeighbors.means,
#                     knn_k=knn_k,
#                     expected_ratings=expected_ratings,
#                     expected_similarity=expected_similarity,
#                     expected_means=expected_means,
#                 )

#             def test_7(self):
#                 x = 2
#                 y = 1

#                 users_neighborhood = np.array([1, 3])
#                 knn_k = 2

#                 expected_ratings = np.array([6.0])
#                 expected_similarity = np.array([0.7])
#                 expected_means = np.array([2.0])

#                 TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                     x=x,
#                     y=y,
#                     dataset=self.square_dataset,
#                     similarity_matrix=self.similarity_matrix,
#                     users_neighborhood=users_neighborhood,
#                     means=TestGetKTopNeighbors.means,
#                     knn_k=knn_k,
#                     expected_ratings=expected_ratings,
#                     expected_similarity=expected_similarity,
#                     expected_means=expected_means,
#                 )

#             def test_8(self):
#                 x = 2
#                 y = 3

#                 users_neighborhood = np.array([0, 1, 3])
#                 knn_k = 2

#                 expected_ratings = np.array([8.0])
#                 expected_similarity = np.array([0.7])
#                 expected_means = np.array([2.0])

#                 TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                     x=x,
#                     y=y,
#                     dataset=self.square_dataset,
#                     similarity_matrix=self.similarity_matrix,
#                     users_neighborhood=users_neighborhood,
#                     means=TestGetKTopNeighbors.means,
#                     knn_k=knn_k,
#                     expected_ratings=expected_ratings,
#                     expected_similarity=expected_similarity,
#                     expected_means=expected_means,
#                 )

#             def test_9(self):
#                 x = 2
#                 y = 3

#                 users_neighborhood = np.array([0, 3])
#                 knn_k = 2

#                 expected_ratings = np.array([])
#                 expected_similarity = np.array([])
#                 expected_means = np.array([])

#                 TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                     x=x,
#                     y=y,
#                     dataset=self.square_dataset,
#                     similarity_matrix=self.similarity_matrix,
#                     users_neighborhood=users_neighborhood,
#                     means=TestGetKTopNeighbors.means,
#                     knn_k=knn_k,
#                     expected_ratings=expected_ratings,
#                     expected_similarity=expected_similarity,
#                     expected_means=expected_means,
#                 )

#     class TestItemScenario:
#         class TestNoNaNs:
#             class TestNoZeroSimilarity:
#                 def test_1(self):
#                     x = 0
#                     y = 0

#                     users_neighborhood = np.array([1, 2, 3])
#                     knn_k = 3

#                     expected_ratings = np.array([3.0, 2.0, 4.0])
#                     expected_similarity = np.array([1.0, 0.5, 0.1])
#                     expected_means = np.array([3.0, 2.0, 4.0])

#                     TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                         x=y,
#                         y=x,
#                         dataset=TestGetKTopNeighbors.square_dataset.T,
#                         users_neighborhood=users_neighborhood,
#                         similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
#                         means=TestGetKTopNeighbors.means,
#                         knn_k=knn_k,
#                         expected_ratings=expected_ratings,
#                         expected_similarity=expected_similarity,
#                         expected_means=expected_means,
#                     )

#                 def test_2(self):
#                     x = 0
#                     y = 0

#                     users_neighborhood = np.array([1, 2, 3])
#                     knn_k = 1

#                     expected_ratings = np.array([3.0])
#                     expected_similarity = np.array([1.0])
#                     expected_means = np.array([3.0])

#                     TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                         x=y,
#                         y=x,
#                         dataset=TestGetKTopNeighbors.square_dataset.T,
#                         users_neighborhood=users_neighborhood,
#                         similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
#                         means=TestGetKTopNeighbors.means,
#                         knn_k=knn_k,
#                         expected_ratings=expected_ratings,
#                         expected_similarity=expected_similarity,
#                         expected_means=expected_means,
#                     )

#                 def test_3(self):
#                     x = 0
#                     y = 0

#                     users_neighborhood = np.array([1, 2, 3])
#                     knn_k = 2

#                     expected_ratings = np.array([3.0, 2.0])
#                     expected_similarity = np.array([1.0, 0.5])
#                     expected_means = np.array([3.0, 2.0])

#                     TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                         x=y,
#                         y=x,
#                         dataset=TestGetKTopNeighbors.square_dataset.T,
#                         users_neighborhood=users_neighborhood,
#                         similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
#                         means=TestGetKTopNeighbors.means,
#                         knn_k=knn_k,
#                         expected_ratings=expected_ratings,
#                         expected_similarity=expected_similarity,
#                         expected_means=expected_means,
#                     )

#                 def test_4(self):
#                     x = 0
#                     y = 0

#                     users_neighborhood = np.array([1, 3])
#                     knn_k = 2

#                     expected_ratings = np.array([2.0, 4.0])
#                     expected_similarity = np.array([0.5, 0.1])
#                     expected_means = np.array([2.0, 4.0])

#                     TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                         x=y,
#                         y=x,
#                         dataset=TestGetKTopNeighbors.square_dataset.T,
#                         users_neighborhood=users_neighborhood,
#                         similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
#                         means=TestGetKTopNeighbors.means,
#                         knn_k=knn_k,
#                         expected_ratings=expected_ratings,
#                         expected_similarity=expected_similarity,
#                         expected_means=expected_means,
#                     )

#                 def test_5(self):
#                     x = 3
#                     y = 1

#                     users_neighborhood = np.array([0, 2, 3])
#                     knn_k = 3

#                     expected_ratings = np.array([15.0, 13.0, 16.0])
#                     expected_similarity = np.array([0.7, 0.5, 0.3])
#                     expected_means = np.array([3.0, 1.0, 4.0])

#                     TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                         x=y,
#                         y=x,
#                         dataset=TestGetKTopNeighbors.square_dataset.T,
#                         users_neighborhood=users_neighborhood,
#                         similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
#                         means=TestGetKTopNeighbors.means,
#                         knn_k=knn_k,
#                         expected_ratings=expected_ratings,
#                         expected_similarity=expected_similarity,
#                         expected_means=expected_means,
#                     )

#                 def test_6(self):
#                     x = 3
#                     y = 1

#                     users_neighborhood = np.array([0, 2, 3])
#                     knn_k = 1

#                     expected_ratings = np.array([15.0])
#                     expected_similarity = np.array([0.7])
#                     expected_means = np.array([3.0])

#                     TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                         x=y,
#                         y=x,
#                         dataset=TestGetKTopNeighbors.square_dataset.T,
#                         users_neighborhood=users_neighborhood,
#                         similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
#                         means=TestGetKTopNeighbors.means,
#                         knn_k=knn_k,
#                         expected_ratings=expected_ratings,
#                         expected_similarity=expected_similarity,
#                         expected_means=expected_means,
#                     )

#                 def test_7(self):
#                     x = 3
#                     y = 1

#                     users_neighborhood = np.array([0, 2, 3])
#                     knn_k = 2

#                     expected_ratings = np.array([15.0, 13.0])
#                     expected_similarity = np.array([0.7, 0.5])
#                     expected_means = np.array([3.0, 1.0])

#                     TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                         x=y,
#                         y=x,
#                         dataset=TestGetKTopNeighbors.square_dataset.T,
#                         users_neighborhood=users_neighborhood,
#                         similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
#                         means=TestGetKTopNeighbors.means,
#                         knn_k=knn_k,
#                         expected_ratings=expected_ratings,
#                         expected_similarity=expected_similarity,
#                         expected_means=expected_means,
#                     )

#                 def test_8(self):
#                     x = 3
#                     y = 1

#                     users_neighborhood = np.array([0, 3])
#                     knn_k = 2

#                     expected_ratings = np.array([13.0, 16.0])
#                     expected_similarity = np.array([0.5, 0.3])
#                     expected_means = np.array([1.0, 4.0])

#                     TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                         x=y,
#                         y=x,
#                         dataset=TestGetKTopNeighbors.square_dataset.T,
#                         users_neighborhood=users_neighborhood,
#                         similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
#                         means=TestGetKTopNeighbors.means,
#                         knn_k=knn_k,
#                         expected_ratings=expected_ratings,
#                         expected_similarity=expected_similarity,
#                         expected_means=expected_means,
#                     )

#             class TestWithZeroSimilarity:
#                 def test_1(self):
#                     x = 3
#                     y = 3

#                     users_neighborhood = np.array([0, 1, 2])
#                     knn_k = 3

#                     expected_ratings = np.array([14.0, 13.0])
#                     expected_similarity = np.array([0.3, 0.1])
#                     expected_means = np.array([2.0, 1.0])

#                     TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                         x=y,
#                         y=x,
#                         dataset=TestGetKTopNeighbors.square_dataset.T,
#                         similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
#                         users_neighborhood=users_neighborhood,
#                         means=TestGetKTopNeighbors.means,
#                         knn_k=knn_k,
#                         expected_ratings=expected_ratings,
#                         expected_similarity=expected_similarity,
#                         expected_means=expected_means,
#                     )

#                 def test_2(self):
#                     x = 3
#                     y = 3

#                     users_neighborhood = np.array([0, 1, 2])
#                     knn_k = 1

#                     expected_ratings = np.array([14.0])
#                     expected_similarity = np.array([0.3])
#                     expected_means = np.array([2.0])

#                     TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                         x=y,
#                         y=x,
#                         dataset=TestGetKTopNeighbors.square_dataset.T,
#                         similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
#                         users_neighborhood=users_neighborhood,
#                         means=TestGetKTopNeighbors.means,
#                         knn_k=knn_k,
#                         expected_ratings=expected_ratings,
#                         expected_similarity=expected_similarity,
#                         expected_means=expected_means,
#                     )

#                 def test_3(self):
#                     x = 3
#                     y = 3

#                     users_neighborhood = np.array([0, 2])
#                     knn_k = 3

#                     expected_ratings = np.array([13.0])
#                     expected_similarity = np.array([0.1])
#                     expected_means = np.array([1.0])

#                     TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                         x=y,
#                         y=x,
#                         dataset=TestGetKTopNeighbors.square_dataset.T,
#                         similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
#                         users_neighborhood=users_neighborhood,
#                         means=TestGetKTopNeighbors.means,
#                         knn_k=knn_k,
#                         expected_ratings=expected_ratings,
#                         expected_similarity=expected_similarity,
#                         expected_means=expected_means,
#                     )

#         class TestWithNaNsOnDataset:
#             class TestNoZeroSimilarity:
#                 # fmt: off
#                 square_dataset = np.array([[ 1,  2,  3,  np.NAN],
#                                            [ 5,  6,  np.NAN,  8],
#                                            [ np.NAN, 10, 11, 12],
#                                            [13, np.NAN, 15, 16]],
#                                            dtype=np.float64)
#                 # fmt: on

#                 def test_1(self):
#                     x = 1
#                     y = 1

#                     users_neighborhood = np.array([0, 2, 3])
#                     knn_k = 2

#                     expected_ratings = np.array([5.0, 8.0])
#                     expected_similarity = np.array([0.5, 0.3])
#                     expected_means = np.array([1.0, 4.0])

#                     TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                         x=y,
#                         y=x,
#                         dataset=self.square_dataset.T,
#                         users_neighborhood=users_neighborhood,
#                         similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
#                         means=TestGetKTopNeighbors.means,
#                         knn_k=knn_k,
#                         expected_ratings=expected_ratings,
#                         expected_similarity=expected_similarity,
#                         expected_means=expected_means,
#                     )

#                 def test_2(self):
#                     x = 1
#                     y = 1

#                     users_neighborhood = np.array([0, 2, 3])
#                     knn_k = 1

#                     expected_ratings = np.array(
#                         [
#                             5.0,
#                         ]
#                     )
#                     expected_similarity = np.array([0.5])
#                     expected_means = np.array([1.0])

#                     TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                         x=y,
#                         y=x,
#                         dataset=self.square_dataset.T,
#                         users_neighborhood=users_neighborhood,
#                         similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
#                         means=TestGetKTopNeighbors.means,
#                         knn_k=knn_k,
#                         expected_ratings=expected_ratings,
#                         expected_similarity=expected_similarity,
#                         expected_means=expected_means,
#                     )

#                 def test_3(self):
#                     x = 1
#                     y = 1

#                     users_neighborhood = np.array([0, 3])
#                     knn_k = 2

#                     expected_ratings = np.array([5.0, 8.0])
#                     expected_similarity = np.array([0.5, 0.3])
#                     expected_means = np.array([1.0, 4.0])

#                     TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                         x=y,
#                         y=x,
#                         dataset=self.square_dataset.T,
#                         users_neighborhood=users_neighborhood,
#                         similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
#                         means=TestGetKTopNeighbors.means,
#                         knn_k=knn_k,
#                         expected_ratings=expected_ratings,
#                         expected_similarity=expected_similarity,
#                         expected_means=expected_means,
#                     )

#                 def test_4(self):
#                     x = 1
#                     y = 1

#                     users_neighborhood = np.array([3])
#                     knn_k = 2

#                     expected_ratings = np.array([8.0])
#                     expected_similarity = np.array([0.3])
#                     expected_means = np.array([4.0])

#                     TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                         x=y,
#                         y=x,
#                         dataset=self.square_dataset.T,
#                         users_neighborhood=users_neighborhood,
#                         similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
#                         means=TestGetKTopNeighbors.means,
#                         knn_k=knn_k,
#                         expected_ratings=expected_ratings,
#                         expected_similarity=expected_similarity,
#                         expected_means=expected_means,
#                     )

#             class TestWithZeroSimilarity:
#                 # fmt: off
#                 square_dataset = np.array([[ 1,  2,  3,  np.NAN],
#                                            [ 5,  6,  np.NAN,  8],
#                                            [ np.NAN, 10, 11, 12],
#                                            [13, np.NAN, 15, 16]],
#                                            dtype=np.float64)
#                 # fmt: on

#                 def test_1(self):
#                     x = 3
#                     y = 3

#                     users_neighborhood = np.array([0, 1, 2])
#                     knn_k = 3

#                     expected_ratings = np.array([13.0])
#                     expected_similarity = np.array([0.1])
#                     expected_means = np.array([1.0])

#                     TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                         x=y,
#                         y=x,
#                         dataset=self.square_dataset.T,
#                         similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
#                         users_neighborhood=users_neighborhood,
#                         means=TestGetKTopNeighbors.means,
#                         knn_k=knn_k,
#                         expected_ratings=expected_ratings,
#                         expected_similarity=expected_similarity,
#                         expected_means=expected_means,
#                     )

#                 def test_2(self):
#                     x = 3
#                     y = 3

#                     users_neighborhood = np.array([0, 1, 2])
#                     knn_k = 1

#                     expected_ratings = np.array([13.0])
#                     expected_similarity = np.array([0.1])
#                     expected_means = np.array([1.0])

#                     TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                         x=y,
#                         y=x,
#                         dataset=self.square_dataset.T,
#                         similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
#                         users_neighborhood=users_neighborhood,
#                         means=TestGetKTopNeighbors.means,
#                         knn_k=knn_k,
#                         expected_ratings=expected_ratings,
#                         expected_similarity=expected_similarity,
#                         expected_means=expected_means,
#                     )

#                 def test_3(self):
#                     x = 3
#                     y = 3

#                     users_neighborhood = np.array([1, 2])
#                     knn_k = 2

#                     expected_ratings = np.array([])
#                     expected_similarity = np.array([])
#                     expected_means = np.array([])

#                     TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                         x=y,
#                         y=x,
#                         dataset=self.square_dataset.T,
#                         similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
#                         users_neighborhood=users_neighborhood,
#                         means=TestGetKTopNeighbors.means,
#                         knn_k=knn_k,
#                         expected_ratings=expected_ratings,
#                         expected_similarity=expected_similarity,
#                         expected_means=expected_means,
#                     )

#         class TestWithNaNsOnSimilarity:
#             class TestNoZeroSimilarity:
#                 # fmt: off
#                 similarity_matrix = np.array([[1.0, np.NAN, 1.0, 0.1],
#                                               [0.5, 1.0, np.NAN, 0.3],
#                                               [1.0, 0.7, np.NAN, 0.0],
#                                               [np.NAN, 0.3, 0.0, 1.0]],
#                                               dtype=np.float64)
#                 # fmt: on

#                 def test_1(self):
#                     x = 0
#                     y = 0

#                     users_neighborhood = np.array([1, 2, 3])
#                     knn_k = 2

#                     expected_ratings = np.array([3.0, 4.0])
#                     expected_similarity = np.array([1.0, 0.1])
#                     expected_means = np.array([3.0, 4.0])

#                     TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                         x=y,
#                         y=x,
#                         dataset=TestGetKTopNeighbors.square_dataset.T,
#                         similarity_matrix=self.similarity_matrix,
#                         users_neighborhood=users_neighborhood,
#                         means=TestGetKTopNeighbors.means,
#                         knn_k=knn_k,
#                         expected_ratings=expected_ratings,
#                         expected_similarity=expected_similarity,
#                         expected_means=expected_means,
#                     )

#                 def test_2(self):
#                     x = 0
#                     y = 0

#                     users_neighborhood = np.array([1, 2, 3])
#                     knn_k = 1

#                     expected_ratings = np.array([3.0])
#                     expected_similarity = np.array([1.0])
#                     expected_means = np.array([3.0])

#                     TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                         x=y,
#                         y=x,
#                         dataset=TestGetKTopNeighbors.square_dataset.T,
#                         similarity_matrix=self.similarity_matrix,
#                         users_neighborhood=users_neighborhood,
#                         means=TestGetKTopNeighbors.means,
#                         knn_k=knn_k,
#                         expected_ratings=expected_ratings,
#                         expected_similarity=expected_similarity,
#                         expected_means=expected_means,
#                     )

#                 def test_3(self):
#                     x = 0
#                     y = 1

#                     users_neighborhood = np.array([2, 3])
#                     knn_k = 3

#                     expected_ratings = np.array([4.0])
#                     expected_similarity = np.array([0.3])
#                     expected_means = np.array([4.0])

#                     TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                         x=y,
#                         y=x,
#                         dataset=TestGetKTopNeighbors.square_dataset.T,
#                         similarity_matrix=self.similarity_matrix,
#                         users_neighborhood=users_neighborhood,
#                         means=TestGetKTopNeighbors.means,
#                         knn_k=knn_k,
#                         expected_ratings=expected_ratings,
#                         expected_similarity=expected_similarity,
#                         expected_means=expected_means,
#                     )

#                 def test_4(self):
#                     x = 0
#                     y = 1

#                     users_neighborhood = np.array([3])
#                     knn_k = 3

#                     expected_ratings = np.array([4.0])
#                     expected_similarity = np.array([0.3])
#                     expected_means = np.array([4.0])

#                     TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                         x=y,
#                         y=x,
#                         dataset=TestGetKTopNeighbors.square_dataset.T,
#                         similarity_matrix=self.similarity_matrix,
#                         users_neighborhood=users_neighborhood,
#                         means=TestGetKTopNeighbors.means,
#                         knn_k=knn_k,
#                         expected_ratings=expected_ratings,
#                         expected_similarity=expected_similarity,
#                         expected_means=expected_means,
#                     )

#             class TestWithZeroSimilarity:
#                 # fmt: off
#                 similarity_matrix = np.array([[1.0, np.NAN, 1.0, 0.1],
#                                               [0.5, 1.0, np.NAN, 0.3],
#                                               [1.0, 0.7, np.NAN, 0.0],
#                                               [np.NAN, 0.3, 0.0, 1.0]],
#                                               dtype=np.float64)
#                 # fmt: on
#                 def test_1(self):
#                     x = 3
#                     y = 3

#                     users_neighborhood = np.array([0, 1, 2])
#                     knn_k = 2

#                     expected_ratings = np.array([14.0])
#                     expected_similarity = np.array([0.3])
#                     expected_means = np.array([2.0])

#                     TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                         x=y,
#                         y=x,
#                         dataset=TestGetKTopNeighbors.square_dataset.T,
#                         similarity_matrix=self.similarity_matrix,
#                         users_neighborhood=users_neighborhood,
#                         means=TestGetKTopNeighbors.means,
#                         knn_k=knn_k,
#                         expected_ratings=expected_ratings,
#                         expected_similarity=expected_similarity,
#                         expected_means=expected_means,
#                     )

#                 def test_2(self):
#                     x = 3
#                     y = 3

#                     users_neighborhood = np.array([0, 2])
#                     knn_k = 1

#                     expected_ratings = np.array([])
#                     expected_similarity = np.array([])
#                     expected_means = np.array([])

#                     TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                         x=y,
#                         y=x,
#                         dataset=TestGetKTopNeighbors.square_dataset.T,
#                         similarity_matrix=self.similarity_matrix,
#                         users_neighborhood=users_neighborhood,
#                         means=TestGetKTopNeighbors.means,
#                         knn_k=knn_k,
#                         expected_ratings=expected_ratings,
#                         expected_similarity=expected_similarity,
#                         expected_means=expected_means,
#                     )

#         class TestWithNaNsOnDatasetAndSimilarity:
#             # fmt: off
#             square_dataset = np.array([[ 1,  2,  3,  np.NAN],
#                                         [ 5,  6,  np.NAN,  8],
#                                         [ np.NAN, 10, 11, 12],
#                                         [13, np.NAN, 15, 16]],
#                                         dtype=np.float64)
#             similarity_matrix = np.array([[1.0, np.NAN, 1.0, 0.1],
#                                             [0.5, 1.0, np.NAN, 0.3],
#                                             [1.0, 0.7, np.NAN, 0.0],
#                                             [np.NAN, 0.3, 0.0, 1.0]],
#                                             dtype=np.float64)
#             # fmt: on

#             def test_1(self):
#                 x = 1
#                 y = 1

#                 users_neighborhood = np.array([0, 2, 3])
#                 knn_k = 2

#                 expected_ratings = np.array([5.0, 8.0])
#                 expected_similarity = np.array([0.5, 0.3])
#                 expected_means = np.array([1.0, 4.0])

#                 TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                     x=y,
#                     y=x,
#                     dataset=self.square_dataset.T,
#                     similarity_matrix=self.similarity_matrix,
#                     users_neighborhood=users_neighborhood,
#                     means=TestGetKTopNeighbors.means,
#                     knn_k=knn_k,
#                     expected_ratings=expected_ratings,
#                     expected_similarity=expected_similarity,
#                     expected_means=expected_means,
#                 )

#             def test_2(self):
#                 x = 1
#                 y = 1

#                 users_neighborhood = np.array([0, 2, 3])
#                 knn_k = 1

#                 expected_ratings = np.array([5.0])
#                 expected_similarity = np.array([0.5])
#                 expected_means = np.array([1.0])

#                 TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                     x=y,
#                     y=x,
#                     dataset=self.square_dataset.T,
#                     similarity_matrix=self.similarity_matrix,
#                     users_neighborhood=users_neighborhood,
#                     means=TestGetKTopNeighbors.means,
#                     knn_k=knn_k,
#                     expected_ratings=expected_ratings,
#                     expected_similarity=expected_similarity,
#                     expected_means=expected_means,
#                 )

#             def test_3(self):
#                 x = 0
#                 y = 0

#                 users_neighborhood = np.array([1, 2, 3])
#                 knn_k = 3

#                 expected_ratings = np.array([3.0])
#                 expected_similarity = np.array([1.0])
#                 expected_means = np.array([3.0])

#                 TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                     x=y,
#                     y=x,
#                     dataset=self.square_dataset.T,
#                     similarity_matrix=self.similarity_matrix,
#                     users_neighborhood=users_neighborhood,
#                     means=TestGetKTopNeighbors.means,
#                     knn_k=knn_k,
#                     expected_ratings=expected_ratings,
#                     expected_similarity=expected_similarity,
#                     expected_means=expected_means,
#                 )

#             def test_4(self):
#                 x = 1
#                 y = 0

#                 users_neighborhood = np.array([2, 3])
#                 knn_k = 3

#                 expected_ratings = np.array([8.0])
#                 expected_similarity = np.array([0.1])
#                 expected_means = np.array([4.0])

#                 TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                     x=y,
#                     y=x,
#                     dataset=self.square_dataset.T,
#                     similarity_matrix=self.similarity_matrix,
#                     users_neighborhood=users_neighborhood,
#                     means=TestGetKTopNeighbors.means,
#                     knn_k=knn_k,
#                     expected_ratings=expected_ratings,
#                     expected_similarity=expected_similarity,
#                     expected_means=expected_means,
#                 )

#             def test_5(self):
#                 x = 1
#                 y = 0

#                 users_neighborhood = np.array([2, 3])
#                 knn_k = 2

#                 expected_ratings = np.array([8.0])
#                 expected_similarity = np.array([0.1])
#                 expected_means = np.array([4.0])

#                 TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                     x=y,
#                     y=x,
#                     dataset=self.square_dataset.T,
#                     similarity_matrix=self.similarity_matrix,
#                     users_neighborhood=users_neighborhood,
#                     means=TestGetKTopNeighbors.means,
#                     knn_k=knn_k,
#                     expected_ratings=expected_ratings,
#                     expected_similarity=expected_similarity,
#                     expected_means=expected_means,
#                 )

#             def test_6(self):
#                 x = 1
#                 y = 0

#                 users_neighborhood = np.array([2, 3])
#                 knn_k = 1

#                 expected_ratings = np.array([8.0])
#                 expected_similarity = np.array([0.1])
#                 expected_means = np.array([4.0])

#                 TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                     x=y,
#                     y=x,
#                     dataset=self.square_dataset.T,
#                     similarity_matrix=self.similarity_matrix,
#                     users_neighborhood=users_neighborhood,
#                     means=TestGetKTopNeighbors.means,
#                     knn_k=knn_k,
#                     expected_ratings=expected_ratings,
#                     expected_similarity=expected_similarity,
#                     expected_means=expected_means,
#                 )

#             def test_7(self):
#                 x = 1
#                 y = 0

#                 users_neighborhood = np.array([3])
#                 knn_k = 2

#                 expected_ratings = np.array([8.0])
#                 expected_similarity = np.array([0.1])
#                 expected_means = np.array([4.0])

#                 TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                     x=y,
#                     y=x,
#                     dataset=self.square_dataset.T,
#                     similarity_matrix=self.similarity_matrix,
#                     users_neighborhood=users_neighborhood,
#                     means=TestGetKTopNeighbors.means,
#                     knn_k=knn_k,
#                     expected_ratings=expected_ratings,
#                     expected_similarity=expected_similarity,
#                     expected_means=expected_means,
#                 )

#             def test_8(self):
#                 x = 3
#                 y = 3

#                 users_neighborhood = np.array([0, 1, 2])
#                 knn_k = 3

#                 expected_ratings = np.array([])
#                 expected_similarity = np.array([])
#                 expected_means = np.array([])

#                 TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                     x=y,
#                     y=x,
#                     dataset=self.square_dataset.T,
#                     similarity_matrix=self.similarity_matrix,
#                     users_neighborhood=users_neighborhood,
#                     means=TestGetKTopNeighbors.means,
#                     knn_k=knn_k,
#                     expected_ratings=expected_ratings,
#                     expected_similarity=expected_similarity,
#                     expected_means=expected_means,
#                 )

#             def test_9(self):
#                 x = 2
#                 y = 3

#                 users_neighborhood = np.array([0, 1, 2])
#                 knn_k = 3

#                 expected_ratings = np.array([10.0])
#                 expected_similarity = np.array([0.3])
#                 expected_means = np.array([2.0])

#                 TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                     x=y,
#                     y=x,
#                     dataset=self.square_dataset.T,
#                     similarity_matrix=self.similarity_matrix,
#                     users_neighborhood=users_neighborhood,
#                     means=TestGetKTopNeighbors.means,
#                     knn_k=knn_k,
#                     expected_ratings=expected_ratings,
#                     expected_similarity=expected_similarity,
#                     expected_means=expected_means,
#                 )

#             def test_10(self):
#                 x = 2
#                 y = 3

#                 users_neighborhood = np.array([0, 2])
#                 knn_k = 3

#                 expected_ratings = np.array([])
#                 expected_similarity = np.array([])
#                 expected_means = np.array([])

#                 TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
#                     x=y,
#                     y=x,
#                     dataset=self.square_dataset.T,
#                     similarity_matrix=self.similarity_matrix,
#                     users_neighborhood=users_neighborhood,
#                     means=TestGetKTopNeighbors.means,
#                     knn_k=knn_k,
#                     expected_ratings=expected_ratings,
#                     expected_similarity=expected_similarity,
#                     expected_means=expected_means,
#                 )


# class TestBiAKNN:
#     class ConcreteBiAKNN(PAkNN):
#         def __init__(self, *args, **kwargs):
#             super().__init__(*args, **kwargs)

#         def compute_biclusters_from_trainset(self):
#             pass

#     class MockTrainset:
#         def __init__(self, ur, ir):
#             self.ur = ur
#             self.ir = ir

#             self.n_users = len(ur)
#             self.n_items = len(ir)

#     def test_init_invalid_args(self):
#         with pytest.raises(AssertionError):
#             self.ConcreteBiAKNN(minimum_bicluster_sparsity="0.5")

#         with pytest.raises(AssertionError):
#             self.ConcreteBiAKNN(minimum_bicluster_sparsity=-0.5)

#         with pytest.raises(AssertionError):
#             self.ConcreteBiAKNN(minimum_bicluster_sparsity=1.5)

#         with pytest.raises(AssertionError):
#             self.ConcreteBiAKNN(minimum_bicluster_coverage="0.8")

#         with pytest.raises(AssertionError):
#             self.ConcreteBiAKNN(minimum_bicluster_coverage=-0.8)

#         with pytest.raises(AssertionError):
#             self.ConcreteBiAKNN(minimum_bicluster_coverage=1.8)

#         with pytest.raises(AssertionError):
#             self.ConcreteBiAKNN(minimum_bicluster_relative_size="0.2")

#         with pytest.raises(AssertionError):
#             self.ConcreteBiAKNN(minimum_bicluster_relative_size=-0.2)

#         with pytest.raises(AssertionError):
#             self.ConcreteBiAKNN(minimum_bicluster_relative_size=1.2)

#         with pytest.raises(AssertionError):
#             self.ConcreteBiAKNN(knn_type="invalid_type")

#         with pytest.raises(AssertionError):
#             self.ConcreteBiAKNN(user_binarization_threshold="0.5")

#         with pytest.raises(AssertionError):
#             self.ConcreteBiAKNN(user_binarization_threshold=-0.5)

#         with pytest.raises(AssertionError):
#             self.ConcreteBiAKNN(number_of_top_k_biclusters="10")

#         with pytest.raises(AssertionError):
#             self.ConcreteBiAKNN(number_of_top_k_biclusters=0)

#         with pytest.raises(AssertionError):
#             self.ConcreteBiAKNN(knn_k="3")

#         with pytest.raises(AssertionError):
#             self.ConcreteBiAKNN(knn_k=0)

#     @patch("recommenders.knn_based_recommenders.apply_bicluster_sparsity_filter")
#     @patch("recommenders.knn_based_recommenders.apply_bicluster_coverage_filter")
#     @patch("recommenders.knn_based_recommenders.apply_bicluster_relative_size_filter")
#     def test_apply_filters_1(
#         self,
#         apply_bicluster_relative_size_filter_mock,
#         apply_bicluster_coverage_filter_mock,
#         apply_bicluster_sparsity_filter_mock,
#     ):
#         biaknn = self.ConcreteBiAKNN(minimum_bicluster_sparsity=0.2)

#         biaknn.dataset = "dataset"
#         biaknn.biclusters = "biclusters"
#         apply_bicluster_sparsity_filter_mock.return_value = "filtered_biclusters"

#         biaknn._apply_filters()

#         apply_bicluster_sparsity_filter_mock.assert_called_once_with("dataset", "biclusters", 0.2)
#         apply_bicluster_coverage_filter_mock.assert_not_called()
#         apply_bicluster_relative_size_filter_mock.assert_not_called()

#         assert biaknn.dataset == "dataset"
#         assert biaknn.biclusters == "filtered_biclusters"
#         assert biaknn.number_of_top_k_biclusters == len("filtered_biclusters")

#     @patch("recommenders.knn_based_recommenders.apply_bicluster_sparsity_filter")
#     @patch("recommenders.knn_based_recommenders.apply_bicluster_coverage_filter")
#     @patch("recommenders.knn_based_recommenders.apply_bicluster_relative_size_filter")
#     def test_apply_filters_2(
#         self,
#         apply_bicluster_relative_size_filter_mock,
#         apply_bicluster_coverage_filter_mock,
#         apply_bicluster_sparsity_filter_mock,
#     ):
#         biaknn = self.ConcreteBiAKNN(minimum_bicluster_sparsity=0.2, number_of_top_k_biclusters=10)

#         biaknn.dataset = "dataset"
#         biaknn.biclusters = "biclusters"
#         apply_bicluster_sparsity_filter_mock.return_value = "filtered_biclusters"

#         biaknn._apply_filters()

#         apply_bicluster_sparsity_filter_mock.assert_called_once_with("dataset", "biclusters", 0.2)
#         apply_bicluster_coverage_filter_mock.assert_not_called()
#         apply_bicluster_relative_size_filter_mock.assert_not_called()

#         assert biaknn.dataset == "dataset"
#         assert biaknn.biclusters == "filtered_biclusters"
#         assert biaknn.number_of_top_k_biclusters == 10

#     @patch("recommenders.knn_based_recommenders.apply_bicluster_sparsity_filter")
#     @patch("recommenders.knn_based_recommenders.apply_bicluster_coverage_filter")
#     @patch("recommenders.knn_based_recommenders.apply_bicluster_relative_size_filter")
#     def test_apply_filters_3(
#         self,
#         apply_bicluster_relative_size_filter_mock,
#         apply_bicluster_coverage_filter_mock,
#         apply_bicluster_sparsity_filter_mock,
#     ):
#         biaknn = self.ConcreteBiAKNN(number_of_top_k_biclusters=10)

#         biaknn.dataset = "dataset"
#         biaknn.biclusters = "biclusters"

#         biaknn._apply_filters()

#         apply_bicluster_sparsity_filter_mock.assert_not_called()
#         apply_bicluster_coverage_filter_mock.assert_not_called()
#         apply_bicluster_relative_size_filter_mock.assert_not_called()

#         assert biaknn.dataset == "dataset"
#         assert biaknn.biclusters == "biclusters"
#         assert biaknn.number_of_top_k_biclusters == 10

#     @patch("recommenders.knn_based_recommenders.apply_bicluster_sparsity_filter")
#     @patch("recommenders.knn_based_recommenders.apply_bicluster_coverage_filter")
#     @patch("recommenders.knn_based_recommenders.apply_bicluster_relative_size_filter")
#     def test_apply_filters_4(
#         self,
#         apply_bicluster_relative_size_filter_mock,
#         apply_bicluster_coverage_filter_mock,
#         apply_bicluster_sparsity_filter_mock,
#     ):
#         biaknn = self.ConcreteBiAKNN()

#         biaknn.dataset = "dataset"
#         biaknn.biclusters = "biclusters"

#         biaknn._apply_filters()

#         apply_bicluster_sparsity_filter_mock.assert_not_called()
#         apply_bicluster_coverage_filter_mock.assert_not_called()
#         apply_bicluster_relative_size_filter_mock.assert_not_called()

#         assert biaknn.dataset == "dataset"
#         assert biaknn.biclusters == "biclusters"
#         assert biaknn.number_of_top_k_biclusters == len("biclusters")

#     @patch("recommenders.knn_based_recommenders.apply_bicluster_sparsity_filter")
#     @patch("recommenders.knn_based_recommenders.apply_bicluster_coverage_filter")
#     @patch("recommenders.knn_based_recommenders.apply_bicluster_relative_size_filter")
#     def test_apply_filters_5(
#         self,
#         apply_bicluster_relative_size_filter_mock,
#         apply_bicluster_coverage_filter_mock,
#         apply_bicluster_sparsity_filter_mock,
#     ):
#         biaknn = self.ConcreteBiAKNN(
#             minimum_bicluster_coverage=0.5,
#         )

#         biaknn.dataset = "dataset"
#         biaknn.biclusters = "biclusters"

#         apply_bicluster_coverage_filter_mock.return_value = "filtered_biclusters"

#         biaknn._apply_filters()

#         apply_bicluster_sparsity_filter_mock.assert_not_called()
#         apply_bicluster_coverage_filter_mock.assert_called_once_with("dataset", "biclusters", 0.5)
#         apply_bicluster_relative_size_filter_mock.assert_not_called()

#         assert biaknn.dataset == "dataset"
#         assert biaknn.biclusters == "filtered_biclusters"
#         assert biaknn.number_of_top_k_biclusters == len("filtered_biclusters")

#     @patch("recommenders.knn_based_recommenders.apply_bicluster_sparsity_filter")
#     @patch("recommenders.knn_based_recommenders.apply_bicluster_coverage_filter")
#     @patch("recommenders.knn_based_recommenders.apply_bicluster_relative_size_filter")
#     def test_apply_filters_6(
#         self,
#         apply_bicluster_relative_size_filter_mock,
#         apply_bicluster_coverage_filter_mock,
#         apply_bicluster_sparsity_filter_mock,
#     ):
#         biaknn = self.ConcreteBiAKNN(
#             minimum_bicluster_relative_size=0.6,
#         )

#         biaknn.dataset = "dataset"
#         biaknn.biclusters = "biclusters"

#         apply_bicluster_relative_size_filter_mock.return_value = "filtered_biclusters"

#         biaknn._apply_filters()

#         apply_bicluster_sparsity_filter_mock.assert_not_called()
#         apply_bicluster_coverage_filter_mock.assert_not_called()
#         apply_bicluster_relative_size_filter_mock.assert_called_once_with(
#             "dataset", "biclusters", 0.6
#         )

#         assert biaknn.dataset == "dataset"
#         assert biaknn.biclusters == "filtered_biclusters"
#         assert biaknn.number_of_top_k_biclusters == len("filtered_biclusters")

#     @patch("recommenders.knn_based_recommenders.apply_bicluster_sparsity_filter")
#     @patch("recommenders.knn_based_recommenders.apply_bicluster_coverage_filter")
#     @patch("recommenders.knn_based_recommenders.apply_bicluster_relative_size_filter")
#     def test_apply_filters_7(
#         self,
#         apply_bicluster_relative_size_filter_mock,
#         apply_bicluster_coverage_filter_mock,
#         apply_bicluster_sparsity_filter_mock,
#     ):
#         biaknn = self.ConcreteBiAKNN(
#             minimum_bicluster_relative_size=0.6,
#             minimum_bicluster_coverage=0.5,
#             minimum_bicluster_sparsity=0.2,
#         )

#         biaknn.dataset = "dataset"
#         biaknn.biclusters = "biclusters"

#         apply_bicluster_sparsity_filter_mock.return_value = "filtered_biclusters_1"
#         apply_bicluster_coverage_filter_mock.return_value = "filtered_biclusters_2"
#         apply_bicluster_relative_size_filter_mock.return_value = "filtered_biclusters_3"

#         biaknn._apply_filters()

#         apply_bicluster_sparsity_filter_mock.assert_called_once_with("dataset", "biclusters", 0.2)
#         apply_bicluster_coverage_filter_mock.assert_called_once_with(
#             "dataset", "filtered_biclusters_1", 0.5
#         )
#         apply_bicluster_relative_size_filter_mock.assert_called_once_with(
#             "dataset", "filtered_biclusters_2", 0.6
#         )

#         assert biaknn.dataset == "dataset"
#         assert biaknn.biclusters == "filtered_biclusters_3"
#         assert biaknn.number_of_top_k_biclusters == len("filtered_biclusters_3")

#     @patch("recommenders.knn_based_recommenders.get_indices_above_threshold")
#     @patch("recommenders.knn_based_recommenders.get_top_k_biclusters_for_user")
#     @patch("recommenders.knn_based_recommenders.merge_biclusters")
#     def test_generate_neighborhood_1(
#         self,
#         mock_merge_biclusters,
#         mock_get_top_k_biclusters_for_user,
#         mock_get_indices_above_threshold,
#     ):
#         mock_get_indices_above_threshold.side_effect = [1, 2, 3]
#         mock_get_top_k_biclusters_for_user.side_effect = [10, 11, 12]
#         mock_merge_biclusters.side_effect = [Concept(1, 2), Concept(3, 4), Concept(5, 6)]

#         biaknn = self.ConcreteBiAKNN()
#         biaknn.dataset = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
#         biaknn.biclusters = "biclusters"
#         biaknn.number_of_top_k_biclusters = 123

#         biaknn._generate_neighborhood()

#         calls = mock_get_indices_above_threshold.call_args_list

#         assert len(calls) == 3

#         assert (calls[0].args[0] == biaknn.dataset[0]).all()
#         assert np.isclose(calls[0].args[1], 1.0)
#         assert (calls[1].args[0] == biaknn.dataset[1]).all()
#         assert np.isclose(calls[1].args[1], 1.0)
#         assert (calls[2].args[0] == biaknn.dataset[2]).all()
#         assert np.isclose(calls[2].args[1], 1.0)

#         calls = mock_get_top_k_biclusters_for_user.call_args_list

#         assert len(calls) == 3

#         assert calls[0].args[0] == "biclusters"
#         assert calls[0].args[1] == 1
#         assert calls[0].args[2] == 123

#         assert calls[1].args[0] == "biclusters"
#         assert calls[1].args[1] == 2
#         assert calls[1].args[2] == 123

#         assert calls[2].args[0] == "biclusters"
#         assert calls[2].args[1] == 3
#         assert calls[2].args[2] == 123

#         calls = mock_merge_biclusters.call_args_list

#         assert len(calls) == 3

#         assert calls[0].args[0] == 10
#         assert calls[1].args[0] == 11
#         assert calls[2].args[0] == 12

#         assert len(calls) == 3
#         assert biaknn.neighborhood[0] == Concept(1, 2).intent
#         assert biaknn.neighborhood[1] == Concept(3, 4).intent
#         assert biaknn.neighborhood[2] == Concept(5, 6).intent

#     @patch("recommenders.knn_based_recommenders.get_indices_above_threshold")
#     @patch("recommenders.knn_based_recommenders.get_top_k_biclusters_for_user")
#     @patch("recommenders.knn_based_recommenders.merge_biclusters")
#     def test_generate_neighborhood_2(
#         self,
#         mock_merge_biclusters,
#         mock_get_top_k_biclusters_for_user,
#         mock_get_indices_above_threshold,
#     ):
#         mock_get_indices_above_threshold.side_effect = [1, 2, 3, 4]
#         mock_get_top_k_biclusters_for_user.side_effect = [10, 11, 12, 13]
#         mock_merge_biclusters.side_effect = [
#             Concept(1, 2),
#             Concept(3, 4),
#             Concept(5, 6),
#             Concept(7, 8),
#         ]

#         biaknn = self.ConcreteBiAKNN()
#         # fmt: off
#         biaknn.dataset = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
#         # fmt: on
#         biaknn.biclusters = "biclusters"
#         biaknn.number_of_top_k_biclusters = 77
#         biaknn.knn_type = "user"

#         biaknn._generate_neighborhood()

#         calls = mock_get_indices_above_threshold.call_args_list

#         assert len(calls) == 4

#         assert (calls[0].args[0] == biaknn.dataset[0]).all()
#         assert np.isclose(calls[0].args[1], 1.0)
#         assert (calls[1].args[0] == biaknn.dataset[1]).all()
#         assert np.isclose(calls[1].args[1], 1.0)
#         assert (calls[2].args[0] == biaknn.dataset[2]).all()
#         assert np.isclose(calls[2].args[1], 1.0)
#         assert (calls[3].args[0] == biaknn.dataset[3]).all()
#         assert np.isclose(calls[3].args[1], 1.0)

#         calls = mock_get_top_k_biclusters_for_user.call_args_list

#         assert len(calls) == 4

#         assert calls[0].args[0] == "biclusters"
#         assert calls[0].args[1] == 1
#         assert calls[0].args[2] == 77

#         assert calls[1].args[0] == "biclusters"
#         assert calls[1].args[1] == 2
#         assert calls[1].args[2] == 77

#         assert calls[2].args[0] == "biclusters"
#         assert calls[2].args[1] == 3
#         assert calls[2].args[2] == 77

#         assert calls[3].args[0] == "biclusters"
#         assert calls[3].args[1] == 4
#         assert calls[3].args[2] == 77

#         calls = mock_merge_biclusters.call_args_list

#         assert len(calls) == 4

#         assert calls[0].args[0] == 10
#         assert calls[1].args[0] == 11
#         assert calls[2].args[0] == 12
#         assert calls[3].args[0] == 13

#         assert len(calls) == 4
#         assert biaknn.neighborhood[0] == Concept(1, 2).extent
#         assert biaknn.neighborhood[1] == Concept(3, 4).extent
#         assert biaknn.neighborhood[2] == Concept(5, 6).extent
#         assert biaknn.neighborhood[3] == Concept(7, 8).extent

#     def test_calculate_means_item_1(self):
#         biaknn = self.ConcreteBiAKNN()
#         biaknn.trainset = self.MockTrainset(
#             ur={
#                 0: [(0, 1), (1, 2)],
#                 1: [(0, 3), (1, 4), (2, 5)],
#                 2: [(0, 6)],
#             },
#             ir={
#                 0: [(0, 1), (1, 2)],
#                 1: [(0, 2), (1, 3), (2, 4)],
#                 2: [(0, 3), (1, 4), (2, 5)],
#             },
#         )

#         biaknn._calculate_means()

#         assert biaknn.n == 3
#         assert (biaknn.means == [1.5, 3, 4]).all()

#     def test_calculate_means_item_2(self):
#         biaknn = self.ConcreteBiAKNN()
#         biaknn.trainset = self.MockTrainset(
#             ur={
#                 0: [(0, 1), (1, 2)],
#                 1: [(0, 3), (1, 4), (2, 5)],
#                 2: [(0, 6)],
#             },
#             ir={
#                 0: [(0, 1), (1, 2)],
#                 1: [(0, 2), (1, 3), (2, 4)],
#                 2: [(0, 3), (1, 4), (2, 5)],
#             },
#         )

#         biaknn._calculate_means()

#         assert biaknn.n == 3
#         assert (biaknn.means == [1.5, 3, 4]).all()

#     def test_calculate_means_item_3(self):
#         biaknn = self.ConcreteBiAKNN()
#         biaknn.trainset = self.MockTrainset(
#             ur={
#                 0: [(0, 2), (1, 3)],
#                 1: [(0, 4), (1, 5), (2, 6)],
#                 2: [(0, 7)],
#             },
#             ir={
#                 0: [(0, 2), (1, 3)],
#                 1: [(0, 3), (1, 4), (2, 5)],
#                 2: [(0, 4), (1, 5), (2, 6)],
#             },
#         )

#         biaknn._calculate_means()

#         assert biaknn.n == 3
#         assert (biaknn.means == [2.5, 4, 5]).all()

#     def test_calculate_means_item_4(self):
#         biaknn = self.ConcreteBiAKNN()
#         biaknn.trainset = self.MockTrainset(
#             ur={
#                 0: [(0, 1), (1, 3), (2, 5), (3, 7)],
#                 1: [(0, 2), (1, 4), (2, 6), (3, 8)],
#                 2: [(0, 3), (1, 5), (2, 7), (3, 9)],
#                 3: [(0, 4), (1, 6), (2, 8), (3, 10)],
#             },
#             ir={
#                 0: [(0, 2), (1, 4), (2, 6), (3, 8)],
#                 1: [(0, 3), (1, 5), (2, 7), (3, 9)],
#                 2: [(0, 4), (1, 6), (2, 8), (3, 10)],
#                 3: [(0, 5), (1, 7), (2, 9), (3, 11)],
#             },
#         )

#         biaknn._calculate_means()

#         assert biaknn.n == 4
#         assert (biaknn.means == [5, 6, 7, 8]).all()

#     def test_calculate_means_user_1(self):
#         biaknn = self.ConcreteBiAKNN()
#         biaknn.trainset = self.MockTrainset(
#             ur={
#                 0: [(0, 1), (1, 3), (2, 5), (3, 7)],
#                 1: [(0, 2), (1, 4), (2, 6), (3, 8)],
#                 2: [(0, 3), (1, 5), (2, 7), (3, 9)],
#                 3: [(0, 4), (1, 6), (2, 8), (3, 10)],
#             },
#             ir={
#                 0: [(0, 2), (1, 4), (2, 6), (3, 8)],
#                 1: [(0, 3), (1, 5), (2, 7), (3, 9)],
#                 2: [(0, 4), (1, 6), (2, 8), (3, 10)],
#                 3: [(0, 5), (1, 7), (2, 9), (3, 11)],
#             },
#         )
#         biaknn.knn_type = "user"

#         biaknn._calculate_means()

#         assert biaknn.n == 4
#         assert (biaknn.means == [4, 5, 6, 7]).all()

#     def test_calculate_means_user_2(self):
#         biaknn = self.ConcreteBiAKNN()
#         biaknn.trainset = self.MockTrainset(
#             ur={
#                 0: [(0, 10), (1, 20), (2, 30), (3, 40)],
#                 1: [(0, 15), (1, 25), (2, 35), (3, 45)],
#                 2: [(0, 20), (1, 30), (2, 40), (3, 50)],
#                 3: [(0, 25), (1, 35), (2, 45), (3, 55)],
#             },
#             ir={
#                 0: [(0, 15), (1, 25), (2, 35), (3, 45)],
#                 1: [(0, 20), (1, 30), (2, 40), (3, 50)],
#                 2: [(0, 25), (1, 35), (2, 45), (3, 55)],
#                 3: [(0, 30), (1, 40), (2, 50), (3, 60)],
#             },
#         )
#         biaknn.knn_type = "user"

#         biaknn._calculate_means()

#         assert biaknn.n == 4
#         assert (biaknn.means == [25, 30, 35, 40]).all()

#     def test_instantiate_similarity_matrix(self):
#         biaknn = self.ConcreteBiAKNN()
#         biaknn.n = 10

#         biaknn._instantiate_similarity_matrix()

#         assert np.isnan(biaknn.similarity_matrix).all()
#         assert biaknn.similarity_matrix.shape == (10, 10)
#         assert biaknn.similarity_matrix.dtype == np.float64

#     class TestEstimate:
#         def test_unknown_user(self):
#             biaknn = TestBiAKNN.ConcreteBiAKNN()

#             mock_trainset = Mock()
#             mock_trainset.knows_user.return_value = False
#             mock_trainset.knows_item.return_value = True

#             biaknn.trainset = mock_trainset

#             with pytest.raises(PredictionImpossible):
#                 biaknn.estimate(1, 1)

#         def test_unknown_item(self):
#             biaknn = TestBiAKNN.ConcreteBiAKNN()

#             mock_trainset = Mock()
#             mock_trainset.knows_user.return_value = True
#             mock_trainset.knows_item.return_value = False

#             biaknn.trainset = mock_trainset

#             with pytest.raises(PredictionImpossible):
#                 biaknn.estimate(1, 1)

#         def test_empty_user_neighborhood_1(self):
#             biaknn = TestBiAKNN.ConcreteBiAKNN()
#             biaknn.dataset = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

#             mock_trainset = Mock()
#             mock_trainset.knows_user.return_value = True
#             mock_trainset.knows_item.return_value = True

#             biaknn.trainset = mock_trainset

#             biaknn.neighborhood = {1: np.array([])}
#             biaknn.means = np.array([3.5, 4.0, 4.5])

#             prediction, info = biaknn.estimate(1, 1)

#             assert np.isclose(prediction, 4.0)
#             assert info["actual_k"] == 0

#         def test_empty_user_neighborhood_2(self):
#             biaknn = TestBiAKNN.ConcreteBiAKNN()
#             biaknn.dataset = None
#             biaknn.knn_type = "user"

#             mock_trainset = Mock()
#             mock_trainset.knows_user.return_value = True
#             mock_trainset.knows_item.return_value = True

#             biaknn.trainset = mock_trainset

#             biaknn.neighborhood = {1: np.array([])}
#             biaknn.means = np.array([3.5, 4.0, 4.5])

#             prediction, info = biaknn.estimate(1, 1)

#             assert np.isclose(prediction, 4.0)
#             assert info["actual_k"] == 0

#         def test_empty_user_neighborhood_3(self):
#             biaknn = TestBiAKNN.ConcreteBiAKNN()
#             biaknn.dataset = None
#             biaknn.knn_type = "user"

#             mock_trainset = Mock()
#             mock_trainset.knows_user.return_value = True
#             mock_trainset.knows_item.return_value = True

#             biaknn.trainset = mock_trainset

#             biaknn.neighborhood = {1: np.array([1])}
#             biaknn.means = np.array([3.5, 4.0, 4.5])

#             prediction, info = biaknn.estimate(1, 1)

#             assert np.isclose(prediction, 4.0)
#             assert info["actual_k"] == 0

#         @patch("recommenders.knn_based_recommenders.get_k_top_neighbors")
#         @patch("recommenders.knn_based_recommenders.compute_neighborhood_cosine_similarity")
#         def test_no_k_top_neighbors_1(
#             self,
#             mock_compute_neighborhood_cosine_similarity,
#             mock_get_k_top_neighbors,
#         ):
#             mock_get_k_top_neighbors.return_value = (np.array([]), np.array([]), np.array([]))

#             biaknn = TestBiAKNN.ConcreteBiAKNN()

#             mock_trainset = Mock()
#             mock_trainset.knows_user.return_value = True
#             mock_trainset.knows_item.return_value = True

#             biaknn.trainset = mock_trainset
#             biaknn.dataset = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
#             biaknn.neighborhood = {1: np.array([1, 2, 3, 4, 5])}
#             biaknn.means = np.array([3.5, 4.0, 4.5])
#             biaknn.knn_type = "user"

#             prediction, info = biaknn.estimate(1, 2)

#             calls = mock_compute_neighborhood_cosine_similarity.call_args_list
#             assert len(calls) == 1
#             assert (calls[0].args[0] == biaknn.dataset).all()
#             assert calls[0].args[1] == biaknn.similarity_matrix
#             assert calls[0].args[2] == 1
#             assert (calls[0].args[3] == np.array([2, 3, 4, 5])).all()

#             calls = mock_get_k_top_neighbors.call_args_list
#             assert len(calls) == 1
#             assert calls[0].args[0] == 1
#             assert calls[0].args[1] == 2
#             assert (calls[0].args[2] == biaknn.dataset).all()
#             assert (calls[0].args[3] == np.array([2, 3, 4, 5])).all()
#             assert calls[0].args[4] == biaknn.similarity_matrix
#             assert (calls[0].args[5] == biaknn.means).all()
#             assert calls[0].args[6] == biaknn.knn_k

#             assert np.isclose(prediction, 4.0)
#             assert info["actual_k"] == 0

#         @patch("recommenders.knn_based_recommenders.get_k_top_neighbors")
#         @patch("recommenders.knn_based_recommenders.compute_neighborhood_cosine_similarity")
#         def test_no_k_top_neighbors_2(
#             self,
#             mock_compute_neighborhood_cosine_similarity,
#             mock_get_k_top_neighbors,
#         ):
#             mock_get_k_top_neighbors.return_value = (np.array([]), np.array([]), np.array([]))

#             biaknn = TestBiAKNN.ConcreteBiAKNN()

#             mock_trainset = Mock()
#             mock_trainset.knows_user.return_value = True
#             mock_trainset.knows_item.return_value = True

#             biaknn.trainset = mock_trainset
#             biaknn.dataset = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
#             biaknn.neighborhood = {1: np.array([1, 2, 3, 4, 5])}
#             biaknn.means = np.array([3.5, 4.0, 4.5])
#             biaknn.knn_type = "item"

#             prediction, info = biaknn.estimate(1, 2)

#             calls = mock_compute_neighborhood_cosine_similarity.call_args_list
#             assert len(calls) == 1
#             assert (calls[0].args[0] == biaknn.dataset.T).all()
#             assert calls[0].args[1] == biaknn.similarity_matrix
#             assert calls[0].args[2] == 2
#             assert (calls[0].args[3] == np.array([1, 3, 4, 5])).all()

#             calls = mock_get_k_top_neighbors.call_args_list
#             assert len(calls) == 1
#             assert calls[0].args[0] == 2
#             assert calls[0].args[1] == 1
#             assert (calls[0].args[2] == biaknn.dataset.T).all()
#             assert (calls[0].args[3] == np.array([1, 3, 4, 5])).all()
#             assert calls[0].args[4] == biaknn.similarity_matrix
#             assert (calls[0].args[5] == biaknn.means).all()
#             assert calls[0].args[6] == biaknn.knn_k

#             assert np.isclose(prediction, 4.5)
#             assert info["actual_k"] == 0

#         @patch("recommenders.knn_based_recommenders.get_k_top_neighbors")
#         @patch("recommenders.knn_based_recommenders.compute_neighborhood_cosine_similarity")
#         @patch("recommenders.knn_based_recommenders.calculate_weighted_rating")
#         def test_success_1(
#             self,
#             mock_calculate_weighted_rating,
#             mock_compute_neighborhood_cosine_similarity,
#             mock_get_k_top_neighbors,
#         ):
#             mock_get_k_top_neighbors.return_value = (
#                 np.array([1, 2, 3], dtype=np.float64),
#                 np.array([0.1, 0.2, 0.3], dtype=np.float64),
#                 np.array([4.0, 4.1, 4.2], dtype=np.float64),
#             )

#             mock_calculate_weighted_rating.return_value = 4.1

#             biaknn = TestBiAKNN.ConcreteBiAKNN()

#             mock_trainset = Mock()
#             mock_trainset.knows_user.return_value = True
#             mock_trainset.knows_item.return_value = True

#             biaknn.trainset = mock_trainset
#             biaknn.dataset = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
#             biaknn.neighborhood = {1: np.array([1, 2, 3, 4, 5])}
#             biaknn.means = np.array([3.5, 4.0, 4.5])
#             biaknn.knn_type = "user"

#             prediction, info = biaknn.estimate(1, 2)

#             calls = mock_compute_neighborhood_cosine_similarity.call_args_list
#             assert len(calls) == 1
#             assert (calls[0].args[0] == biaknn.dataset).all()
#             assert calls[0].args[1] == biaknn.similarity_matrix
#             assert calls[0].args[2] == 1
#             assert (calls[0].args[3] == np.array([2, 3, 4, 5])).all()

#             calls = mock_get_k_top_neighbors.call_args_list
#             assert len(calls) == 1
#             assert calls[0].args[0] == 1
#             assert calls[0].args[1] == 2
#             assert (calls[0].args[2] == biaknn.dataset).all()
#             assert (calls[0].args[3] == np.array([2, 3, 4, 5])).all()
#             assert calls[0].args[4] == biaknn.similarity_matrix
#             assert (calls[0].args[5] == biaknn.means).all()
#             assert calls[0].args[6] == biaknn.knn_k

#             calls = mock_calculate_weighted_rating.call_args_list
#             assert len(calls) == 1
#             assert calls[0].args[0] == 4
#             assert (calls[0].args[1] == np.array([1, 2, 3])).all()
#             assert (calls[0].args[2] == np.array([0.1, 0.2, 0.3])).all()
#             assert (calls[0].args[3] == np.array([4.0, 4.1, 4.2])).all()

#             assert np.isclose(prediction, 4.1)
#             assert info["actual_k"] == 3

#         @patch("recommenders.knn_based_recommenders.get_k_top_neighbors")
#         @patch("recommenders.knn_based_recommenders.compute_neighborhood_cosine_similarity")
#         @patch("recommenders.knn_based_recommenders.calculate_weighted_rating")
#         def test_success_2(
#             self,
#             mock_calculate_weighted_rating,
#             mock_compute_neighborhood_cosine_similarity,
#             mock_get_k_top_neighbors,
#         ):
#             mock_get_k_top_neighbors.return_value = (
#                 np.array([1, 2, 3], dtype=np.float64),
#                 np.array([0.1, 0.2, 0.3], dtype=np.float64),
#                 np.array([4.0, 4.1, 4.2], dtype=np.float64),
#             )

#             mock_calculate_weighted_rating.return_value = 4.1

#             biaknn = TestBiAKNN.ConcreteBiAKNN()

#             mock_trainset = Mock()
#             mock_trainset.knows_user.return_value = True
#             mock_trainset.knows_item.return_value = True

#             biaknn.trainset = mock_trainset
#             biaknn.dataset = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
#             biaknn.neighborhood = {1: np.array([1, 2, 3, 4])}
#             biaknn.means = np.array([3.5, 4.0, 4.5])
#             biaknn.knn_type = "item"

#             prediction, info = biaknn.estimate(1, 2)

#             calls = mock_compute_neighborhood_cosine_similarity.call_args_list
#             assert len(calls) == 1
#             assert (calls[0].args[0] == biaknn.dataset.T).all()
#             assert calls[0].args[1] == biaknn.similarity_matrix
#             assert calls[0].args[2] == 2
#             assert (calls[0].args[3] == np.array([1, 3, 4])).all()

#             calls = mock_get_k_top_neighbors.call_args_list
#             assert len(calls) == 1
#             assert calls[0].args[0] == 2
#             assert calls[0].args[1] == 1
#             assert (calls[0].args[2] == biaknn.dataset.T).all()
#             assert (calls[0].args[3] == np.array([1, 3, 4])).all()
#             assert calls[0].args[4] == biaknn.similarity_matrix
#             assert (calls[0].args[5] == biaknn.means).all()
#             assert calls[0].args[6] == biaknn.knn_k

#             calls = mock_calculate_weighted_rating.call_args_list
#             assert len(calls) == 1
#             assert calls[0].args[0] == 4.5
#             assert (calls[0].args[1] == np.array([1, 2, 3])).all()
#             assert (calls[0].args[2] == np.array([0.1, 0.2, 0.3])).all()
#             assert (calls[0].args[3] == np.array([4.0, 4.1, 4.2])).all()

#             assert np.isclose(prediction, 4.1)
#             assert info["actual_k"] == 3
