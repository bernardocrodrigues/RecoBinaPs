"""
Tests for knn based recommenders from recommenders module.
"""

import numpy as np
import pytest

from unittest.mock import patch

from recommenders.knn_based_recommenders import (
    merge_biclusters,
    calculate_weighted_rating,
    get_k_top_neighbors,
    BiAKNN,
)
from pattern_mining.formal_concept_analysis import Concept, create_concept

# pylint: disable=missing-function-docstring,missing-class-docstring


class TestMergeBiclusters:
    def test_invalid_args(self):
        with pytest.raises(AssertionError):
            merge_biclusters("not a list")

        with pytest.raises(AssertionError):
            merge_biclusters([1, 2, 3])

        with pytest.raises(AssertionError):
            merge_biclusters([])

        with pytest.raises(AssertionError):
            merge_biclusters([Concept(np.array([1, 2, 3]), np.array([4, 5, 6])), 2])

        with pytest.raises(AssertionError):
            merge_biclusters([Concept(np.array([1, 2, 3]), np.array([4, 5, 6])), "not a concept"])

        with pytest.raises(AssertionError):
            merge_biclusters([Concept(np.array([]), np.array([4, 5, 6]))])

        with pytest.raises(AssertionError):
            merge_biclusters([Concept(np.array([1]), np.array([]))])

    def test_merge_single_bicluster(self):
        extent = np.array([1, 2, 3])
        intent = np.array([4, 5, 6])
        bicluster = create_concept(extent, intent)
        biclusters = [bicluster]
        merged_bicluster = merge_biclusters(biclusters)
        assert np.array_equal(merged_bicluster.extent, extent)
        assert np.array_equal(merged_bicluster.intent, intent)

    def test_merge_multiple_biclusters(self):
        bicluster1 = create_concept(np.array([1, 2, 3]), np.array([4, 5, 6]))
        bicluster2 = create_concept(np.array([3, 4, 5]), np.array([6, 7, 8]))
        bicluster3 = create_concept(np.array([5, 6, 7]), np.array([8, 9, 10]))

        biclusters = [bicluster1, bicluster2, bicluster3]

        merged_bicluster = merge_biclusters(biclusters)

        assert np.array_equal(merged_bicluster.extent, np.array([1, 2, 3, 4, 5, 6, 7]))
        assert np.array_equal(merged_bicluster.intent, np.array([4, 5, 6, 7, 8, 9, 10]))

        bicluster1 = create_concept(np.array([1, 2, 3, 8]), np.array([4, 5, 6, 10, 11]))
        bicluster2 = create_concept(np.array([3, 4, 5, 8]), np.array([6, 7, 8]))
        bicluster3 = create_concept(np.array([5, 6, 7, 8, 9]), np.array([8, 9, 10]))

        biclusters = [bicluster1, bicluster2, bicluster3]

        merged_bicluster = merge_biclusters(biclusters)

        assert np.array_equal(merged_bicluster.extent, np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))
        assert np.array_equal(merged_bicluster.intent, np.array([4, 5, 6, 7, 8, 9, 10, 11]))

        bicluster1 = create_concept(np.array([1]), np.array([4]))
        bicluster2 = create_concept(np.array([3]), np.array([6]))
        bicluster3 = create_concept(np.array([5]), np.array([8]))

        biclusters = [bicluster1, bicluster2, bicluster3]

        merged_bicluster = merge_biclusters(biclusters)

        assert np.array_equal(merged_bicluster.extent, np.array([1, 3, 5]))
        assert np.array_equal(merged_bicluster.intent, np.array([4, 6, 8]))

        bicluster1 = create_concept(np.array([1, 2, 3, 4, 5]), np.array([4, 5, 6, 7, 8, 9]))
        bicluster2 = create_concept(np.array([3]), np.array([6]))

        biclusters = [bicluster1, bicluster2, bicluster3]

        merged_bicluster = merge_biclusters(biclusters)

        assert np.array_equal(merged_bicluster.extent, np.array([1, 2, 3, 4, 5]))
        assert np.array_equal(merged_bicluster.intent, np.array([4, 5, 6, 7, 8, 9]))


class TestCalculateWeightedRating:
    def test_invalid_args(self):
        with pytest.raises(AssertionError):
            calculate_weighted_rating(
                "3.0",
                np.array([1, 2, 3], dtype=np.float64),
                np.array([1, 1, 1], dtype=np.float64),
                np.array([1, 2, 3], dtype=np.float64),
            )

        with pytest.raises(AssertionError):
            calculate_weighted_rating(
                1,
                np.array([1, 2, 3], dtype=np.float64),
                np.array([1, 1, 1], dtype=np.float64),
                np.array([1, 2, 3], dtype=np.float64),
            )

        with pytest.raises(AssertionError):
            calculate_weighted_rating(
                1.1,
                np.array([1, 2, 3], dtype=np.int64),
                np.array([1, 1, 1], dtype=np.float64),
                np.array([1, 2, 3], dtype=np.float64),
            )

        with pytest.raises(AssertionError):
            calculate_weighted_rating(
                1.1,
                np.array([1, 2, 3], dtype=np.float64),
                np.array([1, 1, 1], dtype=np.int64),
                np.array([1, 2, 3], dtype=np.float64),
            )

        with pytest.raises(AssertionError):
            calculate_weighted_rating(
                1.1,
                np.array([1, 2, 3], dtype=np.float64),
                np.array([1, 1, 1], dtype=np.float64),
                np.array([1, 2, 3], dtype=np.int64),
            )

        with pytest.raises(AssertionError):
            calculate_weighted_rating(
                1.1,
                np.array([1, 2, 3], dtype=np.float64),
                np.array([0, 1, 1], dtype=np.float64),
                np.array([1, 2, 3], dtype=np.float64),
            )

        with pytest.raises(AssertionError):
            calculate_weighted_rating(
                1.1,
                np.array([1, 2, 3], dtype=np.float64),
                np.array([1, 1], dtype=np.float64),
                np.array([1, 2, 3], dtype=np.float64),
            )

        with pytest.raises(AssertionError):
            calculate_weighted_rating(
                1.1,
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
            )

        with pytest.raises(AssertionError):
            calculate_weighted_rating(
                1.1,
                np.array([1, 2, 3], dtype=np.float64),
                [1, 1, 1],
                np.array([1, 2, 3], dtype=np.float64),
            )

    def test_success_1(self):
        target_mean = 3.0
        neighbors_ratings = np.array([4.0, 2.0, 5.0], dtype=np.float64)
        neighbors_similarities = np.array([0.1, 0.3, 0.6], dtype=np.float64)
        neighbors_means = np.array([3.5, 2.5, 4.0], dtype=np.float64)

        result = calculate_weighted_rating(
            target_mean, neighbors_ratings, neighbors_similarities, neighbors_means
        )
        expected_result = 3.0 + (
            (0.1 * (4.0 - 3.5) + 0.3 * (2.0 - 2.5) + 0.6 * (5.0 - 4.0)) / (0.1 + 0.3 + 0.6)
        )
        assert np.isclose(result, expected_result)

    def test_success_2(self):
        target_mean = 4.2
        neighbors_ratings = np.array([4.0], dtype=np.float64)
        neighbors_similarities = np.array([0.1], dtype=np.float64)
        neighbors_means = np.array([3.5], dtype=np.float64)

        result = calculate_weighted_rating(
            target_mean, neighbors_ratings, neighbors_similarities, neighbors_means
        )
        expected_result = 4.2 + ((0.1 * (4.0 - 3.5)) / (0.1))
        assert np.isclose(result, expected_result)

    def test_success_3(self):
        target_mean = 1.3
        neighbors_ratings = np.array([4.0, 1.1, 3.2, 2.7], dtype=np.float64)
        neighbors_similarities = np.array([0.1, 0.1, 0.2, 0.3], dtype=np.float64)
        neighbors_means = np.array([3.5, 1.2, 4.3, 1.7], dtype=np.float64)

        result = calculate_weighted_rating(
            target_mean, neighbors_ratings, neighbors_similarities, neighbors_means
        )
        expected_result = 1.3 + (
            (0.1 * (4.0 - 3.5) + 0.1 * (1.1 - 1.2) + 0.2 * (3.2 - 4.3) + 0.3 * (2.7 - 1.7))
            / (0.1 + 0.1 + 0.2 + 0.3)
        )
        assert np.isclose(result, expected_result)


class TestGetKTopNeighbors:
    # fmt: off
    square_dataset = np.array([[ 1,  2,  3,  4],
                               [ 5,  6,  7,  8],
                               [ 9, 10, 11, 12],
                               [13, 14, 15, 16]],
                                dtype=np.float64)

    similarity_matrix = np.array([[1.0, 0.5, 1.0, 0.1],
                                  [0.5, 1.0, 0.7, 0.3],
                                  [1.0, 0.7, 1.0, 0.0],
                                  [0.1, 0.3, 0.0, 1.0]])
    
    means = np.array([1.0, 2.0, 3.0, 4.0])
    # fmt: on

    @staticmethod
    def assert_result_get_k_top_neighbors(
        x,
        y,
        dataset,
        users_neighborhood,
        similarity_matrix,
        means,
        knn_k,
        expected_ratings,
        expected_similarity,
        expected_means,
    ):
        k_top_ratings, k_top_similarity, k_top_means = get_k_top_neighbors(
            x, y, dataset, users_neighborhood, similarity_matrix, means, knn_k
        )

        assert np.array_equal(k_top_ratings, expected_ratings)
        assert np.array_equal(k_top_similarity, expected_similarity)
        assert np.array_equal(k_top_means, expected_means)

    def test_invalid_args(self):
        dataset = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
        users_neighborhood = np.array([0, 1], dtype=np.int64)
        similarity_matrix = np.array(
            [[1, 0.9, 0.8], [0.9, 1, 0.7], [0.8, 0.7, 1]], dtype=np.float64
        )
        means = np.array([2, 5, 8], dtype=np.float64)
        knn_k = 2

        with pytest.raises(AssertionError):
            get_k_top_neighbors(
                "0", 1, dataset, users_neighborhood, similarity_matrix, means, knn_k
            )

        with pytest.raises(AssertionError):
            get_k_top_neighbors(
                0, "1", dataset, users_neighborhood, similarity_matrix, means, knn_k
            )

        with pytest.raises(AssertionError):
            get_k_top_neighbors(
                0, 1, dataset.tolist(), users_neighborhood, similarity_matrix, means, knn_k
            )

        with pytest.raises(AssertionError):
            get_k_top_neighbors(
                0, 1, dataset, users_neighborhood.tolist(), similarity_matrix, means, knn_k
            )

        with pytest.raises(AssertionError):
            get_k_top_neighbors(
                0, 1, dataset, users_neighborhood, similarity_matrix.tolist(), means, knn_k
            )

        with pytest.raises(AssertionError):
            get_k_top_neighbors(
                0, 1, dataset, users_neighborhood, similarity_matrix, means.tolist(), knn_k
            )

        with pytest.raises(AssertionError):
            get_k_top_neighbors(0, 1, dataset, users_neighborhood, similarity_matrix, means, "2")

    def test_no_similarity_square_dataset(self):
        x = 0
        y = 0

        # fmt: off
        square_dataset = np.array([[ 1,  2,  3,  4],
                                   [ 5,  6,  7,  8],
                                   [ 9, 10, 11, 12],
                                   [13, 14, 15, 16]],
                                   dtype=np.float64)
        
        similarity_matrix = np.array([[1.0, 0.0, 0.0, 0.0],
                                      [0.0, 1.0, 0.0, 0.0],
                                      [0.0, 0.0, 1.0, 0.0],
                                      [0.0, 0.0, 0.0, 1.0]])
        # fmt: on

        empty_array = np.array([])

        users_neighborhood = np.array([1, 2])
        means = np.array([2.0, 5.0, 8.0, 1.0])
        knn_k = 4

        TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
            x=x,
            y=y,
            dataset=square_dataset,
            users_neighborhood=users_neighborhood,
            similarity_matrix=similarity_matrix,
            means=means,
            knn_k=knn_k,
            expected_ratings=empty_array,
            expected_similarity=empty_array,
            expected_means=empty_array,
        )

        knn_k = 3
        TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
            x=x,
            y=y,
            dataset=square_dataset,
            users_neighborhood=users_neighborhood,
            similarity_matrix=similarity_matrix,
            means=means,
            knn_k=knn_k,
            expected_ratings=empty_array,
            expected_similarity=empty_array,
            expected_means=empty_array,
        )

        knn_k = 2
        TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
            x=x,
            y=y,
            dataset=square_dataset,
            users_neighborhood=users_neighborhood,
            similarity_matrix=similarity_matrix,
            means=means,
            knn_k=knn_k,
            expected_ratings=empty_array,
            expected_similarity=empty_array,
            expected_means=empty_array,
        )

        knn_k = 1
        TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
            x=x,
            y=y,
            dataset=square_dataset,
            users_neighborhood=users_neighborhood,
            similarity_matrix=similarity_matrix,
            means=means,
            knn_k=knn_k,
            expected_ratings=empty_array,
            expected_similarity=empty_array,
            expected_means=empty_array,
        )

        knn_k = 2
        y = 1
        TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
            x=x,
            y=y,
            dataset=square_dataset,
            users_neighborhood=users_neighborhood,
            similarity_matrix=similarity_matrix,
            means=means,
            knn_k=knn_k,
            expected_ratings=empty_array,
            expected_similarity=empty_array,
            expected_means=empty_array,
        )

        y = 2
        TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
            x=x,
            y=y,
            dataset=square_dataset,
            users_neighborhood=users_neighborhood,
            similarity_matrix=similarity_matrix,
            means=means,
            knn_k=knn_k,
            expected_ratings=empty_array,
            expected_similarity=empty_array,
            expected_means=empty_array,
        )

        y = 3
        TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
            x=x,
            y=y,
            dataset=square_dataset,
            users_neighborhood=users_neighborhood,
            similarity_matrix=similarity_matrix,
            means=means,
            knn_k=knn_k,
            expected_ratings=empty_array,
            expected_similarity=empty_array,
            expected_means=empty_array,
        )

        y = 2
        x = 1
        users_neighborhood = np.array([0, 2])
        TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
            x=x,
            y=y,
            dataset=square_dataset,
            users_neighborhood=users_neighborhood,
            similarity_matrix=similarity_matrix,
            means=means,
            knn_k=2,
            expected_ratings=empty_array,
            expected_similarity=empty_array,
            expected_means=empty_array,
        )

        x = 2
        y = 1
        users_neighborhood = np.array([0, 3])
        TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
            x=x,
            y=y,
            dataset=square_dataset,
            users_neighborhood=users_neighborhood,
            similarity_matrix=similarity_matrix,
            means=means,
            knn_k=2,
            expected_ratings=empty_array,
            expected_similarity=empty_array,
            expected_means=empty_array,
        )

        x = 2
        y = 2
        users_neighborhood = np.array([0, 1])
        TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
            x=x,
            y=y,
            dataset=square_dataset,
            users_neighborhood=users_neighborhood,
            similarity_matrix=similarity_matrix,
            means=means,
            knn_k=2,
            expected_ratings=empty_array,
            expected_similarity=empty_array,
            expected_means=empty_array,
        )

        x = 3
        y = 2
        users_neighborhood = np.array([0, 1, 2])
        TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
            x=x,
            y=y,
            dataset=square_dataset,
            users_neighborhood=users_neighborhood,
            similarity_matrix=similarity_matrix,
            means=means,
            knn_k=2,
            expected_ratings=empty_array,
            expected_similarity=empty_array,
            expected_means=empty_array,
        )

    def test_no_similarity_non_square_dataset(self):
        # fmt: off
        non_square_dataset = np.array([[ 1,  2,  3,  4],
                                       [ 5,  6,  7,  8],
                                       [ 9, 10, 11, 12]],
                                       dtype=np.float64)
        
        similarity_matrix = np.array([[1.0, 0.0, 0.0],
                                      [0.0, 1.0, 0.0],
                                      [0.0, 0.0, 1.0]],
                                      dtype=np.float64)
        # fmt: on

        empty_array = np.array([])

        users_neighborhood = np.array([1, 2])
        means = np.array([2.0, 5.0, 8.0])
        knn_k = 3

        x = 0
        y = 0
        TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
            x=x,
            y=y,
            dataset=non_square_dataset,
            users_neighborhood=users_neighborhood,
            similarity_matrix=similarity_matrix,
            means=means,
            knn_k=knn_k,
            expected_ratings=empty_array,
            expected_similarity=empty_array,
            expected_means=empty_array,
        )

        x = 0
        y = 1
        TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
            x=x,
            y=y,
            dataset=non_square_dataset,
            users_neighborhood=users_neighborhood,
            similarity_matrix=similarity_matrix,
            means=means,
            knn_k=knn_k,
            expected_ratings=empty_array,
            expected_similarity=empty_array,
            expected_means=empty_array,
        )

        x = 0
        y = 2
        TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
            x=x,
            y=y,
            dataset=non_square_dataset,
            users_neighborhood=users_neighborhood,
            similarity_matrix=similarity_matrix,
            means=means,
            knn_k=knn_k,
            expected_ratings=empty_array,
            expected_similarity=empty_array,
            expected_means=empty_array,
        )

        x = 1
        y = 0
        users_neighborhood = np.array([0, 2])
        TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
            x=x,
            y=y,
            dataset=non_square_dataset,
            users_neighborhood=users_neighborhood,
            similarity_matrix=similarity_matrix,
            means=means,
            knn_k=knn_k,
            expected_ratings=empty_array,
            expected_similarity=empty_array,
            expected_means=empty_array,
        )

        x = 1
        y = 1
        TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
            x=x,
            y=y,
            dataset=non_square_dataset,
            users_neighborhood=users_neighborhood,
            similarity_matrix=similarity_matrix,
            means=means,
            knn_k=knn_k,
            expected_ratings=empty_array,
            expected_similarity=empty_array,
            expected_means=empty_array,
        )

        x = 1
        y = 2
        users_neighborhood = np.array([2])
        TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
            x=x,
            y=y,
            dataset=non_square_dataset,
            users_neighborhood=users_neighborhood,
            similarity_matrix=similarity_matrix,
            means=means,
            knn_k=1,
            expected_ratings=empty_array,
            expected_similarity=empty_array,
            expected_means=empty_array,
        )

        # fmt: off
        non_square_dataset = np.array([[ 1,  2,  3],
                                       [ 4 , 5,  6],
                                       [ 7,  8,  9],
                                       [10, 11, 12]],
                                       dtype=np.float64)
        
        similarity_matrix = np.array([[1.0, 0.0, 0.0, 0.0],
                                      [0.0, 1.0, 0.0, 0.0],
                                      [0.0, 0.0, 1.0, 0.0],
                                      [0.0, 0.0, 0.0, 1.0]])
        
        means = np.array([2.0, 5.0, 8.0, 1.0])
        # fmt: on

        x = 0
        y = 0
        users_neighborhood = np.array([1, 2, 3])
        TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
            x=x,
            y=y,
            dataset=non_square_dataset,
            users_neighborhood=users_neighborhood,
            similarity_matrix=similarity_matrix,
            means=means,
            knn_k=knn_k,
            expected_ratings=empty_array,
            expected_similarity=empty_array,
            expected_means=empty_array,
        )

        x = 0
        y = 1
        TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
            x=x,
            y=y,
            dataset=non_square_dataset,
            users_neighborhood=users_neighborhood,
            similarity_matrix=similarity_matrix,
            means=means,
            knn_k=knn_k,
            expected_ratings=empty_array,
            expected_similarity=empty_array,
            expected_means=empty_array,
        )

        knn_k = 1
        TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
            x=x,
            y=y,
            dataset=non_square_dataset,
            users_neighborhood=users_neighborhood,
            similarity_matrix=similarity_matrix,
            means=means,
            knn_k=knn_k,
            expected_ratings=empty_array,
            expected_similarity=empty_array,
            expected_means=empty_array,
        )

        knn_k = 3
        x = 0
        y = 2
        users_neighborhood = np.array([1, 3])
        TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
            x=x,
            y=y,
            dataset=non_square_dataset,
            users_neighborhood=users_neighborhood,
            similarity_matrix=similarity_matrix,
            means=means,
            knn_k=3,
            expected_ratings=empty_array,
            expected_similarity=empty_array,
            expected_means=empty_array,
        )

        x = 0
        y = 2
        users_neighborhood = np.array([1, 2, 3])
        TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
            x=x,
            y=y,
            dataset=non_square_dataset,
            users_neighborhood=users_neighborhood,
            similarity_matrix=similarity_matrix,
            means=means,
            knn_k=3,
            expected_ratings=empty_array,
            expected_similarity=empty_array,
            expected_means=empty_array,
        )

        x = 0
        y = 2
        users_neighborhood = np.array([1, 2, 3])
        knn_k = 2
        TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
            x=x,
            y=y,
            dataset=non_square_dataset,
            users_neighborhood=users_neighborhood,
            similarity_matrix=similarity_matrix,
            means=means,
            knn_k=2,
            expected_ratings=empty_array,
            expected_similarity=empty_array,
            expected_means=empty_array,
        )

    def test_no_valid_neighbors(self):
        # fmt: off
        dataset = np.array([[ 1, np.nan,  3],
                            [ 4, np.nan,  6],
                            [ 7, np.nan,  9]],
                            dtype=np.float64)
        
        similarity_matrix = np.array([[1.0, 0.5, 0.2],
                                      [0.5, 1.0, 0.8],
                                      [0.2, 0.8, 1.0]],
                                      dtype=np.float64)
        # fmt: on

        empty_array = np.array([])

        users_neighborhood = np.array([1, 2])
        means = np.array([2.0, 5.0, 8.0])
        knn_k = 3

        x = 0
        y = 1
        TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
            x=x,
            y=y,
            dataset=dataset,
            users_neighborhood=users_neighborhood,
            similarity_matrix=similarity_matrix,
            means=means,
            knn_k=knn_k,
            expected_ratings=empty_array,
            expected_similarity=empty_array,
            expected_means=empty_array,
        )

        users_neighborhood = np.array([2])
        TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
            x=x,
            y=y,
            dataset=dataset,
            users_neighborhood=users_neighborhood,
            similarity_matrix=similarity_matrix,
            means=means,
            knn_k=knn_k,
            expected_ratings=empty_array,
            expected_similarity=empty_array,
            expected_means=empty_array,
        )

    class TestUserScenario:
        class TestNoNaNs:
            class TestNoZeroSimilarity:
                def test_1(self):
                    x = 0
                    y = 0

                    users_neighborhood = np.array([1, 2])
                    knn_k = 3

                    expected_ratings = np.array([9.0, 5.0])
                    expected_similarity = np.array([1.0, 0.5])
                    expected_means = np.array([3.0, 2.0])

                    TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                        x=x,
                        y=y,
                        dataset=TestGetKTopNeighbors.square_dataset,
                        users_neighborhood=users_neighborhood,
                        similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
                        means=TestGetKTopNeighbors.means,
                        knn_k=knn_k,
                        expected_ratings=expected_ratings,
                        expected_similarity=expected_similarity,
                        expected_means=expected_means,
                    )

                def test_2(self):
                    x = 0
                    y = 0

                    users_neighborhood = np.array([1, 2, 3])
                    knn_k = 3

                    expected_ratings = np.array([9.0, 5.0, 13.0])
                    expected_similarity = np.array([1.0, 0.5, 0.1])
                    expected_means = np.array([3.0, 2.0, 4.0])

                    TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                        x=x,
                        y=y,
                        dataset=TestGetKTopNeighbors.square_dataset,
                        users_neighborhood=users_neighborhood,
                        similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
                        means=TestGetKTopNeighbors.means,
                        knn_k=knn_k,
                        expected_ratings=expected_ratings,
                        expected_similarity=expected_similarity,
                        expected_means=expected_means,
                    )

                def test_3(self):
                    x = 0
                    y = 0

                    users_neighborhood = np.array([1, 2, 3])
                    knn_k = 1

                    expected_ratings = np.array([9.0])
                    expected_similarity = np.array([1.0])
                    expected_means = np.array([3.0])

                    TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                        x=x,
                        y=y,
                        dataset=TestGetKTopNeighbors.square_dataset,
                        users_neighborhood=users_neighborhood,
                        similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
                        means=TestGetKTopNeighbors.means,
                        knn_k=knn_k,
                        expected_ratings=expected_ratings,
                        expected_similarity=expected_similarity,
                        expected_means=expected_means,
                    )

                def test_4(self):
                    x = 1
                    y = 2

                    users_neighborhood = np.array([0, 2, 3])
                    knn_k = 1

                    expected_ratings = np.array([11.0])
                    expected_similarity = np.array([0.7])
                    expected_means = np.array([3.0])

                    TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                        x=x,
                        y=y,
                        dataset=TestGetKTopNeighbors.square_dataset,
                        users_neighborhood=users_neighborhood,
                        similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
                        means=TestGetKTopNeighbors.means,
                        knn_k=knn_k,
                        expected_ratings=expected_ratings,
                        expected_similarity=expected_similarity,
                        expected_means=expected_means,
                    )

                def test_5(self):
                    x = 1
                    y = 3

                    users_neighborhood = np.array([0, 2, 3])
                    knn_k = 1

                    expected_ratings = np.array([12.0])
                    expected_similarity = np.array([0.7])
                    expected_means = np.array([3.0])

                    TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                        x=x,
                        y=y,
                        dataset=TestGetKTopNeighbors.square_dataset,
                        users_neighborhood=users_neighborhood,
                        similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
                        means=TestGetKTopNeighbors.means,
                        knn_k=knn_k,
                        expected_ratings=expected_ratings,
                        expected_similarity=expected_similarity,
                        expected_means=expected_means,
                    )

                def test_6(self):
                    x = 1
                    y = 3

                    users_neighborhood = np.array([0, 2])
                    knn_k = 1

                    expected_ratings = np.array([12.0])
                    expected_similarity = np.array([0.7])
                    expected_means = np.array([3.0])

                    TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                        x=x,
                        y=y,
                        dataset=TestGetKTopNeighbors.square_dataset,
                        users_neighborhood=users_neighborhood,
                        similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
                        means=TestGetKTopNeighbors.means,
                        knn_k=knn_k,
                        expected_ratings=expected_ratings,
                        expected_similarity=expected_similarity,
                        expected_means=expected_means,
                    )

                def test_7(self):
                    x = 1
                    y = 3

                    users_neighborhood = np.array([0])
                    knn_k = 2

                    expected_ratings = np.array([4.0])
                    expected_similarity = np.array([0.5])
                    expected_means = np.array([1.0])

                    TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                        x=x,
                        y=y,
                        dataset=TestGetKTopNeighbors.square_dataset,
                        users_neighborhood=users_neighborhood,
                        similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
                        means=TestGetKTopNeighbors.means,
                        knn_k=knn_k,
                        expected_ratings=expected_ratings,
                        expected_similarity=expected_similarity,
                        expected_means=expected_means,
                    )

                def test_8(self):
                    x = 0
                    y = 1

                    users_neighborhood = np.array([1, 2, 3])
                    knn_k = 1

                    expected_ratings = np.array([10.0])
                    expected_similarity = np.array([1.0])
                    expected_means = np.array([3.0])

                    TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                        x=x,
                        y=y,
                        dataset=TestGetKTopNeighbors.square_dataset,
                        users_neighborhood=users_neighborhood,
                        similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
                        means=TestGetKTopNeighbors.means,
                        knn_k=knn_k,
                        expected_ratings=expected_ratings,
                        expected_similarity=expected_similarity,
                        expected_means=expected_means,
                    )

            class TestWithZeroSimilarity:
                def test_1(self):
                    x = 2
                    y = 1

                    users_neighborhood = np.array([0, 1, 3])
                    knn_k = 1

                    expected_ratings = np.array([2.0])
                    expected_similarity = np.array([1.0])
                    expected_means = np.array([1.0])

                    TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                        x=x,
                        y=y,
                        dataset=TestGetKTopNeighbors.square_dataset,
                        users_neighborhood=users_neighborhood,
                        similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
                        means=TestGetKTopNeighbors.means,
                        knn_k=knn_k,
                        expected_ratings=expected_ratings,
                        expected_similarity=expected_similarity,
                        expected_means=expected_means,
                    )

                def test_2(self):
                    x = 2
                    y = 2

                    users_neighborhood = np.array([0, 1, 3])
                    knn_k = 2

                    expected_ratings = np.array([3.0, 7.0])
                    expected_similarity = np.array([1.0, 0.7])
                    expected_means = np.array([1.0, 2.0])

                    TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                        x=x,
                        y=y,
                        dataset=TestGetKTopNeighbors.square_dataset,
                        users_neighborhood=users_neighborhood,
                        similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
                        means=TestGetKTopNeighbors.means,
                        knn_k=knn_k,
                        expected_ratings=expected_ratings,
                        expected_similarity=expected_similarity,
                        expected_means=expected_means,
                    )

                def test_3(self):
                    x = 2
                    y = 2

                    users_neighborhood = np.array([0, 1, 3])
                    knn_k = 3

                    expected_ratings = np.array([3.0, 7.0])
                    expected_similarity = np.array([1.0, 0.7])
                    expected_means = np.array([1.0, 2.0])

                    TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                        x=x,
                        y=y,
                        dataset=TestGetKTopNeighbors.square_dataset,
                        users_neighborhood=users_neighborhood,
                        similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
                        means=TestGetKTopNeighbors.means,
                        knn_k=knn_k,
                        expected_ratings=expected_ratings,
                        expected_similarity=expected_similarity,
                        expected_means=expected_means,
                    )

                def test_4(self):
                    x = 2
                    y = 3

                    users_neighborhood = np.array([0, 1, 3])
                    knn_k = 3

                    expected_ratings = np.array([4.0, 8.0])
                    expected_similarity = np.array([1.0, 0.7])
                    expected_means = np.array([1.0, 2.0])

                    TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                        x=x,
                        y=y,
                        dataset=TestGetKTopNeighbors.square_dataset,
                        users_neighborhood=users_neighborhood,
                        similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
                        means=TestGetKTopNeighbors.means,
                        knn_k=knn_k,
                        expected_ratings=expected_ratings,
                        expected_similarity=expected_similarity,
                        expected_means=expected_means,
                    )

                def test_5(self):
                    x = 2
                    y = 3

                    users_neighborhood = np.array([3])
                    knn_k = 3

                    expected_ratings = np.array([])
                    expected_similarity = np.array([])
                    expected_means = np.array([])

                    TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                        x=x,
                        y=y,
                        dataset=TestGetKTopNeighbors.square_dataset,
                        users_neighborhood=users_neighborhood,
                        similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
                        means=TestGetKTopNeighbors.means,
                        knn_k=knn_k,
                        expected_ratings=expected_ratings,
                        expected_similarity=expected_similarity,
                        expected_means=expected_means,
                    )

        class TestWithNaNsOnDataset:
            class TestNoZeroSimilarity:
                # fmt: off
                square_dataset = np.array([[ 1,  2,  3,  np.NAN],
                                           [ 5,  6,  np.NAN,  8],
                                           [ np.NAN, 10, 11, 12],
                                           [13, np.NAN, 15, 16]],
                                           dtype=np.float64)
                # fmt: on

                def test_1(self):
                    x = 1
                    y = 1

                    users_neighborhood = np.array([0, 2, 3])
                    knn_k = 2

                    expected_ratings = np.array([10.0, 2.0])
                    expected_similarity = np.array([0.7, 0.5])
                    expected_means = np.array([3.0, 1.0])

                    TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                        x=x,
                        y=y,
                        dataset=self.square_dataset,
                        users_neighborhood=users_neighborhood,
                        similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
                        means=TestGetKTopNeighbors.means,
                        knn_k=knn_k,
                        expected_ratings=expected_ratings,
                        expected_similarity=expected_similarity,
                        expected_means=expected_means,
                    )

                def test_2(self):
                    x = 1
                    y = 0

                    users_neighborhood = np.array([0, 2, 3])
                    knn_k = 2

                    expected_ratings = np.array([1.0, 13.0])
                    expected_similarity = np.array([0.5, 0.3])
                    expected_means = np.array([1.0, 4.0])

                    TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                        x=x,
                        y=y,
                        dataset=self.square_dataset,
                        users_neighborhood=users_neighborhood,
                        similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
                        means=TestGetKTopNeighbors.means,
                        knn_k=knn_k,
                        expected_ratings=expected_ratings,
                        expected_similarity=expected_similarity,
                        expected_means=expected_means,
                    )

                def test_3(self):
                    x = 1
                    y = 0

                    users_neighborhood = np.array([0, 3])
                    knn_k = 2

                    expected_ratings = np.array([1.0, 13.0])
                    expected_similarity = np.array([0.5, 0.3])
                    expected_means = np.array([1.0, 4.0])

                    TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                        x=x,
                        y=y,
                        dataset=self.square_dataset,
                        users_neighborhood=users_neighborhood,
                        similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
                        means=TestGetKTopNeighbors.means,
                        knn_k=knn_k,
                        expected_ratings=expected_ratings,
                        expected_similarity=expected_similarity,
                        expected_means=expected_means,
                    )

            class TestWithZeroSimilarity:
                # fmt: off
                square_dataset = np.array([[ 1,  2,  3,  np.NAN],
                                           [ 5,  6,  np.NAN,  8],
                                           [ np.NAN, 10, 11, 12],
                                           [13, np.NAN, 15, 16]],
                                           dtype=np.float64)
                # fmt: on
                def test_1(self):
                    x = 3
                    y = 3

                    users_neighborhood = np.array([0, 1, 2])
                    knn_k = 3

                    expected_ratings = np.array([8.0])
                    expected_similarity = np.array([0.3])
                    expected_means = np.array([2.0])

                    TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                        x=x,
                        y=y,
                        dataset=self.square_dataset,
                        users_neighborhood=users_neighborhood,
                        similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
                        means=TestGetKTopNeighbors.means,
                        knn_k=knn_k,
                        expected_ratings=expected_ratings,
                        expected_similarity=expected_similarity,
                        expected_means=expected_means,
                    )

                def test_2(self):
                    x = 3
                    y = 3

                    users_neighborhood = np.array([0, 1, 2])
                    knn_k = 2

                    expected_ratings = np.array([8.0])
                    expected_similarity = np.array([0.3])
                    expected_means = np.array([2.0])

                    TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                        x=x,
                        y=y,
                        dataset=self.square_dataset,
                        users_neighborhood=users_neighborhood,
                        similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
                        means=TestGetKTopNeighbors.means,
                        knn_k=knn_k,
                        expected_ratings=expected_ratings,
                        expected_similarity=expected_similarity,
                        expected_means=expected_means,
                    )

                def test_3(self):
                    x = 3
                    y = 3

                    users_neighborhood = np.array([0, 1, 2])
                    knn_k = 1

                    expected_ratings = np.array([8.0])
                    expected_similarity = np.array([0.3])
                    expected_means = np.array([2.0])

                    TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                        x=x,
                        y=y,
                        dataset=self.square_dataset,
                        users_neighborhood=users_neighborhood,
                        similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
                        means=TestGetKTopNeighbors.means,
                        knn_k=knn_k,
                        expected_ratings=expected_ratings,
                        expected_similarity=expected_similarity,
                        expected_means=expected_means,
                    )

                def test_5(self):
                    x = 3
                    y = 3

                    users_neighborhood = np.array([1])
                    knn_k = 2

                    expected_ratings = np.array([8.0])
                    expected_similarity = np.array([0.3])
                    expected_means = np.array([2.0])

                    TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                        x=x,
                        y=y,
                        dataset=self.square_dataset,
                        users_neighborhood=users_neighborhood,
                        similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
                        means=TestGetKTopNeighbors.means,
                        knn_k=knn_k,
                        expected_ratings=expected_ratings,
                        expected_similarity=expected_similarity,
                        expected_means=expected_means,
                    )

                def test_6(self):
                    x = 2
                    y = 3

                    users_neighborhood = np.array([0, 1, 3])
                    knn_k = 2

                    expected_ratings = np.array([8.0])
                    expected_similarity = np.array([0.7])
                    expected_means = np.array([2.0])

                    TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                        x=x,
                        y=y,
                        dataset=self.square_dataset,
                        users_neighborhood=users_neighborhood,
                        similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
                        means=TestGetKTopNeighbors.means,
                        knn_k=knn_k,
                        expected_ratings=expected_ratings,
                        expected_similarity=expected_similarity,
                        expected_means=expected_means,
                    )

                def test_7(self):
                    x = 2
                    y = 2

                    users_neighborhood = np.array([0, 1, 3])
                    knn_k = 2

                    expected_ratings = np.array([3.0])
                    expected_similarity = np.array([1.0])
                    expected_means = np.array([1.0])

                    TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                        x=x,
                        y=y,
                        dataset=self.square_dataset,
                        users_neighborhood=users_neighborhood,
                        similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
                        means=TestGetKTopNeighbors.means,
                        knn_k=knn_k,
                        expected_ratings=expected_ratings,
                        expected_similarity=expected_similarity,
                        expected_means=expected_means,
                    )

        class TestWithNaNsOnSimilarity:
            class TestNoZeroSimilarity:
                # fmt: off
                similarity_matrix = np.array([[1.0, np.NAN, 1.0, 0.1],
                                              [0.5, 1.0, np.NAN, 0.3],
                                              [1.0, 0.7, np.NAN, 0.0],
                                              [np.NAN, 0.3, 0.0, 1.0]],
                                              dtype=np.float64)
                # fmt: on

                def test_1(self):
                    x = 0
                    y = 0

                    users_neighborhood = np.array([1, 2, 3])
                    knn_k = 2

                    expected_ratings = np.array([9.0, 13.0])
                    expected_similarity = np.array([1.0, 0.1])
                    expected_means = np.array([3.0, 4.0])

                    TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                        x=x,
                        y=y,
                        dataset=TestGetKTopNeighbors.square_dataset,
                        users_neighborhood=users_neighborhood,
                        similarity_matrix=self.similarity_matrix,
                        means=TestGetKTopNeighbors.means,
                        knn_k=knn_k,
                        expected_ratings=expected_ratings,
                        expected_similarity=expected_similarity,
                        expected_means=expected_means,
                    )

                def test_2(self):
                    x = 0
                    y = 0

                    users_neighborhood = np.array([1, 2, 3])
                    knn_k = 1

                    expected_ratings = np.array([9.0])
                    expected_similarity = np.array([1.0])
                    expected_means = np.array([3.0])

                    TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                        x=x,
                        y=y,
                        dataset=TestGetKTopNeighbors.square_dataset,
                        users_neighborhood=users_neighborhood,
                        similarity_matrix=self.similarity_matrix,
                        means=TestGetKTopNeighbors.means,
                        knn_k=knn_k,
                        expected_ratings=expected_ratings,
                        expected_similarity=expected_similarity,
                        expected_means=expected_means,
                    )

                def test_3(self):
                    x = 0
                    y = 0

                    users_neighborhood = np.array([1, 2, 3])
                    knn_k = 3

                    expected_ratings = np.array([9.0, 13.0])
                    expected_similarity = np.array([1.0, 0.1])
                    expected_means = np.array([3.0, 4.0])

                    TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                        x=x,
                        y=y,
                        dataset=TestGetKTopNeighbors.square_dataset,
                        users_neighborhood=users_neighborhood,
                        similarity_matrix=self.similarity_matrix,
                        means=TestGetKTopNeighbors.means,
                        knn_k=knn_k,
                        expected_ratings=expected_ratings,
                        expected_similarity=expected_similarity,
                        expected_means=expected_means,
                    )

                def test_4(self):
                    x = 0
                    y = 1

                    users_neighborhood = np.array([1, 2, 3])
                    knn_k = 2

                    expected_ratings = np.array([10.0, 14.0])
                    expected_similarity = np.array([1.0, 0.1])
                    expected_means = np.array([3.0, 4.0])

                    TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                        x=x,
                        y=y,
                        dataset=TestGetKTopNeighbors.square_dataset,
                        users_neighborhood=users_neighborhood,
                        similarity_matrix=self.similarity_matrix,
                        means=TestGetKTopNeighbors.means,
                        knn_k=knn_k,
                        expected_ratings=expected_ratings,
                        expected_similarity=expected_similarity,
                        expected_means=expected_means,
                    )

            class TestWithZeroSimilarity:
                # fmt: off
                similarity_matrix = np.array([[1.0, np.NAN, 1.0, 0.1],
                                              [0.5, 1.0, np.NAN, 0.3],
                                              [1.0, 0.7, np.NAN, 0.0],
                                              [np.NAN, 0.3, 0.0, 1.0]],
                                              dtype=np.float64)
                # fmt: on
                def test_1(self):
                    x = 3
                    y = 3

                    users_neighborhood = np.array([0, 1, 2])
                    knn_k = 2

                    expected_ratings = np.array([8.0])
                    expected_similarity = np.array([0.3])
                    expected_means = np.array([2.0])

                    TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                        x=x,
                        y=y,
                        dataset=TestGetKTopNeighbors.square_dataset,
                        similarity_matrix=self.similarity_matrix,
                        users_neighborhood=users_neighborhood,
                        means=TestGetKTopNeighbors.means,
                        knn_k=knn_k,
                        expected_ratings=expected_ratings,
                        expected_similarity=expected_similarity,
                        expected_means=expected_means,
                    )

                def test_2(self):
                    x = 3
                    y = 3

                    users_neighborhood = np.array([0, 1, 2])
                    knn_k = 1

                    expected_ratings = np.array([8.0])
                    expected_similarity = np.array([0.3])
                    expected_means = np.array([2.0])

                    TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                        x=x,
                        y=y,
                        dataset=TestGetKTopNeighbors.square_dataset,
                        similarity_matrix=self.similarity_matrix,
                        users_neighborhood=users_neighborhood,
                        means=TestGetKTopNeighbors.means,
                        knn_k=knn_k,
                        expected_ratings=expected_ratings,
                        expected_similarity=expected_similarity,
                        expected_means=expected_means,
                    )

                def test_3(self):
                    x = 3
                    y = 1

                    users_neighborhood = np.array([0, 1, 2])
                    knn_k = 3

                    expected_ratings = np.array([6.0])
                    expected_similarity = np.array([0.3])
                    expected_means = np.array([2.0])

                    TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                        x=x,
                        y=y,
                        dataset=TestGetKTopNeighbors.square_dataset,
                        similarity_matrix=self.similarity_matrix,
                        users_neighborhood=users_neighborhood,
                        means=TestGetKTopNeighbors.means,
                        knn_k=knn_k,
                        expected_ratings=expected_ratings,
                        expected_similarity=expected_similarity,
                        expected_means=expected_means,
                    )

        class TestWithNaNsOnDatasetAndSimilarity:
            # fmt: off
            square_dataset = np.array([[ 1,  2,  3,  np.NAN],
                                       [ 5,  6,  np.NAN,  8],
                                       [ np.NAN, 10, 11, 12],
                                       [13, np.NAN, 15, 16]],
                                       dtype=np.float64)
            similarity_matrix = np.array([[1.0, np.NAN, 1.0, 0.1],
                                          [0.5, 1.0, np.NAN, 0.3],
                                          [1.0, 0.7, np.NAN, 0.0],
                                          [np.NAN, 0.3, 0.0, 1.0]],
                                          dtype=np.float64)
            # fmt: on

            def test_1(self):
                x = 1
                y = 1

                users_neighborhood = np.array([0, 2, 3])
                knn_k = 2

                expected_ratings = np.array([2.0])
                expected_similarity = np.array([0.5])
                expected_means = np.array([1.0])

                TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                    x=x,
                    y=y,
                    dataset=self.square_dataset,
                    similarity_matrix=self.similarity_matrix,
                    users_neighborhood=users_neighborhood,
                    means=TestGetKTopNeighbors.means,
                    knn_k=knn_k,
                    expected_ratings=expected_ratings,
                    expected_similarity=expected_similarity,
                    expected_means=expected_means,
                )

            def test_2(self):
                x = 1
                y = 1

                users_neighborhood = np.array([2, 3])
                knn_k = 2

                expected_ratings = np.array([])
                expected_similarity = np.array([])
                expected_means = np.array([])

                TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                    x=x,
                    y=y,
                    dataset=self.square_dataset,
                    similarity_matrix=self.similarity_matrix,
                    users_neighborhood=users_neighborhood,
                    means=TestGetKTopNeighbors.means,
                    knn_k=knn_k,
                    expected_ratings=expected_ratings,
                    expected_similarity=expected_similarity,
                    expected_means=expected_means,
                )

            def test_3(self):
                x = 1
                y = 3

                users_neighborhood = np.array([2, 3])
                knn_k = 2

                expected_ratings = np.array([16.0])
                expected_similarity = np.array([0.3])
                expected_means = np.array([4.0])

                TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                    x=x,
                    y=y,
                    dataset=self.square_dataset,
                    similarity_matrix=self.similarity_matrix,
                    users_neighborhood=users_neighborhood,
                    means=TestGetKTopNeighbors.means,
                    knn_k=knn_k,
                    expected_ratings=expected_ratings,
                    expected_similarity=expected_similarity,
                    expected_means=expected_means,
                )

            def test_4(self):
                x = 1
                y = 3

                users_neighborhood = np.array([3])
                knn_k = 2

                expected_ratings = np.array([16.0])
                expected_similarity = np.array([0.3])
                expected_means = np.array([4.0])

                TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                    x=x,
                    y=y,
                    dataset=self.square_dataset,
                    similarity_matrix=self.similarity_matrix,
                    users_neighborhood=users_neighborhood,
                    means=TestGetKTopNeighbors.means,
                    knn_k=knn_k,
                    expected_ratings=expected_ratings,
                    expected_similarity=expected_similarity,
                    expected_means=expected_means,
                )

            def test_5(self):
                x = 1
                y = 3

                users_neighborhood = np.array([2])
                knn_k = 2

                expected_ratings = np.array([])
                expected_similarity = np.array([])
                expected_means = np.array([])

                TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                    x=x,
                    y=y,
                    dataset=self.square_dataset,
                    similarity_matrix=self.similarity_matrix,
                    users_neighborhood=users_neighborhood,
                    means=TestGetKTopNeighbors.means,
                    knn_k=knn_k,
                    expected_ratings=expected_ratings,
                    expected_similarity=expected_similarity,
                    expected_means=expected_means,
                )

            def test_6(self):
                x = 2
                y = 1

                users_neighborhood = np.array([0, 1, 3])
                knn_k = 2

                expected_ratings = np.array([2.0, 6.0])
                expected_similarity = np.array([1.0, 0.7])
                expected_means = np.array([1.0, 2.0])

                TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                    x=x,
                    y=y,
                    dataset=self.square_dataset,
                    similarity_matrix=self.similarity_matrix,
                    users_neighborhood=users_neighborhood,
                    means=TestGetKTopNeighbors.means,
                    knn_k=knn_k,
                    expected_ratings=expected_ratings,
                    expected_similarity=expected_similarity,
                    expected_means=expected_means,
                )

            def test_7(self):
                x = 2
                y = 1

                users_neighborhood = np.array([1, 3])
                knn_k = 2

                expected_ratings = np.array([6.0])
                expected_similarity = np.array([0.7])
                expected_means = np.array([2.0])

                TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                    x=x,
                    y=y,
                    dataset=self.square_dataset,
                    similarity_matrix=self.similarity_matrix,
                    users_neighborhood=users_neighborhood,
                    means=TestGetKTopNeighbors.means,
                    knn_k=knn_k,
                    expected_ratings=expected_ratings,
                    expected_similarity=expected_similarity,
                    expected_means=expected_means,
                )

            def test_8(self):
                x = 2
                y = 3

                users_neighborhood = np.array([0, 1, 3])
                knn_k = 2

                expected_ratings = np.array([8.0])
                expected_similarity = np.array([0.7])
                expected_means = np.array([2.0])

                TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                    x=x,
                    y=y,
                    dataset=self.square_dataset,
                    similarity_matrix=self.similarity_matrix,
                    users_neighborhood=users_neighborhood,
                    means=TestGetKTopNeighbors.means,
                    knn_k=knn_k,
                    expected_ratings=expected_ratings,
                    expected_similarity=expected_similarity,
                    expected_means=expected_means,
                )

            def test_9(self):
                x = 2
                y = 3

                users_neighborhood = np.array([0, 3])
                knn_k = 2

                expected_ratings = np.array([])
                expected_similarity = np.array([])
                expected_means = np.array([])

                TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                    x=x,
                    y=y,
                    dataset=self.square_dataset,
                    similarity_matrix=self.similarity_matrix,
                    users_neighborhood=users_neighborhood,
                    means=TestGetKTopNeighbors.means,
                    knn_k=knn_k,
                    expected_ratings=expected_ratings,
                    expected_similarity=expected_similarity,
                    expected_means=expected_means,
                )

    class TestItemScenario:
        class TestNoNaNs:
            class TestNoZeroSimilarity:
                def test_1(self):
                    x = 0
                    y = 0

                    users_neighborhood = np.array([1, 2, 3])
                    knn_k = 3

                    expected_ratings = np.array([3.0, 2.0, 4.0])
                    expected_similarity = np.array([1.0, 0.5, 0.1])
                    expected_means = np.array([3.0, 2.0, 4.0])

                    TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                        x=y,
                        y=x,
                        dataset=TestGetKTopNeighbors.square_dataset.T,
                        users_neighborhood=users_neighborhood,
                        similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
                        means=TestGetKTopNeighbors.means,
                        knn_k=knn_k,
                        expected_ratings=expected_ratings,
                        expected_similarity=expected_similarity,
                        expected_means=expected_means,
                    )

                def test_2(self):
                    x = 0
                    y = 0

                    users_neighborhood = np.array([1, 2, 3])
                    knn_k = 1

                    expected_ratings = np.array([3.0])
                    expected_similarity = np.array([1.0])
                    expected_means = np.array([3.0])

                    TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                        x=y,
                        y=x,
                        dataset=TestGetKTopNeighbors.square_dataset.T,
                        users_neighborhood=users_neighborhood,
                        similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
                        means=TestGetKTopNeighbors.means,
                        knn_k=knn_k,
                        expected_ratings=expected_ratings,
                        expected_similarity=expected_similarity,
                        expected_means=expected_means,
                    )

                def test_3(self):
                    x = 0
                    y = 0

                    users_neighborhood = np.array([1, 2, 3])
                    knn_k = 2

                    expected_ratings = np.array([3.0, 2.0])
                    expected_similarity = np.array([1.0, 0.5])
                    expected_means = np.array([3.0, 2.0])

                    TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                        x=y,
                        y=x,
                        dataset=TestGetKTopNeighbors.square_dataset.T,
                        users_neighborhood=users_neighborhood,
                        similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
                        means=TestGetKTopNeighbors.means,
                        knn_k=knn_k,
                        expected_ratings=expected_ratings,
                        expected_similarity=expected_similarity,
                        expected_means=expected_means,
                    )

                def test_4(self):
                    x = 0
                    y = 0

                    users_neighborhood = np.array([1, 3])
                    knn_k = 2

                    expected_ratings = np.array([2.0, 4.0])
                    expected_similarity = np.array([0.5, 0.1])
                    expected_means = np.array([2.0, 4.0])

                    TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                        x=y,
                        y=x,
                        dataset=TestGetKTopNeighbors.square_dataset.T,
                        users_neighborhood=users_neighborhood,
                        similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
                        means=TestGetKTopNeighbors.means,
                        knn_k=knn_k,
                        expected_ratings=expected_ratings,
                        expected_similarity=expected_similarity,
                        expected_means=expected_means,
                    )

                def test_5(self):
                    x = 3
                    y = 1

                    users_neighborhood = np.array([0, 2, 3])
                    knn_k = 3

                    expected_ratings = np.array([15.0, 13.0, 16.0])
                    expected_similarity = np.array([0.7, 0.5, 0.3])
                    expected_means = np.array([3.0, 1.0, 4.0])

                    TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                        x=y,
                        y=x,
                        dataset=TestGetKTopNeighbors.square_dataset.T,
                        users_neighborhood=users_neighborhood,
                        similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
                        means=TestGetKTopNeighbors.means,
                        knn_k=knn_k,
                        expected_ratings=expected_ratings,
                        expected_similarity=expected_similarity,
                        expected_means=expected_means,
                    )

                def test_6(self):
                    x = 3
                    y = 1

                    users_neighborhood = np.array([0, 2, 3])
                    knn_k = 1

                    expected_ratings = np.array([15.0])
                    expected_similarity = np.array([0.7])
                    expected_means = np.array([3.0])

                    TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                        x=y,
                        y=x,
                        dataset=TestGetKTopNeighbors.square_dataset.T,
                        users_neighborhood=users_neighborhood,
                        similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
                        means=TestGetKTopNeighbors.means,
                        knn_k=knn_k,
                        expected_ratings=expected_ratings,
                        expected_similarity=expected_similarity,
                        expected_means=expected_means,
                    )

                def test_7(self):
                    x = 3
                    y = 1

                    users_neighborhood = np.array([0, 2, 3])
                    knn_k = 2

                    expected_ratings = np.array([15.0, 13.0])
                    expected_similarity = np.array([0.7, 0.5])
                    expected_means = np.array([3.0, 1.0])

                    TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                        x=y,
                        y=x,
                        dataset=TestGetKTopNeighbors.square_dataset.T,
                        users_neighborhood=users_neighborhood,
                        similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
                        means=TestGetKTopNeighbors.means,
                        knn_k=knn_k,
                        expected_ratings=expected_ratings,
                        expected_similarity=expected_similarity,
                        expected_means=expected_means,
                    )

                def test_8(self):
                    x = 3
                    y = 1

                    users_neighborhood = np.array([0, 3])
                    knn_k = 2

                    expected_ratings = np.array([13.0, 16.0])
                    expected_similarity = np.array([0.5, 0.3])
                    expected_means = np.array([1.0, 4.0])

                    TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                        x=y,
                        y=x,
                        dataset=TestGetKTopNeighbors.square_dataset.T,
                        users_neighborhood=users_neighborhood,
                        similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
                        means=TestGetKTopNeighbors.means,
                        knn_k=knn_k,
                        expected_ratings=expected_ratings,
                        expected_similarity=expected_similarity,
                        expected_means=expected_means,
                    )

            class TestWithZeroSimilarity:
                def test_1(self):
                    x = 3
                    y = 3

                    users_neighborhood = np.array([0, 1, 2])
                    knn_k = 3

                    expected_ratings = np.array([14.0, 13.0])
                    expected_similarity = np.array([0.3, 0.1])
                    expected_means = np.array([2.0, 1.0])

                    TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                        x=y,
                        y=x,
                        dataset=TestGetKTopNeighbors.square_dataset.T,
                        similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
                        users_neighborhood=users_neighborhood,
                        means=TestGetKTopNeighbors.means,
                        knn_k=knn_k,
                        expected_ratings=expected_ratings,
                        expected_similarity=expected_similarity,
                        expected_means=expected_means,
                    )

                def test_2(self):
                    x = 3
                    y = 3

                    users_neighborhood = np.array([0, 1, 2])
                    knn_k = 1

                    expected_ratings = np.array([14.0])
                    expected_similarity = np.array([0.3])
                    expected_means = np.array([2.0])

                    TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                        x=y,
                        y=x,
                        dataset=TestGetKTopNeighbors.square_dataset.T,
                        similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
                        users_neighborhood=users_neighborhood,
                        means=TestGetKTopNeighbors.means,
                        knn_k=knn_k,
                        expected_ratings=expected_ratings,
                        expected_similarity=expected_similarity,
                        expected_means=expected_means,
                    )

                def test_3(self):
                    x = 3
                    y = 3

                    users_neighborhood = np.array([0, 2])
                    knn_k = 3

                    expected_ratings = np.array([13.0])
                    expected_similarity = np.array([0.1])
                    expected_means = np.array([1.0])

                    TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                        x=y,
                        y=x,
                        dataset=TestGetKTopNeighbors.square_dataset.T,
                        similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
                        users_neighborhood=users_neighborhood,
                        means=TestGetKTopNeighbors.means,
                        knn_k=knn_k,
                        expected_ratings=expected_ratings,
                        expected_similarity=expected_similarity,
                        expected_means=expected_means,
                    )

        class TestWithNaNsOnDataset:
            class TestNoZeroSimilarity:
                # fmt: off
                square_dataset = np.array([[ 1,  2,  3,  np.NAN],
                                           [ 5,  6,  np.NAN,  8],
                                           [ np.NAN, 10, 11, 12],
                                           [13, np.NAN, 15, 16]],
                                           dtype=np.float64)
                # fmt: on

                def test_1(self):
                    x = 1
                    y = 1

                    users_neighborhood = np.array([0, 2, 3])
                    knn_k = 2

                    expected_ratings = np.array([5.0, 8.0])
                    expected_similarity = np.array([0.5, 0.3])
                    expected_means = np.array([1.0, 4.0])

                    TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                        x=y,
                        y=x,
                        dataset=self.square_dataset.T,
                        users_neighborhood=users_neighborhood,
                        similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
                        means=TestGetKTopNeighbors.means,
                        knn_k=knn_k,
                        expected_ratings=expected_ratings,
                        expected_similarity=expected_similarity,
                        expected_means=expected_means,
                    )

                def test_2(self):
                    x = 1
                    y = 1

                    users_neighborhood = np.array([0, 2, 3])
                    knn_k = 1

                    expected_ratings = np.array(
                        [
                            5.0,
                        ]
                    )
                    expected_similarity = np.array([0.5])
                    expected_means = np.array([1.0])

                    TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                        x=y,
                        y=x,
                        dataset=self.square_dataset.T,
                        users_neighborhood=users_neighborhood,
                        similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
                        means=TestGetKTopNeighbors.means,
                        knn_k=knn_k,
                        expected_ratings=expected_ratings,
                        expected_similarity=expected_similarity,
                        expected_means=expected_means,
                    )

                def test_3(self):
                    x = 1
                    y = 1

                    users_neighborhood = np.array([0, 3])
                    knn_k = 2

                    expected_ratings = np.array([5.0, 8.0])
                    expected_similarity = np.array([0.5, 0.3])
                    expected_means = np.array([1.0, 4.0])

                    TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                        x=y,
                        y=x,
                        dataset=self.square_dataset.T,
                        users_neighborhood=users_neighborhood,
                        similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
                        means=TestGetKTopNeighbors.means,
                        knn_k=knn_k,
                        expected_ratings=expected_ratings,
                        expected_similarity=expected_similarity,
                        expected_means=expected_means,
                    )

                def test_4(self):
                    x = 1
                    y = 1

                    users_neighborhood = np.array([3])
                    knn_k = 2

                    expected_ratings = np.array([8.0])
                    expected_similarity = np.array([0.3])
                    expected_means = np.array([4.0])

                    TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                        x=y,
                        y=x,
                        dataset=self.square_dataset.T,
                        users_neighborhood=users_neighborhood,
                        similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
                        means=TestGetKTopNeighbors.means,
                        knn_k=knn_k,
                        expected_ratings=expected_ratings,
                        expected_similarity=expected_similarity,
                        expected_means=expected_means,
                    )

            class TestWithZeroSimilarity:
                # fmt: off
                square_dataset = np.array([[ 1,  2,  3,  np.NAN],
                                           [ 5,  6,  np.NAN,  8],
                                           [ np.NAN, 10, 11, 12],
                                           [13, np.NAN, 15, 16]],
                                           dtype=np.float64)
                # fmt: on

                def test_1(self):
                    x = 3
                    y = 3

                    users_neighborhood = np.array([0, 1, 2])
                    knn_k = 3

                    expected_ratings = np.array([13.0])
                    expected_similarity = np.array([0.1])
                    expected_means = np.array([1.0])

                    TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                        x=y,
                        y=x,
                        dataset=self.square_dataset.T,
                        similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
                        users_neighborhood=users_neighborhood,
                        means=TestGetKTopNeighbors.means,
                        knn_k=knn_k,
                        expected_ratings=expected_ratings,
                        expected_similarity=expected_similarity,
                        expected_means=expected_means,
                    )

                def test_2(self):
                    x = 3
                    y = 3

                    users_neighborhood = np.array([0, 1, 2])
                    knn_k = 1

                    expected_ratings = np.array([13.0])
                    expected_similarity = np.array([0.1])
                    expected_means = np.array([1.0])

                    TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                        x=y,
                        y=x,
                        dataset=self.square_dataset.T,
                        similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
                        users_neighborhood=users_neighborhood,
                        means=TestGetKTopNeighbors.means,
                        knn_k=knn_k,
                        expected_ratings=expected_ratings,
                        expected_similarity=expected_similarity,
                        expected_means=expected_means,
                    )

                def test_3(self):
                    x = 3
                    y = 3

                    users_neighborhood = np.array([1, 2])
                    knn_k = 2

                    expected_ratings = np.array([])
                    expected_similarity = np.array([])
                    expected_means = np.array([])

                    TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                        x=y,
                        y=x,
                        dataset=self.square_dataset.T,
                        similarity_matrix=TestGetKTopNeighbors.similarity_matrix,
                        users_neighborhood=users_neighborhood,
                        means=TestGetKTopNeighbors.means,
                        knn_k=knn_k,
                        expected_ratings=expected_ratings,
                        expected_similarity=expected_similarity,
                        expected_means=expected_means,
                    )

        class TestWithNaNsOnSimilarity:
            class TestNoZeroSimilarity:
                # fmt: off
                similarity_matrix = np.array([[1.0, np.NAN, 1.0, 0.1],
                                              [0.5, 1.0, np.NAN, 0.3],
                                              [1.0, 0.7, np.NAN, 0.0],
                                              [np.NAN, 0.3, 0.0, 1.0]],
                                              dtype=np.float64)
                # fmt: on

                def test_1(self):
                    x = 0
                    y = 0

                    users_neighborhood = np.array([1, 2, 3])
                    knn_k = 2

                    expected_ratings = np.array([3.0, 4.0])
                    expected_similarity = np.array([1.0, 0.1])
                    expected_means = np.array([3.0, 4.0])

                    TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                        x=y,
                        y=x,
                        dataset=TestGetKTopNeighbors.square_dataset.T,
                        similarity_matrix=self.similarity_matrix,
                        users_neighborhood=users_neighborhood,
                        means=TestGetKTopNeighbors.means,
                        knn_k=knn_k,
                        expected_ratings=expected_ratings,
                        expected_similarity=expected_similarity,
                        expected_means=expected_means,
                    )

                def test_2(self):
                    x = 0
                    y = 0

                    users_neighborhood = np.array([1, 2, 3])
                    knn_k = 1

                    expected_ratings = np.array([3.0])
                    expected_similarity = np.array([1.0])
                    expected_means = np.array([3.0])

                    TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                        x=y,
                        y=x,
                        dataset=TestGetKTopNeighbors.square_dataset.T,
                        similarity_matrix=self.similarity_matrix,
                        users_neighborhood=users_neighborhood,
                        means=TestGetKTopNeighbors.means,
                        knn_k=knn_k,
                        expected_ratings=expected_ratings,
                        expected_similarity=expected_similarity,
                        expected_means=expected_means,
                    )

                def test_3(self):
                    x = 0
                    y = 1

                    users_neighborhood = np.array([2, 3])
                    knn_k = 3

                    expected_ratings = np.array([4.0])
                    expected_similarity = np.array([0.3])
                    expected_means = np.array([4.0])

                    TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                        x=y,
                        y=x,
                        dataset=TestGetKTopNeighbors.square_dataset.T,
                        similarity_matrix=self.similarity_matrix,
                        users_neighborhood=users_neighborhood,
                        means=TestGetKTopNeighbors.means,
                        knn_k=knn_k,
                        expected_ratings=expected_ratings,
                        expected_similarity=expected_similarity,
                        expected_means=expected_means,
                    )

                def test_4(self):
                    x = 0
                    y = 1

                    users_neighborhood = np.array([3])
                    knn_k = 3

                    expected_ratings = np.array([4.0])
                    expected_similarity = np.array([0.3])
                    expected_means = np.array([4.0])

                    TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                        x=y,
                        y=x,
                        dataset=TestGetKTopNeighbors.square_dataset.T,
                        similarity_matrix=self.similarity_matrix,
                        users_neighborhood=users_neighborhood,
                        means=TestGetKTopNeighbors.means,
                        knn_k=knn_k,
                        expected_ratings=expected_ratings,
                        expected_similarity=expected_similarity,
                        expected_means=expected_means,
                    )

            class TestWithZeroSimilarity:
                # fmt: off
                similarity_matrix = np.array([[1.0, np.NAN, 1.0, 0.1],
                                              [0.5, 1.0, np.NAN, 0.3],
                                              [1.0, 0.7, np.NAN, 0.0],
                                              [np.NAN, 0.3, 0.0, 1.0]],
                                              dtype=np.float64)
                # fmt: on
                def test_1(self):
                    x = 3
                    y = 3

                    users_neighborhood = np.array([0, 1, 2])
                    knn_k = 2

                    expected_ratings = np.array([14.0])
                    expected_similarity = np.array([0.3])
                    expected_means = np.array([2.0])

                    TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                        x=y,
                        y=x,
                        dataset=TestGetKTopNeighbors.square_dataset.T,
                        similarity_matrix=self.similarity_matrix,
                        users_neighborhood=users_neighborhood,
                        means=TestGetKTopNeighbors.means,
                        knn_k=knn_k,
                        expected_ratings=expected_ratings,
                        expected_similarity=expected_similarity,
                        expected_means=expected_means,
                    )

                def test_2(self):
                    x = 3
                    y = 3

                    users_neighborhood = np.array([0, 2])
                    knn_k = 1

                    expected_ratings = np.array([])
                    expected_similarity = np.array([])
                    expected_means = np.array([])

                    TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                        x=y,
                        y=x,
                        dataset=TestGetKTopNeighbors.square_dataset.T,
                        similarity_matrix=self.similarity_matrix,
                        users_neighborhood=users_neighborhood,
                        means=TestGetKTopNeighbors.means,
                        knn_k=knn_k,
                        expected_ratings=expected_ratings,
                        expected_similarity=expected_similarity,
                        expected_means=expected_means,
                    )

        class TestWithNaNsOnDatasetAndSimilarity:
            # fmt: off
            square_dataset = np.array([[ 1,  2,  3,  np.NAN],
                                        [ 5,  6,  np.NAN,  8],
                                        [ np.NAN, 10, 11, 12],
                                        [13, np.NAN, 15, 16]],
                                        dtype=np.float64)
            similarity_matrix = np.array([[1.0, np.NAN, 1.0, 0.1],
                                            [0.5, 1.0, np.NAN, 0.3],
                                            [1.0, 0.7, np.NAN, 0.0],
                                            [np.NAN, 0.3, 0.0, 1.0]],
                                            dtype=np.float64)
            # fmt: on

            def test_1(self):
                x = 1
                y = 1

                users_neighborhood = np.array([0, 2, 3])
                knn_k = 2

                expected_ratings = np.array([5.0, 8.0])
                expected_similarity = np.array([0.5, 0.3])
                expected_means = np.array([1.0, 4.0])

                TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                    x=y,
                    y=x,
                    dataset=self.square_dataset.T,
                    similarity_matrix=self.similarity_matrix,
                    users_neighborhood=users_neighborhood,
                    means=TestGetKTopNeighbors.means,
                    knn_k=knn_k,
                    expected_ratings=expected_ratings,
                    expected_similarity=expected_similarity,
                    expected_means=expected_means,
                )

            def test_2(self):
                x = 1
                y = 1

                users_neighborhood = np.array([0, 2, 3])
                knn_k = 1

                expected_ratings = np.array([5.0])
                expected_similarity = np.array([0.5])
                expected_means = np.array([1.0])

                TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                    x=y,
                    y=x,
                    dataset=self.square_dataset.T,
                    similarity_matrix=self.similarity_matrix,
                    users_neighborhood=users_neighborhood,
                    means=TestGetKTopNeighbors.means,
                    knn_k=knn_k,
                    expected_ratings=expected_ratings,
                    expected_similarity=expected_similarity,
                    expected_means=expected_means,
                )

            def test_3(self):
                x = 0
                y = 0

                users_neighborhood = np.array([1, 2, 3])
                knn_k = 3

                expected_ratings = np.array([3.0])
                expected_similarity = np.array([1.0])
                expected_means = np.array([3.0])

                TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                    x=y,
                    y=x,
                    dataset=self.square_dataset.T,
                    similarity_matrix=self.similarity_matrix,
                    users_neighborhood=users_neighborhood,
                    means=TestGetKTopNeighbors.means,
                    knn_k=knn_k,
                    expected_ratings=expected_ratings,
                    expected_similarity=expected_similarity,
                    expected_means=expected_means,
                )

            def test_4(self):
                x = 1
                y = 0

                users_neighborhood = np.array([2, 3])
                knn_k = 3

                expected_ratings = np.array([8.0])
                expected_similarity = np.array([0.1])
                expected_means = np.array([4.0])

                TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                    x=y,
                    y=x,
                    dataset=self.square_dataset.T,
                    similarity_matrix=self.similarity_matrix,
                    users_neighborhood=users_neighborhood,
                    means=TestGetKTopNeighbors.means,
                    knn_k=knn_k,
                    expected_ratings=expected_ratings,
                    expected_similarity=expected_similarity,
                    expected_means=expected_means,
                )

            def test_5(self):
                x = 1
                y = 0

                users_neighborhood = np.array([2, 3])
                knn_k = 2

                expected_ratings = np.array([8.0])
                expected_similarity = np.array([0.1])
                expected_means = np.array([4.0])

                TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                    x=y,
                    y=x,
                    dataset=self.square_dataset.T,
                    similarity_matrix=self.similarity_matrix,
                    users_neighborhood=users_neighborhood,
                    means=TestGetKTopNeighbors.means,
                    knn_k=knn_k,
                    expected_ratings=expected_ratings,
                    expected_similarity=expected_similarity,
                    expected_means=expected_means,
                )

            def test_6(self):
                x = 1
                y = 0

                users_neighborhood = np.array([2, 3])
                knn_k = 1

                expected_ratings = np.array([8.])
                expected_similarity = np.array([0.1])
                expected_means = np.array([4.0])

                TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                    x=y,
                    y=x,
                    dataset=self.square_dataset.T,
                    similarity_matrix=self.similarity_matrix,
                    users_neighborhood=users_neighborhood,
                    means=TestGetKTopNeighbors.means,
                    knn_k=knn_k,
                    expected_ratings=expected_ratings,
                    expected_similarity=expected_similarity,
                    expected_means=expected_means,
                )

            def test_7(self):
                x = 1
                y = 0

                users_neighborhood = np.array([3])
                knn_k = 2

                expected_ratings = np.array([8.0])
                expected_similarity = np.array([0.1])
                expected_means = np.array([4.0])

                TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                    x=y,
                    y=x,
                    dataset=self.square_dataset.T,
                    similarity_matrix=self.similarity_matrix,
                    users_neighborhood=users_neighborhood,
                    means=TestGetKTopNeighbors.means,
                    knn_k=knn_k,
                    expected_ratings=expected_ratings,
                    expected_similarity=expected_similarity,
                    expected_means=expected_means,
                )

            def test_8(self):
                x = 3
                y = 3

                users_neighborhood = np.array([0, 1, 2])
                knn_k = 3

                expected_ratings = np.array([])
                expected_similarity = np.array([])
                expected_means = np.array([])

                TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                    x=y,
                    y=x,
                    dataset=self.square_dataset.T,
                    similarity_matrix=self.similarity_matrix,
                    users_neighborhood=users_neighborhood,
                    means=TestGetKTopNeighbors.means,
                    knn_k=knn_k,
                    expected_ratings=expected_ratings,
                    expected_similarity=expected_similarity,
                    expected_means=expected_means,
                )

            def test_9(self):
                x = 2
                y = 3

                users_neighborhood = np.array([0, 1, 2])
                knn_k = 3

                expected_ratings = np.array([10.0])
                expected_similarity = np.array([0.3])
                expected_means = np.array([2.0])

                TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                    x=y,
                    y=x,
                    dataset=self.square_dataset.T,
                    similarity_matrix=self.similarity_matrix,
                    users_neighborhood=users_neighborhood,
                    means=TestGetKTopNeighbors.means,
                    knn_k=knn_k,
                    expected_ratings=expected_ratings,
                    expected_similarity=expected_similarity,
                    expected_means=expected_means,
                )

            def test_10(self):
                x = 2
                y = 3

                users_neighborhood = np.array([0, 2])
                knn_k = 3

                expected_ratings = np.array([])
                expected_similarity = np.array([])
                expected_means = np.array([])

                TestGetKTopNeighbors.assert_result_get_k_top_neighbors(
                    x=y,
                    y=x,
                    dataset=self.square_dataset.T,
                    similarity_matrix=self.similarity_matrix,
                    users_neighborhood=users_neighborhood,
                    means=TestGetKTopNeighbors.means,
                    knn_k=knn_k,
                    expected_ratings=expected_ratings,
                    expected_similarity=expected_similarity,
                    expected_means=expected_means,
                )


class TestBiAKNN:
    class ConcreteBiAKNN(BiAKNN):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def compute_biclusters_from_trainset(self):
            pass

    def test_init_invalid_args(self):
        with pytest.raises(AssertionError):
            self.ConcreteBiAKNN(minimum_bicluster_sparsity="0.5")

        with pytest.raises(AssertionError):
            self.ConcreteBiAKNN(minimum_bicluster_sparsity=-0.5)

        with pytest.raises(AssertionError):
            self.ConcreteBiAKNN(minimum_bicluster_sparsity=1.5)

        with pytest.raises(AssertionError):
            self.ConcreteBiAKNN(minimum_bicluster_coverage="0.8")

        with pytest.raises(AssertionError):
            self.ConcreteBiAKNN(minimum_bicluster_coverage=-0.8)

        with pytest.raises(AssertionError):
            self.ConcreteBiAKNN(minimum_bicluster_coverage=1.8)

        with pytest.raises(AssertionError):
            self.ConcreteBiAKNN(minimum_bicluster_relative_size="0.2")

        with pytest.raises(AssertionError):
            self.ConcreteBiAKNN(minimum_bicluster_relative_size=-0.2)

        with pytest.raises(AssertionError):
            self.ConcreteBiAKNN(minimum_bicluster_relative_size=1.2)

        with pytest.raises(AssertionError):
            self.ConcreteBiAKNN(knn_type="invalid_type")

        with pytest.raises(AssertionError):
            self.ConcreteBiAKNN(user_binarization_threshold="0.5")

        with pytest.raises(AssertionError):
            self.ConcreteBiAKNN(user_binarization_threshold=-0.5)

        with pytest.raises(AssertionError):
            self.ConcreteBiAKNN(number_of_top_k_biclusters="10")

        with pytest.raises(AssertionError):
            self.ConcreteBiAKNN(number_of_top_k_biclusters=0)

        with pytest.raises(AssertionError):
            self.ConcreteBiAKNN(knn_k="3")

        with pytest.raises(AssertionError):
            self.ConcreteBiAKNN(knn_k=0)
