"""
Tests for recommenders module.
"""

from unittest.mock import Mock
import numpy as np
from pytest import approx
from recommenders.common import jaccard_distance, get_similarity_matrix
from tests.toy_datasets import zaki_binary_dataset


# pylint: disable=missing-function-docstring



def test_jaccard_distance():
    vector_a = [True, False, True]
    vector_b = [False, True, False]

    distance = jaccard_distance(vector_a, vector_b)

    assert distance == 1


def test_jaccard_distance_2():
    vector_a = [True, False, True]
    vector_b = [True, False, True]

    distance = jaccard_distance(vector_a, vector_b)

    assert distance == 0


def test_get_similarity_matrix():
    distance_strategy_mock = Mock(side_effect=np.arange(0, 0.21, 0.01))
    similarity_matrix = get_similarity_matrix(zaki_binary_dataset, distance_strategy_mock)

    expected_similarity_matrix = [
        [1.0, 0.99, 0.98, 0.97, 0.96, 0.95],
        [0.99, 0.94, 0.93, 0.92, 0.91, 0.9],
        [0.98, 0.93, 0.89, 0.88, 0.87, 0.86],
        [0.97, 0.92, 0.88, 0.85, 0.84, 0.83],
        [0.96, 0.91, 0.87, 0.84, 0.82, 0.81],
        [0.95, 0.9, 0.86, 0.83, 0.81, 0.8],
    ]

    for i, _ in enumerate(similarity_matrix):
        for j, _ in enumerate(similarity_matrix):
            assert similarity_matrix[i][j] == similarity_matrix[j][i]
            assert similarity_matrix[i][j] == approx(expected_similarity_matrix[i][j], 0.01)
