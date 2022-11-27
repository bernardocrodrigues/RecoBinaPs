from lib.BooleanMatrixBasedRecomenders import jaccard_distance, cosine_distance, get_similarity_matrix
from tests.ToyDatasets import zaki_binary_dataset

import numpy as np

from pytest import approx

from unittest.mock import Mock


def test_cosine_distance_of_booleans():

    a = [1, 1, 0, 0]
    b = [0, 1, 1, 1]

    distance = cosine_distance(a, b)

    assert distance == cosine_distance([True, True, False, False], [False, True, True, True])

    alternative_consine_distance = 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    assert distance == approx(alternative_consine_distance, 0.00000001)
    assert distance == approx(0.592, 0.001)


def test_cosine_distance_of_booleans_2():

    a = [1, 0, 1]
    b = [0, 1, 1]

    distance = cosine_distance(a, b)

    assert distance == cosine_distance([True, False, True], [False, True, True])

    alternative_consine_distance = 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    assert distance == approx(alternative_consine_distance, 0.00000001)
    assert distance == 0.5


def test_cosine_distance():

    a = [True, False, True]
    b = [False, True, False]

    distance = cosine_distance(a, b)

    assert distance == 1


def test_cosine_distance_2():

    a = [True, False, True]
    b = [True, False, True]

    distance = cosine_distance(a, b)

    assert distance == 0


def test_jaccard_distance():

    a = [True, False, True]
    b = [False, True, False]

    distance = jaccard_distance(a, b)

    assert distance == 1


def test_jaccard_distance_2():

    a = [True, False, True]
    b = [True, False, True]

    distance = jaccard_distance(a, b)

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
