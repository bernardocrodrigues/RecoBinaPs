"""
Tests for the fca module.
"""

import pytest
from unittest.mock import patch
import numpy as np
from pattern_mining.formal_concept_analysis import (
    get_factor_matrices_from_concepts,
    create_concept,
    grecond,
    construct_context_from_binaps_patterns,
)

from dataset.mushroom_dataset import get_mushroom_dataset

from tests.toy_datasets import (
    my_toy_binary_dataset,
    my_toy_binary_2_dataset,
    zaki_binary_dataset,
    belohlavek_binary_dataset,
    belohlavek_binary_dataset_2,
    nenova_dataset_dataset,
)

# pylint: disable=missing-function-docstring


def test_get_matrices_belohlavek():
    # example from belohlavek paper page 14 and 15

    formal_context = [
        create_concept(np.array([0, 3, 4]), np.array([2, 5])),
        create_concept(np.array([2, 4]), np.array([1, 3, 5])),
        create_concept(np.array([0, 2]), np.array([0, 4, 5])),
        create_concept(np.array([0, 1, 3, 4]), np.array([2])),
    ]

    A, B = get_factor_matrices_from_concepts(
        formal_context, belohlavek_binary_dataset.shape[0], belohlavek_binary_dataset.shape[1]
    )

    assert np.array_equal(
        A,
        [
            [True, False, True, True],
            [False, False, False, True],
            [False, True, True, False],
            [True, False, False, True],
            [True, True, False, True],
        ],
    )

    assert np.array_equal(
        B,
        [
            [False, False, True, False, False, True],
            [False, True, False, True, False, True],
            [True, False, False, False, True, True],
            [False, False, True, False, False, False],
        ],
    )

    I = np.matmul(A, B)

    assert (I == belohlavek_binary_dataset).all()


def test_get_matrices_belohlavek_2():
    # example from belohlavek paper page 9 to 11

    concept_1 = create_concept(np.array([0, 4, 8, 10]), np.array([0, 1, 2, 4]))
    concept_2 = create_concept(np.array([1, 3, 11]), np.array([0, 1, 5, 7]))
    concept_3 = create_concept(np.array([2, 5, 6]), np.array([1, 4, 6]))
    concept_4 = create_concept(np.array([2, 5, 6, 7, 9]), np.array([6]))
    concept_5 = create_concept(np.array([0, 2, 4, 5, 6, 8, 10]), np.array([1, 4]))

    formal_context_1 = [concept_1, concept_2, concept_3, concept_4]

    A, B = get_factor_matrices_from_concepts(
        formal_context_1, belohlavek_binary_dataset_2.shape[0], belohlavek_binary_dataset_2.shape[1]
    )

    assert np.array_equal(
        A,
        [
            [True, False, False, False],
            [False, True, False, False],
            [False, False, True, True],
            [False, True, False, False],
            [True, False, False, False],
            [False, False, True, True],
            [False, False, True, True],
            [False, False, False, True],
            [True, False, False, False],
            [False, False, False, True],
            [True, False, False, False],
            [False, True, False, False],
        ],
    )

    assert np.array_equal(
        B,
        [
            [True, True, True, False, True, False, False, False],
            [True, True, False, False, False, True, False, True],
            [False, True, False, False, True, False, True, False],
            [False, False, False, False, False, False, True, False],
        ],
    )

    I = np.matmul(A, B)

    assert (I == belohlavek_binary_dataset_2).all()

    formal_context_2 = [concept_1, concept_2, concept_4, concept_5]
    A, B = get_factor_matrices_from_concepts(
        formal_context_2, belohlavek_binary_dataset_2.shape[0], belohlavek_binary_dataset_2.shape[1]
    )
    I = np.matmul(A, B)
    assert (I == belohlavek_binary_dataset_2).all()


def test_get_matrices_nenova():
    # example from nenova paper at page 62
    formal_context = [
        create_concept(np.array([0, 1]), np.array([0, 1, 2])),
        create_concept(np.array([1, 2, 3]), np.array([3, 4])),
        create_concept(np.array([3, 4, 5]), np.array([5, 6])),
    ]

    A, B = get_factor_matrices_from_concepts(
        formal_context, nenova_dataset_dataset.shape[0], nenova_dataset_dataset.shape[1]
    )

    assert np.array_equal(
        A,
        [
            [True, False, False],
            [True, True, False],
            [False, True, False],
            [False, True, True],
            [False, False, True],
            [False, False, True],
        ],
    )

    assert np.array_equal(
        B,
        [
            [True, True, True, False, False, False, False],
            [False, False, False, True, True, False, False],
            [False, False, False, False, False, True, True],
        ],
    )

    I = np.matmul(A, B)

    assert (I == nenova_dataset_dataset).all()


def test_grecond_my_toy_dataset():
    formal_context, coverage = grecond(my_toy_binary_dataset)

    assert coverage == 1
    A, B = get_factor_matrices_from_concepts(
        formal_context, my_toy_binary_dataset.shape[0], my_toy_binary_dataset.shape[1]
    )
    I = np.matmul(A, B)

    assert (I == my_toy_binary_dataset).all()


def test_grecond_my_toy_2_dataset():
    formal_context, coverage = grecond(my_toy_binary_2_dataset)

    assert coverage == 1
    A, B = get_factor_matrices_from_concepts(
        formal_context, my_toy_binary_2_dataset.shape[0], my_toy_binary_2_dataset.shape[1]
    )
    I = np.matmul(A, B)

    assert (I == my_toy_binary_2_dataset).all()


def test_grecond_zaki():
    formal_context, coverage = grecond(zaki_binary_dataset)

    assert coverage == 1
    A, B = get_factor_matrices_from_concepts(
        formal_context, zaki_binary_dataset.shape[0], zaki_binary_dataset.shape[1]
    )
    I = np.matmul(A, B)

    assert (I == zaki_binary_dataset).all()


def test_grecond_belohlavek():
    formal_context, coverage = grecond(belohlavek_binary_dataset)

    assert coverage == 1
    assert len(formal_context) == 4

    assert np.array_equal(formal_context[0].extent, [0, 2])
    assert np.array_equal(formal_context[0].intent, [0, 4, 5])

    assert np.array_equal(formal_context[1].extent, [2, 4])
    assert np.array_equal(formal_context[1].intent, [1, 3, 5])

    assert np.array_equal(formal_context[2].extent, [0, 1, 3, 4])
    assert np.array_equal(formal_context[2].intent, [2])

    assert np.array_equal(formal_context[3].extent, [0, 2, 3, 4])
    assert np.array_equal(formal_context[3].intent, [5])

    A, B = get_factor_matrices_from_concepts(
        formal_context, belohlavek_binary_dataset.shape[0], belohlavek_binary_dataset.shape[1]
    )
    I = np.matmul(A, B)

    assert (I == belohlavek_binary_dataset).all()


def test_grecond_nenova():
    formal_context, coverage = grecond(nenova_dataset_dataset)

    assert coverage == 1
    assert len(formal_context) == 3

    assert np.array_equal(formal_context[0].extent, [0, 1])
    assert np.array_equal(formal_context[0].intent, [0, 1, 2])

    assert np.array_equal(formal_context[1].extent, [1, 2, 3])
    assert np.array_equal(formal_context[1].intent, [3, 4])

    assert np.array_equal(formal_context[2].extent, [3, 4, 5])
    assert np.array_equal(formal_context[2].intent, [5, 6])

    A, B = get_factor_matrices_from_concepts(
        formal_context, nenova_dataset_dataset.shape[0], nenova_dataset_dataset.shape[1]
    )
    I = np.matmul(A, B)

    assert (I == nenova_dataset_dataset).all()


def test_grecond_partial():
    formal_context, _ = grecond(nenova_dataset_dataset, coverage=0.1)
    A, B = get_factor_matrices_from_concepts(
        formal_context, nenova_dataset_dataset.shape[0], nenova_dataset_dataset.shape[1]
    )
    I = np.matmul(A, B)

    real_coverage = np.count_nonzero(I) / np.count_nonzero(I == nenova_dataset_dataset)

    assert I.shape == nenova_dataset_dataset.shape
    assert real_coverage <= 0.6


def test_grecond_partial_2():
    formal_context, _ = grecond(nenova_dataset_dataset, coverage=0.1)
    A, B = get_factor_matrices_from_concepts(
        formal_context, nenova_dataset_dataset.shape[0], nenova_dataset_dataset.shape[1]
    )
    I = np.matmul(A, B)

    real_coverage = np.count_nonzero(I) / np.count_nonzero(nenova_dataset_dataset)

    assert I.shape == nenova_dataset_dataset.shape
    assert real_coverage >= 0.1
    assert real_coverage < 0.34

    formal_context, _ = grecond(nenova_dataset_dataset, coverage=0.2)
    A, B = get_factor_matrices_from_concepts(
        formal_context, nenova_dataset_dataset.shape[0], nenova_dataset_dataset.shape[1]
    )
    I = np.matmul(A, B)

    real_coverage = np.count_nonzero(I) / np.count_nonzero(nenova_dataset_dataset)

    assert I.shape == nenova_dataset_dataset.shape
    assert real_coverage >= 0.1
    assert real_coverage <= 0.34

    formal_context, _ = grecond(nenova_dataset_dataset, coverage=0.3)
    A, B = get_factor_matrices_from_concepts(
        formal_context, nenova_dataset_dataset.shape[0], nenova_dataset_dataset.shape[1]
    )
    I = np.matmul(A, B)

    real_coverage = np.count_nonzero(I) / np.count_nonzero(nenova_dataset_dataset)

    assert I.shape == nenova_dataset_dataset.shape
    assert real_coverage >= 0.1
    assert real_coverage <= 0.34

    formal_context, _ = grecond(nenova_dataset_dataset, coverage=0.4)
    A, B = get_factor_matrices_from_concepts(
        formal_context, nenova_dataset_dataset.shape[0], nenova_dataset_dataset.shape[1]
    )
    I = np.matmul(A, B)

    real_coverage = np.count_nonzero(I) / np.count_nonzero(nenova_dataset_dataset)

    assert I.shape == nenova_dataset_dataset.shape
    assert real_coverage >= 0.4
    assert real_coverage <= 0.7

    formal_context, _ = grecond(nenova_dataset_dataset, coverage=0.7)
    A, B = get_factor_matrices_from_concepts(
        formal_context, nenova_dataset_dataset.shape[0], nenova_dataset_dataset.shape[1]
    )
    I = np.matmul(A, B)

    real_coverage = np.count_nonzero(I) / np.count_nonzero(nenova_dataset_dataset)

    assert I.shape == nenova_dataset_dataset.shape
    assert real_coverage >= 0.7
    assert real_coverage <= 1


def test_grecond_mushroom():
    coverages = np.arange(0.01, 1.01, 0.01)
    # fmt: off
    # pylint: disable=line-too-long
    mushroom_factors_per_coverage = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3,
                                     3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 8, 8, 9, 9, 10,
                                     10, 11, 11, 12, 12, 13, 13, 14, 15, 16, 16, 17, 18, 18, 19, 20, 21, 22, 23, 24, 25,
                                     26, 27, 28, 28, 29, 31, 32, 34, 35, 37, 38, 40, 42, 44, 46, 49, 52, 55, 58, 62, 66,
                                     70, 75, 85, 120]
    # pylint: enable=line-too-long
    # fmt: on
    found_factors_per_coverage = []

    dataset = get_mushroom_dataset()

    for coverage in coverages:
        concepts, _ = grecond(dataset, coverage=coverage)
        found_factors_per_coverage.append(len(concepts))

    assert mushroom_factors_per_coverage == found_factors_per_coverage


class TestConstructContextFromBinapsPatterns:
    def test_with_invalid_args(self):
        patterns = [np.array([0, 2]), np.array([1]), np.array([0, 1, 3])]

        with pytest.raises(AssertionError):
            construct_context_from_binaps_patterns(123, patterns)
        with pytest.raises(AssertionError):
            construct_context_from_binaps_patterns("aasdas", patterns)
        with pytest.raises(AssertionError):
            construct_context_from_binaps_patterns(np.array([0, 2]), patterns)

        dataset = np.array(
            [
                [True, False, True, True],
                [False, False, False, True],
                [False, True, True, False],
                [True, False, False, True],
                [True, True, False, True],
            ]
        )

        patterns = [
            np.array([1, 2, 3], dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
            np.array([10], dtype=np.int64),
        ]

        with pytest.raises(AssertionError):
            construct_context_from_binaps_patterns(dataset, patterns)

    @patch("pattern_mining.formal_concept_analysis.t")
    @patch("pattern_mining.formal_concept_analysis.i")
    def test_success_without_closed_itemset(self, i_mock, t_mock):
        t_mock.side_effect = [
            np.array([4], dtype=np.int64),
            np.array([5], dtype=np.int64),
            np.array([6], dtype=np.int64),
        ]

        dataset = np.array(
            [
                [True, False, True, True],
                [False, False, False, True],
                [False, True, True, False],
                [True, False, False, True],
                [True, True, False, True],
            ]
        )

        patterns = [
            np.array([1, 2, 3], dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
            np.array([1], dtype=np.int64),
        ]

        # Call the function under test
        context = construct_context_from_binaps_patterns(dataset, patterns, closed_itemsets=False)

        assert i_mock.call_count == 0
        assert t_mock.call_count == 3

        assert context == [
            create_concept(extent=np.array([4], dtype=np.int64), intent=patterns[0]),
            create_concept(extent=np.array([5], dtype=np.int64), intent=patterns[1]),
            create_concept(extent=np.array([6], dtype=np.int64), intent=patterns[2]),
        ]

    @patch("pattern_mining.formal_concept_analysis.t")
    @patch("pattern_mining.formal_concept_analysis.i")
    def test_success_with_closed_itemset(self, i_mock, t_mock):
        t_mock.side_effect = [
            np.array([4], dtype=np.int64),
            np.array([5], dtype=np.int64),
            np.array([6], dtype=np.int64),
            np.array([400], dtype=np.int64),
            np.array([500], dtype=np.int64),
            np.array([600], dtype=np.int64),
        ]

        i_mock.side_effect = [
            np.array([40], dtype=np.int64),
            np.array([50], dtype=np.int64),
            np.array([60], dtype=np.int64),
        ]

        dataset = np.array(
            [
                [True, False, True, True],
                [False, False, False, True],
                [False, True, True, False],
                [True, False, False, True],
                [True, True, False, True],
            ]
        )

        patterns = [
            np.array([1, 2, 3], dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
            np.array([1], dtype=np.int64),
        ]

        # Call the function under test
        context = construct_context_from_binaps_patterns(dataset, patterns, closed_itemsets=True)

        assert i_mock.call_count == 3
        assert t_mock.call_count == 6

        assert context == [
            create_concept(extent=np.array([5], dtype=np.int64), intent=np.array([40], dtype=np.int64)),
            create_concept(extent=np.array([400], dtype=np.int64), intent=np.array([50], dtype=np.int64)),
            create_concept(extent=np.array([600], dtype=np.int64), intent=np.array([60], dtype=np.int64)),
        ]
