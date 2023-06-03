import numpy as np

from unittest.mock import Mock, call
from lib.FormalConceptAnalysis import get_factor_matrices_from_concepts, Concept, GreConD, construct_context_from_binaps_patterns
from lib.BinaryDataset import BinaryDataset

from tests.ToyDatasets import my_toy_binary_dataset, my_toy_binary_2_dataset, zaki_binary_dataset, belohlavek_binary_dataset, belohlavek_binary_dataset_2, nenova_dataset_dataset
from tests.mushroom import MushroomDataset

def test_get_matrices_belohlavek():
    # example from belohlavek paper page 14 and 15

    formal_context = [
        Concept(np.array([0, 3, 4]), np.array([2, 5])),
        Concept(np.array([2, 4]), np.array([1, 3, 5])),
        Concept(np.array([0, 2]), np.array([0, 4, 5])),
        Concept(np.array([0, 1, 3, 4]), np.array([2])),
    ]

    Af, Bf = get_factor_matrices_from_concepts(formal_context, belohlavek_binary_dataset.shape[0], belohlavek_binary_dataset.shape[1])

    assert np.array_equal(
        Af,
        [
            [True, False, True, True],
            [False, False, False, True],
            [False, True, True, False],
            [True, False, False, True],
            [True, True, False, True],
        ],
    )

    assert np.array_equal(
        Bf,
        [
            [False, False, True, False, False, True],
            [False, True, False, True, False, True],
            [True, False, False, False, True, True],
            [False, False, True, False, False, False],
        ],
    )

    I = np.matmul(Af, Bf)

    assert (I == belohlavek_binary_dataset._binary_dataset).all()


def test_get_matrices_belohlavek_2():
    # example from belohlavek paper page 9 to 11

    C1 = Concept(np.array([0, 4, 8, 10]), np.array([0, 1, 2, 4]))
    C2 = Concept(np.array([1, 3, 11]), np.array([0, 1, 5, 7]))
    C3 = Concept(np.array([2, 5, 6]), np.array([1, 4, 6]))
    C4 = Concept(np.array([2, 5, 6, 7, 9]), np.array([6]))
    C5 = Concept(np.array([0, 2, 4, 5, 6, 8, 10]), np.array([1, 4]))

    formal_context_1 = [C1, C2, C3, C4]

    Af, Bf = get_factor_matrices_from_concepts(formal_context_1, belohlavek_binary_dataset_2.shape[0], belohlavek_binary_dataset_2.shape[1])

    assert np.array_equal(
        Af,
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
        Bf,
        [
            [True, True, True, False, True, False, False, False],
            [True, True, False, False, False, True, False, True],
            [False, True, False, False, True, False, True, False],
            [False, False, False, False, False, False, True, False],
        ],
    )

    I = np.matmul(Af, Bf)

    assert (I == belohlavek_binary_dataset_2._binary_dataset).all()

    formal_context_2 = [C1, C2, C4, C5]
    Af, Bf = get_factor_matrices_from_concepts(formal_context_2, belohlavek_binary_dataset_2.shape[0], belohlavek_binary_dataset_2.shape[1])
    I = np.matmul(Af, Bf)
    assert (I == belohlavek_binary_dataset_2._binary_dataset).all()


def test_get_matrices_nenova():
    # example from nenova paper at page 62
    formal_context = [
        Concept(np.array([0, 1]), np.array([0, 1, 2])),
        Concept(np.array([1, 2, 3]), np.array([3, 4])),
        Concept(np.array([3, 4, 5]), np.array([5, 6])),
    ]

    Af, Bf = get_factor_matrices_from_concepts(formal_context, nenova_dataset_dataset.shape[0], nenova_dataset_dataset.shape[1])

    assert np.array_equal(
        Af,
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
        Bf,
        [
            [True, True, True, False, False, False, False],
            [False, False, False, True, True, False, False],
            [False, False, False, False, False, True, True],
        ],
    )

    I = np.matmul(Af, Bf)

    assert (I == nenova_dataset_dataset._binary_dataset).all()


def test_GreConD_my_toy_dataset():
    formal_context, coverage = GreConD(my_toy_binary_dataset)

    assert coverage == 1
    Af, Bf = get_factor_matrices_from_concepts(formal_context, my_toy_binary_dataset.shape[0], my_toy_binary_dataset.shape[1])
    I = np.matmul(Af, Bf)

    assert (I == my_toy_binary_dataset._binary_dataset).all()


def test_GreConD_my_toy_2_dataset():
    formal_context, coverage = GreConD(my_toy_binary_2_dataset)

    assert coverage == 1
    Af, Bf = get_factor_matrices_from_concepts(formal_context, my_toy_binary_2_dataset.shape[0], my_toy_binary_2_dataset.shape[1])
    I = np.matmul(Af, Bf)

    assert (I == my_toy_binary_2_dataset._binary_dataset).all()


def test_GreConD_zaki():
    formal_context, coverage = GreConD(zaki_binary_dataset)

    assert coverage == 1
    Af, Bf = get_factor_matrices_from_concepts(formal_context, zaki_binary_dataset.shape[0], zaki_binary_dataset.shape[1])
    I = np.matmul(Af, Bf)

    assert (I == zaki_binary_dataset._binary_dataset).all()


def test_GreConD_belohlavek():
    formal_context, coverage = GreConD(belohlavek_binary_dataset)

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

    Af, Bf = get_factor_matrices_from_concepts(formal_context, belohlavek_binary_dataset.shape[0], belohlavek_binary_dataset.shape[1])
    I = np.matmul(Af, Bf)

    assert (I == belohlavek_binary_dataset._binary_dataset).all()


def test_GreConD_nenova():
    formal_context, coverage = GreConD(nenova_dataset_dataset)

    assert coverage == 1
    assert len(formal_context) == 3

    assert np.array_equal(formal_context[0].extent, [0, 1])
    assert np.array_equal(formal_context[0].intent, [0, 1, 2])

    assert np.array_equal(formal_context[1].extent, [1, 2, 3])
    assert np.array_equal(formal_context[1].intent, [3, 4])

    assert np.array_equal(formal_context[2].extent, [3, 4, 5])
    assert np.array_equal(formal_context[2].intent, [5, 6])

    Af, Bf = get_factor_matrices_from_concepts(formal_context, nenova_dataset_dataset.shape[0], nenova_dataset_dataset.shape[1])
    I = np.matmul(Af, Bf)

    assert (I == nenova_dataset_dataset._binary_dataset).all()


def test_GreConD_partial():

    formal_context, _ = GreConD(nenova_dataset_dataset, coverage=0.1)
    Af, Bf = get_factor_matrices_from_concepts(formal_context, nenova_dataset_dataset.shape[0], nenova_dataset_dataset.shape[1])
    I = np.matmul(Af, Bf)

    real_coverage = np.count_nonzero(I) / np.count_nonzero(I == nenova_dataset_dataset._binary_dataset)

    assert I.shape == nenova_dataset_dataset.shape
    assert real_coverage <= 0.6


def test_GreConD_partial_2():

    formal_context, _ = GreConD(nenova_dataset_dataset, coverage=0.1)
    Af, Bf = get_factor_matrices_from_concepts(formal_context, nenova_dataset_dataset.shape[0], nenova_dataset_dataset.shape[1])
    I = np.matmul(Af, Bf)

    real_coverage = np.count_nonzero(I) / np.count_nonzero(nenova_dataset_dataset._binary_dataset)

    assert I.shape == nenova_dataset_dataset.shape
    assert real_coverage >= 0.1
    assert real_coverage < 0.34

    formal_context, _ = GreConD(nenova_dataset_dataset, coverage=0.2)
    Af, Bf = get_factor_matrices_from_concepts(formal_context, nenova_dataset_dataset.shape[0], nenova_dataset_dataset.shape[1])
    I = np.matmul(Af, Bf)

    real_coverage = np.count_nonzero(I) / np.count_nonzero(nenova_dataset_dataset._binary_dataset)

    assert I.shape == nenova_dataset_dataset.shape
    assert real_coverage >= 0.1
    assert real_coverage <= 0.34

    formal_context, _ = GreConD(nenova_dataset_dataset, coverage=0.3)
    Af, Bf = get_factor_matrices_from_concepts(formal_context, nenova_dataset_dataset.shape[0], nenova_dataset_dataset.shape[1])
    I = np.matmul(Af, Bf)

    real_coverage = np.count_nonzero(I) / np.count_nonzero(nenova_dataset_dataset._binary_dataset)

    assert I.shape == nenova_dataset_dataset.shape
    assert real_coverage >= 0.1
    assert real_coverage <= 0.34

    formal_context, _ = GreConD(nenova_dataset_dataset, coverage=0.4)
    Af, Bf = get_factor_matrices_from_concepts(formal_context, nenova_dataset_dataset.shape[0], nenova_dataset_dataset.shape[1])
    I = np.matmul(Af, Bf)

    real_coverage = np.count_nonzero(I) / np.count_nonzero(nenova_dataset_dataset._binary_dataset)

    assert I.shape == nenova_dataset_dataset.shape
    assert real_coverage >= 0.4
    assert real_coverage <= 0.7

    formal_context, _ = GreConD(nenova_dataset_dataset, coverage=0.7)
    Af, Bf = get_factor_matrices_from_concepts(formal_context, nenova_dataset_dataset.shape[0], nenova_dataset_dataset.shape[1])
    I = np.matmul(Af, Bf)

    real_coverage = np.count_nonzero(I) / np.count_nonzero(nenova_dataset_dataset._binary_dataset)

    assert I.shape == nenova_dataset_dataset.shape
    assert real_coverage >= 0.7
    assert real_coverage <= 1


def test_GreConD_mushroom():

    coverages =  np.arange(0, 1.01, 0.01)
    mushroom_factors_per_coverage = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3,
                                     3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 8, 8, 9, 9, 10,
                                     10, 11, 11, 12, 12, 13, 13, 14, 15, 16, 16, 17, 18, 18, 19, 20, 21, 22, 23, 24, 25,
                                     26, 27, 28, 28, 29, 31, 32, 34, 35, 37, 38, 40, 42, 44, 46, 49, 52, 55, 58, 62, 66,
                                     70, 75, 85, 120]

    found_factors_per_coverage = []

    for coverage in coverages:
        concepts, _ = GreConD(MushroomDataset, coverage=coverage)
        found_factors_per_coverage.append(len(concepts))

    assert mushroom_factors_per_coverage == found_factors_per_coverage


def test_construct_context_from_binaps_patterns_with_closed_itemsets():
    # Mock the binary_dataset
    binary_dataset = Mock(spec=BinaryDataset)
    binary_dataset.t.side_effect = lambda itemset: [i * 10 for i in itemset]
    binary_dataset.i.side_effect = lambda tidset: [i * 100 for i in tidset]

    # Define the input patterns
    patterns = [[1, 2, 3], [4, 5], [2, 4, 6]]

    # Call the function under test
    context = construct_context_from_binaps_patterns(binary_dataset, patterns, closed_itemsets=True)

    # Assert the calls to binary_dataset.i
    binary_dataset.i.assert_has_calls([call([10, 20, 30]), call([40, 50]), call([20, 40, 60])])

    # Assert the calls to binary_dataset.t
    binary_dataset.t.assert_has_calls([call([1, 2, 3]), call([1000, 2000, 3000]), call([4, 5]), call([4000, 5000]), call([2, 4, 6]),   call([2000, 4000, 6000])])

    # Assert the output
    expected_context = [
        Concept(extent=[10000, 20000, 30000], intent=[1000, 2000, 3000]),
        Concept(extent=[40000, 50000], intent=[4000, 5000]),
        Concept(extent=[20000, 40000, 60000], intent=[2000, 4000, 6000])
    ]
    assert context == expected_context


def test_construct_context_from_binaps_patterns_without_closed_itemsets():
    # Mock the binary_dataset
    binary_dataset = Mock(spec=BinaryDataset)
    binary_dataset.t.side_effect = lambda itemset: [i * 10 for i in itemset]
    binary_dataset.i.side_effect = lambda tidset: [i * 100 for i in tidset]

    # Define the input patterns
    patterns = [[1, 2, 3], [4, 5], [2, 4, 6]]

    # Call the function under test
    context = construct_context_from_binaps_patterns(binary_dataset, patterns, closed_itemsets=False)

    # Assert the calls to binary_dataset.i
    binary_dataset.i.assert_not_called()

    # Assert the calls to binary_dataset.t
    binary_dataset.t.assert_has_calls([call([1, 2, 3]), call([4, 5]), call([2, 4, 6])])

    # Assert the output
    expected_context = [
        Concept(extent=[10, 20, 30], intent=[1, 2, 3]),
        Concept(extent=[40, 50], intent=[4, 5]),
        Concept(extent=[20, 40, 60], intent=[2, 4, 6])
    ]
    assert context == expected_context
