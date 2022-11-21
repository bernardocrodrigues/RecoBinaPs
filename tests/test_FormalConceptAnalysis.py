import numpy as np

from lib.FormalConceptAnalysis import get_factor_matrices_from_concepts, Concept, GreConD

from tests.ToyDatasets import my_toy_binary_dataset, my_toy_binary_2_dataset, zaki_binary_dataset, belohlavek_binary_dataset, belohlavek_binary_dataset_2, nenova_dataset_dataset


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
