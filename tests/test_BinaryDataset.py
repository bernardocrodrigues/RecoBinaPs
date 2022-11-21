import numpy as np

from tests.ToyDatasets import my_toy_binary_dataset, zaki_binary_dataset, belohlavek_binary_dataset


def test_i_on_my_toy_binary_dataset():
    assert np.array_equal(my_toy_binary_dataset.i(np.array([0])), [0, 1, 3])
    assert np.array_equal(my_toy_binary_dataset.i(np.array([1])), [0, 1, 2, 3])
    assert np.array_equal(my_toy_binary_dataset.i(np.array([2])), [0, 1, 2, 3])
    assert np.array_equal(my_toy_binary_dataset.i(np.array([3])), [4, 5])
    assert np.array_equal(my_toy_binary_dataset.i(np.array([4])), [4, 5])
    assert np.array_equal(my_toy_binary_dataset.i(np.array([5])), [0, 1, 2, 3, 4, 5])

    assert np.array_equal(my_toy_binary_dataset.i(np.array([0, 1])), [0, 1, 3])
    assert np.array_equal(my_toy_binary_dataset.i(np.array([0, 2])), [0, 1, 3])
    assert np.array_equal(my_toy_binary_dataset.i(np.array([0, 3])), [])
    assert np.array_equal(my_toy_binary_dataset.i(np.array([0, 4])), [])
    assert np.array_equal(my_toy_binary_dataset.i(np.array([0, 5])), [0, 1, 3])
    assert np.array_equal(my_toy_binary_dataset.i(np.array([0, 6])), [])

    assert np.array_equal(my_toy_binary_dataset.i(np.array([0, 1, 2])), [0, 1, 3])
    assert np.array_equal(my_toy_binary_dataset.i(np.array([0, 3, 5])), [])
    assert np.array_equal(my_toy_binary_dataset.i(np.array([1, 2, 5])), [0, 1, 2, 3])

    assert np.array_equal(my_toy_binary_dataset.i(np.array([0, 1, 2, 5])), [0, 1, 3])

    assert np.array_equal(my_toy_binary_dataset.i(np.array([0, 1, 2, 3, 4, 5])), [])


def test_t_on_my_toy_binary_dataset():
    assert np.array_equal(my_toy_binary_dataset.t(np.array([0])), [0, 1, 2, 5])
    assert np.array_equal(my_toy_binary_dataset.t(np.array([0])), [0, 1, 2, 5])
    assert np.array_equal(my_toy_binary_dataset.t(np.array([1])), [0, 1, 2, 5])
    assert np.array_equal(my_toy_binary_dataset.t(np.array([2])), [1, 2, 5])
    assert np.array_equal(my_toy_binary_dataset.t(np.array([3])), [0, 1, 2, 5])
    assert np.array_equal(my_toy_binary_dataset.t(np.array([4])), [3, 4, 5])
    assert np.array_equal(my_toy_binary_dataset.t(np.array([5])), [3, 4, 5])
    assert np.array_equal(my_toy_binary_dataset.t(np.array([6])), [6])

    assert np.array_equal(my_toy_binary_dataset.t(np.array([0, 1])), [0, 1, 2, 5])
    assert np.array_equal(my_toy_binary_dataset.t(np.array([0, 5])), [5])
    assert np.array_equal(my_toy_binary_dataset.t(np.array([2, 3])), [1, 2, 5])
    assert np.array_equal(my_toy_binary_dataset.t(np.array([4, 5])), [3, 4, 5])

    assert np.array_equal(my_toy_binary_dataset.t(np.array([0, 2, 5])), [5])

    assert np.array_equal(my_toy_binary_dataset.t(np.array([0, 1, 2, 3])), [1, 2, 5])
    assert np.array_equal(my_toy_binary_dataset.t(np.array([1, 2, 3, 4])), [5])

    assert np.array_equal(my_toy_binary_dataset.t(np.array([0, 1, 2, 3, 4])), [5])
    assert np.array_equal(my_toy_binary_dataset.t(np.array([0, 1, 2, 3, 4, 6])), [])


def test_i_on_zaki_binary_dataset():
    assert np.array_equal(zaki_binary_dataset.i(np.array([0])), [0, 1, 3, 4])
    assert np.array_equal(zaki_binary_dataset.i(np.array([1])), [1, 2, 4])
    assert np.array_equal(zaki_binary_dataset.i(np.array([2])), [0, 1, 4])
    assert np.array_equal(zaki_binary_dataset.i(np.array([3])), [0, 1, 2, 4])
    assert np.array_equal(zaki_binary_dataset.i(np.array([4])), [0, 1, 2, 3, 4])
    assert np.array_equal(zaki_binary_dataset.i(np.array([5])), [1, 2, 3])

    assert np.array_equal(zaki_binary_dataset.i(np.array([0, 1])), [1, 4])
    assert np.array_equal(zaki_binary_dataset.i(np.array([1, 3])), [1, 2, 4])
    assert np.array_equal(zaki_binary_dataset.i(np.array([3, 4])), [0, 1, 2, 4])
    assert np.array_equal(zaki_binary_dataset.i(np.array([0, 5])), [1, 3])

    assert np.array_equal(zaki_binary_dataset.i(np.array([0, 1, 5])), [1])
    assert np.array_equal(zaki_binary_dataset.i(np.array([1, 2, 4])), [1, 4])
    assert np.array_equal(zaki_binary_dataset.i(np.array([2, 4, 1])), [1, 4])

    assert np.array_equal(zaki_binary_dataset.i(np.array([0, 1, 2, 3])), [1, 4])
    assert np.array_equal(zaki_binary_dataset.i(np.array([0, 1, 2, 3, 4])), [1, 4])
    assert np.array_equal(zaki_binary_dataset.i(np.array([0, 1, 2, 3, 4, 5])), [1])


def test_t_on_zaki_binary_dataset():
    assert np.array_equal(zaki_binary_dataset.t(np.array([0])), [0, 2, 3, 4])
    assert np.array_equal(zaki_binary_dataset.t(np.array([1])), [0, 1, 2, 3, 4, 5])
    assert np.array_equal(zaki_binary_dataset.t(np.array([2])), [1, 3, 4, 5])
    assert np.array_equal(zaki_binary_dataset.t(np.array([3])), [0, 4, 5])
    assert np.array_equal(zaki_binary_dataset.t(np.array([4])), [0, 1, 2, 3, 4])

    assert np.array_equal(zaki_binary_dataset.t(np.array([0, 4])), [0, 2, 3, 4])
    assert np.array_equal(zaki_binary_dataset.t(np.array([1, 4])), [0, 1, 2, 3, 4])
    assert np.array_equal(zaki_binary_dataset.t(np.array([1, 3])), [0, 4, 5])
    assert np.array_equal(zaki_binary_dataset.t(np.array([3, 4])), [0, 4])

    assert np.array_equal(zaki_binary_dataset.t(np.array([0, 1, 4])), [0, 2, 3, 4])
    assert np.array_equal(zaki_binary_dataset.t(np.array([0, 2, 4])), [3, 4])
    assert np.array_equal(zaki_binary_dataset.t(np.array([0, 2, 3])), [4])

    assert np.array_equal(zaki_binary_dataset.t(np.array([0, 1, 2, 4])), [3, 4])
    assert np.array_equal(zaki_binary_dataset.t(np.array([0, 1, 2, 3])), [4])

    assert np.array_equal(zaki_binary_dataset.t(np.array([0, 1, 2, 3, 4])), [4])


def test_closed_itemsets_on_zaki_binary_dataset():
    # example from zaki page 245

    def get_closure(X):
        return zaki_binary_dataset.i(zaki_binary_dataset.t(X))

    closed_itemset = np.array([1])
    assert np.array_equal(get_closure(closed_itemset), closed_itemset)

    closed_itemset = np.array([0, 1, 3, 4])
    assert np.array_equal(get_closure(closed_itemset), closed_itemset)

    closed_itemset = np.array([0, 1, 4])
    assert np.array_equal(get_closure(closed_itemset), closed_itemset)

    closed_itemset = np.array([1, 3])
    assert np.array_equal(get_closure(closed_itemset), closed_itemset)

    closed_itemset = np.array([1, 4])
    assert np.array_equal(get_closure(closed_itemset), closed_itemset)

    closed_itemset = np.array([1, 2])
    assert np.array_equal(get_closure(closed_itemset), closed_itemset)

    closed_itemset = np.array([1,2,4])
    assert np.array_equal(get_closure(closed_itemset), closed_itemset)

    closed_itemset = np.array([0])
    assert np.array_equal(get_closure(closed_itemset), closed_itemset) == False

    closed_itemset = np.array([2])
    assert np.array_equal(get_closure(closed_itemset), closed_itemset) == False

    closed_itemset = np.array([3])
    assert np.array_equal(get_closure(closed_itemset), closed_itemset) == False

    closed_itemset = np.array([4])
    assert np.array_equal(get_closure(closed_itemset), closed_itemset) == False

    closed_itemset = np.array([0, 3])
    assert np.array_equal(get_closure(closed_itemset), closed_itemset) == False

    closed_itemset = np.array([3, 4])
    assert np.array_equal(get_closure(closed_itemset), closed_itemset) == False

    closed_itemset = np.array([0 ,1])
    assert np.array_equal(get_closure(closed_itemset), closed_itemset) == False

    closed_itemset = np.array([0 ,4])
    assert np.array_equal(get_closure(closed_itemset), closed_itemset) == False

    closed_itemset = np.array([2 ,4])
    assert np.array_equal(get_closure(closed_itemset), closed_itemset) == False

    closed_itemset = np.array([0, 1, 3])
    assert np.array_equal(get_closure(closed_itemset), closed_itemset) == False

    closed_itemset = np.array([0, 3, 4])
    assert np.array_equal(get_closure(closed_itemset), closed_itemset) == False

    closed_itemset = np.array([1, 3, 4])
    assert np.array_equal(get_closure(closed_itemset), closed_itemset) == False


def test_i_on_belohlavek_binary_dataset():
    assert np.array_equal(belohlavek_binary_dataset.i(np.array([0, 2])), [0, 4, 5])
    assert np.array_equal(belohlavek_binary_dataset.i(np.array([2, 4])), [1, 3, 5])


def test_t_on_belohlavek_binary_dataset():
    assert np.array_equal(belohlavek_binary_dataset.t(np.array([0])), [0, 2])
    assert np.array_equal(belohlavek_binary_dataset.t(np.array([1])), [2, 4])
