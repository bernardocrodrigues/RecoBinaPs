import numpy as np

from tests.ToyDatasets import (
    convert_raw_rating_list_into_trainset,
    my_toy_binary_dataset,
    my_toy_dataset_raw_rating,
    zaki_binary_dataset,
    zaki_dataset_raw_rating,
    belohlavek_binary_dataset,
    belohlavek_dataset_raw_rating,
)
from lib.BinaryDataset import BinaryDataset

CONVERT_DATASET_SHUFFLE_TIMES = 10


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

    closed_itemset = np.array([1, 2, 4])
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

    closed_itemset = np.array([0, 1])
    assert np.array_equal(get_closure(closed_itemset), closed_itemset) == False

    closed_itemset = np.array([0, 4])
    assert np.array_equal(get_closure(closed_itemset), closed_itemset) == False

    closed_itemset = np.array([2, 4])
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


def assert_dataset_and_trainset_are_equal(converted_binary_dataset, trainset, original_dataset):

    assert np.count_nonzero(converted_binary_dataset._binary_dataset) == trainset.n_ratings
    assert np.count_nonzero(converted_binary_dataset._binary_dataset) == np.count_nonzero(original_dataset._binary_dataset)

    for iuid, row in enumerate(converted_binary_dataset._binary_dataset):
        for iiid, item in enumerate(row):
            if item:
                uid = trainset.to_raw_uid(iuid)
                iid = trainset.to_raw_iid(iiid)

                assert original_dataset._binary_dataset[uid][iid]

                user_ratings = trainset.ur[iuid]
                for this_iiid, rating in user_ratings:
                    if trainset.to_raw_iid(this_iiid) == iid:
                        assert rating
                        break
                else:
                    raise Exception(iuid, iiid, uid, iid, user_ratings)


def test_load_from_trainset_my_toy():
    for _ in range(CONVERT_DATASET_SHUFFLE_TIMES):
        my_toy_trainset = convert_raw_rating_list_into_trainset(my_toy_dataset_raw_rating)
        binary_dataset = BinaryDataset.load_from_trainset(my_toy_trainset)
        assert_dataset_and_trainset_are_equal(binary_dataset, my_toy_trainset, my_toy_binary_dataset)


def test_load_from_trainset_zaki():
    for _ in range(CONVERT_DATASET_SHUFFLE_TIMES):
        zaki_trainset = convert_raw_rating_list_into_trainset(zaki_dataset_raw_rating)
        binary_dataset = BinaryDataset.load_from_trainset(zaki_trainset)
        assert_dataset_and_trainset_are_equal(binary_dataset, zaki_trainset, zaki_binary_dataset)


def test_load_from_trainset_belohlavek():
    for _ in range(CONVERT_DATASET_SHUFFLE_TIMES):
        belohlavek_trainset = convert_raw_rating_list_into_trainset(belohlavek_dataset_raw_rating)
        binary_dataset = BinaryDataset.load_from_trainset(belohlavek_trainset)
        assert_dataset_and_trainset_are_equal(binary_dataset, belohlavek_trainset, belohlavek_binary_dataset)


# todo: add sparsity and number of zeros tests