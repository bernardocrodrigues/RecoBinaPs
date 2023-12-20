""" 
tests for BinaryDataset class
"""
import os
import random
import tempfile
from pathlib import Path
import numpy as np
from surprise import Dataset, Trainset
from pattern_mining.binaps.binaps_wrapper import generate_synthetic_data
from dataset.binary_dataset import (
    load_binary_dataset_from_trainset,
    load_from_binaps_compatible_input,
    save_as_binaps_compatible_input,
)

from tests.toy_datasets import (
    convert_raw_rating_list_into_trainset,
    my_toy_binary_dataset,
    my_toy_dataset_raw_rating,
    zaki_binary_dataset,
    zaki_dataset_raw_rating,
    belohlavek_binary_dataset,
    belohlavek_dataset_raw_rating,
    nenova_dataset_dataset,
)
from dataset.binary_dataset import i, t

CONVERT_DATASET_SHUFFLE_TIMES = 10


def assert_dataset_and_trainset_are_equal(
    converted_binary_dataset: np.ndarray, trainset: Trainset, original_dataset: np.ndarray
):
    """
    This method asserts that the converted_binary_dataset and the original_dataset are equal to the
    trainset. This enforces that the conversion from the trainset to the binary_dataset and back
    to the trainset is correct.

    Args:
        converted_binary_dataset: The binary_dataset that was converted from the trainset
        trainset: The trainset that was converted to the binary_dataset
        original_dataset: The original binary_dataset that was converted to the trainset

    Returns:
        None

    Raises:
        AssertionError: If the converted_binary_dataset and the original_dataset are not equal to
        the trainset

    """
    assert np.count_nonzero(converted_binary_dataset) == trainset.n_ratings
    assert np.count_nonzero(converted_binary_dataset) == np.count_nonzero(original_dataset)

    for iuid, row in enumerate(converted_binary_dataset):
        for iiid, item in enumerate(row):
            if item:
                uid = trainset.to_raw_uid(iuid)
                iid = trainset.to_raw_iid(iiid)

                assert original_dataset[uid][iid]

                user_ratings = trainset.ur[iuid]
                for this_iiid, rating in user_ratings:
                    if trainset.to_raw_iid(this_iiid) == iid:
                        assert rating
                        break
                else:
                    raise AssertionError(iuid, iiid, uid, iid, user_ratings)


def assert_i_success(dataset: np.ndarray, itemset: np.ndarray, expected_result):
    """
    This method asserts that the i(dataset, itemset) is equal to the expected_result
    """
    closure = i(dataset, itemset)
    assert isinstance(closure, np.ndarray)
    assert np.array_equal(closure, expected_result)


def assert_t_success(dataset: np.ndarray, itemset: np.ndarray, expected_result):
    """
    This method asserts that the t(dataset, itemset) is equal to the expected_result
    """
    closure = t(dataset, itemset)
    assert isinstance(closure, np.ndarray)
    assert np.array_equal(closure, expected_result)


class TestIT:
    # pylint: disable=missing-function-docstring

    def test_i_wrong_argument_types(self):
        with np.testing.assert_raises(AssertionError):
            i(my_toy_binary_dataset, "1")

        with np.testing.assert_raises(AssertionError):
            i(my_toy_binary_dataset, 1.0)

        with np.testing.assert_raises(AssertionError):
            i(my_toy_binary_dataset, True)

        with np.testing.assert_raises(AssertionError):
            i(my_toy_binary_dataset, [1, 2, 3])

    def test_i_on_my_toy_binary_dataset(self):
        assert_i_success(
            dataset=my_toy_binary_dataset, itemset=np.array([0]), expected_result=[0, 1, 3]
        )
        assert_i_success(
            dataset=my_toy_binary_dataset, itemset=np.array([1]), expected_result=[0, 1, 2, 3]
        )
        assert_i_success(
            dataset=my_toy_binary_dataset, itemset=np.array([2]), expected_result=[0, 1, 2, 3]
        )
        assert_i_success(
            dataset=my_toy_binary_dataset, itemset=np.array([3]), expected_result=[4, 5]
        )
        assert_i_success(
            dataset=my_toy_binary_dataset, itemset=np.array([4]), expected_result=[4, 5]
        )
        assert_i_success(
            dataset=my_toy_binary_dataset, itemset=np.array([5]), expected_result=[0, 1, 2, 3, 4, 5]
        )

        assert_i_success(
            dataset=my_toy_binary_dataset, itemset=np.array([0, 1]), expected_result=[0, 1, 3]
        )
        assert_i_success(
            dataset=my_toy_binary_dataset, itemset=np.array([0, 2]), expected_result=[0, 1, 3]
        )
        assert_i_success(
            dataset=my_toy_binary_dataset, itemset=np.array([0, 3]), expected_result=[]
        )
        assert_i_success(
            dataset=my_toy_binary_dataset, itemset=np.array([0, 4]), expected_result=[]
        )
        assert_i_success(
            dataset=my_toy_binary_dataset, itemset=np.array([0, 5]), expected_result=[0, 1, 3]
        )
        assert_i_success(
            dataset=my_toy_binary_dataset, itemset=np.array([0, 6]), expected_result=[]
        )

        assert_i_success(
            dataset=my_toy_binary_dataset, itemset=np.array([0, 1, 2]), expected_result=[0, 1, 3]
        )
        assert_i_success(
            dataset=my_toy_binary_dataset, itemset=np.array([0, 3, 5]), expected_result=[]
        )
        assert_i_success(
            dataset=my_toy_binary_dataset, itemset=np.array([1, 2, 5]), expected_result=[0, 1, 2, 3]
        )

        assert_i_success(
            dataset=my_toy_binary_dataset, itemset=np.array([0, 1, 2, 5]), expected_result=[0, 1, 3]
        )

        assert_i_success(
            dataset=my_toy_binary_dataset, itemset=np.array([0, 1, 2, 3, 4, 5]), expected_result=[]
        )

    def test_t_on_my_toy_binary_dataset(self):
        assert_t_success(
            dataset=my_toy_binary_dataset, itemset=np.array([0]), expected_result=[0, 1, 2, 5]
        )
        assert_t_success(
            dataset=my_toy_binary_dataset, itemset=np.array([0]), expected_result=[0, 1, 2, 5]
        )
        assert_t_success(
            dataset=my_toy_binary_dataset, itemset=np.array([1]), expected_result=[0, 1, 2, 5]
        )
        assert_t_success(
            dataset=my_toy_binary_dataset, itemset=np.array([2]), expected_result=[1, 2, 5]
        )
        assert_t_success(
            dataset=my_toy_binary_dataset, itemset=np.array([3]), expected_result=[0, 1, 2, 5]
        )
        assert_t_success(
            dataset=my_toy_binary_dataset, itemset=np.array([4]), expected_result=[3, 4, 5]
        )
        assert_t_success(
            dataset=my_toy_binary_dataset, itemset=np.array([5]), expected_result=[3, 4, 5]
        )
        assert_t_success(dataset=my_toy_binary_dataset, itemset=np.array([6]), expected_result=[6])

        assert_t_success(
            dataset=my_toy_binary_dataset, itemset=np.array([0, 1]), expected_result=[0, 1, 2, 5]
        )
        assert_t_success(
            dataset=my_toy_binary_dataset, itemset=np.array([0, 5]), expected_result=[5]
        )
        assert_t_success(
            dataset=my_toy_binary_dataset, itemset=np.array([2, 3]), expected_result=[1, 2, 5]
        )
        assert_t_success(
            dataset=my_toy_binary_dataset, itemset=np.array([4, 5]), expected_result=[3, 4, 5]
        )

        assert_t_success(
            dataset=my_toy_binary_dataset, itemset=np.array([0, 2, 5]), expected_result=[5]
        )

        assert_t_success(
            dataset=my_toy_binary_dataset, itemset=np.array([0, 1, 2, 3]), expected_result=[1, 2, 5]
        )
        assert_t_success(
            dataset=my_toy_binary_dataset, itemset=np.array([1, 2, 3, 4]), expected_result=[5]
        )

        assert_t_success(
            dataset=my_toy_binary_dataset, itemset=np.array([0, 1, 2, 3, 4]), expected_result=[5]
        )
        assert_t_success(
            dataset=my_toy_binary_dataset, itemset=np.array([0, 1, 2, 3, 4, 6]), expected_result=[]
        )

    def test_i_on_zaki_binary_dataset(self):
        assert_i_success(
            dataset=zaki_binary_dataset, itemset=np.array([0]), expected_result=[0, 1, 3, 4]
        )
        assert_i_success(
            dataset=zaki_binary_dataset, itemset=np.array([1]), expected_result=[1, 2, 4]
        )
        assert_i_success(
            dataset=zaki_binary_dataset, itemset=np.array([2]), expected_result=[0, 1, 4]
        )
        assert_i_success(
            dataset=zaki_binary_dataset, itemset=np.array([3]), expected_result=[0, 1, 2, 4]
        )
        assert_i_success(
            dataset=zaki_binary_dataset, itemset=np.array([4]), expected_result=[0, 1, 2, 3, 4]
        )
        assert_i_success(
            dataset=zaki_binary_dataset, itemset=np.array([5]), expected_result=[1, 2, 3]
        )

        assert_i_success(
            dataset=zaki_binary_dataset, itemset=np.array([0, 1]), expected_result=[1, 4]
        )
        assert_i_success(
            dataset=zaki_binary_dataset, itemset=np.array([1, 3]), expected_result=[1, 2, 4]
        )
        assert_i_success(
            dataset=zaki_binary_dataset, itemset=np.array([3, 4]), expected_result=[0, 1, 2, 4]
        )
        assert_i_success(
            dataset=zaki_binary_dataset, itemset=np.array([0, 5]), expected_result=[1, 3]
        )

        assert_i_success(
            dataset=zaki_binary_dataset, itemset=np.array([0, 1, 5]), expected_result=[1]
        )
        assert_i_success(
            dataset=zaki_binary_dataset, itemset=np.array([1, 2, 4]), expected_result=[1, 4]
        )
        assert_i_success(
            dataset=zaki_binary_dataset, itemset=np.array([2, 4, 1]), expected_result=[1, 4]
        )

        assert_i_success(
            dataset=zaki_binary_dataset, itemset=np.array([0, 1, 2, 3]), expected_result=[1, 4]
        )
        assert_i_success(
            dataset=zaki_binary_dataset, itemset=np.array([0, 1, 2, 3, 4]), expected_result=[1, 4]
        )
        assert_i_success(
            dataset=zaki_binary_dataset, itemset=np.array([0, 1, 2, 3, 4, 5]), expected_result=[1]
        )

    def test_t_on_zaki_binary_dataset(self):
        assert_t_success(
            dataset=zaki_binary_dataset, itemset=np.array([0]), expected_result=[0, 2, 3, 4]
        )
        assert_t_success(
            dataset=zaki_binary_dataset, itemset=np.array([1]), expected_result=[0, 1, 2, 3, 4, 5]
        )
        assert_t_success(
            dataset=zaki_binary_dataset, itemset=np.array([2]), expected_result=[1, 3, 4, 5]
        )
        assert_t_success(
            dataset=zaki_binary_dataset, itemset=np.array([3]), expected_result=[0, 4, 5]
        )
        assert_t_success(
            dataset=zaki_binary_dataset, itemset=np.array([4]), expected_result=[0, 1, 2, 3, 4]
        )

        assert_t_success(
            dataset=zaki_binary_dataset, itemset=np.array([0, 4]), expected_result=[0, 2, 3, 4]
        )
        assert_t_success(
            dataset=zaki_binary_dataset, itemset=np.array([1, 4]), expected_result=[0, 1, 2, 3, 4]
        )
        assert_t_success(
            dataset=zaki_binary_dataset, itemset=np.array([1, 3]), expected_result=[0, 4, 5]
        )
        assert_t_success(
            dataset=zaki_binary_dataset, itemset=np.array([3, 4]), expected_result=[0, 4]
        )

        assert_t_success(
            dataset=zaki_binary_dataset, itemset=np.array([0, 1, 4]), expected_result=[0, 2, 3, 4]
        )
        assert_t_success(
            dataset=zaki_binary_dataset, itemset=np.array([0, 2, 4]), expected_result=[3, 4]
        )
        assert_t_success(
            dataset=zaki_binary_dataset, itemset=np.array([0, 2, 3]), expected_result=[4]
        )

        assert_t_success(
            dataset=zaki_binary_dataset, itemset=np.array([0, 1, 2, 4]), expected_result=[3, 4]
        )
        assert_t_success(
            dataset=zaki_binary_dataset, itemset=np.array([0, 1, 2, 3]), expected_result=[4]
        )

        assert_t_success(
            dataset=zaki_binary_dataset, itemset=np.array([0, 1, 2, 3, 4]), expected_result=[4]
        )

    def test_closed_itemsets_on_zaki_binary_dataset(self):
        """
        Example from zaki page 245
        """

        def get_closure(itemset):
            return i(zaki_binary_dataset, t(zaki_binary_dataset, itemset))

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
        assert np.array_equal(get_closure(closed_itemset), closed_itemset) is False

        closed_itemset = np.array([2])
        assert np.array_equal(get_closure(closed_itemset), closed_itemset) is False

        closed_itemset = np.array([3])
        assert np.array_equal(get_closure(closed_itemset), closed_itemset) is False

        closed_itemset = np.array([4])
        assert np.array_equal(get_closure(closed_itemset), closed_itemset) is False

        closed_itemset = np.array([0, 3])
        assert np.array_equal(get_closure(closed_itemset), closed_itemset) is False

        closed_itemset = np.array([3, 4])
        assert np.array_equal(get_closure(closed_itemset), closed_itemset) is False

        closed_itemset = np.array([0, 1])
        assert np.array_equal(get_closure(closed_itemset), closed_itemset) is False

        closed_itemset = np.array([0, 4])
        assert np.array_equal(get_closure(closed_itemset), closed_itemset) is False

        closed_itemset = np.array([2, 4])
        assert np.array_equal(get_closure(closed_itemset), closed_itemset) is False

        closed_itemset = np.array([0, 1, 3])
        assert np.array_equal(get_closure(closed_itemset), closed_itemset) is False

        closed_itemset = np.array([0, 3, 4])
        assert np.array_equal(get_closure(closed_itemset), closed_itemset) is False

        closed_itemset = np.array([1, 3, 4])
        assert np.array_equal(get_closure(closed_itemset), closed_itemset) is False

    def test_i_on_belohlavek_binary_dataset(self):
        assert_i_success(
            dataset=belohlavek_binary_dataset, itemset=np.array([0, 2]), expected_result=[0, 4, 5]
        )
        assert_i_success(
            dataset=belohlavek_binary_dataset, itemset=np.array([2, 4]), expected_result=[1, 3, 5]
        )

    def test_t_on_belohlavek_binary_dataset(self):
        assert_t_success(
            dataset=belohlavek_binary_dataset, itemset=np.array([0]), expected_result=[0, 2]
        )
        assert_t_success(
            dataset=belohlavek_binary_dataset, itemset=np.array([1]), expected_result=[2, 4]
        )


def test_load_from_trainset_my_toy():
    for _ in range(CONVERT_DATASET_SHUFFLE_TIMES):
        my_toy_trainset = convert_raw_rating_list_into_trainset(my_toy_dataset_raw_rating)
        binary_dataset = load_binary_dataset_from_trainset(my_toy_trainset)
        assert_dataset_and_trainset_are_equal(
            binary_dataset, my_toy_trainset, my_toy_binary_dataset
        )


def test_load_from_trainset_zaki():
    for _ in range(CONVERT_DATASET_SHUFFLE_TIMES):
        zaki_trainset = convert_raw_rating_list_into_trainset(zaki_dataset_raw_rating)
        binary_dataset = load_binary_dataset_from_trainset(zaki_trainset)
        assert_dataset_and_trainset_are_equal(binary_dataset, zaki_trainset, zaki_binary_dataset)


def test_load_from_trainset_belohlavek():
    for _ in range(CONVERT_DATASET_SHUFFLE_TIMES):
        belohlavek_trainset = convert_raw_rating_list_into_trainset(belohlavek_dataset_raw_rating)
        binary_dataset = load_binary_dataset_from_trainset(belohlavek_trainset)
        assert_dataset_and_trainset_are_equal(
            binary_dataset, belohlavek_trainset, belohlavek_binary_dataset
        )


def test_save_as_binaps_compatible_input():
    def write_and_assert_file_output(dataset, line_results):
        with tempfile.TemporaryFile(mode="w+t") as file_object:
            save_as_binaps_compatible_input(dataset, file_object)

            file_object.seek(0)

            for file_line, reference_line in zip(file_object, line_results):
                assert file_line == reference_line

    my_toy_binary_dataset_line_results = [
        "1 2 4\n",
        "1 2 3 4\n",
        "1 2 3 4\n",
        "5 6\n",
        "5 6\n",
        "1 2 3 4 5 6\n",
        "7\n",
    ]

    zaki_line_results = [
        "1 2 4 5\n",
        "2 3 5\n",
        "1 2 5\n",
        "1 2 3 5\n",
        "1 2 3 4 5\n",
        "2 3 4\n",
    ]

    nenova_line_results = [
        "1 2 3\n",
        "1 2 3 4 5\n",
        "4 5\n",
        "4 5 6 7\n",
        "6 7\n",
        "6 7\n",
    ]

    write_and_assert_file_output(my_toy_binary_dataset, my_toy_binary_dataset_line_results)
    write_and_assert_file_output(zaki_binary_dataset, zaki_line_results)
    write_and_assert_file_output(nenova_dataset_dataset, nenova_line_results)


def test_save_as_binaps_compatible_input_on_movielens():
    threshold = 1.0

    dataset = Dataset.load_builtin("ml-1m", prompt=False)
    trainset = dataset.build_full_trainset()
    binary_dataset = load_binary_dataset_from_trainset(trainset, threshold=threshold)

    with tempfile.TemporaryFile(mode="r+") as file_object:
        save_as_binaps_compatible_input(binary_dataset, file_object)

        file_object.seek(0)
        file_lines = file_object.read().split("\n")

        for user, item, rating in trainset.all_ratings():
            if rating >= threshold:
                binaps_item = str(item + 1)
                assert binaps_item in file_lines[user].split(" ")

    dataset = Dataset.load_builtin("ml-100k", prompt=False)
    trainset = dataset.build_full_trainset()
    binary_dataset = load_binary_dataset_from_trainset(trainset, threshold=threshold)

    with tempfile.TemporaryFile(mode="r+") as file_object:
        save_as_binaps_compatible_input(binary_dataset, file_object)

        file_object.seek(0)
        file_lines = file_object.read().split("\n")

        for user, item, rating in trainset.all_ratings():
            if rating >= threshold:
                assert str(item + 1) in file_lines[user].split(" ")


def test_load_from_binaps_compatible_input_on_example_datasets():
    def write_and_assert_file_output(dataset):
        file_object = tempfile.NamedTemporaryFile(mode="w+t", delete=False)
        save_as_binaps_compatible_input(dataset, file_object)
        file_object.close()

        saved_dataset = load_from_binaps_compatible_input(file_object.name)
        np.testing.assert_array_equal(saved_dataset, dataset)

    write_and_assert_file_output(my_toy_binary_dataset)
    write_and_assert_file_output(zaki_binary_dataset)
    write_and_assert_file_output(nenova_dataset_dataset)
    write_and_assert_file_output(belohlavek_binary_dataset)


def test_load_and_save_procedures_on_synthetic_data():
    def test_against_synthetic_data(synthetic_data_path):
        synthetic_dataset = load_from_binaps_compatible_input(synthetic_data_path)

        file_object = tempfile.NamedTemporaryFile(mode="w+t", delete=False)
        save_as_binaps_compatible_input(synthetic_dataset, file_object)
        file_object.close()

        with open(file_object.name, encoding="UTF-8") as generated, open(
            synthetic_data_path, encoding="UTF-8"
        ) as reference:
            assert generated.read() == reference.read()

        yet_another_synthetic_dataset = load_from_binaps_compatible_input(file_object.name)

        np.testing.assert_array_equal(synthetic_dataset, yet_another_synthetic_dataset)

        yet_another_file_object = tempfile.NamedTemporaryFile(mode="w+t", delete=False)
        save_as_binaps_compatible_input(yet_another_synthetic_dataset, yet_another_file_object)
        yet_another_file_object.close()

        with open(yet_another_file_object.name, encoding="UTF-8") as generated, open(
            synthetic_data_path, encoding="UTF-8"
        ) as reference:
            assert generated.read() == reference.read()

        os.remove(file_object.name)
        os.remove(yet_another_file_object.name)

    prefix = "/tmp/data"

    non_overlapping_data_path = f"{prefix}.dat"
    non_overlapping_data_patterns_path = f"{prefix}.dat_patterns.txt"
    overlapping_data_path = f"{prefix}_itemOverlap.dat"
    overlapping_data_patterns_path = f"{prefix}_itemOverlap.dat_patterns.txt"

    row_quantity = random.randint(100, 20000)
    column_quantity = random.randint(1, 100)
    max_pattern_size = random.randint(1, 100)
    noise = random.uniform(0, 1)
    density = random.uniform(0, 1)

    tmp_dir = Path(tempfile.mkdtemp())

    generate_synthetic_data(
        tmp_dir, row_quantity, column_quantity, prefix, max_pattern_size, noise, density
    )

    test_against_synthetic_data(non_overlapping_data_path)
    test_against_synthetic_data(overlapping_data_path)

    os.remove(non_overlapping_data_path)
    os.remove(non_overlapping_data_patterns_path)
    os.remove(overlapping_data_path)
    os.remove(overlapping_data_patterns_path)


def test_load_and_save_procedures_on_movielens():
    threshold = 1.0
    dataset = Dataset.load_builtin("ml-100k", prompt=False)
    trainset = dataset.build_full_trainset()
    binary_dataset = load_binary_dataset_from_trainset(trainset, threshold=threshold)

    with tempfile.NamedTemporaryFile(mode="w+t") as file_object:
        save_as_binaps_compatible_input(binary_dataset, file_object)
        file_object.seek(0)
        file_lines = file_object.read().split("\n")

        for user, item, rating in trainset.all_ratings():
            if rating >= threshold:
                assert str(item + 1) in file_lines[user]

    dataset = Dataset.load_builtin("ml-1m", prompt=False)
    trainset = dataset.build_full_trainset()
    binary_dataset = load_binary_dataset_from_trainset(trainset, threshold=threshold)

    with tempfile.NamedTemporaryFile(mode="w+t") as file_object:
        save_as_binaps_compatible_input(binary_dataset, file_object)
        file_object.seek(0)
        file_lines = file_object.read().split("\n")

        for user, item, rating in trainset.all_ratings():
            if rating >= threshold:
                assert str(item + 1) in file_lines[user]


# pylint: enable=missing-function-docstring

# todo: add sparsity and number of zeros tests
