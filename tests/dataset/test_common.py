import numpy as np
import pandas as pd
import pytest

from surprise import Dataset
from surprise import Reader
from surprise import Trainset
from dataset.common import (
    convert_trainset_to_matrix,
    generate_random_dataset,
    convert_to_raw_ratings,
)


class TestConvertTrainsetToMatrix:
    def df_matches_matrix(
        self, dataframe: pd.DataFrame, matrix: np.ndarray, trainset: Trainset
    ) -> bool:
        number_of_users_df = dataframe["user"].unique().shape[0]
        number_of_items_df = dataframe["item"].unique().shape[0]

        number_of_users_matrix = matrix.shape[0]
        number_of_items_matrix = matrix.shape[1]

        if number_of_users_df != number_of_users_matrix:
            return False

        if number_of_items_df != number_of_items_matrix:
            return False

        number_of_ratings_df = dataframe.shape[0]
        number_of_ratings_matrix = np.count_nonzero(~np.isnan(matrix))

        if number_of_ratings_df != number_of_ratings_matrix:
            return False

        for _, row in dataframe.iterrows():
            ruid = row["user"]
            riid = row["item"]
            rating = row["rating"]

            iuid = trainset.to_inner_uid(ruid)
            iiid = trainset.to_inner_iid(riid)

            if matrix[iuid][iiid] != rating:
                return False

        return True

    def test_invalid_input(self):
        with pytest.raises(AssertionError):
            convert_trainset_to_matrix(None)

        with pytest.raises(AssertionError):
            convert_trainset_to_matrix(1)

        with pytest.raises(AssertionError):
            convert_trainset_to_matrix("abc")

        with pytest.raises(AssertionError):
            convert_trainset_to_matrix(1.0)

        with pytest.raises(AssertionError):
            convert_trainset_to_matrix([1, 2, 3])

        with pytest.raises(AssertionError):
            convert_trainset_to_matrix((1, 2, 3))

        with pytest.raises(AssertionError):
            convert_trainset_to_matrix(np.array([1, 2, 3]))

    def test_empty_trainset(self):
        raw_dataset_as_pandas_df = pd.DataFrame(
            data={
                "user": [],
                "item": [],
                "rating": [],
            }
        )
        dataset = Dataset.load_from_df(raw_dataset_as_pandas_df, reader=Reader(rating_scale=(0, 5)))
        trainset = dataset.build_full_trainset()

        result = convert_trainset_to_matrix(trainset)

        assert result.shape == (0, 0)

    def test_single_rating(self):
        raw_dataset_as_pandas_df = pd.DataFrame(
            data={
                "user": [0],
                "item": [0],
                "rating": [3],
            }
        )
        dataset = Dataset.load_from_df(raw_dataset_as_pandas_df, reader=Reader(rating_scale=(0, 5)))
        trainset = dataset.build_full_trainset()

        result = convert_trainset_to_matrix(trainset)

        assert self.df_matches_matrix(raw_dataset_as_pandas_df, result, trainset)

    def test_multiple_ratings_1(self):
        raw_dataset_as_pandas_df = pd.DataFrame(
            data={
                "user": [0, 0, 1, 1, 2, 2],
                "item": [0, 1, 0, 1, 0, 1],
                "rating": [3, 4, 1, 2, 5, 3],
            }
        )
        dataset = Dataset.load_from_df(raw_dataset_as_pandas_df, reader=Reader(rating_scale=(0, 5)))
        trainset = dataset.build_full_trainset()

        result = convert_trainset_to_matrix(trainset)

        assert self.df_matches_matrix(raw_dataset_as_pandas_df, result, trainset)

    def test_multiple_ratings_2(self):
        raw_dataset_as_pandas_df = pd.DataFrame(
            data={
                "user": [1, 1, 1, 0, 0, 0],
                "item": [0, 1, 2, 0, 1, 2],
                "rating": [3, 4, 1, 2, 5, 3],
            }
        )
        dataset = Dataset.load_from_df(raw_dataset_as_pandas_df, reader=Reader(rating_scale=(0, 5)))
        trainset = dataset.build_full_trainset()

        result = convert_trainset_to_matrix(trainset)

        assert self.df_matches_matrix(raw_dataset_as_pandas_df, result, trainset)

    def test_multiple_ratings_3(self):
        raw_dataset_as_pandas_df = pd.DataFrame(
            data={
                "user": [1, 1, 1, 0, 0, 0],
                "item": [2, 1, 0, 2, 1, 0],
                "rating": [3, 4, 1, 2, 5, 3],
            }
        )
        dataset = Dataset.load_from_df(raw_dataset_as_pandas_df, reader=Reader(rating_scale=(0, 5)))
        trainset = dataset.build_full_trainset()

        result = convert_trainset_to_matrix(trainset)

        assert self.df_matches_matrix(raw_dataset_as_pandas_df, result, trainset)

    def test_multiple_ratings_4(self):
        raw_dataset_as_pandas_df = pd.DataFrame(
            data={
                "user": [2, 1, 0, 2, 1, 0],
                "item": [2, 1, 0, 0, 2, 1],
                "rating": [1, 2, 3, 4, 1.1, 1.5],
            }
        )
        dataset = Dataset.load_from_df(raw_dataset_as_pandas_df, reader=Reader(rating_scale=(0, 5)))
        trainset = dataset.build_full_trainset()

        result = convert_trainset_to_matrix(trainset)

        assert self.df_matches_matrix(raw_dataset_as_pandas_df, result, trainset)


class TestGenerateRandomDataset:
    def test_invalid_input(self):
        with pytest.raises(AssertionError):
            generate_random_dataset(0, 5, 5, 0.5)

        with pytest.raises(AssertionError):
            generate_random_dataset(10, 0, 5, 0.5)

        with pytest.raises(AssertionError):
            generate_random_dataset(10, 5, 0, 0.5)

        with pytest.raises(AssertionError):
            generate_random_dataset(10, 5, 5, 0)

        with pytest.raises(AssertionError):
            generate_random_dataset(10, 5, 5, 1.5)

    def test_valid_input(self):
        number_of_users = 1000
        number_of_items = 5000
        rating_scale = 5
        sparsity_target = 0.3

        dataset = generate_random_dataset(
            number_of_users, number_of_items, rating_scale, sparsity_target
        )

        sparsity = 1 - (np.isnan(dataset).sum() / dataset.size)

        assert isinstance(dataset, np.ndarray)
        assert dataset.shape == (number_of_users, number_of_items)
        assert np.nanmin(dataset) >= 0
        assert np.nanmax(dataset) <= rating_scale
        assert np.isclose(sparsity, sparsity_target, atol=0.01)


class TestConvertToRawRatings:
    def test_invalid_input(self):
        with pytest.raises(AssertionError):
            convert_to_raw_ratings(None)

        with pytest.raises(AssertionError):
            convert_to_raw_ratings(1)

        with pytest.raises(AssertionError):
            convert_to_raw_ratings("abc")

        with pytest.raises(AssertionError):
            convert_to_raw_ratings(1.0)

        with pytest.raises(AssertionError):
            convert_to_raw_ratings([1, 2, 3])

        with pytest.raises(AssertionError):
            convert_to_raw_ratings((1, 2, 3))

    def test_convert_to_raw_ratings_all_nan(self):
        dataset = np.array([[np.nan, np.nan], [np.nan, np.nan]])

        expected_raw_ratings = []

        raw_ratings = convert_to_raw_ratings(dataset)

        assert raw_ratings == expected_raw_ratings

    def test_convert_to_raw_ratings_empty_dataset(self):
        dataset = np.empty((0, 0))

        expected_raw_ratings = []

        raw_ratings = convert_to_raw_ratings(dataset)

        assert raw_ratings == expected_raw_ratings

    def test_convert_to_raw_ratings_single_value(self):
        dataset = np.array([[3]])

        expected_raw_ratings = [(0, 0, 3.0, None)]

        raw_ratings = convert_to_raw_ratings(dataset)

        assert raw_ratings == expected_raw_ratings

    def test_convert_to_raw_ratings(self):
        dataset = np.array([[1, np.nan, 3], [4, 5, np.nan], [np.nan, 2, 6]])

        expected_raw_ratings = [
            (0, 0, 1.0, None),
            (0, 2, 3.0, None),
            (1, 0, 4.0, None),
            (1, 1, 5.0, None),
            (2, 1, 2.0, None),
            (2, 2, 6.0, None),
        ]

        raw_ratings = convert_to_raw_ratings(dataset)

        assert raw_ratings == expected_raw_ratings
