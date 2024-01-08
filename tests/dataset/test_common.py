import numpy as np
import pandas as pd
import pytest

from surprise import Dataset
from surprise import Reader
from surprise import Trainset
from dataset.common import convert_trainset_to_matrix


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
