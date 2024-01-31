""" test_evaluation.py 

This module contains unit tests for the evaluation module.

Copyright 2023 Bernardo C. Rodrigues
See LICENSE file for license details

"""
import statistics
from typing import List
from collections import defaultdict
import pytest
import numpy as np

from surprise.prediction_algorithms.predictions import Prediction

from evaluation import (
    get_micro_averaged_precision,
    get_macro_averaged_precision,
    get_precision_at_k,
    get_micro_averaged_recall,
    get_macro_averaged_recall,
    get_recall_at_k,
)

RANDOM_NUMBER_GENERATOR = np.random.default_rng(seed=42)


def generate_predictions_list(
    num_predictions: int, num_users: int, num_items: int
) -> List[Prediction]:
    """
    Generate a list of random Surprise predictions.

    Args:
        num_predictions (int): Number of predictions to generate.
        num_users (int): Number of users in the dataset.
        num_items (int): Number of items in the dataset.

    Returns:
        List[Prediction]: List of random predictions.
    """
    predictions = []
    user_item_pairs = set()

    for _ in range(num_predictions):
        while True:
            user = RANDOM_NUMBER_GENERATOR.integers(0, num_users)
            item = RANDOM_NUMBER_GENERATOR.integers(0, num_items)

            if (user, item) not in user_item_pairs:
                break

        new_prediction = Prediction(
            uid=user,
            iid=item,
            r_ui=RANDOM_NUMBER_GENERATOR.uniform(0, 5),
            est=RANDOM_NUMBER_GENERATOR.uniform(0, 5),
            details=None,
        )

        predictions.append(new_prediction)

    return predictions


def group_predictions_by_user(predictions: List[Prediction]):
    """
    Group a list of predictions by user.

    Args:
        predictions (List[Prediction]): List of predictions.

    Returns:
        Dict[int, List[Prediction]]: Dictionary of predictions grouped by user. Dictionary keys are
                                     user IDs.

    """
    predictions_per_user = defaultdict(list)
    for prediction in predictions:
        predictions_per_user[prediction.uid].append(prediction)

    return predictions_per_user


def get_relevant_items(predictions: List[Prediction], threshold: float):
    """
    Get a list of relevant items from a list of predictions.

    Relevant items are those whose true rating is greater than or equal to the threshold. In other
    words, relevant items are those that is known that the user has rated positively.

    Args:
        predictions (List[Prediction]): List of predictions.
        threshold (float): Threshold for relevance.

    Returns:
        List[Prediction]: List of relevant items.
    """
    relevant_items = []

    for prediction in predictions:
        if prediction.r_ui >= threshold:
            relevant_items.append(prediction)

    return relevant_items


def get_selected_items(predictions: List[Prediction], threshold: float):
    """
    Get a list of selected items from a list of predictions.

    Selected items are those whose estimated rating is greater than or equal to the threshold. In
    other words, selected items are those that the algorithm has predicted that the user will rate
    positively.

    Args:
        predictions (List[Prediction]): List of predictions.
        threshold (float): Threshold for selection.

    Returns:
        List[Prediction]: List of selected items.
    """
    selected_items = []

    for prediction in predictions:
        if prediction.est >= threshold:
            selected_items.append(prediction)

    return selected_items


def get_intersection(relevant_items: List[Prediction], selected_items: List[Prediction]):
    """
    Get the intersection of two lists of predictions.

    Args:
        relevant_items (List[Prediction]): List of relevant items.
        selected_items (List[Prediction]): List of selected items.

    Returns:
        List[Prediction]: List of predictions that are both relevant and selected.

    """
    intersection = []

    for relevant_item in relevant_items:
        if relevant_item in selected_items:
            intersection.append(relevant_item)

    return intersection


# pylint: disable=missing-class-docstring missing-function-docstring unused-argument


class TestMicroAveragedPrecision:
    def test_micro_averaged_precision_all_predictions_are_relevant_1(self):
        predictions = [
            Prediction(uid=None, iid=None, r_ui=1, est=1, details=None),
            Prediction(uid=None, iid=None, r_ui=1, est=1, details=None),
            Prediction(uid=None, iid=None, r_ui=1, est=1, details=None),
        ]

        # True Positives = 3, False Positives = 0
        # Precision = 3 / (3 + 0) = 1.0
        assert np.isclose(get_micro_averaged_precision(predictions), 1.0)

    def test_micro_averaged_precision_all_predictions_are_relevant_2(self):
        predictions = [
            Prediction(uid=None, iid=None, r_ui=0, est=1, details=None),
            Prediction(uid=None, iid=None, r_ui=1, est=1, details=None),
            Prediction(uid=None, iid=None, r_ui=0, est=1, details=None),
            Prediction(uid=None, iid=None, r_ui=1, est=1, details=None),
        ]

        # True Positives = 2, False Positives = 0
        # Precision = 2 / (2 + 0) = 0.5
        assert get_micro_averaged_precision(predictions), 0.5

    def test_micro_averaged_precision_all_predictions_are_relevant_3(self):
        predictions = [
            Prediction(uid=None, iid=None, r_ui=1, est=5, details=None),
            Prediction(uid=None, iid=None, r_ui=1, est=2, details=None),
            Prediction(uid=None, iid=None, r_ui=0, est=2, details=None),
            Prediction(uid=None, iid=None, r_ui=4, est=2, details=None),
            Prediction(uid=None, iid=None, r_ui=5, est=5, details=None),
        ]

        # True Positives = 2, False Positives = 3
        # Precision = 2 / (2 + 3) = 0.4
        assert np.isclose(get_micro_averaged_precision(predictions, threshold=2), 0.4)

    def test_micro_averaged_precision_all_predictions_are_relevant_4(self):
        predictions = [
            Prediction(uid=None, iid=None, r_ui=1, est=5, details=None),
            Prediction(uid=None, iid=None, r_ui=3, est=3, details=None),
            Prediction(uid=None, iid=None, r_ui=3, est=3, details=None),
            Prediction(uid=None, iid=None, r_ui=4, est=3, details=None),
            Prediction(uid=None, iid=None, r_ui=0, est=5, details=None),
        ]

        # True Positives = 3, False Positives = 2
        # Precision = 3 / (3 + 2) = 0.6
        assert np.isclose(get_micro_averaged_precision(predictions, threshold=3), 0.6)

    def test_micro_averaged_precision_no_predictions_are_relevant_1(self):
        predictions = [
            Prediction(uid=None, iid=None, r_ui=1, est=0, details=None),
            Prediction(uid=None, iid=None, r_ui=2, est=0, details=None),
            Prediction(uid=None, iid=None, r_ui=3, est=0, details=None),
        ]

        # True Positives = 0, False Positives = 0
        # Precision = 0 / (0 + 0) = 0.0
        assert np.isclose(get_micro_averaged_precision(predictions), 0.0)

    def test_micro_averaged_precision_no_predictions_are_relevant_2(self):
        predictions = [
            Prediction(uid=None, iid=None, r_ui=3, est=1, details=None),
            Prediction(uid=None, iid=None, r_ui=3, est=1, details=None),
            Prediction(uid=None, iid=None, r_ui=3, est=1, details=None),
            Prediction(uid=None, iid=None, r_ui=3, est=1, details=None),
        ]

        # True Positives = 0, False Positives = 0
        # Precision = 0 / (0 + 0) = 0.0
        assert np.isclose(get_micro_averaged_precision(predictions, threshold=2), 0.0)

    def test_micro_averaged_precision_no_predictions_are_relevant_3(self):
        predictions = [
            Prediction(uid=None, iid=None, r_ui=2, est=1, details=None),
            Prediction(uid=None, iid=None, r_ui=2, est=1, details=None),
            Prediction(uid=None, iid=None, r_ui=3, est=1, details=None),
            Prediction(uid=None, iid=None, r_ui=4, est=1, details=None),
            Prediction(uid=None, iid=None, r_ui=5, est=1, details=None),
        ]

        # True Positives = 0, False Positives = 0
        # Precision = 0 / (0 + 0) = 0.0
        assert np.isclose(get_micro_averaged_precision(predictions, threshold=2), 0.0)

    def test_micro_averaged_precision_no_predictions_are_relevant_4(self):
        predictions = [
            Prediction(uid=None, iid=None, r_ui=1, est=1, details=None),
            Prediction(uid=None, iid=None, r_ui=1, est=1, details=None),
            Prediction(uid=None, iid=None, r_ui=1, est=1, details=None),
            Prediction(uid=None, iid=None, r_ui=1, est=1, details=None),
            Prediction(uid=None, iid=None, r_ui=1, est=1, details=None),
        ]

        # True Positives = 0, False Positives = 0
        # Precision = 0 / (0 + 0) = 0.0
        assert np.isclose(get_micro_averaged_precision(predictions, threshold=3), 0.0)

    def test_micro_averaged_precision_some_predictions_are_relevant_no_false_positives_1(self):
        predictions = [
            Prediction(uid=None, iid=None, r_ui=1, est=1, details=None),
            Prediction(uid=None, iid=None, r_ui=2, est=1, details=None),
            Prediction(uid=None, iid=None, r_ui=3, est=1, details=None),
            Prediction(uid=None, iid=None, r_ui=0, est=0, details=None),
            Prediction(uid=None, iid=None, r_ui=0, est=0, details=None),
        ]

        # True Positives = 3, False Positives = 0
        # Precision = 3 / (3 + 0) = 1.0
        assert np.isclose(get_micro_averaged_precision(predictions, threshold=1), 1.0)

    def test_micro_averaged_precision_some_predictions_are_relevant_no_false_positives_2(self):
        predictions = [
            Prediction(uid=None, iid=None, r_ui=1, est=1, details=None),
            Prediction(uid=None, iid=None, r_ui=3, est=5, details=None),
            Prediction(uid=None, iid=None, r_ui=1, est=1, details=None),
            Prediction(uid=None, iid=None, r_ui=5, est=2, details=None),
            Prediction(uid=None, iid=None, r_ui=3, est=5, details=None),
        ]

        # True Positives = 2, False Positives = 0
        # Precision = 2 / (2 + 0) = 1.0
        assert np.isclose(get_micro_averaged_precision(predictions, threshold=3), 1.0)

    def test_micro_averaged_precision_some_predictions_are_relevant_no_false_positives_3(self):
        predictions = [
            Prediction(uid=None, iid=None, r_ui=1, est=1, details=None),
            Prediction(uid=None, iid=None, r_ui=1, est=1, details=None),
            Prediction(uid=None, iid=None, r_ui=1, est=0, details=None),
            Prediction(uid=None, iid=None, r_ui=1, est=0, details=None),
            Prediction(uid=None, iid=None, r_ui=1, est=1, details=None),
        ]

        # True Positives = 3, False Positives = 0
        # Precision = 3 / (3 + 0) = 1.0
        assert np.isclose(get_micro_averaged_precision(predictions), 1.0)

    def test_micro_averaged_precision_some_predictions_are_relevant_no_false_positives_4(self):
        predictions = [
            Prediction(uid=None, iid=None, r_ui=1, est=1, details=None),
            Prediction(uid=None, iid=None, r_ui=1, est=0, details=None),
            Prediction(uid=None, iid=None, r_ui=1, est=0, details=None),
            Prediction(uid=None, iid=None, r_ui=1, est=1, details=None),
            Prediction(uid=None, iid=None, r_ui=1, est=0, details=None),
        ]

        # True Positives = 2, False Positives = 0
        # Precision = 2 / (2 + 0) = 1.0
        assert np.isclose(get_micro_averaged_precision(predictions), 1.0)

    def test_micro_averaged_precision_all_predictions_are_relevant_and_some_false_positives_1(self):
        predictions = [
            Prediction(uid=None, iid=None, r_ui=1, est=1, details=None),
            Prediction(uid=None, iid=None, r_ui=2, est=1, details=None),
            Prediction(uid=None, iid=None, r_ui=3, est=1, details=None),
            Prediction(uid=None, iid=None, r_ui=0, est=1, details=None),
            Prediction(uid=None, iid=None, r_ui=0, est=1, details=None),
        ]

        # True Positives = 3, False Positives = 2
        # Precision = 3 / (3 + 2) = 0.6
        assert np.isclose(get_micro_averaged_precision(predictions), 0.6)

    def test_micro_averaged_precision_all_predictions_are_relevant_and_some_false_positives_2(self):
        predictions = [
            Prediction(uid=None, iid=None, r_ui=1, est=3, details=None),
            Prediction(uid=None, iid=None, r_ui=3, est=5, details=None),
            Prediction(uid=None, iid=None, r_ui=1, est=3, details=None),
            Prediction(uid=None, iid=None, r_ui=5, est=3, details=None),
            Prediction(uid=None, iid=None, r_ui=3, est=5, details=None),
        ]

        # True Positives = 3, False Positives = 2
        # Precision = 3 / (3 + 2) = 0.6
        assert np.isclose(get_micro_averaged_precision(predictions, threshold=3), 0.6)

    def test_micro_averaged_precision_some_predictions_are_relevant_and_some_false_positives_1(
        self,
    ):
        predictions = [
            Prediction(uid=None, iid=None, r_ui=1, est=1, details=None),
            Prediction(uid=None, iid=None, r_ui=0, est=1, details=None),
            Prediction(uid=None, iid=None, r_ui=3, est=1, details=None),
            Prediction(uid=None, iid=None, r_ui=0, est=1, details=None),
            Prediction(uid=None, iid=None, r_ui=0, est=0, details=None),
        ]

        # True Positives = 2, False Positives = 2
        # Precision = 2 / (2 + 2) = 0.6
        assert np.isclose(get_micro_averaged_precision(predictions), 0.5)

    @pytest.mark.parametrize("execution_number", range(100))
    def test_micro_averaged_precision_fuzzy(self, execution_number):
        predictions = generate_predictions_list(10000, 1000, 1000)

        threshold = RANDOM_NUMBER_GENERATOR.uniform(0, 5)

        relevant_items = get_relevant_items(predictions, threshold=threshold)
        selected_items = get_selected_items(predictions, threshold=threshold)
        intersection = get_intersection(relevant_items, selected_items)

        precision = len(intersection) / len(selected_items)

        assert np.isclose(get_micro_averaged_precision(predictions, threshold=threshold), precision)


class TestMacroAveragedPrecision:
    @pytest.mark.parametrize(
        "predictions, threshold, expected",
        [
            (
                [
                    Prediction(uid=1, iid=None, r_ui=1, est=1, details=None),
                    Prediction(uid=1, iid=None, r_ui=1, est=1, details=None),
                    Prediction(uid=1, iid=None, r_ui=1, est=1, details=None),
                    Prediction(uid=2, iid=None, r_ui=1, est=1, details=None),
                    Prediction(uid=2, iid=None, r_ui=1, est=1, details=None),
                    Prediction(uid=2, iid=None, r_ui=1, est=1, details=None),
                ],
                1,
                1.0,
            ),
            # User 1: True Positives = 3, False Positives = 0
            # User 2: True Positives = 3, False Positives = 0
            # Precision = (3 / (3 + 0) + 3 / (3 + 0)) / 2 = 1.0
            (
                [
                    Prediction(uid=1, iid=None, r_ui=1, est=1, details=None),
                    Prediction(uid=1, iid=None, r_ui=1, est=1, details=None),
                    Prediction(uid=1, iid=None, r_ui=1, est=1, details=None),
                    Prediction(uid=2, iid=None, r_ui=1, est=1, details=None),
                    Prediction(uid=2, iid=None, r_ui=1, est=1, details=None),
                    Prediction(uid=2, iid=None, r_ui=1, est=1, details=None),
                    Prediction(uid=3, iid=None, r_ui=1, est=1, details=None),
                    Prediction(uid=3, iid=None, r_ui=1, est=1, details=None),
                    Prediction(uid=3, iid=None, r_ui=1, est=1, details=None),
                ],
                1,
                1.0,
            ),
            # User 1: True Positives = 3, False Positives = 0
            # User 2: True Positives = 3, False Positives = 0
            # User 3: True Positives = 3, False Positives = 0
            # Precision = (3 / (3 + 0) + 3 / (3 + 0) + 3 / (3 + 0)) / 3 = 1.0
            (
                [
                    Prediction(uid=1, iid=None, r_ui=1, est=3, details=None),
                    Prediction(uid=1, iid=None, r_ui=3, est=2, details=None),
                    Prediction(uid=2, iid=None, r_ui=2, est=1, details=None),
                    Prediction(uid=2, iid=None, r_ui=5, est=4, details=None),
                    Prediction(uid=2, iid=None, r_ui=1, est=1, details=None),
                    Prediction(uid=3, iid=None, r_ui=1, est=5, details=None),
                    Prediction(uid=3, iid=None, r_ui=2, est=5, details=None),
                    Prediction(uid=3, iid=None, r_ui=1, est=1, details=None),
                    Prediction(uid=3, iid=None, r_ui=3, est=5, details=None),
                    Prediction(uid=3, iid=None, r_ui=2, est=4, details=None),
                ],
                2,
                0.75,
            ),
            # User 1: True Positives = 1, False Positives = 1
            # User 2: True Positives = 1, False Positives = 0
            # User 3: True Positives = 3, False Positives = 1
            # Precision = (1 / (1 + 1) + 1 / (1 + 0) + 3 / (3 + 1)) / 3 = 0.75
            (
                [
                    Prediction(uid=1, iid=None, r_ui=1, est=0, details=None),
                    Prediction(uid=1, iid=None, r_ui=2, est=0, details=None),
                    Prediction(uid=1, iid=None, r_ui=3, est=0, details=None),
                    Prediction(uid=2, iid=None, r_ui=1, est=0, details=None),
                    Prediction(uid=2, iid=None, r_ui=2, est=0, details=None),
                    Prediction(uid=2, iid=None, r_ui=3, est=0, details=None),
                ],
                1,
                0.0,
            ),
        ],
        #     # User 1: True Positives = 0, False Positives = 0
        #     # User 2: True Positives = 0, False Positives = 0
        #     # Precision = (0 / (0 + 0) + 0 / (0 + 0)) / 2 = 0.0
    )
    def test_macro_averaged_precision_all_predictions_are_relevant(
        self, predictions, threshold, expected
    ):
        assert np.isclose(get_macro_averaged_precision(predictions, threshold=threshold), expected)

    @pytest.mark.parametrize("execution_number", range(100))
    def test_macro_averaged_precision_fuzzy(self, execution_number):
        predictions = generate_predictions_list(10000, 1000, 1000)

        threshold = RANDOM_NUMBER_GENERATOR.uniform(0, 5)

        ratings_per_user = group_predictions_by_user(predictions)

        precisions = []
        for user_ratings in ratings_per_user.values():
            relevant_items = get_relevant_items(user_ratings, threshold=threshold)
            selected_items = get_selected_items(user_ratings, threshold=threshold)
            intersection = get_intersection(relevant_items, selected_items)

            try:
                precision = len(intersection) / len(selected_items)
            except ZeroDivisionError:
                precision = 0

            precisions.append(precision)

        assert np.isclose(
            get_macro_averaged_precision(predictions, threshold=threshold),
            statistics.mean(precisions),
        )

    def test_macro_averaged_precision_no_predictions(self):
        predictions = {}

        precision = get_macro_averaged_precision(predictions, threshold=1)

        assert np.isclose(precision, 0)


class TestPrecisionAtK:
    @pytest.mark.parametrize("execution_number", range(100))
    def test_precision_at_k_fuzzy(self, execution_number):
        predictions = generate_predictions_list(10000, 1000, 1000)

        threshold = RANDOM_NUMBER_GENERATOR.uniform(0, 5)
        k = int(RANDOM_NUMBER_GENERATOR.integers(1, 20))

        predictions_per_user = group_predictions_by_user(predictions)

        precisions = set()

        for user_predictions in predictions_per_user.values():
            relevant_items = get_relevant_items(user_predictions, threshold=threshold)
            selected_items = get_selected_items(user_predictions, threshold=threshold)

            if len(selected_items) < k:
                continue

            selected_items.sort(key=lambda prediction: prediction.est, reverse=True)

            top_k_ratings = selected_items[:k]

            assert len(top_k_ratings) == k

            intersection = get_intersection(relevant_items, top_k_ratings)
            try:
                precision = len(intersection) / len(top_k_ratings)
            except ZeroDivisionError:
                precision = 0

            precisions.add(precision)

        try:
            precision_at_k = statistics.mean(precisions)
        except statistics.StatisticsError:
            precision_at_k = 0

        assert np.isclose(
            get_precision_at_k(
                predictions,
                threshold=threshold,
                k=k,
            ),
            precision_at_k,
        )


class TestMicroAveragedRecall:
    @pytest.mark.parametrize("execution_number", range(100))
    def test_micro_averaged_recall_fuzzy(self, execution_number):
        predictions = generate_predictions_list(10000, 1000, 1000)

        threshold = RANDOM_NUMBER_GENERATOR.uniform(0, 5)

        relevant_items = get_relevant_items(predictions, threshold=threshold)
        selected_items = get_selected_items(predictions, threshold=threshold)
        intersection = get_intersection(relevant_items, selected_items)

        recall = len(intersection) / len(relevant_items)

        assert np.isclose(get_micro_averaged_recall(predictions, threshold=threshold), recall)

    def test_get_micro_averaged_recall_zero_division(self):
        predictions = []

        recall = get_micro_averaged_recall(predictions)

        assert np.isclose(recall, 0)


class TestMacroAveragedRecall:
    @pytest.mark.parametrize(
        "predictions, threshold, expected",
        [
            (
                [
                    Prediction(uid=1, iid=None, r_ui=1, est=1, details=None),
                    Prediction(uid=1, iid=None, r_ui=1, est=1, details=None),
                    Prediction(uid=1, iid=None, r_ui=1, est=1, details=None),
                    Prediction(uid=2, iid=None, r_ui=1, est=1, details=None),
                    Prediction(uid=2, iid=None, r_ui=1, est=1, details=None),
                    Prediction(uid=2, iid=None, r_ui=1, est=1, details=None),
                ],
                1,
                1.0,
            ),
            (
                [
                    Prediction(uid=1, iid=None, r_ui=1, est=1, details=None),
                    Prediction(uid=1, iid=None, r_ui=1, est=1, details=None),
                    Prediction(uid=1, iid=None, r_ui=1, est=1, details=None),
                ],
                1,
                1.0,
            ),
            (
                [
                    Prediction(uid=0, iid=None, r_ui=1, est=0, details=None),
                    Prediction(uid=1, iid=None, r_ui=1, est=1, details=None),
                    Prediction(uid=0, iid=None, r_ui=1, est=0, details=None),
                    Prediction(uid=1, iid=None, r_ui=1, est=1, details=None),
                ],
                1,
                0.5,
            ),
            (
                [
                    Prediction(uid=0, iid=0, r_ui=1, est=0, details=None),
                    Prediction(uid=0, iid=1, r_ui=1, est=1, details=None),
                    Prediction(uid=0, iid=2, r_ui=1, est=0, details=None),
                    Prediction(uid=0, iid=3, r_ui=1, est=1, details=None),
                ],
                1,
                0.5,
            ),
            (
                [
                    Prediction(uid=0, iid=0, r_ui=1, est=0, details=None),
                    Prediction(uid=0, iid=1, r_ui=1, est=1, details=None),
                    Prediction(uid=0, iid=2, r_ui=1, est=0, details=None),
                    Prediction(uid=1, iid=3, r_ui=1, est=1, details=None),
                ],
                1,
                2 / 3,
            ),
            (
                [
                    Prediction(uid=0, iid=0, r_ui=2, est=2, details=None),  # True Negative
                    Prediction(uid=0, iid=1, r_ui=1, est=4, details=None),  # False Positive
                    Prediction(uid=0, iid=2, r_ui=4, est=4, details=None),  # True Positive
                    Prediction(uid=1, iid=3, r_ui=2, est=5, details=None),  # False Positive
                    Prediction(uid=1, iid=4, r_ui=3, est=3, details=None),  # True Positive
                    Prediction(uid=2, iid=5, r_ui=1, est=2, details=None),  # True Negative
                ],
                3,
                2 / 3,
            ),
        ],
    )
    def test_mixed(self, predictions, threshold, expected):
        assert np.isclose(get_macro_averaged_recall(predictions, threshold=threshold), expected)

    @pytest.mark.parametrize("execution_number", range(100))
    def test_macro_averaged_recall_fuzzy(self, execution_number):
        predictions = generate_predictions_list(10000, 1000, 1000)

        threshold = RANDOM_NUMBER_GENERATOR.uniform(0, 5)

        ratings_per_user = group_predictions_by_user(predictions)

        recalls = []
        for user_ratings in ratings_per_user.values():
            relevant_items = get_relevant_items(user_ratings, threshold=threshold)
            selected_items = get_selected_items(user_ratings, threshold=threshold)
            intersection = get_intersection(relevant_items, selected_items)

            recall = 0
            try:
                recall = len(intersection) / len(relevant_items)
            except ZeroDivisionError:
                pass

            recalls.append(recall)

        assert np.isclose(
            get_macro_averaged_recall(predictions, threshold=threshold), statistics.mean(recalls)
        )


class TestRecallAtK:
    @pytest.mark.parametrize("execution_number", range(100))
    def test_recall_at_k_fuzzy(self, execution_number):
        predictions = generate_predictions_list(10000, 1000, 1000)

        threshold = RANDOM_NUMBER_GENERATOR.uniform(0, 5)
        k = int(RANDOM_NUMBER_GENERATOR.integers(1, 20))

        predictions_per_user = group_predictions_by_user(predictions)

        recalls = set()

        for user_predictions in predictions_per_user.values():
            relevant_items = get_relevant_items(user_predictions, threshold=threshold)
            selected_items = get_selected_items(user_predictions, threshold=threshold)

            if len(selected_items) < k:
                continue

            selected_items.sort(key=lambda prediction: prediction.est, reverse=True)

            top_k_ratings = selected_items[:k]

            assert len(top_k_ratings) == k

            intersection = get_intersection(relevant_items, top_k_ratings)
            try:
                recall = len(intersection) / len(relevant_items)
            except ZeroDivisionError:
                recall = 0

            recalls.add(recall)

        try:
            recall_at_k = statistics.mean(recalls)
        except statistics.StatisticsError:
            recall_at_k = 0

        assert np.isclose(
            get_recall_at_k(
                predictions,
                threshold=threshold,
                k=k,
            ),
            recall_at_k,
        )
