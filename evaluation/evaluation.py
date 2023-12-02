""" evaluation.py

This module contains functions to evaluate the performance of a recommender system.

Copyright 2023 Bernardo C. Rodrigues
See LICENSE file for license details
"""
import statistics
from typing import List
from collections import defaultdict

from collections import namedtuple
from surprise.prediction_algorithms import Prediction

ContingencyTable = namedtuple(
    "ContingencyTable", ["true_positives", "false_positives", "true_negatives", "false_negatives"]
)


def generate_contingency_table(
    predictions: List[Prediction], threshold: float = 1
) -> ContingencyTable:
    """
    Returns a contingency table from a list of predictions.

    A contingency table is a table that shows the distribution of the predicted ratings and the
    true ratings. It is used to calculate the precision and recall metrics.

    The contingency table is defined as:

    true_positive = number of relevant items that were retrieved
    false_positive = number of irrelevant items that were retrieved
    true_negative = number of irrelevant items that were not retrieved
    false_negative = number of relevant items that were not retrieved

    In recommender systems, the relevant items are the items that the user liked and the retrieved
    items are the items that were recommended by the system.

    Args:
        predictions (List[Prediction]): A list of predictions.
        threshold (float, optional): The threshold used to determine if an item is relevant and
                                     should be retrieved. Defaults to 1.

    Returns:
        ContingencyTable: A contingency table.
    """
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for _, _, true_rating, estimate, _ in predictions:
        if estimate >= threshold:
            if true_rating >= threshold:
                true_positives += 1
            else:
                false_positives += 1
        else:
            if true_rating >= threshold:
                false_negatives += 1
            else:
                true_negatives += 1

    return ContingencyTable(
        true_positives=true_positives,
        false_positives=false_positives,
        true_negatives=true_negatives,
        false_negatives=false_negatives,
    )


def get_micro_averaged_precision(predictions: List[Prediction], threshold: float = 1):
    """
    Returns the micro-averaged, or global, precision from a predictions list.

    Precision is defined as the fraction of retrieved (recommended) instances that are relevant
    (liked by the user). It is the conditional probability that a recommended item is relevant,
    given that it was recommended - P(relevant|retrieved).

    Micro-averaged precision calculates the precision globally, by considering all the items
    recommended to all users.

        micro_averaged_precision = true_positives / (true_positives + false_positives)

    Where true_positives is the number of relevant items that were retrieved, false_positives is
    the number of irrelevant items that were retrieved and (true_positives + false_positives) is
    the total number of retrieved items.

    Args:
        predictions (List[Prediction]): A list of predictions.
        threshold (float, optional): The threshold used to determine if an item is relevant and
                                     should be retrieved. Defaults to 1.

    Returns:
        float: The micro-averaged precision.
    """

    contingency_table = generate_contingency_table(predictions, threshold)

    try:
        return contingency_table.true_positives / (
            contingency_table.true_positives + contingency_table.false_positives
        )
    except ZeroDivisionError:
        return 0


def get_macro_averaged_precision(predictions: List[Prediction], threshold: float = 1):
    """
     Returns the macro-averaged, or the user-averaged, precision for a list of predictions.

    Precision is defined as the fraction of retrieved (recommended) instances that are relevant
    (liked by the user). It is the conditional probability that a recommended item is relevant,
    given that it was recommended - P(relevant|retrieved).

    Macro-averaged precision calculates the precision for each user and then averages the results.

    The precision for the k-th user (Precision[k]) is defined as:

        precision[k] = true_positives[k] / (true_positives[k] + false_positives[k])

    Where precision[k] is the precision for the k-th user, true_positives[k] is the number of
    relevant items that were retrieved for the k-th user, and false_positives[k] is the number of
    irrelevant items that were retrieved for the k-th user and n is the number of users.

    Macro-averaged precision (macro_averaged_precision) is defined as:

        macro_averaged_precision = (precision[1] + precision[2] + ... + precision[n]) / n

    Where n is the number of users.

    Args:
        predictions (List[Prediction]): A list of predictions.
        threshold (float, optional): The threshold used to determine if an item is
                                               relevant and should be retrieved. Defaults to 1.

    Returns:
        float: The macro-averaged precision.

    """

    predictions_per_user = defaultdict(list)
    for prediction in predictions:
        predictions_per_user[prediction.uid].append(prediction)

    precisions = []
    for user_ratings in predictions_per_user.values():
        precisions.append(get_micro_averaged_precision(user_ratings, threshold))

    try:
        return statistics.mean(precisions)
    except statistics.StatisticsError:
        return 0


def get_precision_at_k(predictions: List[Prediction], threshold: float = 1, k: int = 20):
    """
    Calculate the precision at K (Precision@K) for a list of predictions.

    Precision@K is the macro-averaged precision for the top K recommendations. In other words,
    it is the precision calculated for the top K recommendations for each user, and then averaged.If
    there are not enough selected items to reach K, the user is skipped.

    The precision for the i-th user in the top K recommendations (precision_at_k[i]) is defined as:

        precision_at_k[i] = true_positives_at_k[i] / (true_positives_at_k[i] +
                                                                    false_positives_at_k[i])

    Where True true_positives_at_k[i] is the number of relevant items that were retrieved for the
    i-th user in the top K recommendations, and false_positives_at_k[i] is the number of irrelevant
    items that were retrieved for the i-th user in the top K recommendations. Since we are limiting
    the number of recommendations to K, the value of (true_positives_at_k[i] +
    false_positives_at_k[i]) is always equal to K.

    Therefore, the formula can be simplified to:

        precision_at_k[i] = true_positives_at_k[i] / k

    Finally, the macro-averaged precision at K (precision_at_k) is defined as:

        precision_at_k = (precision_at_k[1] + precision_at_k[2] + ... + precision_at_k[n]) / n

    Where n is the number of users.

    Args:
        predictions (List[Prediction]): A list of predictions.
        threshold (float, optional): The threshold used to determine if an item is relevant and
                                        should be retrieved. Defaults to 1.
        k (int, optional): The number of recommendations to consider. Defaults to 20.

    Returns:
        float: The precision at K.
    """

    predictions_per_user = defaultdict(list)
    for prediction in predictions:
        predictions_per_user[prediction.uid].append(prediction)

    precisions = set()
    for user_predictions in predictions_per_user.values():
        if len(user_predictions) < k:
            # skip this user if there are not enough recommendations to reach k
            continue

        user_predictions.sort(key=lambda prediction: prediction.est, reverse=True)

        if user_predictions[k - 1].est < threshold:
            # skip this user if the k-th item is not relevant
            continue

        top_k_ratings = user_predictions[:k]

        precisions.add(get_micro_averaged_precision(top_k_ratings, threshold))

    try:
        return statistics.mean(precisions)
    except statistics.StatisticsError:
        return 0


def get_micro_averaged_recall(predictions: List[Prediction], threshold: float = 1):
    """
    Returns the micro-averaged, or global, recall from a predictions list.

    Recall is defined as the fraction of relevant (liked by the user) instances that are retrieved
    (recommended). It is the conditional probability that a relevant item is retrieved, given that
    it was relevant - P(retrieved|relevant).

    Micro-averaged recall calculates the recall globally, by considering all the items recommended
    to all users.

        micro_averaged_recall = true_positives / (true_positives + false_negatives)

    Where true_positives is the number of relevant items that were retrieved, false_negatives is
    the number of relevant items that were not retrieved and (true_positives + false_negatives) is
    the total number of relevant items.

    Args:
        predictions (List[Prediction]): A list of predictions.
        threshold (float, optional): The threshold used to determine if an item is relevant and
                                     should be retrieved. Defaults to 1.

    Returns:
        float: The micro-averaged recall.
    """

    contingency_table = generate_contingency_table(predictions, threshold)

    try:
        return contingency_table.true_positives / (
            contingency_table.true_positives + contingency_table.false_negatives
        )
    except ZeroDivisionError:
        return 0


def get_macro_averaged_recall(predictions, threshold=1):
    """
    Returns the macro-averaged, or the user-averaged, recall for a list of predictions.

    Recall is defined as the fraction of relevant (liked by the user) instances that are retrieved
    (recommended). It is the conditional probability that a relevant item is retrieved, given that
    it was relevant - P(retrieved|relevant).

    Macro-averaged recall calculates the recall for each user and then averages the results.

    The recall for the k-th user (recall[k]) is defined as:

        recall[k] = true_positives[k] / (true_positives[k] + false_negatives[k])

    Where true_positives[k] is the number of relevant items that were retrieved for the k-th user,
    and false_negatives[k] is the number of relevant items that were not retrieved for the k-th user
    and n is the number of users.

    Macro-averaged recall (macro_averaged_recall) is defined as:

        macro_averaged_recall = (recall[1] + recall[2] + ... + recall[n]) / n

    Where n is the number of users.

    Args:
        predictions (List[Prediction]): A list of predictions.
        threshold (float, optional): The threshold used to determine if an item is relevant and
                                        should be retrieved. Defaults to 1.
    Returns:
        float: The macro-averaged recall.

    """

    predictions_per_user = defaultdict(list)
    for prediction in predictions:
        predictions_per_user[prediction.uid].append(prediction)

    recalls = []
    for user_ratings in predictions_per_user.values():
        recalls.append(get_micro_averaged_recall(user_ratings, threshold))

    return statistics.mean(recalls)


def get_recall_at_k(predictions, threshold=1, k=20):
    """
    Calculate the recall at K (Recall@K) for a list of predictions.

    Recall@K is the macro-averaged recall for the top K recommendations. In other words,
    it is the recall calculated for the top K recommendations for each user, and then averaged.If
    there are not enough selected items to reach K, the user is skipped.

    The recall for the i-th user in the top K recommendations (recall_at_k[i]) is defined as:

        recall_at_k[i] = true_positives_at_k[i] / true_positives + false_negatives

    Where True true_positives_at_k[i] is the number of relevant items that were retrieved for the
    i-th user in the top K recommendations, and false_negatives is the number of relevant items that
    were not retrieved for the i-th user. Note that the value of (true_positives + false_negatives)
    is the subset of items that the i-th user found relevant. Therefore, is is computed over the
    entire predictions list, not just the top K recommendations.

    Finally, the macro-averaged recall at K (recall_at_k) is defined as:

        recall_at_k = (recall_at_k[1] + recall_at_k[2] + ... + recall_at_k[n]) / n

    Where n is the number of users.

    Args:
        predictions (List[Prediction]): A list of predictions.
        threshold (float, optional): The threshold used to determine if an item is relevant and
                                        should be retrieved. Defaults to 1.
        k (int, optional): The number of recommendations to consider. Defaults to 20.

    Returns:
        float: The recall at K.
    """

    predictions_per_user = defaultdict(list)
    for prediction in predictions:
        predictions_per_user[prediction.uid].append(prediction)

    recalls = set()
    for user_predictions in predictions_per_user.values():
        if len(user_predictions) < k:
            # skip this user if there are not enough recommendations to reach k
            continue

        user_predictions.sort(key=lambda prediction: prediction.est, reverse=True)

        if user_predictions[k - 1].est < threshold:
            # skip this user if the k-th item is not relevant
            continue

        top_k_ratings = user_predictions[:k]

        predictions_contingency_table = generate_contingency_table(user_predictions, threshold)
        top_k_contingency_table = generate_contingency_table(top_k_ratings, threshold)

        try:
            recall = top_k_contingency_table.true_positives / (
                predictions_contingency_table.true_positives
                + predictions_contingency_table.false_negatives
            )
        except ZeroDivisionError:
            recall = 0

        recalls.add(recall)

    try:
        return statistics.mean(recalls)
    except statistics.StatisticsError:
        return 0


def count_impossible_predictions(predictions: List[Prediction]) -> int:
    """
    Returns the number of predictions that were impossible to generate.

    Args:
        predictions (List[Prediction]): A list of predictions.

    Returns:
        int: The number of impossible predictions.
    """
    count = 0
    for prediction in predictions:
        if prediction.details["was_impossible"]:
            count += 1
    return count
