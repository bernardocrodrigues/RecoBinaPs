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


def get_micro_averaged_precision(predictions: List[Prediction], relevance_threshold: float = 1):
    """
    Returns the micro-averaged, or global, precision from a predictions list.

    Precision is defined as the fraction of retrieved (recommended) instances that are relevant
    (liked by the user). It is the conditional probability that a recommended item is relevant,
    given that it was recommended - P(relevant|retrieved).

    Micro-averaged precision calculates the precision globally, by considering all the items
    recommended to all users.

    Precision = True Positives / (True Positives + False Positives)

    Args:
        predictions (List[Prediction]): A list of predictions.
        relevance_threshold (float, optional): The threshold used to determine if an item is
                                                  relevant and should be retrieved. Defaults to 1.

    Returns:
        float: The micro-averaged precision.
    """

    contingency_table = generate_contingency_table(predictions, relevance_threshold)

    try:
        return contingency_table.true_positives / (
            contingency_table.true_positives + contingency_table.false_positives
        )
    except ZeroDivisionError:
        return 0


def get_macro_averaged_precision(predictions: List[Prediction], relevance_threshold: float = 1):
    """
     Returns the macro-averaged, or the user-averaged, precision for a list of predictions.

    Precision is defined as the fraction of retrieved (recommended) instances that are relevant
    (liked by the user). It is the conditional probability that a recommended item is relevant,
    given that it was recommended - P(relevant|retrieved).

    Macro-averaged precision calculates the precision for each user and then averages the results.

    Precision[k] = True Positives[k] / (True Positives[k] + False Positives[k])
    Macro-averaged Precision = (Precision[1] + Precision[2] + ... + Precision[n]) / n

    Where Precision[k] is the precision for the k-th user, True Positives[k] is the number of
    relevant items that were retrieved for the k-th user, and False Positives[k] is the number of
    irrelevant items that were retrieved for the k-th user and n is the number of users.

    Args:
        predictions (List[Prediction]): A list of predictions.
        relevance_threshold (float, optional): The threshold used to determine if an item is
                                               relevant and should be retrieved. Defaults to 1.
    """

    ratings_per_user = defaultdict(list)
    for prediction in predictions:
        ratings_per_user[prediction.uid].append(prediction)

    precisions = []
    for user_ratings in ratings_per_user.values():
        precisions.append(get_micro_averaged_precision(user_ratings, relevance_threshold))

    try:
        return statistics.mean(precisions)
    except statistics.StatisticsError:
        return 0


def get_precision_at_k(predictions: List[Prediction], relevance_threshold: float = 1, k: int = 20):
    """
    Calculate the precision at K (Precision@K) for a list of predictions.

    Precision@K is the macro-averaged precision for the top K recommendations. In other words,
    it is the precision calculated for the top K recommendations for each user, and then averaged.

    If there are not enough recommendations to reach K, the user is skipped.

    Precision@K_i = True Positives[i] / (True Positives[i] + False Positives[i])

    """

    uid_index = 0

    ratings_per_user = defaultdict(list)
    for prediction in predictions:
        ratings_per_user[prediction[uid_index]].append(prediction)

    precisions = []
    for user_ratings in ratings_per_user.values():

        user_ratings.sort(key=lambda prediction: prediction.est, reverse=True)

        top_k_ratings = user_ratings[:k]

        if len(top_k_ratings) < k:
            # skip this user if there are not enough recommendations to reach k
            continue

        precisions.append(get_micro_averaged_precision(top_k_ratings, relevance_threshold))

    return statistics.mean(precisions)


def get_global_recall(predictions, relevance_threshold=1):
    """
    Returns the recall from a predictions list.

    Recall is defined as the fraction of relevant instances that were retrieved. In recommender systems,
    it measures the fraction of items that are liked by the user among all the items that are available.

    Recall = True Positives / (True Positives + False Negatives)

    For example, if you have a recommender system that suggests movies to a user, and you have 100 movies in total,
    10 of which are liked by the user. If your system misses 6 movies that are liked by the user (false negatives), then
    your recall is 4 / (4 + 6) = 0.4. This means that 40% of the movies that are actually liked by the user were
    recommended by your system.

    """

    def is_relevant(measure):
        return measure >= relevance_threshold

    true_positives = 0
    false_negatives = 0

    for _, _, true_rating, estimate, _ in predictions:
        if is_relevant(estimate):
            if is_relevant(true_rating):
                true_positives += 1
        else:
            if is_relevant(true_rating):
                false_negatives += 1

    return true_positives / (true_positives + false_negatives)


def get_user_averaged_recall(predictions, relevance_threshold=1):
    def is_relevant(measure):
        return measure >= relevance_threshold

    recalls = []
    ratings_per_user = defaultdict(list)
    for user_id, _, true_rating, estimate, _ in predictions:
        ratings_per_user[user_id].append((estimate, true_rating))

    for _, user_ratings in ratings_per_user.items():
        true_positives = 0
        false_negatives = 0

        for estimate, true_rating in user_ratings:
            if is_relevant(estimate):
                if is_relevant(true_rating):
                    true_positives += 1
            else:
                if is_relevant(true_rating):
                    false_negatives += 1
        try:
            recall = true_positives / (true_positives + false_negatives)
        except ZeroDivisionError:
            pass
        else:
            recalls.append(recall)

    return statistics.mean(recalls)


def get_recall_at_k(predictions, relevance_threshold=1, k=20):
    def is_relevant(measure):
        return measure >= relevance_threshold

    recalls = []
    ratings_per_user = defaultdict(list)
    for user_id, _, true_rating, estimate, _ in predictions:
        ratings_per_user[user_id].append((estimate, true_rating))

    for _, user_ratings in ratings_per_user.items():
        relevant_itens_in_the_top_k = 0
        total_relevant_itens = 0

        user_ratings.sort(key=lambda x: x[0], reverse=True)

        for estimate, true_rating in user_ratings[:k]:
            if is_relevant(true_rating):
                relevant_itens_in_the_top_k += 1

        for estimate, true_rating in user_ratings:
            if is_relevant(true_rating):
                total_relevant_itens += 1
        try:
            recalls.append(relevant_itens_in_the_top_k / total_relevant_itens)
        except ZeroDivisionError:
            pass

    return statistics.mean(recalls)
