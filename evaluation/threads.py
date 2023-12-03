import time
import statistics
from collections import namedtuple
from typing import Tuple, List
from surprise import Trainset, AlgoBase
from surprise.accuracy import mae, rmse

from evaluation import (
    get_micro_averaged_recall,
    get_macro_averaged_recall,
    get_recall_at_k,
    get_micro_averaged_precision,
    get_macro_averaged_precision,
    get_precision_at_k,
    count_impossible_predictions,
)

from recommenders import grecond_recommender


BenchmarkResult = namedtuple("BenchmarkResult", "fold_index variation metrics")
"""
A named tuple representing the result of a benchmark.

Attributes:
    fold_index (int): The index of the fold.
    variation (str): The name of the recommender variation.
    metrics (dict): The raw results of the benchmark.
"""

RecommenderVariation = namedtuple("RecommenderVariation", "variation recommender")
"""
A named tuple representing a recommender variation. The variation is a particular configuration of
a recommender. For example, the RecommenderVariation for a UBCF recommender with cosine similarity
and k=10 could be RecommenderVariation("UBCF_cosine_10", UBCF(k=10,sim_options={"name": 
"cosine"})). The variation is used to sort the results of the benchmark afterwards. 

Attributes:
    variation (str): The name of the recommender variation.
    recommender (AlgoBase): The recommender object.
"""

GENERIC_METRIC_NAMES = [
    "mae",
    "rmse",
    "micro_averaged_recall",
    "macro_averaged_recall",
    "recall_at_k",
    "micro_averaged_precision",
    "macro_averaged_precision",
    "precision_at_k",
    "fit_time",
    "test_time",
    "impossible_predictions",
]

GRECOND_METRIC_NAMES = [
    "mean_pattern_size",
    "actual_coverage",
]


def generic_benchmark_thread(
    fold: Tuple[int, Tuple[Trainset, List[Tuple[int, int, float]]]],
    recommender: Tuple[str, AlgoBase],
    threshold: float = 5.0,
    number_of_top_recommendations: int = 20,
):
    """
    Benchmarks a recommender system and returns the raw results. Even though you can call it
    directly, this function is expected to be used in a multiprocessing test bench
    (e.g. multiprocessing.Pool.starmap).

    Args:
        fold (Tuple[int, Tuple[Trainset, List[Tuple[int, int, float]]]]): The fold to be processed.
            It is a tuple of the fold index and the trainset and testset to be used.
        recommender (Tuple[str, AlgoBase]): The recommender to be evaluated. It is a tuple of the
            recommender name and the recommender object. The recommender must implement Surprise's
            AlgoBase API (fit and test methods).
        threshold (float): The threshold that determines whether a rating is relevant or not. This
            is used for calculating metrics that assume a binary prediction (e.g. recall).
        number_of_top_recommendations (int): The number of top recommendations to be considered
            when calculating metrics that assume a top-k list of recommendations (e.g. precision@k).

    Returns:
        Tuple[int, str, dict]: The results of the benchmark. It is a tuple of the fold index, the
            recommender name and the raw results.
    """
    fold_index, (trainset, testset) = fold
    recommender_name, recommender_object = recommender

    start_time = time.time()
    recommender_object.fit(trainset)
    elapsed_fit_time = time.time() - start_time

    start_time = time.time()
    predictions = recommender_object.test(testset)
    elapsed_test_time = time.time() - start_time

    output = {
        "mae": mae(predictions=predictions, verbose=False),
        "rmse": rmse(predictions=predictions, verbose=False),
        "micro_averaged_recall": get_micro_averaged_recall(
            predictions=predictions, threshold=threshold
        ),
        "macro_averaged_recall": get_macro_averaged_recall(
            predictions=predictions, threshold=threshold
        ),
        "recall_at_k": get_recall_at_k(
            predictions=predictions,
            threshold=threshold,
            k=number_of_top_recommendations,
        ),
        "micro_averaged_precision": get_micro_averaged_precision(
            predictions=predictions, threshold=threshold
        ),
        "macro_averaged_precision": get_macro_averaged_precision(
            predictions=predictions, threshold=threshold
        ),
        "precision_at_k": get_precision_at_k(
            predictions=predictions,
            threshold=threshold,
            k=number_of_top_recommendations,
        ),
        "impossible_predictions": count_impossible_predictions(predictions=predictions),
        "fit_time": elapsed_fit_time,
        "test_time": elapsed_test_time,
    }

    return fold_index, recommender_name, output


def GreConDKNNRecommender2_benchmark_thread(
    fold: Tuple[int, Tuple[Trainset, List[Tuple[int, int, float]]]],
    recommender: Tuple[str, grecond_recommender.GreConDKNNRecommender2],
    threshold: float = 5.0,
    number_of_top_recommendations: int = 20,
):
    """
    Benchmarks a GreConDKNNRecommender2 recommender system and returns the raw results. It builds
    upon the generic_benchmark_thread function. It collects additional metrics that are specific
    to the GreConDKNNRecommender2 recommender.

    Args:
        fold (Tuple[int, Tuple[Trainset, List[Tuple[int, int, float]]]]): The fold to be processed.
            It is a tuple of the fold index and the trainset and testset to be used.
        recommender (Tuple[str, grecond_recommender.GreConDKNNRecommender2]): The recommender to be
            evaluated. It is a tuple of the recommender name and the recommender object.
        threshold (float): The threshold that determines whether a rating is relevant or not. This
            is used for calculating metrics that assume a binary prediction (e.g. recall).
        number_of_top_recommendations (int): The number of top recommendations to be considered
            when calculating metrics that assume a top-k list of recommendations (e.g. precision@k).

    Returns:
        Tuple[int, str, dict]: The results of the benchmark. It is a tuple of the fold index, the
            recommender name and the raw results.
    """

    _, recommender_object = recommender

    assert isinstance(recommender_object, grecond_recommender.GreConDKNNRecommender2)

    fold_index, recommender_name, output = generic_benchmark_thread(
        fold=fold,
        recommender=recommender,
        threshold=threshold,
        number_of_top_recommendations=number_of_top_recommendations,
    )

    mean_pattern_size = statistics.mean([pattern.size for pattern in recommender_object.patterns])

    output["mean_pattern_size"] = mean_pattern_size
    output["actual_coverage"] = recommender_object.actual_coverage

    return fold_index, recommender_name, output
