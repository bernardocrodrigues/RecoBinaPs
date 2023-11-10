import time
from typing import Tuple, List, Dict
from surprise import Trainset, AlgoBase
from surprise.accuracy import mae, rmse

from evaluation import (
    get_micro_averaged_recall,
    get_macro_averaged_recall,
    get_recall_at_k,
    get_micro_averaged_precision,
    get_macro_averaged_precision,
    get_precision_at_k,
)


def generic_thread(
    fold: Tuple[int, Tuple[Trainset, List[Tuple[int, int, float]]]],
    recommender: Tuple[str, AlgoBase],
    threshold: float = 5.0,
    number_of_top_recommendations: int = 20,
):
    """
    This function is used to parallelize the GreConD recommender. It puts the results on a
    dictionary called 'output'. 'output' is expected to be a Manager().dict() object since it is
    shared between processes.

    Args:
        fold (Tuple[int, Tuple[Trainset, List[Tuple[int, int, float]]]]): The fold to be processed.
        output (Dict): The dictionary to put the results on.
        recommender (Tuple[str, AlgoBase]): The recommender to use. It is a tuple of the name of
            the recommender and the recommender itself.
        threshold (float): The relevance threshold to use.
        number_of_top_recommendations (int): The number of top recommendations to use.

    Returns:
        None
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
        "fit_time": elapsed_fit_time,
        "test_time": elapsed_test_time,
    }

    return fold_index, recommender_name, output
