import time

from typing import List, Tuple

from surprise import AlgoBase, Trainset
from pydantic import validate_call, ConfigDict

from evaluation.strategies import TestMeasureStrategy, TrainMeasureStrategy


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True, validate_return=True))
def fit_and_score(
    recommender_system: AlgoBase,
    trainset: Trainset,
    testset: List[Tuple[str, str, float]],
    test_measures: List[TestMeasureStrategy] = [],
    train_measures: List[TrainMeasureStrategy] = [],
) -> Tuple[dict, dict, float, float]:
    """
    Fit a recommender system on a trainset and evaluate it on a testset.

    This function is based on Surprise's fit_and_score 
    (surprise.model_selection.validation.fit_and_score) however it is somewhat
    simplified and adapted to:
    - work with our custom measures (modeled as strategies)
    - process precomputed trainset and testset

    Args:
    recommender_system: AlgoBase
        The recommender system to fit and evaluate.
    trainset: Trainset
        The training dataset.
    testset: List[Tuple[str, str, float]]
        The test dataset.
    test_measures: List[TestMeasureStrategy]
        The test measures to compute. A list
    train_measures: List[TrainMeasureStrategy]
        The train measures to compute.

    Returns:
    Tuple[dict, dict, float, float]
        The test measurements, train measurements, fit time and test time.

    """
    start_fit = time.time()
    recommender_system.fit(trainset)
    fit_time = time.time() - start_fit

    train_measurements = {}
    for measure in train_measures:
        measure_name = measure.get_name()
        train_measurements[measure_name] = measure.calculate(recommender_system)

    start_test = time.time()
    predictions = recommender_system.test(testset)
    test_time = time.time() - start_test

    test_measurements = {}

    for measure in test_measures:
        measure_name = measure.get_name()
        test_measurements[measure_name] = measure.calculate(predictions)

    return test_measurements, train_measurements, fit_time, test_time


