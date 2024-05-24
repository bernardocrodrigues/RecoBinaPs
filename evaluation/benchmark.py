import time

from typing import List, Tuple

import numpy as np
from surprise import AlgoBase, Trainset
from pydantic import validate_call, ConfigDict

from evaluation.strategies import TestMeasureStrategy, TrainMeasureStrategy


def print_progress(tasks: List) -> None:
    """
    Print progress of a list of tasks.
    """

    total = len(tasks)
    start_time = time.time()

    # enumeration starts at 1 to avoid division by zero
    for i, _ in enumerate(as_completed(tasks), 1):

        now = time.time()
        elapsed_time = now - start_time

        average_time = elapsed_time / i
        estimated_time_left = average_time * (total - i)

        print(
            f"Completed {i}/{total} | Avg. time/task: {average_time/60:.2f} min | Est. time left: {estimated_time_left/60:.2f} min"
        )


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


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True, validate_return=True))
def cross_validate(
    recommender_system: AlgoBase,
    folds: List[Tuple[Trainset, List[Tuple[str, str, float]]]],
    test_measures: List[TestMeasureStrategy] = [],
    train_measures: List[TrainMeasureStrategy] = [],
    max_workers=1,
    verbose=True,
):
    """
    Cross validate a recommender system.

    This function is based on Surprise's cross_validate
    (surprise.model_selection.validation.cross_validate) however it is somewhat
    simplified and adapted to:
    - work with our custom measures (modeled as strategies)
    - process precomputed trainset and testset
    - uses futures instead of parallel for parallelization
    """

    with ProcessPoolExecutor(max_workers=max_workers) as executor:

        futures = [
            executor.submit(
                fit_and_score,
                recommender_system,
                trainset,
                testset,
                test_measures,
                train_measures,
            )
            for trainset, testset in folds
        ]

        if verbose:
            print_progress(futures)

        out = [future.result() for future in futures]

    (test_measures_dicts, train_measures_dicts, fit_times, test_times) = zip(*out)

    measurements = {}

    for train_measure in train_measures:
        train_measure_name = train_measure.get_name()
        measurements[train_measure_name] = np.asarray(
            [d[train_measure_name] for d in train_measures_dicts]
        )

    for test_measure in test_measures:
        test_measure_name = test_measure.get_name()
        measurements[test_measure_name] = np.asarray(
            [d[test_measure_name] for d in test_measures_dicts]
        )

    measurements["fit_time"] = list(fit_times)
    measurements["test_time"] = list(test_times)

    return measurements
