import time

from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

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

        average_minutes = average_time // 60
        average_seconds = average_time % 60

        time_left_minutes = estimated_time_left // 60
        time_left_seconds = estimated_time_left % 60

        print(
            f"Completed {i}/{total} | "
            f"Avg. time/task: {int(average_minutes)}m {average_seconds:.1f}s | "
            f"Time left: {int(time_left_minutes)}m {time_left_seconds:.1f}s"
        )

    print("All tasks completed.")
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60
    print(f"Total time: {hours}h {minutes}m {seconds:.1f}s")


def determine_worker_split(tasks: List, max_workers: int):
    """
    Determine the number of parent and child workers for parallelization.

    This function is used to determine how to split the workers when parallelizing
    tasks in a nested fashion. The idea is to have a number of parent workers that
    is less than or equal to the number of tasks, and then split the remaining
    workers among the tasks.

    Args:
    tasks: List
        The tasks to parallelize.
    max_workers: int
        The maximum number of workers to use.

    Returns:
    Tuple[int, List[int]]
        The number of parent workers and a list of child workers.
    """
    if len(tasks) < max_workers:
        parent_workers = len(tasks)
        whole_child_workers = max_workers // len(tasks)
        workers_left = max_workers % len(tasks)

        child_workers = [whole_child_workers] * len(tasks)

        for i in range(workers_left):
            child_workers[i] += 1
    else:
        parent_workers = max_workers
        child_workers = [1] * len(tasks)
    return parent_workers, child_workers


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


def cross_validade_recommenders(
    recommenders: List[AlgoBase],
    folds: List[Tuple[Trainset, List[Tuple[str, str, float]]]],
    test_measures: List[TestMeasureStrategy] = [],
    train_measures: List[TrainMeasureStrategy] = [],
    max_workers=1,
    verbose=True,
):
    """
    Cross validate a list of recommender systems.

    Args:
    recommenders: List[AlgoBase]
        The recommender systems to cross validate.
    folds: List[Tuple[Trainset, List[Tuple[str, str, float]]]]
        The train-test splits.
    test_measures: List[TestMeasureStrategy]
        The test measures to compute.
    train_measures: List[TrainMeasureStrategy]
        The train measures to compute.
    max_workers: int
        The maximum number of workers to use.
    verbose: bool
        Whether to print progress.

    Returns:
    Dict[AlgoBase, dict]
        A dictionary mapping each recommender to its measurements.
    """

    parent_workers, child_workers = determine_worker_split(recommenders, max_workers)

    if verbose:
        print("Parent workers: ", parent_workers)
        print("Child workers: ", child_workers)

    with ProcessPoolExecutor(max_workers=parent_workers) as executor:

        futures = [
            executor.submit(
                cross_validate,
                recommender,
                folds,
                test_measures,
                train_measures,
                child_workers[id],
                False,
            )
            for id, recommender in enumerate(recommenders)
        ]

        if verbose:
            print_progress(futures)

        output = [future.result() for future in futures]

    measurements = {}

    for recommender, recommender_measurements in zip(recommenders, output):
        measurements[recommender] = recommender_measurements

    return measurements
