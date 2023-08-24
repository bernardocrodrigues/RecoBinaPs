#!/usr/bin/env python3
""" generate_binaps_result.py

This script runs the experiments for the Binaps model and stores the results in a SQLite database.
The results are stored in the database after each experiment, so if the script is interrupted, 
the results of the experiments that have already been run will be stored in the database.
The script can be run multiple times to run the experiments that have not been run yet.

The experiments are stored in the binaps_experiments and for each experiment, 
the following information is stored as a row in the table:
- Dataset path (DATASET_PATH)
- Train set size (TRAIN_SET_SIZE)
- Batch size (BATCH_SIZE)
- Test batch size (TEST_BATCH_SIZE)
- Number of epochs (EPOCHS)
- Learning rate (LEARNING_RATE)
- Weight decay (WEIGHT_DECAY)
- Gamma (GAMMA)
- Seed (SEED)
- Hidden dimension (HIDDEN_DIMENSION)
- Weights (WEIGHTS)
- Training losses (TRAINING_LOSSES)
- Test losses (TEST_LOSSES)
- Runtime (RUNTIME)

The weights, training losses and test losses are stored as compressed pickled objects. The
weights are stored as a list of numpy arrays, the training losses and test losses are stored
as lists of floats. The runtime is stored as a float. The other parameters are stored as they
are.

Usage:
    The preferred way to run this script withing the accompanying project container. To do so,
    run the following command from the project environment directory:

    docker-compose run --rm --entrypoint="scripts/generate_binaps_result.py" notebook-cuda

Copyright 2023 Bernardo C. Rodrigues
See COPYING file for license details

"""

import itertools
import time
import sqlite3
import pickle
import lzma
from rich.progress import track

from binaps.binaps_wrapper import run_binaps


# Experiment parameters.

# Alter these to change the experiments. The parameters are
# combined through a Cartesian product, so all combinations will be tested. For example,
# if DATASET = ["ml-100k", "ml-1m"] and BATCH_SIZE = [64, 128], then the experiments
# will be run for ml-100k with batch size 64, ml-100k with batch size 128, ml-1m with
# batch size 64 and ml-1m with batch size 128.

##### Change from this line on!

# Path to the SQLite database where the results will be stored. Use either absolute or relative
# paths to the project root directory.
DATABASE_PATH = "experiments.db"

# List of dataset paths to be used in the experiments. Use either absolute or relative paths
# to the project root directory.
DATASET_PATH = [
    "datasets/ml-100k/u0.base.dat",
    "datasets/ml-100k/u1.base.dat",
    "datasets/ml-100k/u2.base.dat",
    "datasets/ml-100k/u3.base.dat",
    "datasets/ml-100k/u4.base.dat",
]

# Binaps parameters. See binaps/binaps_wrapper.py for more information on these parameters.
TRAIN_SET_SIZE = [0.9]
BATCH_SIZE = [64]
TEST_BATCH_SIZE = [64]
EPOCHS = [100]
LEARNING_RATE = [0.01]
WEIGHT_DECAY = [0]
GAMMA = [0.1]
SEED = [1]
HIDDEN_DIMENSION = [-1]

##### Attention! Do not change bellow this line!

DATABASE = sqlite3.connect(DATABASE_PATH)

EXPERIMENT_PARAMETERS = list(
    itertools.product(
        DATASET_PATH,
        TRAIN_SET_SIZE,
        BATCH_SIZE,
        TEST_BATCH_SIZE,
        EPOCHS,
        LEARNING_RATE,
        WEIGHT_DECAY,
        GAMMA,
        SEED,
        HIDDEN_DIMENSION,
    )
)


def commit_result_to_database(
    dataset,
    train_set_size,
    batch_size,
    test_batch_size,
    epochs,
    learning_rate,
    weight_decay,
    gamma,
    seed,
    hidden_dimension,
    weights,
    training_losses,
    test_losses,
    runtime,
):
    """
    Commits the results of the experiment to the database.

    Parameters


    """
    cursor = DATABASE.cursor()

    serialized_weights = pickle.dumps(weights)
    serialized_training_losses = pickle.dumps(training_losses)
    serialized_test_losses = pickle.dumps(test_losses)

    compressed_weights = lzma.compress(serialized_weights)
    compressed_training_losses = lzma.compress(serialized_training_losses)
    compressed_test_losses = lzma.compress(serialized_test_losses)

    cursor.execute(
        (
            "INSERT INTO binaps_experiments (DATASET, TRAIN_SET_SIZE, BATCH_SIZE, TEST_BATCH_SIZE, "
            "EPOCHS, LEARNING_RATE, WEIGHT_DECAY, GAMMA, SEED, HIDDEN_DIMENSION, WEIGHTS, "
            "TRAINING_LOSSES, TEST_LOSSES, RUNTIME) VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        ),
        (
            dataset,
            train_set_size,
            batch_size,
            test_batch_size,
            epochs,
            learning_rate,
            weight_decay,
            gamma,
            seed,
            hidden_dimension,
            compressed_weights,
            compressed_training_losses,
            compressed_test_losses,
            runtime,
        ),
    )

    DATABASE.commit()


def prepare_database():
    """
    Creates the database tables if they do not exist.
    """
    cursor = DATABASE.cursor()

    cursor.execute(
        "CREATE TABLE IF NOT EXISTS binaps_experiments "
        "(ID INTEGER PRIMARY KEY AUTOINCREMENT, DATASET TEXT, TRAIN_SET_SIZE REAL, BATCH_SIZE"
        "INTEGER, TEST_BATCH_SIZE INTEGER, EPOCHS INTEGER, LEARNING_RATE REAL, WEIGHT_DECAY REAL, "
        "GAMMA REAL, SEED INTEGER, HIDDEN_DIMENSION INTEGER, WEIGHTS BLOB, TRAINING_LOSSES BLOB, "
        "TEST_LOSSES BLOB, RUNTIME REAL)"
    )

    DATABASE.commit()


def main():
    """
    Main


    """
    print("Number of experiments to be run:", len(EXPERIMENT_PARAMETERS))

    prepare_database()

    for (
        dataset_path,
        train_set_size,
        batch_size,
        test_batch_size,
        epochs,
        learning_rate,
        weight_decay,
        gamma,
        seed,
        hidden_dimension,
    ) in track(EXPERIMENT_PARAMETERS):
        start_time = time.time()
        weights, training_losses, test_losses = run_binaps(
            dataset_path,
            train_set_size,
            batch_size,
            test_batch_size,
            epochs,
            learning_rate,
            weight_decay,
            gamma,
            seed,
            hidden_dimension,
        )
        end_time = time.time()
        execution_time = end_time - start_time

        commit_result_to_database(
            dataset_path,
            train_set_size,
            batch_size,
            test_batch_size,
            epochs,
            learning_rate,
            weight_decay,
            gamma,
            seed,
            hidden_dimension,
            weights,
            training_losses,
            test_losses,
            execution_time,
        )
        print(
            f"Experiment finished. ({dataset_path}, {train_set_size}, {batch_size}, "
            f"{test_batch_size}, {epochs}, {learning_rate}, {weight_decay}, {gamma}, {seed}, "
            f"{hidden_dimension}, {execution_time:.2f}s) "
        )

    DATABASE.close()


if __name__ == "__main__":
    main()
