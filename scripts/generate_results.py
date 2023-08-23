import itertools
import time
import sqlite3
import pickle
from rich.progress import track

# from lib.BinaryDataset import BinaryDataset
from binaps.binaps_wrapper import run_binaps


DATABASE = sqlite3.connect("your_database.db")


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
    cursor = DATABASE.cursor()

    serialized_weights = pickle.dumps(weights)
    serialized_training_losses = pickle.dumps(training_losses)
    serialized_test_losses = pickle.dumps(test_losses)

    cursor.execute(
        (
            "INSERT INTO binaps_experiments (DATASET, TRAIN_SET_SIZE, BATCH_SIZE, TEST_BATCH_SIZE, EPOCHS, LEARNING_RATE, "
            "WEIGHT_DECAY, GAMMA, SEED, HIDDEN_DIMENSION, WEIGHTS, TRAINING_LOSSES, TEST_LOSSES, RUNTIME) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
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
            serialized_weights,
            serialized_training_losses,
            serialized_test_losses,
            runtime,
        ),
    )

    DATABASE.commit()


import os
from surprise import Dataset, Reader
from surprise.model_selection import PredefinedKFold

fold_files_dir = os.path.expanduser("/workdir/datasets/ml-100k")
folds_files = [
    (f"{fold_files_dir}/u{i}.base", f"{fold_files_dir}/u{i}.test") for i in (1, 2, 3, 4, 5)
]

reader = Reader("ml-100k")
dataset = Dataset.load_from_folds(folds_files, reader=reader)
pkf = PredefinedKFold()


DATASET_FOLDER_PATH = "/workdir/datasets"
DATASET = [
    "ml-100k/u0.base.dat",
    "ml-100k/u1.base.dat",
    "ml-100k/u2.base.dat",
    "ml-100k/u3.base.dat",
    "ml-100k/u4.base.dat",
]


# Binaps meta parameters
TRAIN_SET_SIZE = [0.9]
BATCH_SIZE = [64]
TEST_BATCH_SIZE = [64]
EPOCHS = [50000]
LEARNING_RATE = [0.01]
WEIGHT_DECAY = [0]
GAMMA = [0.1]
SEED = [1]
HIDDEN_DIMENSION = [-1]

meta_parameters = list(
    itertools.product(
        DATASET,
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


print(len(meta_parameters))

for (
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
) in track(meta_parameters):
    start_time = time.time()

    weights, training_losses, test_losses = run_binaps(
        f"{DATASET_FOLDER_PATH}/{dataset}",
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
        execution_time,
    )

    print(
        "done",
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
    )

DATABASE.close()

