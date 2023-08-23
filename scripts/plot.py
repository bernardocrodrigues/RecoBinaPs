import sqlite3
import pickle
import plotext as plt

from recomenders.formal_context_based_recommender import BinapsRecommender, FcaBmf

DATABASE = sqlite3.connect("your_database.db")
cursor = DATABASE.cursor()

cursor.execute("SELECT * FROM experiments WHERE id = ?", (32,))
results = cursor.fetchone()

(
    id,
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
    serialized_patterns,
    serialized_training_losses,
    serialized_test_losses,
    runtime
) = results

patterns = pickle.loads(serialized_patterns)
training_losses = pickle.loads(serialized_training_losses)
test_losses = pickle.loads(serialized_test_losses)


from surprise import Dataset
from surprise.model_selection import KFold
from surprise.prediction_algorithms import KNNBasic
from surprise.accuracy import mae, rmse


dataset = Dataset.load_builtin("ml-100k")

K=5
THRESHOLD = 1

binaps_recommender = BinapsRecommender.from_previously_computed_patterns(patterns, k=K, threshold=THRESHOLD)
grecond_recommender = FcaBmf(k=K, threshold=THRESHOLD)
KNN_recommender = KNNBasic(k=5, sim_options={"name": "cosine"})


# trainset = dataset.build_full_trainset()
kf = KFold(n_splits=5)
fold_generator = kf.split(dataset)
trainset, testset = next(fold_generator)

binaps_recommender.fit(trainset)
grecond_recommender.fit(trainset)
KNN_recommender.fit(trainset)

binaps_predictions = binaps_recommender.test(testset)
grecond_predictions = grecond_recommender.test(testset)
knn_predictions = KNN_recommender.test(testset)


# # Overall quality of the predictions

mae(predictions=binaps_predictions)
rmse(predictions=binaps_predictions)
mae(predictions=grecond_predictions)
rmse(predictions=grecond_predictions)
mae(predictions=knn_predictions)
rmse(predictions=knn_predictions)
