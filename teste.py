import numpy as np

seed = 1
np.random.seed(seed)

from surprise import Dataset
from surprise.model_selection import KFold
from surprise.accuracy import mae, rmse

from recommenders import DEBUG_LOGGER
from recommenders.pedro import PedroRecommender 

dataset = Dataset.load_builtin("ml-100k", prompt=False)

kf = KFold(n_splits=5)
folds = [(fold_index, fold) for fold_index, fold in enumerate(kf.split(dataset))]

fold_index, (trainset, testset) = folds[0]

recommender = PedroRecommender(epochs=5000)
recommender.fit(trainset)
predictions = recommender.test(testset)

mae(predictions=predictions)
rmse(predictions=predictions)
