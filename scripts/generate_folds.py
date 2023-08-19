import os

from surprise import Dataset, Reader
from surprise.model_selection import PredefinedKFold

from lib.BinaryDataset import BinaryDataset

files_dir = os.path.expanduser("/workdir/datasets/ml-100k/")

reader = Reader("ml-100k")

train_file = files_dir + "u%d.base"
test_file = files_dir + "u%d.test"
folds_files = [(train_file % i, test_file % i) for i in (1, 2, 3, 4, 5)]

data = Dataset.load_from_folds(folds_files, reader=reader)
pkf = PredefinedKFold()



for index, (trainset, testset) in enumerate(pkf.split(data)):

    binary_dataset = BinaryDataset.load_from_trainset(trainset)

    with open(f"/workdir/datasets/u{index}.base.dat", 'w')  as file_object:
        binary_dataset.save_as_binaps_compatible_input(file_object)


    # dataset.save_as_binaps_compatible_input()