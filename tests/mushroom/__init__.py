from .MushroomDataset import parse_mushroom_dataset

MushroomDataset = parse_mushroom_dataset("tests/mushroom/agaricus-lepiota.data")

__all__ = ["MushroomDataset"]