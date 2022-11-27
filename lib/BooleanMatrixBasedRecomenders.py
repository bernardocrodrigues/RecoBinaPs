import numpy as np

from lib.FormalConceptAnalysis import BinaryDataset
from scipy.spatial import distance


def jaccard_distance(A: np.array, B: np.array):
    return distance.jaccard(A, B)


def cosine_distance(A: np.array, B: np.array):
    return distance.cosine(A, B)


def get_similarity_matrix(dataset: BinaryDataset, distance_strategy=jaccard_distance):
    """
    Given a BinaryDataset and some method that calculates some distance between two vector, computes the similarity
    matrix between all users (rows).

    The distance strategy must compute the distance between two numpy arrays. A return value of 1 implies that the
    vectors are completely different (maximum distance) while a return value of 0 implies that the vectors are identical
    (minimum distance).
    """
    similarity_matrix = np.ones((dataset.shape[0], dataset.shape[0]), np.double)

    similarity_matrix = -1 * similarity_matrix

    for i, row1 in enumerate(dataset._binary_dataset):
        for j, row2 in enumerate(dataset._binary_dataset):

            if similarity_matrix[i, j] != -1:
                continue

            dissimilarity = distance_strategy(row1, row2)
            similarity = 1 - dissimilarity

            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity

    return similarity_matrix


