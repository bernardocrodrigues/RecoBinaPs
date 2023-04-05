import numpy as np

from lib.FormalConceptAnalysis import BinaryDataset, GreConD, get_factor_matrices_from_concepts
from surprise import AlgoBase, PredictionImpossible
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


class FcaBmf(AlgoBase):
    def __init__(self, k=30, coverage=1.0, threshold=1, distance_strategy=jaccard_distance, verbose=False):
        AlgoBase.__init__(self)
        self.verbose = verbose
        self.k = k
        self.coverage = coverage
        self.threshold = threshold
        self.distance_strategy = distance_strategy

        self.actual_coverage = None
        self.number_of_factors = None

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        if self.verbose:
            print("[FcaBmf] Generating binary dataset...")

        self.binary_dataset = BinaryDataset.load_from_trainset(trainset,threshold=self.threshold)

        if self.verbose:
            print("[FcaBmf] Generating binary dataset OK!")
            print(f"[FcaBmf] Resulting binary dataset is {self.binary_dataset.shape[0]} rows x {self.binary_dataset.shape[1]} columns")

        if self.verbose:
            print("[FcaBmf] Generating Formal Context...")

        self.formal_context, self.actual_coverage = GreConD(self.binary_dataset, coverage=self.coverage, verbose=self.verbose)
        self.number_of_factors = len(self.formal_context)

        if self.verbose:
            print("[FcaBmf] Generating Formal Context OK")

        self.Af, self.Bf = get_factor_matrices_from_concepts(self.formal_context, self.binary_dataset.shape[0], self.binary_dataset.shape[1])
        latent_binary_dataset = BinaryDataset(self.Af)

        if self.verbose:
            print("[FcaBmf] Generating Similarity Matrix...")

        self.sim = get_similarity_matrix(latent_binary_dataset, self.distance_strategy)

        if self.verbose:
            print("[FcaBmf] Generating Similarity Matrix Ok")

        return self

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible("User and/or item is unknown.")

        ruid = self.trainset.to_raw_uid(u)
        riid = self.trainset.to_raw_iid(u)

        rating = 0
        weight_sum = 0
        neighbors_used = []

        neighbors = [(x2, self.sim[u, x2], r) for (x2, r) in self.trainset.ir[i]]
        nearest_neighbors = sorted(neighbors, key=lambda d: d[1], reverse=True)

        for neighbor in nearest_neighbors:

            if len(neighbors_used) >= self.k:
                break

            neighbor_iid = neighbor[0]
            neighbor_similarity = neighbor[1]
            neighbor_rating = neighbor[2]

            neighbor_ruid = self.trainset.to_raw_uid(neighbor_iid)

            if neighbor_similarity == 0:
                continue

            rating += neighbor_similarity * neighbor_rating
            weight_sum += neighbor_similarity
            neighbors_used.append((neighbor_ruid, neighbor_similarity, neighbor_rating))

        if not neighbors_used:
            raise PredictionImpossible("Not enough neighbors.")

        rating /= weight_sum

        details = {"actual_k": len(neighbors_used), "neighbors_used": neighbors_used}

        return rating, details
