from abc import abstractmethod

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, cosine_similarity

from fsfc.base import KBestFeatureSelector


class SPEC(KBestFeatureSelector):
    """
    SPEC feature selection algorithm

    Creates graph representation of how samples are distributed in a high-dimensional space and
    uses Spectral Graph Theory to calculate metrics

    Based on http://www.public.asu.edu/~huanliu/papers/icml07.pdf
    """

    @abstractmethod
    def _calc_spec_scores(self, degree, laplacian, normalised_features, normaliser):
        pass

    def _calc_scores(self, x):
        similarity = rbf_kernel(x)
        adjacency = similarity
        degree_vector = np.sum(adjacency, 1)
        degree = np.diag(degree_vector)
        laplacian = degree - adjacency
        normaliser_vector = np.reciprocal(np.sqrt(degree_vector))
        normaliser = np.diag(normaliser_vector)

        normalised_laplacian = normaliser.dot(laplacian).dot(normaliser)

        weighted_features = np.matmul(normaliser, x)

        normalised_features = weighted_features / np.linalg.norm(weighted_features, axis=0)
        return self._calc_spec_scores(degree, normalised_laplacian, normalised_features, normaliser)


class NormalizedCut(SPEC):
    """
    Feature selection algorithm that represents samples as vertices of graph. Weight of edges of this graph are
    equal to distances between points. Algorithm attempts to find minimal cut in this graph.

    Features that were `separated` by it are considered better for clustering
    """
    def _calc_spec_scores(self, degree, laplacian, normalised_features, normaliser):
        all_to_all = normalised_features.transpose().dot(laplacian).dot(normalised_features)
        return np.diag(all_to_all)


class GenericSPEC(NormalizedCut):
    """
    Feature selection algorithm that uses spectral graph theory to find features with the best separability.
    """
    def _calc_spec_scores(self, degree, laplacian, normalised_features, normaliser):
        normalised_cut = super()._calc_spec_scores(
            degree, laplacian, normalised_features, normaliser
        )
        trivial_eugenvector = normaliser.dot(np.ones([normalised_features.shape[0], 1]))
        norm = 1 - normalised_features.transpose().dot(trivial_eugenvector).squeeze().transpose()

        return normalised_cut / norm


class FixedSPEC(SPEC):
    """
    Feature selection algorithm that uses spectral graph theory to find features with the best separability
    in assumption that points are separated to fixed number of clusters
    """
    def __init__(self, k, clusters):
        super().__init__(k)
        self.clusters = clusters

    def _calc_spec_scores(self, degree, laplacian, normalised_features, normaliser):
        values, vectors = np.linalg.eigh(laplacian)

        result = np.zeros([normalised_features.shape[1]])
        for k in range(1, self.clusters):
            eigenvalue = values[k]
            eigenvector = vectors[:, k]
            cosines = np.zeros_like(result)
            for i in range(normalised_features.shape[1]):
                feature = normalised_features[:, i]
                similarity = cosine_similarity([eigenvector], [feature])[0]
                cosines[i] = similarity * similarity
            result += (2 - eigenvalue) * cosines
        return -result

