from abc import abstractmethod

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, cosine_similarity

from fsfc.base import KBestFeatureSelector


class SPEC(KBestFeatureSelector):
    """
    SPEC-family of feature selection algorithms.

    Every algorithm of this family creates graph representation of distribution of samples in a high-dimensional space.
    Samples becomes vertices and RBF-distance between them becomes weight of the edge. Algorithms use Spectral Graph
    Theory to calculate features scores.

    Based on the article `"Spectral feature selection for supervised and unsupervised learning." <http://www.public.asu.edu/~huanliu/papers/icml07.pdf>`_.
    """

    @abstractmethod
    def _calc_spec_scores(self, degree, laplacian, normalised_features, normaliser):
        """
        Calculate SPEC scores for the method

        Parameters
        ----------
        degree: ndarray
            Degree matrix of the graph
        laplacian: ndarray
            Laplacian of the graph
        normalised_features: ndarray
            Feature vectors, normalised by normaliser
        normaliser: ndarray
            Vector computed by the degree matrix and used for normalisation of feature vectors and laplacian

        Returns
        -------
        scores: ndarray
            Scores of features
        """
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
    Feature selection algorithm that represents samples as vertices of graph.
    Weights of edges of this graph are equal to RBF-distances between points.
    Algorithm attempts to find minimal cut in this graph.

    Algorithm selects k features which were `separated` by this cut as they are considered to be better for explanation
    of the dataset.

    Parameters
    ----------
    k: int
        Number of features to select
    """

    def _calc_spec_scores(self, degree, laplacian, normalised_features, normaliser):
        all_to_all = normalised_features.transpose().dot(laplacian).dot(normalised_features)
        return np.diag(all_to_all)


class GenericSPEC(NormalizedCut):
    """
    Feature selection algorithm that represents samples as vertices of graph.
    Weights of edges of this graph are equal to RBF-distances between points.

    Algorithm uses Spectral Graph Theory to find features with the best separability.

    To do this, algorithm finds the trivial eugenvector of the Laplacian of the graph and uses it to
    normalise scores computed using :class:`NormalizedCut`. Such normalisation helps to improve
    accuracy of feature selection according to the article SPEC family is based on.

    Algorithm selects top k features according to this scores

    Parameters
    ----------
    k: int
        Number of features to select
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
    Feature selection algorithm that represents samples as vertices of graph.
    Weights of edges of this graph are equal to RBF-distances between points.

    Algorithm uses Spectral Graph Theory to find features with the best separability in assumption that
    points are separated to the predefined number of clusters

    To do this, algorithm finds eigenvectors corresponding to K smallest eigenvalues of the Laplacian of the graph,
    except the trivial one, and uses cosine distance between them and feature vectors to detect the most explaining
    features. Algorithm selects k features using this score.

    Parameters
    ----------
    k: int
        Number of features to select
    clusters: int
        Expected number of clusters
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

