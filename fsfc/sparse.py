import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from scipy.optimize import fixed_point

from .base import ClusteringFeatureSelector


class Lasso(ClusteringFeatureSelector):
    """
    Feature selection and clustering algorithm, based on idea of Tibshirani's lasso

    Simultaneously does clustering and computes "importance" of each feature for it

    Based on https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2930825/
    """
    def __init__(self, k, norm_constraint, iterations=100):
        super().__init__(k)
        self.norm_constraint = norm_constraint
        self.iterations = iterations

    def _calc_scores_and_labels(self, x):
        feature_weights = np.full([1, x.shape[1]], 1/np.sqrt(x.shape[1]))
        labels = None
        for it in range(self.iterations):
            weighted_samples = feature_weights * x
            labels = KMeans().fit_predict(weighted_samples)
            objective = Lasso._calc_objective_vector(x, labels)
            root = fixed_point(
                lambda delta: self._function_to_optimize(objective, delta),
                0
            )
            old_weights = feature_weights.copy()
            feature_weights = Lasso._calc_new_feature_weights(objective, root)
            if np.linalg.norm(feature_weights - old_weights) < 1e4:
                break
        return feature_weights[0], labels

    @staticmethod
    def _calc_objective_vector(x, labels):
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(i)
        result = np.zeros([1, x.shape[1]])
        for i in range(x.shape[1]):
            feature = 0
            samples = x[:, i].T.reshape([x.shape[0], 1])
            for label, cluster in clusters.items():
                size = len(cluster)
                cluster_samples = samples[cluster]
                distances = euclidean_distances(cluster_samples)
                feature += np.sum(distances) / size
            result[0, i] = np.sum(euclidean_distances(samples)) / x.shape[0] - feature
        return result

    def _function_to_optimize(self, objective, delta):
        weights = Lasso._calc_new_feature_weights(objective, delta)
        return np.linalg.norm(weights, 1) - self.norm_constraint

    @staticmethod
    def _soft_threshold(x, delta):
        positive = np.clip(x, a_min=0, a_max=None)
        return np.sign(positive) * np.clip(np.abs(positive) - delta, a_min=0, a_max=None)

    @staticmethod
    def _calc_new_feature_weights(objective, delta):
        soft = Lasso._soft_threshold(objective, delta)
        normed = soft / np.linalg.norm(soft)
        return normed
