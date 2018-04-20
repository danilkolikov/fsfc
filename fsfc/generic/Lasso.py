import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from scipy.optimize import fixed_point

from fsfc.base import ClusteringFeatureSelector


class Lasso(ClusteringFeatureSelector):
    """
    Feature selection and clustering algorithm exploting the idea of L1-norm

    Simultaneously does clustering and computes "importance" of each feature for it

    Based on the article `"A framework for feature selection in clustering." <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2930825/>`_

    Algorithm does clustering and selects features in the following way:
        1. Assigns equal weights to every feature
        2. Finds clusters according to weights of features
        3. Computes objective of the method
        4. Optimizes L1-penalty for found objective
        5. Computes new feature weights using objective
        6. If new weights equal to the old ones, break. Otherwise repeat steps 2-6
        7. Select top k features according to weights

    Parameters
    ----------
    k: int
        Number of features to select
    norm_constraint: float
        Constraint of L1-norm
    eps: float (default 1e-4)
        Precision of the algorithm
    max_iterations: int (default 100)
        Maximal number of iterations of algorithm
    """
    def __init__(self, k, norm_constraint, eps=1e-4, max_iterations=100):
        super().__init__(k)
        self.norm_constraint = norm_constraint
        self.max_iterations = max_iterations
        self.eps = eps

    def _calc_scores_and_labels(self, x):
        # Assign equal weights to features
        feature_weights = np.full([1, x.shape[1]], 1/np.sqrt(x.shape[1]))
        labels = None
        for it in range(self.max_iterations):
            # Find clusters in the feature space, normalised by weights
            weighted_samples = feature_weights * x
            labels = KMeans().fit_predict(weighted_samples)

            # Compute objective vector of the method
            objective = Lasso._calc_objective_vector(x, labels)

            # Find parameter of threshold that optimizes L1-norm
            root = fixed_point(
                lambda delta: self._function_to_optimize(objective, delta),
                0
            )
            # Compute new feature weights
            new_weights = Lasso._calc_new_feature_weights(objective, root)
            if np.linalg.norm(new_weights - feature_weights) < self.eps:
                break
            feature_weights = new_weights
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
