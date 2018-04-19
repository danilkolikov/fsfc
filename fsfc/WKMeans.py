import numpy as np
from sklearn.cluster import KMeans
from .base import ClusteringFeatureSelector


class WKMeans(ClusteringFeatureSelector):
    """
    Weighted K-Means algorithm.

    Assigns a weight parameter to every feature, runs N iterations of K-means and adjusts this parameter to
    explain the data better
    """

    def __init__(self, k, beta, eps=1e-3, n_iterations=10):
        """
        Creates instance of weighted K-means algorithm

        :param k: Number of clusters in expected clustering
        :param beta: Degree parameter for feature weights
        :param eps: Precision of algorithm
        :param n_iterations: Maximal number of iterations
        """
        super().__init__(k)
        self.beta = beta
        self.eps = eps
        self.n_iterations = n_iterations

    def _calc_scores_and_labels(self, x):
        k_means = KMeans(n_clusters=self.k)

        random_weights = np.random.rand(x.shape[1])
        exp_weights = np.exp(random_weights)
        weights = exp_weights / np.sum(exp_weights)

        modified_x = x * weights ** self.beta
        clusters = k_means.fit_predict(modified_x)
        old_score = WKMeans._calc_objective(k_means, modified_x)
        for i in range(self.n_iterations):
            divider = weights.copy() ** self.beta
            divider[abs(divider) < self.eps] = 1
            centroids = k_means.cluster_centers_ / divider

            d = np.zeros(x.shape[1])
            for k in range(d.size):
                d[k] = 0
                for j in range(x.shape[0]):
                    d[k] += abs(x[j][k] - centroids[clusters[j]][k])
            new_weights = np.zeros(weights.size)
            if self.beta == 1:
                new_weights[np.argmin(d)] = 1
            else:
                for k in range(new_weights.size):
                    if abs(d[k]) < self.eps:
                        continue
                    for current_d in d:
                        if abs(current_d) < self.eps:
                            continue
                        new_weights[k] += (d[k] / current_d) ** (1 / (self.beta - 1))
                    new_weights[k] = 1 / new_weights[k]
            weights = new_weights
            modified_x = x * weights ** self.beta
            clusters = k_means.fit_predict(modified_x)
            new_score = WKMeans._calc_objective(k_means, modified_x)
            if abs(new_score - old_score) < self.eps:
                break
            old_score = new_score
        return weights, clusters

    @staticmethod
    def _calc_objective(k_means, x):
        return -k_means.score(x)
