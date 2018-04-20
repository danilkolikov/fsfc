import numpy as np
from sklearn.cluster import KMeans
from fsfc.base import ClusteringFeatureSelector


class WKMeans(ClusteringFeatureSelector):
    """
    Weighted K-Means algorithm.

    Assigns a weight parameter to every feature, runs N iterations of K-means and adjusts this parameter to
    explain the data better. Selects K best features according to their weights.

    Based on the article `"Automated variable weighting in k-means type clustering." <https://ieeexplore.ieee.org/document/1407871/>`_.

    Algorithm selects features in the following way:
        1. Randomly assigns values for feature weights in a way that keeps their sum equal to 1.
        2. Find clusters using samples multiplied by weights in the power of *beta*
        3. Compute score for clustering.
        4. Recompute weights using approach from the article.
        5. Find clustering using new weights. If score of new clustering didn't change, stop algorithm.
           Otherwise got to step 2.
        6. Select top k features according to their weights.

    Parameters
    ----------
    k: int
        Number of features to select.
    beta: float
        Degree parameter for features weights.
    eps: float (default 1e-3)
        Precision of the algorithm.
    max_iterations: int (default 10)
        Maximal number of iterations of algorithm.
    """

    def __init__(self, k, beta, eps=1e-3, max_iterations=10):
        super().__init__(k)
        self.beta = beta
        self.eps = eps
        self.max_iterations = max_iterations

    def _calc_scores_and_labels(self, x):
        k_means = KMeans(n_clusters=self.k)

        # Assign random weights to each feature. Sum of weights should be equal to 1
        random_weights = np.random.rand(x.shape[1])
        exp_weights = np.exp(random_weights)
        weights = exp_weights / np.sum(exp_weights)

        # Compute clusters using samples scaled by weight in the power of beta
        modified_x = x * weights ** self.beta
        clusters = k_means.fit_predict(modified_x)

        # Compute score for clustering
        old_score = WKMeans._calc_objective(k_means, modified_x)
        for i in range(self.max_iterations):

            # Find position of centroids
            divider = weights.copy() ** self.beta
            divider[abs(divider) < self.eps] = 1
            centroids = k_means.cluster_centers_ / divider

            # Compute D-array used for computation of new weights
            d = np.zeros(x.shape[1])
            for k in range(d.size):
                d[k] = 0
                for j in range(x.shape[0]):
                    d[k] += abs(x[j][k] - centroids[clusters[j]][k])

            # Compute new weights using D-array
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

            # Recompute clusters and check convergence of algorithm
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
