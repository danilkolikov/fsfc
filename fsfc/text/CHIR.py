import numpy as np
from fsfc.base import KBestFeatureSelector
from sklearn.cluster import KMeans


class CHIR(KBestFeatureSelector):
    """
    Chi-R feature selection algorithm for text clustering.

    Uses Chi-square statistics to evaluate the importance of each feature and R-coefficient
    that normalises statistics features across the corpus.

    Based on the article
    `"Text clustering with feature selection by using statistical data." <https://ieeexplore.ieee.org/document/4408578/>`_.

    Algorithm selects features in the following way:
        1. Find initial clustering of dataset.
        2. Compute Chi-R scores for each feature according to the article.
        3. Select top *k* features according to scores.
        4. Set weights for top features equal to 1 and for others to alpha.
        5. Recompute clustering. If it changes, repeat steps 2-5, otherwise return weights of features.

    Parameters
    ----------
    k: int
        Number of features to select.
    clusters: int
        Expected number of clusters.
    alpha: float (default 0.1)
        Value of weight of irrelevant feature.
    max_iter: int (default 1000)
        Maximal number of iterations of the algorithm.

    """
    def __init__(self, k, clusters, alpha=0.1, max_iter=1000):
        super().__init__(k)
        self.clusters = clusters
        self.alpha = alpha
        self.max_iter = max_iter

    def fit(self, x, *rest):
        """
        Fit algorithm to dataset and select relevant features.

        Parameters
        ----------
        x: csr_matrix
            SciPy Sparse Matrix representing terms contained in every sample. May be created by vectorizers from
            sklearn. Preferred choice is TF-IDF vectorizer.

        Returns
        -------
        self: FTC
            Returns itself to support chaining.
        """
        return super().fit(x, *rest)

    def _calc_scores(self, x):
        n_samples, n_features = x.shape

        k_means = KMeans(n_clusters=self.clusters)
        clusters = k_means.fit_predict(x)
        weight = np.ones([n_features])
        for i in range(self.max_iter):
            scores = self._compute_chir_scores(x, clusters)
            sorted_args = np.argsort(scores)

            # Weight for top scores should be equal to 1, for others to alpha
            new_weight = np.zeros_like(weight)
            new_weight[sorted_args[0:n_features - self.k]] = self.alpha
            new_weight[sorted_args[-self.k:]] = 1

            # Recompute clusters
            new_clusters = k_means.fit_predict(x.multiply(new_weight))
            if np.equal(weight, new_weight).all():
                break
            clusters = new_clusters
            weight = new_weight
        return weight

    @staticmethod
    def _compute_chir_scores(x, labels):
        # Compute matrix W where w[i, j] = amount of occurrences of j-th word in i-th cluster
        n_samples, n_features = x.get_shape()
        n_clusters = np.max(labels)
        w = np.zeros([n_clusters, n_features])
        rows, cols = x.nonzero()
        for (sample, word) in zip(rows, cols):
            cluster = labels[sample] - 1
            w[cluster, word] += 1
        row_sum = np.sum(w, 1)  # Number of words in i-th cluster
        column_sum = np.sum(w, 0)  # Number of clusters that contain j-th word
        w_sum = np.sum(row_sum)  # Total number of words

        # Compute Chi-R statistics using this matrix
        scores = np.zeros([n_features])
        for j in range(n_features):
            chi = np.zeros([n_clusters])  # Chi-square statistics
            r = np.zeros([n_clusters])  # R coefficient
            for i in range(n_clusters):
                # Contingency matrix
                cont = np.array([
                    [w[i, j], row_sum[i] - w[i, j]],
                    [column_sum[j] - w[i, j], w_sum - column_sum[j] - row_sum[i] + w[i, j]]
                ])

                # Calculation of expected frequency
                def calc_e(wrd, c):
                    return (cont[wrd, 0] + cont[wrd, 1]) * (cont[0, c] + cont[1, c]) / w_sum

                # Compute R coefficient for cluster and Chi-square statistics
                r[i] = w[i, j] / calc_e(0, 0)
                for a in [0, 1]:
                    for b in [0, 1]:
                        expected = calc_e(a, b)
                        chi[i] += (cont[a, b] - expected) ** 2 / expected
            # Select clusters with positive dependence with this term
            dependent = r > 1
            r_dependent = r[dependent]
            chi_dependent = chi[dependent]
            # Compute score
            scores[j] = (r_dependent / np.sum(r_dependent)).dot(chi_dependent)
        return scores
