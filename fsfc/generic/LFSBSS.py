import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import euclidean_distances

from fsfc.base import BaseEstimator, ClusterMixin


class LFSBSS(BaseEstimator, ClusterMixin):
    """
    Localised Feature Selection, Based on Scattered Separability.

    Selects features and simultaneously builds clustering in an iterative way.
    Every cluster has it's local set of selected features, and we project input data
    to a subspace defined by it to predict cluster of a point.

    This implementation doesn't take into account importance of overlay of clusters and unassigned points
    for the sake of performance.

    Based on the article `"Localized feature selection for clustering." <http://www.cs.wayne.edu/~jinghua/publication/PRL-LocalizedFeatureSelection.pdf>`_.

    Algorithm builds clustering in the following way:
        1. Find initial clustering using k-Means algorithm
        2. For each cluster:
            1. Find feature which can be dropped to improve the Scatter Separability Score
            2. Recompute clusters without this feature
            3. Find cluster that is the most similar to the current one
            4. Compute normalized value of Scatter Separability Score for two clusters - current and new
            5. If scores were improved, drop the feature and update clustering
        3. Repeat step 2 until no changes were made
        4. Return found clusters

    Algorithm predicts clusters for new points in the following way:
        1. Project points to the feature subspace of each cluster
        2. Find the cluster whose center is the closest to projected point

    Parameters
    ----------
    clusters: int
        Number of clusters to find
    max_iterations: int (default 100)
        Maximal number of iterations of the algorithm
    """

    def __init__(self, clusters, max_iterations=100):
        self.clusters = clusters
        self.labels_ = None
        self.features_ = None
        self.means_ = None
        self.vars_ = None
        self.max_iterations = max_iterations

    def fit(self, x):
        """
        Fit algorithm to dataset, find clusters and set of features for every cluster

        Parameters
        ----------
        x: ndarray
            The dataset

        Returns
        -------
        self: LFSBSS
            Returns itself to support chaining
        """

        n_samples, n_features = x.shape
        # Build initial clustering
        k_means = KMeans(n_clusters=self.clusters)
        features = [set(range(n_features)) for _ in range(self.clusters)]
        clusters, means = self._find_clusters_and_means(k_means, x)

        for it in range(self.max_iterations):
            was_changed = False
            for i in range(self.clusters):
                cluster = clusters[i]
                mean = means[i]
                this_features = features[i]
                if len(this_features) == 1:
                    # Can't drop anything
                    continue

                # Find a feature which we can drop and have the highest scatter separability score
                max_score = None
                new_features = None
                for feature in this_features:
                    without_feature = list(this_features - {feature})
                    score = self._compute_score(x, means, cluster, mean, without_feature)
                    if max_score is None or score > max_score:
                        max_score = score
                        new_features = without_feature
                if new_features is None:
                    # Nothing to remove from this cluster
                    continue

                # Repartition dataset using new features, find a cluster that is the most similar to a current one
                new_x = x[:, new_features]
                new_clusters, new_means = self._find_clusters_and_means(k_means, new_x)

                # Use Jaccard difference as the measure of similarity
                max_score = None
                most_similar = None
                new_mean = None
                for j in range(self.clusters):
                    new_cluster = clusters[j]
                    score = LFSBSS._jaccard_score(new_cluster, cluster)
                    if max_score is None or score > max_score:
                        max_score = score
                        most_similar = new_cluster
                        new_mean = new_means[i]
                if most_similar is None or new_mean is None:
                    # Nothing to select
                    continue

                # Compute normalized value of scatter separability
                nv_old = self._compute_score(x, means, cluster, mean, list(this_features)) * \
                         self._compute_score(x, means, cluster, mean, new_features)
                nv_new = self._compute_score(x, means, most_similar, mean, list(this_features)) * \
                         self._compute_score(new_x, new_means, most_similar, new_mean, range(len(new_features)))

                if nv_new >= nv_old:
                    # It's better to drop this feature
                    was_changed = True
                    features[i] = set(new_features)
                    clusters[i] = most_similar
                    # means[i] = new_mean
            if not was_changed:
                break

        self.features_ = features
        self.means_ = means
        self.vars_ = [None] * x.shape[0]

        # Compute variances for clusters
        for (idx, cluster) in enumerate(clusters):
            feature = np.array(list(features[idx]))[:, np.newaxis]
            self.vars_[idx] = np.var(x[cluster, feature])

        self.labels_ = [None] * x.shape[0]
        for (idx, cluster) in enumerate(clusters):
            for sample in cluster:
                self.labels_[sample] = idx
        for i in range(x.shape[0]):
            if self.labels_[i] is None:
                self.labels_[i] = self.predict(x[i])
        return self

    def predict(self, x):
        """
        Predict clusters for one sample

        Parameters
        ----------
        x: ndarray
            Samples to predict

        Returns
        -------
        label: int
            Predicted cluster
        """

        # Find the closest cluster to samples
        # To do it, project x to appropriate subspace, find distance to mean value and norm by variance
        min_score = None
        closest = None
        for i in range(self.clusters):
            projection = x[:, self.features_[i]]
            norm = euclidean_distances(projection, self.means_[i])
            score = norm / self.vars_[i]
            if min_score is None or score < min_score:
                min_score = score
                closest = i
        return closest

    def _find_clusters_and_means(self, k_means, x):
        initial = k_means.fit_predict(x)
        clusters = [[] for _ in range(self.clusters)]
        means = k_means.cluster_centers_.copy()
        for (idx, c) in enumerate(initial):
            clusters[c].append(idx)
        return clusters, means

    @staticmethod
    def _compute_score(x, means, cluster, mean, features):
        cluster = np.array(cluster)
        features = np.array(features)
        means = means[:, features]
        x = x[:, features]
        mean = mean[features]

        total_mean = np.mean(x, axis=0)
        cluster_values = x[cluster, :]
        cluster_diff = cluster_values - mean
        in_cluster = LFSBSS._tensor_product_sum(cluster_diff)

        means_diff = means - total_mean
        between_cluster = LFSBSS._tensor_product_sum(means_diff)

        try:
            separability = np.trace(np.linalg.inv(in_cluster).dot(between_cluster))
        except np.linalg.LinAlgError as _:
            separability = np.trace(between_cluster) / np.trace(in_cluster)
        return separability / cluster_values.shape[0]

    @staticmethod
    def _tensor_product_sum(x):
        res = np.zeros([x.shape[1], x.shape[1]])
        for i in range(x.shape[0]):
            res = np.add(res, np.tensordot(x[i], x[i], axes=0))
        return res

    @staticmethod
    def _jaccard_score(a, b):
        a = set(a)
        b = set(b)
        return 1.0 * len(a.intersection(b)) / len(a.union(b))
