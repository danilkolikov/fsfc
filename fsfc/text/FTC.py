from math import log2, inf

import numpy as np

from fsfc.base import ClusteringFeatureSelector
from fsfc.utils import apriori


class FTC(ClusteringFeatureSelector):
    """
    Frequent Terms-based Clustering algorithm. Uses frequent termsets to find clusters and simultaneously
    select features which determine every cluster.

    Based on the article `"Frequent term-based text clustering" <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.12.7997&rep=rep1&type=pdf>`_.

    **FTS** is a set of terms that appear in some part of all samples in dataset. We will say that FTS
    *covers* sample if every term of FTS is contained in the sample.

    Algorithm does clustering in the following way:
        1. Find all FTS for dataset with specified *minsup*. Elements of FTS are terms, i.e. features of dataset.
        2. Find FTS that has the lowest Entropy Overlap with the rest clusters with respect to dataset.
           It's shown in the paper that such FTS will explain data the best.
        3. Add this FTS to clustering, remove from dataset samples covered by it, repeat steps 2 and 3.
        4. Assign clusters to samples. Sample belongs to a cluster defined by FTS if FTS covers sample.
        5. Scores of features are 1 if feature belongs to any FTS and 0 otherwise.

    Parameters
    ----------
    minsup: float
        Part of the dataset which should be covered by each FTS.
    """

    def __init__(self, minsup):
        super().__init__(-1)
        self.minsup = minsup

    def fit(self, x, *rest):
        """
        Fit algorithm to dataset, find clusters and select relevant features.

        Parameters
        ----------
        x: csr_matrix
            SciPy Sparse Matrix representing terms contained in every sample. May be created by vectorizers from sklearn.

        Returns
        -------
        self: FTC
            Returns itself to support chaining.
        """
        return super().fit(x, *rest)

    def _calc_scores_and_labels(self, x):
        n_samples, n_features = x.get_shape()
        # Construct dataset for frequent itemsets extractions
        rows, columns = x.nonzero()
        dataset = [[] for _ in range(n_samples)]
        for (i, j) in zip(rows, columns):
            if x[i, j] != 0:
                # noinspection PyTypeChecker
                dataset[i].append(int(j))
        remaining = apriori(dataset, self.minsup)
        if len(FTC._calculate_coverage(remaining, dataset)) != n_samples:
            # Dataset can't be fully clustered using this method
            raise ValueError(
                'Can\'t cluster a dataset using FTC method - it isn\'t covered by the chosen itemset '
                + str(remaining)
            )
        dataset_copy = [sample for sample in dataset]
        result = []

        # Until all samples are covered
        while len(dataset) != 0:
            min_overlap = None
            best_cluster = None
            # Find feature set with the smallest overlap with rest
            for itemset in remaining:
                overlap = FTC._calculate_overlap(itemset, remaining, dataset)
                if min_overlap is None or overlap < min_overlap:
                    min_overlap = overlap
                    best_cluster = itemset
            # Add it to result
            result.append(best_cluster)
            remaining.remove(best_cluster)

            # Remove all covered samples from dataset
            for sample in FTC._calculate_coverage([best_cluster], dataset):
                dataset.remove(sample)

        # Select features used in final clustering, assign labels according to chosen itemsets
        used_features = {f for itemset in result for f in itemset}
        scores = [1 if i in used_features else 0 for i in range(n_features)]
        labels = [None for _ in range(n_samples)]
        for i in range(n_samples):
            features = dataset_copy[i]
            for cluster_idx in range(len(result)):
                if all(c in features for c in result[cluster_idx]):
                    labels[i] = cluster_idx
                    break
        self.k = len(used_features)

        return np.array(scores), np.array(labels)

    @staticmethod
    def _calculate_coverage(clusters, dataset):
        """
        Finds samples of dataset covered by clusters

        Parameters
        ----------
        clusters: list
            List of FTS defining clusters
        dataset: list
            List of sets defining samples

        Returns
        -------
        covered: list
            List of samples covered by clusters
        """
        # Find samples covered by specified clusters
        covered = []
        for sample in dataset:
            for cluster in clusters:
                if all(c in sample for c in cluster):
                    covered.append(sample)
                    break
        return covered

    @staticmethod
    def _calculate_overlap(cluster, rest, dataset):
        """
        Calculates Entropy Overlap of cluster with rest FTS

        Parameters
        ----------
        cluster: set
            Termset defining cluster
        rest: list
            List of other termsets
        dataset: list
            List of sets defining samples

        Returns
        -------
        overlap: float
            Value of overlap of cluster with the rest of clusters
        """
        f = [0 for _ in dataset]
        # Calculate how much clusters from `rest` cover each sample
        for i in range(len(dataset)):
            sample = dataset[i]
            for cur_cluster in rest:
                if all(c in sample for c in cur_cluster):
                    f[i] += 1
        overlap = inf
        # Compute entropy-overlap for cluster
        for i in range(len(dataset)):
            sample = dataset[i]
            if all(c in sample for c in cluster):
                if overlap == inf:
                    overlap = 0
                overlap += -1.0 / f[i] * log2(1.0 / f[i])
        return overlap
