from math import log2, inf

import numpy as np

from fsfc.base import ClusteringFeatureSelector
from fsfc.utils import apriori


class FTC(ClusteringFeatureSelector):
    """
    Frequent Terms-based Clustering

    Based on http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.12.7997&rep=rep1&type=pdf
    """

    def __init__(self, minsup):
        super().__init__(-1)
        self.minsup = minsup

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
