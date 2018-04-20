import numpy as np
import math
from sklearn.neighbors import kneighbors_graph
from sklearn.linear_model import LassoLars, Lars
from scipy.linalg import eigh
from fsfc.base import KBestFeatureSelector


class MCFS(KBestFeatureSelector):
    """
    Multi-Class Feature selection.

    Selects features in 5 steps:

        1. Computes k-NN graph for dataset
        2. Computes heat matrix for this graph, degree and laplacian matrix
        3. Solves eigen-problem: L y = `lambda` D y, selects K smallest eigen-values and corresponding vectors
        4. Solves K regression problems, trying to predict every eigenvector using dataset
        5. Computes score of each feature using found regression coefficients
    """

    def __init__(self, clusters, k, p=8, sigma=1, mode='default', alpha=0.01):
        """
        Creates instance of MCFS-algorithm

        :param clusters: Expected number of clusters
        :param k: Number of features to select
        :param p: Initial number of clusters for K-NN graph
        :param sigma: Parameter for computation of heat-matrix
        :param mode: 'default' - use Lars method, 'lasso' - user Lars method with L1-penalty
        :param alpha: Importance of L1-penalty for 'lasso' mode
        """
        super().__init__(k)
        self.clusters = clusters
        self.p = p
        self.mode = mode
        self.sigma = sigma
        self.alpha = alpha

    def _create_regressor(self):
        if self.mode == 'default':
            return Lars()
        if self.mode == 'lasso':
            return LassoLars(alpha=self.alpha)
        raise ValueError('Unexpected mode ' + self.mode + '. Expected "default" or "lasso"')

    def _calc_scores(self, x):
        graph = kneighbors_graph(
            x,
            n_neighbors=self.p,
        )
        # Construct heat matrix
        w = np.zeros([x.shape[0], x.shape[0]])
        rows, cols = graph.nonzero()
        for i, j in zip(rows, cols):
            w[i, j] = math.exp(-np.linalg.norm(x[i] - x[j])**2/self.sigma)

        # Compute degree and Laplacian matrices
        degree_vector = np.sum(w, 1)
        degree = np.diag(degree_vector)
        laplacian = degree - w

        # Solve eigen-problem
        values, vectors = eigh(laplacian, degree)
        smallest = vectors[:, 0:self.clusters].T

        # Find coefficients for each cluster
        coefs = []
        for i in range(self.clusters):
            this_coefs = self._create_regressor().fit(x, smallest[i]).coef_
            coefs.append(this_coefs)
        coefs = np.array(coefs)

        # Compute MCFS-scores
        scores = np.max(coefs, 0)
        return scores
