import numpy as np
import math
from sklearn.neighbors import kneighbors_graph
from sklearn.linear_model import LassoLars, Lars
from scipy.linalg import eigh
from fsfc.base import KBestFeatureSelector


class MCFS(KBestFeatureSelector):
    """
    Multi-Class Feature selection algorithm.

    Uses k-NN graph of samples in dataset and Spectral Graph Theory to find the most explaining features.

    Based on the article `"Unsupervised feature selection for multi-cluster data." <https://dl.acm.org/citation.cfm?id=1835848>`_.

    Algorithm selects features in the following way:

        1. Computes k-NN graph for the dataset.
        2. Computes heat matrix for this graph, degree and laplacian matrix.
        3. Solves eigen-problem: L y = `lambda` D y, selects K smallest eigen-values and corresponding vectors.
        4. Solves K regression problems, trying to predict every eigenvector by regression using dataset.
        5. Computes score of each feature using found regression coefficients.
        6. Select k features with the top scores.

    Parameters
    ----------
    k: int
        Number of features to select.
    clusters: int
        Expected number of datasets.
    p: int (default 8)
        Number of nearest neighbours for construction of k-NN graph.
    sigma: int (default 1)
        Coefficient for computation of heat matrix.
    mode: 'default' or 'lasso' (default 'default')
        Type of penalty for the method: with 'default' algorithm uses no penalty, with 'lasso' it uses L1-penalty.
    alpha: float (default 0.01)
        Importance of penalty for algorithm with **mode='lasso'**.
    """

    def __init__(self, k, clusters, p=8, sigma=1, mode='default', alpha=0.01):
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
        # Construct the heat matrix
        w = np.zeros([x.shape[0], x.shape[0]])
        rows, cols = graph.nonzero()
        for i, j in zip(rows, cols):
            w[i, j] = math.exp(-np.linalg.norm(x[i] - x[j])**2/self.sigma)

        # Compute degree and Laplacian matrices
        degree_vector = np.sum(w, 1)
        degree = np.diag(degree_vector)
        laplacian = degree - w

        # Solve the eigen-problem
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
