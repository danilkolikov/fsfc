from abc import abstractmethod

from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin
from sklearn.base import ClusterMixin

from fsfc.mixins import KBestSelectorMixin


class BaseFeatureSelector(BaseEstimator, SelectorMixin):
    """
    Base class for all feature selection algorithms. It's SKLearn-compliant, so it can be used in pipelines.

    Successors should override methods :meth:`fit`  and :meth:`_get_support_mask`.
    """

    @abstractmethod
    def fit(self, x, *rest):
        """
        Fit selector to a dataset.

        Parameters
        ----------
        x: ndarray
            The input samples - array of shape [n_samples, n_features].
        rest: list
            List of miscellaneous arguments.

        Returns
        -------
        selector: BaseFeatureSelector
            Returns self to support chaining.
        """

        return self


class KBestFeatureSelector(KBestSelectorMixin, BaseFeatureSelector):
    """
    Base class for algorithms that selects K best features according to features scores.

    Successors should override method :meth:`_calc_scores` for computation of the score.

    Parameters
    ----------
    k: int
        Number of features to select
    """

    def __init__(self, k):
        self.k = k
        self.scores = None

    @abstractmethod
    def _calc_scores(self, x):
        """
        Calculate scores for features in dataset.

        Parameters
        ----------
        x: ndarray
            The input samples - array of shape [n_samples, n_features].

        Returns
        -------
        scores: ndarray
            Array of shape [n_features]. i-th element is a score of i-th feature.
        """
        pass

    def fit(self, x, *rest):
        self.scores = self._calc_scores(x)
        return self

    def _get_k(self):
        return self.k

    def _get_scores(self):
        return self.scores


class ClusteringFeatureSelector(KBestSelectorMixin, BaseFeatureSelector, ClusterMixin):
    """
    Clusters samples and simultaneously finds relevant features. Allows to transform dataset
    and select K best features according to features scores.

    Parameters
    ----------
    k: int
        Number of features to select
    """

    def __init__(self, k):
        self.k = k
        self.scores = None
        self.labels_ = None

    @abstractmethod
    def _calc_scores_and_labels(self, x):
        """
        Calculate scores and labels for samples.

        Parameters
        ----------
        x: ndarray
            The input samples - array of shape [n_samples, n_features].

        Returns
        -------
        scores_and_labels: tuple
            Tuple where first element is scores of features and second element are cluster labels for samples
        """
        pass

    def fit(self, x, *rest):
        scores, labels = self._calc_scores_and_labels(x)
        self.scores = scores
        self.labels_ = labels
        return self

    def _get_scores(self):
        return self.scores

    def _get_k(self):
        return self.k
