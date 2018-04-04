from abc import abstractmethod

from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin
from sklearn.base import ClusterMixin

from fsfc.mixins import KBestSelectorMixin


class BaseFeatureSelector(BaseEstimator, SelectorMixin):
    """
    Base class for all feature selection algorithms. It's SKLearn-compliant, so it can be used in pipelines

    Successors should override methods *fit(x)*  and *_get_support_mask()*
    """
    @abstractmethod
    def fit(self, x, *rest):
        pass


class KBestFeatureSelector(KBestSelectorMixin, BaseFeatureSelector):
    """
    Base class for algorithms that selects K best features according to some score.

    Successors should override method `_calc_scores(x)` that computes score
    """
    def __init__(self, k):
        self.k = k
        self.scores = None

    @abstractmethod
    def _calc_scores(self, x):
        pass

    def fit(self, x, *rest):
        self.scores = self._calc_scores(x)

    def _get_k(self):
        return self.k

    def _get_scores(self):
        return self.scores


class ClusteringFeatureSelector(KBestSelectorMixin, BaseFeatureSelector, ClusterMixin):
    """
    Clusters samples and simultaneously finds relevant features
    """

    def __init__(self, k):
        self.k = k
        self.scores = None
        self.labels_ = None

    @abstractmethod
    def _calc_scores_and_labels(self, x):
        """
        Calculate scores and labels for samples

        :rtype: tuple
        """
        pass

    def fit(self, x, *rest):
        scores, labels = self._calc_scores_and_labels(x)
        self.scores = scores
        self.labels_ = labels

    def _get_scores(self):
        return self.scores

    def _get_k(self):
        return self.k