from abc import abstractmethod

from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin
from sklearn.exceptions import NotFittedError
import numpy as np


class BaseFeatureSelector(BaseEstimator, SelectorMixin):
    """
    Base class for all feature selection algorithms. It's SKLearn-compliant, so it can be used in pipelines

    Successors should override methods *fit(x)*  and *_get_support_mask()*
    """
    @abstractmethod
    def fit(self, x, *rest):
        pass


class KBestFeatureSelector(BaseFeatureSelector):
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

    def _check_scores_set(self):
        return self.scores is not None

    def _get_support_mask(self):
        if not self._check_scores_set():
            raise NotFittedError('Feature Selector is not fitted')
        mask = np.zeros(self.scores.shape, dtype=bool)
        mask[np.argsort(self.scores, kind="mergesort")[-self.k:]] = 1
        return mask
