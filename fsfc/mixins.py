from abc import abstractmethod

import numpy as np
from sklearn.exceptions import NotFittedError


class ScoreSelectorMixin:
    """
    Mixin that adds getter for calculation of scores of features and checks that scores are calculated
    """

    @abstractmethod
    def _get_scores(self):
        """
        Get calculated scores for features

        Returns
        ------
        scores: ndarray
            Numpy Array with length equal to the initial number of features in dataset. Every element
            of the array is the score of this feature
        """
        pass

    def _check_scores_set(self):
        """
        Checks that scores for features are computed

        Returns
        -------
        check: bool
            Flag that indicates that scores are computed
        """

        return self._get_scores() is not None


class ThresholdSelectorMixin(ScoreSelectorMixin):
    """
    Mixin that selects features according to some threshold. That means that
    all features whose score is higher than threshold are selected
    """

    @abstractmethod
    def _get_threshold(self):
        """
        Get the value of threshold

        Returns
        -------
        threshold: float
            The value of threshold
        """
        pass

    def _get_support_mask(self):
        if not self._check_scores_set():
            raise NotFittedError('Feature Selector is not fitted')
        return self._get_scores() > self._get_threshold()


class KBestSelectorMixin(ScoreSelectorMixin):
    """
    Mixin that selects K best features according to their scores
    """

    @abstractmethod
    def _get_k(self):
        """
        Get number of features to select

        Returns
        -------
        k: int
            Number of features to select
        """
        pass

    def _get_support_mask(self):
        if not self._check_scores_set():
            raise NotFittedError('Feature Selector is not fitted')
        mask = np.zeros(self._get_scores().shape, dtype=bool)
        mask[np.argsort(self._get_scores(), kind="mergesort")[-self._get_k():]] = 1
        return mask
