from abc import abstractmethod

import numpy as np
from sklearn.exceptions import NotFittedError


class ScoreSelectorMixin:
    @abstractmethod
    def _get_scores(self):
        """
        Get calculated scores for features

        :rtype: ndarray
        """
        pass

    def _check_scores_set(self):
        return self._get_scores() is not None


class ThresholdSelectorMixin(ScoreSelectorMixin):

    @abstractmethod
    def _get_threshold(self):
        pass

    def _get_support_mask(self):
        if not self._check_scores_set():
            raise NotFittedError('Feature Selector is not fitted')
        return self._get_scores() > self._get_threshold()


class KBestSelectorMixin(ScoreSelectorMixin):

    @abstractmethod
    def _get_k(self):
        pass

    def _get_support_mask(self):
        if not self._check_scores_set():
            raise NotFittedError('Feature Selector is not fitted')
        mask = np.zeros(self._get_scores().shape, dtype=bool)
        mask[np.argsort(self._get_scores(), kind="mergesort")[-self._get_k():]] = 1
        return mask
