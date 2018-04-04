from .AlgorithmTest import AlgorithmTest
from ..sparse import Lasso

FEATURES_COUNT = 20
CLUSTERS_COUNT = 5


class LassoTest(AlgorithmTest):
    def create_selector(self):
        return Lasso(FEATURES_COUNT, CLUSTERS_COUNT)

    def _check_consistent(self, selector, dataset):
        # Selector uses random distribution, so it selects different features on each iteration
        pass
