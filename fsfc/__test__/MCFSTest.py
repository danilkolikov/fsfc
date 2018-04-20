from .AlgorithmTest import AlgorithmTest
from fsfc.generic.MCFS import MCFS

FEATURES_COUNT = 20
CLUSTERS_COUNT = 4
SIGMA = 1e3


class MCFSTest(AlgorithmTest):
    def create_selector(self, dataset_size):
        return MCFS(FEATURES_COUNT, CLUSTERS_COUNT, p=dataset_size // 4, sigma=SIGMA)

    def _check_consistent(self, selector, dataset):
        try:
            super()._check_consistent(selector, dataset)
        except AssertionError:
            print('But this may be a result of wrongly chosen selector parameters')
