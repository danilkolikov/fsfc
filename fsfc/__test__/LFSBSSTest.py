from .AlgorithmTest import AlgorithmTest
from ..LFSBSS import LFSBSS

CLUSTERS_COUNT = 5


class LFSBSSTest(AlgorithmTest):
    def create_selector(self, dataset_size):
        return LFSBSS(CLUSTERS_COUNT)

    def _check_dataset_transformation(self, selector, dataset):
        # Dataset isn't transformed
        pass

    def _check_consistent(self, selector, dataset):
        # Dataset isn't transformed
        pass

