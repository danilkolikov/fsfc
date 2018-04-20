from .AlgorithmTest import AlgorithmTest
from fsfc.generic.WKMeans import WKMeans

CLUSTERS_COUNT = 4
BETA = 1


class WKMeansTest(AlgorithmTest):
    def create_selector(self, dataset_size):
        return WKMeans(CLUSTERS_COUNT, BETA)

    def _check_consistent(self, selector, dataset):
        # Uses EM-algorithm that is random by design
        pass

