from .AlgorithmTest import AlgorithmTest
from fsfc.generic.SPEC import NormalizedCut, GenericSPEC, FixedSPEC

FEATURES_COUNT = 20
CLUSTERS_COUNT = 5


class NormalizedCutTest(AlgorithmTest):
    def create_selector(self, dataset_size):
        return NormalizedCut(FEATURES_COUNT)


class ArbitraryClusteringTest(AlgorithmTest):
    def create_selector(self, dataset_size):
        return GenericSPEC(FEATURES_COUNT)


class FixedClusteringTest(AlgorithmTest):
    def create_selector(self, dataset_size):
        return FixedSPEC(FEATURES_COUNT, CLUSTERS_COUNT)