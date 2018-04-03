from .AlgorithmTest import AlgorithmTest
from ..SPEC import NormalizedCut, ArbitraryClustering, FixedClustering

FEATURES_COUNT = 20
CLUSTERS_COUNT = 5


class NormalizedCutTest(AlgorithmTest):
    def create_selector(self):
        return NormalizedCut(FEATURES_COUNT)


class ArbitraryClusteringTest(AlgorithmTest):
    def create_selector(self):
        return ArbitraryClustering(FEATURES_COUNT)


class FixedClusteringTest(AlgorithmTest):
    def create_selector(self):
        return FixedClustering(FEATURES_COUNT, CLUSTERS_COUNT)