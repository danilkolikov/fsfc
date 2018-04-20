from .AlgorithmTest import AlgorithmTest
from fsfc.generic.LFSBSS import LFSBSS

CLUSTERS_COUNT = 5


class LFSBSSTest(AlgorithmTest):
    def prepare_dataset(self, dataset):
        print('Select 50 first features, because algorithm can barely handle bigger dimensionality')
        return super().prepare_dataset(dataset)[:, :50]

    def create_selector(self, dataset_size):
        return LFSBSS(CLUSTERS_COUNT)

    def _check_dataset_transformation(self, selector, dataset):
        # Dataset isn't transformed
        pass

    def _check_consistent(self, selector, dataset):
        # Dataset isn't transformed
        pass

