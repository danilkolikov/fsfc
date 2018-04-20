from .AlgorithmTest import AlgorithmTest, DatasetType
from fsfc.text.CHIR import CHIR
from sklearn.feature_extraction.text import TfidfVectorizer


FEATURES_COUNT = 4
CLUSTERS_COUNT = 4
ALPHA = 0.1


class CHIRTest(AlgorithmTest):
    def create_selector(self, dataset_size):
        return CHIR(FEATURES_COUNT, CLUSTERS_COUNT, ALPHA)

    def get_dataset_type(self):
        return DatasetType.TEXT

    def prepare_dataset(self, dataset):
        return TfidfVectorizer().fit_transform(dataset[:1000])

    def _check_consistent(self, selector, dataset):
        # Uses EM-algorithm that is random by design
        pass



