from .AlgorithmTest import AlgorithmTest, DatasetType
from fsfc.text.FTC import FTC
from sklearn.feature_extraction.text import TfidfVectorizer


MIN_SPAN = 0.06


class FTCTest(AlgorithmTest):
    def create_selector(self, dataset_size):
        return FTC(MIN_SPAN)

    def get_dataset_type(self):
        return DatasetType.TEXT

    def prepare_dataset(self, dataset):
        return TfidfVectorizer().fit_transform(dataset[:50])

    def _check_consistent(self, selector, dataset):
        # Dataset is too small for consistency testing
        pass



