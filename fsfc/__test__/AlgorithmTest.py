import unittest
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import silhouette_score, calinski_harabaz_score
from abc import abstractmethod
from enum import Enum

DATASETS_FOLDER = 'fsfc/__test__/datasets/'

# Uses highly-dimensional points datasets downloaded from http://cs.uef.fi/sipu/datasets/
DATASETS = {
    '32': 'dim032.txt',
    '64': 'dim064.txt',
    '128': 'dim128.txt',
    '256': 'dim256.txt',
    '512': 'dim512.txt',
    '1024': 'dim1024.txt',
}

# Uses text datasets downloaded from https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
TEXT_DATASETS = {
    'SMS': 'SMSSpamCollection',
}


class DatasetType(Enum):
    POINTS = 1
    TEXT = 2


class AlgorithmTest(unittest.TestCase):

    @abstractmethod
    def create_selector(self, dataset_size):
        """
        :rtype: Selector
        """
        pass

    def get_dataset_type(self):
        return DatasetType.POINTS

    def setUp(self):
        print('Starting: ' + self.__class__.__name__)

    def prepare_dataset(self, dataset):
        return dataset

    def test_selector(self):
        print('Testing that selector works on all datasets...')
        for dataset_name, dataset in AlgorithmTest.load_datasets(self.get_dataset_type()):
            print('Testing on dataset: ' + dataset_name)
            prepared = self.prepare_dataset(dataset)
            selector = self.create_selector(prepared.shape[0])
            selector.fit(prepared)

            self._check_dataset_transformation(selector, prepared)
            self._check_consistent(selector, prepared)
            print('ok')
    print()

    def _check_dataset_transformation(self, selector, dataset):
        print('Checking dataset transformation...')
        transformed = selector.transform(dataset)
        self.assertEqual(
            transformed.shape[0],
            dataset.shape[0],
            "Some samples were missed"
        )
        self.assertLess(
            transformed.shape[1],
            dataset.shape[1],
            "Dimension wasn't reduced"
        )
        self._check_silhouette(dataset, transformed)

    def _check_consistent(self, selector, dataset):
        print('Checking consistency of feature selection...')
        mask = selector._get_support_mask()
        # Shuffle and reselect features
        permutation = np.random.permutation(dataset.shape[1])
        reverse = np.zeros(permutation.size, permutation.dtype)
        reverse[permutation] = range(permutation.size)

        selector = self.create_selector(dataset.shape[0])
        selector.fit(dataset[:, permutation])
        new_mask = selector._get_support_mask()[reverse]
        self.assertTrue(
            np.array_equal(mask, new_mask),
            "Algorithm is inconsistent - different features are selected after shuffle"
        )

    def _check_silhouette(self, dataset, transformed):
        expected = KMeans().fit_predict(dataset)
        got = KMeans().fit_predict(transformed)

        if type(dataset) is not np.ndarray:
            dataset = dataset.toarray()
        if type(expected) is not np.ndarray:
            expected = expected.toarray()
        if type(got) is not np.ndarray:
            got = got.toarray()

        print(
            "Silhouette Index: expected:",
            silhouette_score(dataset, expected),
            "got:",
            silhouette_score(dataset, got)
        )
        print(
            "Calinski-Harabaz Index: expected:",
            calinski_harabaz_score(dataset, expected),
            "got:",
            calinski_harabaz_score(dataset, got)
        )

    @staticmethod
    def load_datasets(dataset_type):
        if dataset_type is DatasetType.POINTS:
            for name, location in DATASETS.items():
                with open(DATASETS_FOLDER + location) as f:
                    dataset = f.readlines()
                    data = [[int(s) for s in line.split()] for line in dataset]
                    yield name + ' dimensions', np.array(data)
        if dataset_type is DatasetType.TEXT:
            for name, location in TEXT_DATASETS.items():
                with open(DATASETS_FOLDER + location) as f:
                    dataset = f.readlines()
                    # First word of each line is 'spam' or 'ham'
                    data = [line.split('\t', 1)[1] for line in dataset]
                    yield name, data
