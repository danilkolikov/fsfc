import unittest
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import silhouette_score, calinski_harabaz_score
from abc import abstractmethod

# Uses datasets from downloaded from http://cs.uef.fi/sipu/datasets/
DATASETS_FOLDER = 'fsfc/__test__/datasets/'
DATASETS = {
    '32': 'dim032.txt',
    '64': 'dim064.txt',
    '128': 'dim128.txt',
    '256': 'dim256.txt',
    '512': 'dim512.txt',
    '1024': 'dim1024.txt',
}


class AlgorithmTest(unittest.TestCase):

    @abstractmethod
    def create_selector(self):
        """
        :rtype: Selector
        """
        pass

    def setUp(self):
        print('Starting: ' + self.__class__.__name__)

    def test_selector(self):
        print('Testing that selector works on all datasets...')
        for dataset_name in DATASETS.keys():
            print('Testing on dataset with ' + dataset_name + ' dimensions')
            dataset = AlgorithmTest._load_dataset(dataset_name)
            selector = self.create_selector()
            selector.fit(dataset)

            self._check_dataset_transformation(selector, dataset)
            self._check_consistent(selector, dataset)
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

        selector = self.create_selector()
        selector.fit(dataset[:, permutation])
        new_mask = selector._get_support_mask()[reverse]
        self.assertTrue(
            np.array_equal(mask, new_mask),
            "Algorithm is inconsistent - different features are selected after shuffle"
        )

    def _check_silhouette(self, dataset, transformed):
        expected = KMeans().fit_predict(dataset)
        got = KMeans().fit_predict(transformed)

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
    def _load_dataset(name):
        location = DATASETS[name]
        with open(DATASETS_FOLDER + location) as f:
            dataset = f.readlines()
            data = [[int(s) for s in line.split()] for line in dataset]
            return np.array(data)