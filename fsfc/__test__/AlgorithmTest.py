import unittest
import numpy as np
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
            mask = selector._get_support_mask()

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
    print()

    @staticmethod
    def _load_dataset(name):
        location = DATASETS[name]
        with open(DATASETS_FOLDER + location) as f:
            dataset = f.readlines()
            data = [[int(s) for s in line.split()] for line in dataset]
            return np.array(data)