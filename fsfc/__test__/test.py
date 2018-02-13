import unittest

class DummyTest(unittest.TestCase):
    def test_main(self):
        print('Hello, World!')
        self.assertEqual(2 + 2, 4)
