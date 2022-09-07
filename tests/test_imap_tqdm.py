import unittest
from acvl_utils.miscellaneous.imap_tqdm import imap_tqdm


class TestImapTqdm(unittest.TestCase):
    def test_single_argument(self, iterations=30, processes=10):
        y = list(range(0, iterations))
        y_hat = imap_tqdm(method, range(0, iterations), processes)
        self.assertEqual(y_hat, y)

    def test_multiple_arguments(self, iterations=30, processes=10, j=5):
        y = list(range(0, iterations))
        y = [element * j for element in y]
        y_hat = imap_tqdm(method, range(0, iterations), processes, j=j)
        self.assertEqual(y_hat, y)


def method(i, j=None):
    if j is None:
        return i
    else:
        return i * j