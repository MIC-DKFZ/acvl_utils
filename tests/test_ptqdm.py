import unittest
from acvl_utils.miscellaneous.ptqdm import ptqdm


class TestPtqdm(unittest.TestCase):
    def test_single_iterable_single_argument(self, iterations=30, processes=10):
        y = list(range(0, iterations))
        y_hat = ptqdm(method, range(0, iterations), processes)
        self.assertEqual(y_hat, y)

    def test_single_iterable_multiple_arguments(self, iterations=30, processes=10, j=5):
        y = list(range(0, iterations))
        y = [element * j for element in y]
        y_hat = ptqdm(method, range(0, iterations), processes, j=j)
        self.assertEqual(y_hat, y)

    def test_multiple_iterables_single_argument(self, iterations=30, processes=10):
        x_1 = range(0, iterations)
        x_2 = range(1, iterations+1)
        y = [x_1[i] * x_2[i] for i in range(len(x_1))]
        y_hat = ptqdm(method, (x_1, x_2), processes, zipped=True)
        self.assertEqual(y_hat, y)


def method(i, j=None):
    if j is None:
        return i
    else:
        return i * j
