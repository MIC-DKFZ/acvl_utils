from acvl_utils.array_manipulation.slicer import slicer
import numpy as np
import copy
import unittest


class TestSlicer(unittest.TestCase):
    def test_ellipsis(self):
        arr = np.random.rand(10, 10, 10)
        sub_arr1 = arr[...]
        sub_arr2 = arr[slicer(arr, [None])]
        self.assertTrue(np.array_equal(sub_arr1, sub_arr2))

        arr = np.random.rand(10, 10, 10)
        sub_arr1 = arr[..., 0, 0]
        sub_arr2 = arr[slicer(arr, [None, 0, 0])]
        self.assertTrue(np.array_equal(sub_arr1, sub_arr2))

        arr = np.random.rand(10, 10, 10)
        sub_arr1 = arr[..., 0]
        sub_arr2 = arr[slicer(arr, [None, 0])]
        self.assertTrue(np.array_equal(sub_arr1, sub_arr2))

        arr = np.random.rand(10, 10, 10)
        sub_arr1 = arr[0, ...]
        sub_arr2 = arr[slicer(arr, [0, None])]
        self.assertTrue(np.array_equal(sub_arr1, sub_arr2))

        arr = np.random.rand(10, 10, 10)
        sub_arr1 = arr[..., 0, 0]
        sub_arr2 = arr[slicer(arr, [None, 0, 0])]
        self.assertTrue(np.array_equal(sub_arr1, sub_arr2))

    def test_colon(self):
        arr = np.random.rand(10, 10, 10)
        sub_arr1 = arr[:]
        sub_arr2 = arr[slicer(arr, [[None]])]
        self.assertTrue(np.array_equal(sub_arr1, sub_arr2))

        arr = np.random.rand(10, 10, 10)
        sub_arr1 = arr[:, 0]
        sub_arr2 = arr[slicer(arr, [[None], 0])]
        self.assertTrue(np.array_equal(sub_arr1, sub_arr2))

        arr = np.random.rand(10, 10, 10)
        sub_arr1 = arr[:, 0, 0]
        sub_arr2 = arr[slicer(arr, [[None], 0, 0])]
        self.assertTrue(np.array_equal(sub_arr1, sub_arr2))

    def test_single(self):
        arr = np.random.rand(10, 10, 10)
        sub_arr1 = arr[0]
        sub_arr2 = arr[slicer(arr, [0])]
        self.assertTrue(np.array_equal(sub_arr1, sub_arr2))

        arr = np.random.rand(10, 10, 10)
        sub_arr1 = arr[0, 0, 0]
        sub_arr2 = arr[slicer(arr, [0, 0, 0])]
        self.assertTrue(np.array_equal(sub_arr1, sub_arr2))

    def test_range(self):
        arr = np.random.rand(10, 10, 10)
        sub_arr1 = arr[5:8, 3:4, 1:7]
        sub_arr2 = arr[slicer(arr, [[5, 8], [3, 4], [1, 7]])]
        self.assertTrue(np.array_equal(sub_arr1, sub_arr2))

        arr = np.random.rand(10, 10, 10)
        sub_arr1 = arr[5:-1, -7:8, -1:-1]
        sub_arr2 = arr[slicer(arr, [[5, -1], [-7, 8], [-1, -1]])]
        self.assertTrue(np.array_equal(sub_arr1, sub_arr2))

        arr = np.random.rand(10, 10, 10)
        sub_arr1 = arr[5:, :4, :]
        sub_arr2 = arr[slicer(arr, [[5, None], [None, 4], [None]])]
        self.assertTrue(np.array_equal(sub_arr1, sub_arr2))

    def test_assign(self):
        arr = np.random.rand(10, 10, 10)
        arr1 = copy.deepcopy(arr)
        arr2 = copy.deepcopy(arr)
        arr1[1, 2, 3] = 7
        arr2[slicer(arr2, [1, 2, 3])] = 7
        self.assertTrue(np.array_equal(arr1, arr2))

        arr = np.random.rand(10, 10, 10)
        arr1 = copy.deepcopy(arr)
        arr2 = copy.deepcopy(arr)
        arr1[5] = 7
        arr2[slicer(arr2, [5])] = 7
        self.assertTrue(np.array_equal(arr1, arr2))

        arr = np.random.rand(10, 10, 10)
        sub_arr = np.random.rand(3, 3, 3)
        arr1 = copy.deepcopy(arr)
        arr2 = copy.deepcopy(arr)
        arr1[3:6, 1:4, 5:8] = sub_arr
        arr2[slicer(arr2, [[3, 6], [1, 4], [5, 8]])] = sub_arr
        self.assertTrue(np.array_equal(arr1, arr2))
