import unittest

from acvl_utils.cropping_and_padding.bounding_boxes import pad_bbox


class TestPadBBox(unittest.TestCase):

    def test_no_array_shape(self):
        bbox = [[2, 10], [5, 15]]
        pad_amount = 2
        expected = [[0, 12], [3, 17]]
        result = pad_bbox(bbox, pad_amount)
        self.assertEqual(result, expected)

    def test_with_array_shape(self):
        bbox = [[2, 10], [5, 15]]
        pad_amount = 2
        array_shape = (12, 20)
        expected = [[0, 12], [3, 17]]
        result = pad_bbox(bbox, pad_amount, array_shape)
        self.assertEqual(result, expected)

    def test_dimension_specific_padding(self):
        bbox = [[2, 10], [5, 15]]
        pad_amount = [3, 1]
        expected = [[0, 13], [4, 16]]
        result = pad_bbox(bbox, pad_amount)
        self.assertEqual(result, expected)

    def test_clipping_with_array_shape(self):
        bbox = [[2, 10], [5, 15]]
        pad_amount = 5
        array_shape = (12, 18)
        expected = [[0, 12], [0, 18]]
        result = pad_bbox(bbox, pad_amount, array_shape)
        self.assertEqual(result, expected)

    def test_no_padding(self):
        bbox = [[2, 10], [5, 15]]
        pad_amount = 0
        expected = [[2, 10], [5, 15]]
        result = pad_bbox(bbox, pad_amount)
        self.assertEqual(result, expected)

if __name__ == "__main__":
    unittest.main()