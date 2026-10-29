import tracemalloc
import unittest

import blosc2
import numpy as np
import torch

from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd


class TestCropAndPadND(unittest.TestCase):

    def setUp(self):
        # Set up sample images in both numpy and torch formats
        self.image_np = np.random.randn(5, 10, 10, 10)  # 4D numpy array with shape (5, 10, 10, 10)
        self.image_torch = torch.tensor(self.image_np)  # Convert to torch.Tensor for testing

    def test_crop_within_bounds_np(self):
        """Test cropping a bounding box fully within the bounds of an np.ndarray, including upper bound."""
        bbox = [[3, 8], [1, 6], [5, 10]]

        cropped_patch = crop_and_pad_nd(self.image_np, bbox)

        # Expected output shape should maintain initial dimensions, affecting only the last 3 dimensions
        self.assertEqual(cropped_patch.shape, (5, 5, 5, 5))

        # Verify content by slicing only the last 3 dimensions as per bounding box
        expected_patch = self.image_np[:, 3:8, 1:6, 5:10]
        np.testing.assert_array_equal(cropped_patch, expected_patch)

    def test_crop_within_bounds_torch(self):
        """Test cropping a bounding box fully within the bounds of a torch.Tensor, including upper bound."""
        bbox = [[3, 8], [1, 6], [5, 10]]

        cropped_patch = crop_and_pad_nd(self.image_torch, bbox)

        # Expected output shape should maintain initial dimensions, affecting only the last 3 dimensions
        self.assertEqual(cropped_patch.shape, (5, 5, 5, 5))

        # Verify content by slicing only the last 3 dimensions as per bounding box
        expected_patch = self.image_torch[:, 3:8, 1:6, 5:10]
        torch.testing.assert_close(cropped_patch, expected_patch)

    def test_crop_with_padding_np(self):
        """Test cropping a bounding box that extends beyond bounds of np.ndarray and requires padding, including upper bound."""
        bbox = [[-2, 3], [-4, 3], [-3, 3]]

        cropped_patch = crop_and_pad_nd(self.image_np, bbox)

        # Expected shape includes initial dimension, with padding extending to (5, 5, 5)
        self.assertEqual(cropped_patch.shape, (5, 5, 7, 6))

        # Check if padding values (0) are correctly applied at the boundaries
        self.assertTrue(np.all(cropped_patch[:, :2, :, :] == 0))
        self.assertTrue(np.all(cropped_patch[:, :, :4, :] == 0))
        self.assertTrue(np.all(cropped_patch[:, :, :, :3] == 0))

    def test_crop_with_padding_torch(self):
        """Test cropping a bounding box that extends beyond bounds of torch.Tensor and requires padding, including upper bound."""
        bbox = [[-2, 3], [-4, 3], [-3, 3]]

        cropped_patch = crop_and_pad_nd(self.image_torch, bbox)

        # Expected shape includes initial dimension, with padding extending to (5, 5, 5)
        self.assertEqual(cropped_patch.shape, (5, 5, 7, 6))

        # Check if padding values (0) are correctly applied at the boundaries
        self.assertTrue(torch.all(cropped_patch[:, :2, :, :] == 0))
        self.assertTrue(torch.all(cropped_patch[:, :, :4, :] == 0))
        self.assertTrue(torch.all(cropped_patch[:, :, :, :3] == 0))

    def test_nd_support_with_less_crop_dims_np(self):
        """Test cropping a 4D np.ndarray with bounding box applied only to the last 3 dimensions, including upper bound."""
        image_np = np.random.randn(3, 10, 10, 10)
        bbox = [[1, 6], [3, 10]]

        cropped_patch = crop_and_pad_nd(image_np, bbox)

        # Expected output shape includes unaffected initial dimension
        self.assertEqual(cropped_patch.shape, (3, 10, 5, 7))

    def test_nd_support_with_less_crop_dims_torch(self):
        """Test cropping a 4D torch.Tensor with bounding box applied only to the last 3 dimensions, including upper bound."""
        image_torch = torch.randn(3, 10, 10, 10)
        bbox = [[1, 6], [3, 10]]

        cropped_patch = crop_and_pad_nd(image_torch, bbox)

        # Expected output shape includes unaffected initial dimension
        self.assertEqual(cropped_patch.shape, (3, 10, 5, 7))

    def test_return_type_np(self):
        """Check that np.ndarray input returns an np.ndarray output."""
        bbox = [[3, 8], [3, 8], [3, 8]]

        cropped_patch = crop_and_pad_nd(self.image_np, bbox)
        self.assertIsInstance(cropped_patch, np.ndarray)

    def test_return_type_torch(self):
        """Check that torch.Tensor input returns a torch.Tensor output."""
        bbox = [[3, 8], [3, 8], [3, 8]]

        cropped_patch = crop_and_pad_nd(self.image_torch, bbox)
        self.assertIsInstance(cropped_patch, torch.Tensor)

    def test_outside_negative_bounds(self):
        """Test a bounding box that is entirely outside the image on the negative side."""
        bbox = [[-15, -9], [-25, -15], [-20, -11]]
        cropped_patch = crop_and_pad_nd(self.image_np, bbox)

        # Expected shape is fully padded with zeros, matching the bbox size.
        self.assertEqual(cropped_patch.shape, (5, 6, 10, 9))
        self.assertTrue(np.all(cropped_patch == 0), "Expected fully padded array of zeros")

    def test_outside_positive_bounds(self):
        """Test a bounding box that is entirely outside the image on the positive side."""
        bbox = [[17, 26], [18, 20], [21, 22]]
        cropped_patch = crop_and_pad_nd(self.image_np, bbox)

        # Expected shape is fully padded with zeros, matching the bbox size.
        self.assertEqual(cropped_patch.shape, (5, 9, 2, 1))
        self.assertTrue(np.all(cropped_patch == 0), "Expected fully padded array of zeros")

    def test_at_dimension_end(self):
        """Test a bounding box where min_val = max_val = img_shape[i] (at the end of the dimension)."""
        bbox = [[10, 11], [10, 11], [10, 11]]
        cropped_patch = crop_and_pad_nd(self.image_np, bbox)

        # Expected shape is (1, 1, 1) with only padding, so it should be all zeros.
        self.assertEqual(cropped_patch.shape, (5, 1, 1, 1))
        self.assertTrue(np.all(cropped_patch == 0), "Expected fully padded array of zeros")

    def test_at_dimension_end2(self):
        """Test a bounding box where min_val = max_val = img_shape[i] (at the end of the dimension)."""
        bbox = [[9, 10], [9, 10], [9, 10]]
        cropped_patch = crop_and_pad_nd(self.image_np, bbox)

        # Expected shape is (1, 1, 1) with only padding, so it should be all zeros.
        self.assertEqual(cropped_patch.shape, (5, 1, 1, 1))
        self.assertTrue(np.all(cropped_patch == self.image_np[:, 9:10, 9:10, 9:10]), "Expected fully padded array of zeros")

    def test_at_dimension_end_torch(self):
        """Test a bounding box where min_val = max_val = img_shape[i] (at the end of the dimension)."""
        bbox = [[10, 11], [10, 11], [10, 11]]
        cropped_patch = crop_and_pad_nd(self.image_torch, bbox)

        # Expected shape is (1, 1, 1) with only padding, so it should be all zeros.
        self.assertEqual(cropped_patch.shape, (5, 1, 1, 1))
        self.assertTrue(torch.all(cropped_patch == 0), "Expected fully padded array of zeros")

    def test_at_dimension_end2_torch(self):
        """Test a bounding box where min_val = max_val = img_shape[i] (at the end of the dimension)."""
        bbox = [[9, 10], [9, 10], [9, 10]]
        cropped_patch = crop_and_pad_nd(self.image_torch, bbox)

        # Expected shape is (1, 1, 1) with only padding, so it should be all zeros.
        self.assertEqual(cropped_patch.shape, (5, 1, 1, 1))
        self.assertTrue(torch.all(cropped_patch == self.image_torch[:, 9:10, 9:10, 9:10]), "Expected fully padded array of zeros")

    def test_max_val_zero(self):
        """Test a bounding box where max_val = 0, ending at the start of the dimension."""
        bbox = [[-2, 1], [-3, 9], [-5, 4]]
        cropped_patch = crop_and_pad_nd(self.image_np, bbox)

        # Expected shape is (3, 3, 3), with padding at the start of each dimension.
        self.assertEqual(cropped_patch.shape, (5, 3, 12, 9))

        # Check that the boundary areas are correctly padded with zeros.
        self.assertTrue(np.all(cropped_patch[:, :2, :, :] == 0), "Expected padding at the start of first dimension")
        self.assertTrue(np.all(cropped_patch[:, :, :3, :] == 0), "Expected padding at the start of second dimension")
        self.assertTrue(np.all(cropped_patch[:, :, :, :5] == 0), "Expected padding at the start of third dimension")

    def test_mixed_bounds_partial_padding(self):
        """Test a bounding box that partially overlaps the image, requiring partial padding."""
        bbox = [[-5, 6], [-5, 6], [-5, 6]]
        cropped_patch = crop_and_pad_nd(self.image_np, bbox)

        # Expected shape is (11, 11, 11) since we have a range of -5 to 5 in each dimension.
        self.assertEqual(cropped_patch.shape, (5, 11, 11, 11))

        # Verify that padding is applied correctly on one side and data on the other.
        self.assertTrue(np.all(cropped_patch[:, :5, :, :] == 0), "Expected padding in first dimension")
        self.assertTrue(np.all(cropped_patch[:, :, :5, :] == 0), "Expected padding in second dimension")
        self.assertTrue(np.all(cropped_patch[:, :, :, :5] == 0), "Expected padding in third dimension")
        self.assertTrue(np.all(cropped_patch[:, 5:, 5:, 5:] == self.image_np[:, :6, :6, :6]),
                        "Expected valid data from image")

    def test_bbox_entire_image_npy(self):
        """Test a bounding box that exactly matches the full image dimensions."""
        bbox = [[0, 10], [0, 10], [0, 10]]
        cropped_patch = crop_and_pad_nd(self.image_np, bbox)

        # The output should be the same as the original image
        self.assertEqual(cropped_patch.shape, self.image_np.shape)
        np.testing.assert_array_equal(cropped_patch, self.image_np)

    def test_bbox_entire_image_torch(self):
        """Test a bounding box that exactly matches the full image dimensions."""
        bbox = [[0, 10], [0, 10], [0, 10]]
        cropped_patch = crop_and_pad_nd(self.image_torch, bbox)

        # The output should be the same as the original image
        self.assertEqual(cropped_patch.shape, self.image_torch.shape)
        self.assertTrue(torch.all(cropped_patch == self.image_torch))

    def test_empty_bbox_npy(self):
        """Test a bounding box where min_val == max_val in each dimension, resulting in a single voxel."""
        bbox = [[3, 4], [5, 6], [7, 8]]
        cropped_patch = crop_and_pad_nd(self.image_np, bbox)

        # Expected shape is (1, 1, 1) with the single voxel's value
        self.assertEqual(cropped_patch.shape, (5, 1, 1, 1))
        np.testing.assert_array_equal(cropped_patch, self.image_np[:, 3:4, 5:6, 7:8])

    def test_empty_bbox_torch(self):
        """Test a bounding box where min_val == max_val in each dimension, resulting in a single voxel."""
        bbox = [[3, 4], [5, 6], [7, 8]]
        cropped_patch = crop_and_pad_nd(self.image_torch, bbox)

        # Expected shape is (1, 1, 1) with the single voxel's value
        self.assertEqual(cropped_patch.shape, (5, 1, 1, 1))
        self.assertTrue(torch.all(cropped_patch == self.image_torch[:, 3:4, 5:6, 7:8]))

    def test_partial_dimension_crop_npy(self):
        """Test cropping only some dimensions, leaving others at full bounds."""
        bbox = [[1, 6], [0, 10], [4, 7]]
        cropped_patch = crop_and_pad_nd(self.image_np, bbox)

        # Expected shape: original first dim, cropped in last three dimensions
        self.assertEqual(cropped_patch.shape, (5, 5, 10, 3))
        np.testing.assert_array_equal(cropped_patch, self.image_np[:, 1:6, :, 4:7])

    def test_partial_dimension_crop_torch(self):
        """Test cropping only some dimensions, leaving others at full bounds."""
        bbox = [[1, 6], [0, 10], [4, 7]]
        cropped_patch = crop_and_pad_nd(self.image_torch, bbox)

        # Expected shape: original first dim, cropped in last three dimensions
        self.assertEqual(cropped_patch.shape, (5, 5, 10, 3))
        self.assertTrue(torch.all(cropped_patch ==  self.image_torch[:, 1:6, :, 4:7]))

    def test_non_cubic_bbox_npy(self):
        """Test a non-cubic bounding box to ensure rectangular patches are handled correctly."""
        bbox = [[1, 6], [2, 7], [0, 4]]
        cropped_patch = crop_and_pad_nd(self.image_np, bbox)

        # Expected shape for the cropped patch based on bbox
        self.assertEqual(cropped_patch.shape, (5, 5, 5, 4))
        np.testing.assert_array_equal(cropped_patch, self.image_np[:, 1:6, 2:7, 0:4])

    def test_non_cubic_bbox_torch(self):
        """Test a non-cubic bounding box to ensure rectangular patches are handled correctly."""
        bbox = [[1, 6], [2, 7], [0, 4]]
        cropped_patch = crop_and_pad_nd(self.image_torch, bbox)

        # Expected shape for the cropped patch based on bbox
        self.assertEqual(cropped_patch.shape, (5, 5, 5, 4))
        self.assertTrue(torch.all(cropped_patch == self.image_torch[:, 1:6, 2:7, 0:4]))

    def test_single_dimension_crop_np(self):
        """Test cropping only the last dimension of a 4D np.ndarray."""
        bbox = [[3, 8]]
        cropped_patch = crop_and_pad_nd(self.image_np, bbox)

        # Expected shape includes initial dimensions unaffected, and only the last dimension cropped
        self.assertEqual(cropped_patch.shape, (5, 10, 10, 5))
        np.testing.assert_array_equal(cropped_patch, self.image_np[:, :, :, 3:8])

    def test_two_dimensions_crop_np(self):
        """Test cropping the last two dimensions of a 4D np.ndarray."""
        bbox = [[3, 8], [1, 7]]
        cropped_patch = crop_and_pad_nd(self.image_np, bbox)

        # Expected shape includes initial dimensions unaffected, cropping last two dimensions
        self.assertEqual(cropped_patch.shape, (5, 10, 5, 6))
        np.testing.assert_array_equal(cropped_patch, self.image_np[:, :, 3:8, 1:7])

    def test_three_dimensions_crop_np(self):
        """Test cropping the last three dimensions of a 4D np.ndarray."""
        bbox = [[1, 6], [2, 9], [3, 8]]
        cropped_patch = crop_and_pad_nd(self.image_np, bbox)

        # Expected shape with last three dimensions cropped as per bbox
        self.assertEqual(cropped_patch.shape, (5, 5, 7, 5))
        np.testing.assert_array_equal(cropped_patch, self.image_np[:, 1:6, 2:9, 3:8])

    def test_all_dimensions_crop_np(self):
        """Test cropping all dimensions of a 4D np.ndarray."""
        bbox = [[1, 4], [2, 7], [0, 6], [4, 9]]
        cropped_patch = crop_and_pad_nd(self.image_np, bbox)

        # Expected shape with all dimensions cropped as per bbox
        self.assertEqual(cropped_patch.shape, (3, 5, 6, 5))
        np.testing.assert_array_equal(cropped_patch, self.image_np[1:4, 2:7, 0:6, 4:9])

    def test_single_dimension_crop_torch(self):
        """Test cropping only the last dimension of a 4D torch.Tensor."""
        bbox = [[3, 8]]
        cropped_patch = crop_and_pad_nd(self.image_torch, bbox)

        # Expected shape includes initial dimensions unaffected, and only the last dimension cropped
        self.assertEqual(cropped_patch.shape, (5, 10, 10, 5))
        torch.testing.assert_close(cropped_patch, self.image_torch[:, :, :, 3:8])

    def test_two_dimensions_crop_torch(self):
        """Test cropping the last two dimensions of a 4D torch.Tensor."""
        bbox = [[3, 8], [1, 7]]
        cropped_patch = crop_and_pad_nd(self.image_torch, bbox)

        # Expected shape includes initial dimensions unaffected, cropping last two dimensions
        self.assertEqual(cropped_patch.shape, (5, 10, 5, 6))
        torch.testing.assert_close(cropped_patch, self.image_torch[:, :, 3:8, 1:7])

    def test_three_dimensions_crop_torch(self):
        """Test cropping the last three dimensions of a 4D torch.Tensor."""
        bbox = [[1, 6], [2, 9], [3, 8]]
        cropped_patch = crop_and_pad_nd(self.image_torch, bbox)

        # Expected shape with last three dimensions cropped as per bbox
        self.assertEqual(cropped_patch.shape, (5, 5, 7, 5))
        torch.testing.assert_close(cropped_patch, self.image_torch[:, 1:6, 2:9, 3:8])

    def test_all_dimensions_crop_torch(self):
        """Test cropping all dimensions of a 4D torch.Tensor."""
        bbox = [[1, 4], [2, 7], [0, 6], [4, 9]]
        cropped_patch = crop_and_pad_nd(self.image_torch, bbox)

        # Expected shape with all dimensions cropped as per bbox
        self.assertEqual(cropped_patch.shape, (3, 5, 6, 5))
        torch.testing.assert_close(cropped_patch, self.image_torch[1:4, 2:7, 0:6, 4:9])

    def test_pad_all_dims(self):
        seg = np.ones((128, 128))
        bbox = [[-5, 133], [-5, 133]]
        out = crop_and_pad_nd(seg, bbox)
        self.assertEqual(out.shape, (138, 138))
        bbox_reverse = [[5, 133], [5, 133]]
        out2 = crop_and_pad_nd(out, bbox_reverse)
        self.assertEqual(out2.shape, (128, 128))
        self.assertTrue(np.all(out2==1))

    def test_pad_all_dims_torch(self):
        seg = torch.ones((128, 128))
        bbox = [[-5, 133], [-5, 133]]
        out = crop_and_pad_nd(seg, bbox)
        self.assertEqual(out.shape, (138, 138))
        bbox_reverse = [[5, 133], [5, 133]]
        out2 = crop_and_pad_nd(out, bbox_reverse)
        self.assertEqual(out2.shape, (128, 128))
        self.assertTrue(torch.all(out2==1))


class TestCropAndPadCastConstant(unittest.TestCase):
    """numpy and blosc2 inputs with constant padding: one output array, cast and padded in one pass."""

    def setUp(self):
        self.seg = np.random.default_rng(0).integers(-1, 4, (2, 10, 12, 14)).astype(np.int8)
        self.bbox = [[-3, 6], [4, 16], [-2, 9]]
        self.expected = np.pad(self.seg[:, 0:6, 4:12, 0:9].astype(np.int16), [(0, 0), (3, 0), (0, 4), (2, 0)],
                               constant_values=-1)

    def test_cast_and_pad_np(self):
        out = crop_and_pad_nd(self.seg, self.bbox, -1, cast_cropped_to=np.int16)
        self.assertEqual(out.dtype, np.int16)
        np.testing.assert_array_equal(out, self.expected)

    def test_cast_and_pad_blosc2(self):
        out = crop_and_pad_nd(blosc2.asarray(self.seg), self.bbox, -1, cast_cropped_to=np.int16)
        self.assertEqual(out.dtype, np.int16)
        np.testing.assert_array_equal(out, self.expected)

    def test_cast_and_pad_allocates_only_the_output_np(self):
        # casting the crop first and padding it afterwards allocated a second, cast copy of the crop
        seg = np.zeros((1, 64, 64, 64), dtype=np.int8)
        tracemalloc.start()
        out = crop_and_pad_nd(seg, [[-8, 56], [0, 64], [0, 64]], -1, cast_cropped_to=np.int16)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.assertLess(peak, 1.1 * out.nbytes)

    def test_bbox_entirely_outside_is_all_padding(self):
        out = crop_and_pad_nd(self.seg, [[20, 24], [0, 4], [3, 5]], -1, cast_cropped_to=np.int16)
        self.assertEqual(out.dtype, np.int16)
        np.testing.assert_array_equal(out, np.full((2, 4, 4, 2), -1, dtype=np.int16))
        out = crop_and_pad_nd(blosc2.asarray(self.seg), [[-9, -1], [0, 4], [3, 5]], -1, cast_cropped_to=np.int16)
        np.testing.assert_array_equal(out, np.full((2, 8, 4, 2), -1, dtype=np.int16))
        out = crop_and_pad_nd(torch.from_numpy(self.seg), [[20, 24], [0, 4], [3, 5]], -1, cast_cropped_to=torch.int16)
        self.assertEqual(out.dtype, torch.int16)
        self.assertTrue(torch.all(out == -1))

    def test_out_of_range_pad_value_wraps_like_np_pad(self):
        img = np.arange(24, dtype=np.uint8).reshape(4, 6)
        out = crop_and_pad_nd(img, [[-1, 3], [2, 8]], -1)
        np.testing.assert_array_equal(out, np.pad(img[0:3, 2:6], [(1, 0), (0, 2)], constant_values=-1))


if __name__ == "__main__":
    unittest.main()
    # t = TestCropAndPadND()
    # t.setUp()
    # t.test_outside_positive_bounds()
    # t.test_max_val_zero()
    # t.test_outside_negative_bounds()
    # t.test_mixed_bounds_partial_padding()