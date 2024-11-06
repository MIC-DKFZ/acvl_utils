import unittest
import torch
import numpy as np
from typing import Union

from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd, insert_crop_into_image


class TestInsertCropIntoImage(unittest.TestCase):

    def setUp(self):
        # Set up sample images in both numpy and torch formats
        self.image_np = np.random.randn(5, 10, 10, 10)  # 4D numpy array with shape (5, 10, 10, 10)
        self.image_torch = torch.tensor(self.image_np)  # Convert to torch.Tensor for testing

    def test_within_bounds_reinsertion(self):
        """Test reinserting a crop that is fully within the bounds of the original image."""
        bbox = [[3, 8], [1, 6], [4, 9]]

        # Crop and then reinsert
        crop_np = crop_and_pad_nd(self.image_np, bbox)
        result_np = insert_crop_into_image(self.image_np.copy(), crop_np, bbox)
        np.testing.assert_array_almost_equal(result_np, self.image_np)

        crop_torch = crop_and_pad_nd(self.image_torch, bbox)
        result_torch = insert_crop_into_image(self.image_torch.clone(), crop_torch, bbox)
        torch.testing.assert_close(result_torch, self.image_torch)

    def test_outside_bounds_reinsertion(self):
        """Test reinserting a crop that extends outside the bounds of the original image."""
        bbox = [[-3, 6], [2, 9], [7, 15]]

        # Crop and then reinsert
        crop_np = crop_and_pad_nd(self.image_np, bbox)
        result_np = insert_crop_into_image(self.image_np.copy(), crop_np, bbox)
        np.testing.assert_array_almost_equal(result_np, self.image_np)

        crop_torch = crop_and_pad_nd(self.image_torch, bbox)
        result_torch = insert_crop_into_image(self.image_torch.clone(), crop_torch, bbox)
        torch.testing.assert_close(result_torch, self.image_torch)

    def test_single_dimension_crop_reinsertion(self):
        """Test reinserting a crop that affects only the last dimension."""
        bbox = [[5, 10]]

        # Crop and then reinsert
        crop_np = crop_and_pad_nd(self.image_np, bbox)
        result_np = insert_crop_into_image(self.image_np.copy(), crop_np, bbox)
        np.testing.assert_array_almost_equal(result_np, self.image_np)

        crop_torch = crop_and_pad_nd(self.image_torch, bbox)
        result_torch = insert_crop_into_image(self.image_torch.clone(), crop_torch, bbox)
        torch.testing.assert_close(result_torch, self.image_torch)

    def test_partial_dimension_crop_reinsertion(self):
        """Test reinserting a crop that affects the last two dimensions."""
        bbox = [[3, 8], [4, 10]]

        # Crop and then reinsert
        crop_np = crop_and_pad_nd(self.image_np, bbox)
        result_np = insert_crop_into_image(self.image_np.copy(), crop_np, bbox)
        np.testing.assert_array_almost_equal(result_np, self.image_np)

        crop_torch = crop_and_pad_nd(self.image_torch, bbox)
        result_torch = insert_crop_into_image(self.image_torch.clone(), crop_torch, bbox)
        torch.testing.assert_close(result_torch, self.image_torch)

    def test_full_image_reinsertion(self):
        """Test reinserting a crop that covers the full image dimensions."""
        bbox = [[0, 10], [0, 10], [0, 10]]

        # Crop and then reinsert
        crop_np = crop_and_pad_nd(self.image_np, bbox)
        result_np = insert_crop_into_image(self.image_np.copy(), crop_np, bbox)
        np.testing.assert_array_almost_equal(result_np, self.image_np)

        crop_torch = crop_and_pad_nd(self.image_torch, bbox)
        result_torch = insert_crop_into_image(self.image_torch.clone(), crop_torch, bbox)
        torch.testing.assert_close(result_torch, self.image_torch)

    def test_non_cubic_reinsertion(self):
        """Test reinserting a non-cubic crop."""
        bbox = [[1, 5], [2, 8], [3, 6]]

        # Crop and then reinsert
        crop_np = crop_and_pad_nd(self.image_np, bbox)
        result_np = insert_crop_into_image(self.image_np.copy(), crop_np, bbox)
        np.testing.assert_array_almost_equal(result_np, self.image_np)

        crop_torch = crop_and_pad_nd(self.image_torch, bbox)
        result_torch = insert_crop_into_image(self.image_torch.clone(), crop_torch, bbox)
        torch.testing.assert_close(result_torch, self.image_torch)

    def test_empty_reinsertion(self):
        """Test reinserting an empty crop (bbox out of bounds)."""
        bbox = [[-15, -9], [-25, -19], [-20, -14]]

        # Crop and then reinsert
        crop_np = crop_and_pad_nd(self.image_np, bbox)
        result_np = insert_crop_into_image(self.image_np.copy(), crop_np, bbox)
        np.testing.assert_array_almost_equal(result_np, self.image_np)

        crop_torch = crop_and_pad_nd(self.image_torch, bbox)
        result_torch = insert_crop_into_image(self.image_torch.clone(), crop_torch, bbox)
        torch.testing.assert_close(result_torch, self.image_torch)


if __name__ == "__main__":
    unittest.main()
    # t = TestInsertCropIntoImage()
    # t.setUp()
    # t.test_within_bounds_reinsertion()