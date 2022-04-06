import unittest
from typing import List, Union, Tuple

import numpy as np
from acvl_utils.cropping_and_padding.padding import pad_nd_image
import torch


class TestPadding(unittest.TestCase):
    def setUp(self) -> None:
        self.torch_tensor_2d = torch.rand((32, 23))
        self.numpy_array_2d = self.torch_tensor_2d.numpy()
        self.torch_tensor_3d = torch.rand((61, 16, 37))
        self.numpy_array_3d = self.torch_tensor_3d.numpy()
        self.torch_tensor_4d = torch.rand((3, 34, 55, 3))
        self.numpy_array_4d = self.torch_tensor_4d.numpy()
        self.torch_tensor_5d = torch.rand((1, 3, 57, 18, 10))
        self.numpy_array_5d = self.torch_tensor_5d.numpy()

        self.torch_tensor_2d_padded = self.numpy_array_2d_padded = \
            self.torch_tensor_3d_padded = self.numpy_array_3d_padded = \
            self.torch_tensor_4d_padded = self.numpy_array_4d_padded = \
            self.torch_tensor_5d_padded = self.numpy_array_5d_padded = None
        self.torch_tensor_2d_padded_slicer = self.numpy_array_2d_padded_slicer = \
            self.torch_tensor_3d_padded_slicer = self.numpy_array_3d_padded_slicer = \
            self.torch_tensor_4d_padded_slicer = self.numpy_array_4d_padded_slicer = \
            self.torch_tensor_5d_padded_slicer = self.numpy_array_5d_padded_slicer = None

    def _pad_tensors(self, new_shape=None, must_be_divisible_by=None):
        if new_shape is None or len(new_shape) < 3:
            self.torch_tensor_2d_padded, self.torch_tensor_2d_padded_slicer = \
                pad_nd_image(self.torch_tensor_2d,
                             new_shape=new_shape,
                             shape_must_be_divisible_by=must_be_divisible_by,
                             return_slicer=True)
            self.numpy_array_2d_padded, self.numpy_array_2d_padded_slicer = \
                pad_nd_image(self.numpy_array_2d,
                             new_shape=new_shape,
                             shape_must_be_divisible_by=must_be_divisible_by,
                             return_slicer=True)
        if new_shape is None or len(new_shape) < 4:
            self.torch_tensor_3d_padded, self.torch_tensor_3d_padded_slicer = \
                pad_nd_image(self.torch_tensor_3d,
                             new_shape=new_shape,
                             shape_must_be_divisible_by=must_be_divisible_by,
                             return_slicer=True)
            self.numpy_array_3d_padded, self.numpy_array_3d_padded_slicer = \
                pad_nd_image(self.numpy_array_3d,
                             new_shape=new_shape,
                             shape_must_be_divisible_by=must_be_divisible_by,
                             return_slicer=True)
        if new_shape is None or len(new_shape) < 5:
            self.torch_tensor_4d_padded, self.torch_tensor_4d_padded_slicer = \
                pad_nd_image(self.torch_tensor_4d,
                             new_shape=new_shape,
                             shape_must_be_divisible_by=must_be_divisible_by,
                             return_slicer=True)
            self.numpy_array_4d_padded, self.numpy_array_4d_padded_slicer = \
                pad_nd_image(self.numpy_array_4d,
                             new_shape=new_shape,
                             shape_must_be_divisible_by=must_be_divisible_by,
                             return_slicer=True)
        if new_shape is None or len(new_shape) < 6:
            self.torch_tensor_5d_padded, self.torch_tensor_5d_padded_slicer = \
                pad_nd_image(self.torch_tensor_5d,
                             new_shape=new_shape,
                             shape_must_be_divisible_by=must_be_divisible_by,
                             return_slicer=True)
            self.numpy_array_5d_padded, self.numpy_array_5d_padded_slicer = \
                pad_nd_image(self.numpy_array_5d,
                             new_shape=new_shape,
                             shape_must_be_divisible_by=must_be_divisible_by,
                             return_slicer=True)

    def _verify_shape(self, tensor_or_array, expected_shape):
        self.assertTrue(all([i == j for i, j in zip(tensor_or_array.shape, expected_shape)]),
                        f'expected shape was {expected_shape} but got {tensor_or_array.shape}')

    def _verify_equal_content(self, a, b):
        if isinstance(a, torch.Tensor):
            a = a.numpy()
        if isinstance(b, torch.Tensor):
            b = b.numpy()
        self.assertTrue(np.all(a == b))

    def _verify_divisible_by(self, tensor_or_array, divisible_by: Union[int, Tuple, List], original_tensor):
        if isinstance(divisible_by, int):
            divisible_by = [divisible_by] * len(tensor_or_array.shape)
        if len(divisible_by) < len(tensor_or_array.shape):
            divisible_by = [1] * (len(tensor_or_array.shape) - len(divisible_by)) + list(divisible_by)
        self.assertTrue(all([i % j == 0 for i, j in zip(tensor_or_array.shape, divisible_by)]))
        self.assertTrue(all([i - j < k for i, j, k in zip(tensor_or_array.shape, divisible_by, original_tensor.shape)]))

    def _verify_slicer(self, padded, orig, slicer):
        reverted = padded[slicer]
        self._verify_shape(reverted, orig.shape)
        self._verify_equal_content(reverted, orig)

    def _verify_all_slicers(self):
        if self.torch_tensor_2d_padded is not None:
            self._verify_slicer(self.torch_tensor_2d_padded, self.torch_tensor_2d, self.torch_tensor_2d_padded_slicer)
        if self.numpy_array_2d_padded is not None:
            self._verify_slicer(self.numpy_array_2d_padded, self.numpy_array_2d, self.numpy_array_2d_padded_slicer)
        if self.torch_tensor_3d_padded is not None:
            self._verify_slicer(self.torch_tensor_3d_padded, self.torch_tensor_3d, self.torch_tensor_3d_padded_slicer)
        if self.numpy_array_3d_padded is not None:
            self._verify_slicer(self.numpy_array_3d_padded, self.numpy_array_3d, self.numpy_array_3d_padded_slicer)
        if self.torch_tensor_4d_padded is not None:
            self._verify_slicer(self.torch_tensor_4d_padded, self.torch_tensor_4d, self.torch_tensor_4d_padded_slicer)
        if self.numpy_array_4d_padded is not None:
            self._verify_slicer(self.numpy_array_4d_padded, self.numpy_array_4d, self.numpy_array_4d_padded_slicer)
        if self.torch_tensor_5d_padded is not None:
            self._verify_slicer(self.torch_tensor_5d_padded, self.torch_tensor_5d, self.torch_tensor_5d_padded_slicer)
        if self.numpy_array_5d_padded is not None:
            self._verify_slicer(self.numpy_array_5d_padded, self.numpy_array_5d, self.numpy_array_5d_padded_slicer)

    def test_padding_to_new_shape(self):
        new_shape = (96, 64)
        self._pad_tensors(new_shape=new_shape)
        self._verify_shape(self.torch_tensor_2d_padded, new_shape)
        self._verify_shape(self.numpy_array_2d_padded, new_shape)
        self._verify_equal_content(self.torch_tensor_2d_padded, self.numpy_array_2d_padded)

        self._verify_shape(self.torch_tensor_3d_padded, [61, *new_shape])
        self._verify_shape(self.numpy_array_3d_padded, [61, *new_shape])
        self._verify_equal_content(self.torch_tensor_3d_padded, self.numpy_array_3d_padded)

        self._verify_shape(self.torch_tensor_4d_padded, [3, 34, *new_shape])
        self._verify_shape(self.numpy_array_4d_padded, [3, 34, *new_shape])
        self._verify_equal_content(self.torch_tensor_4d_padded, self.numpy_array_4d_padded)

        self._verify_shape(self.torch_tensor_5d_padded, [1, 3, 57, *new_shape])
        self._verify_shape(self.numpy_array_5d_padded, [1, 3, 57, *new_shape])
        self._verify_equal_content(self.torch_tensor_5d_padded, self.numpy_array_5d_padded)

        self._verify_all_slicers()

    def test_padding_to_new_shape2(self):
        new_shape = (78, 96, 64)
        self._pad_tensors(new_shape=new_shape)
        # dropping 2d because 3d new shape
        self._verify_shape(self.torch_tensor_3d_padded, new_shape)
        self._verify_shape(self.numpy_array_3d_padded, new_shape)
        self._verify_equal_content(self.torch_tensor_3d_padded, self.numpy_array_3d_padded)

        self._verify_shape(self.torch_tensor_4d_padded, [3, *new_shape])
        self._verify_shape(self.numpy_array_4d_padded, [3, *new_shape])
        self._verify_equal_content(self.torch_tensor_4d_padded, self.numpy_array_4d_padded)

        self._verify_shape(self.torch_tensor_5d_padded, [1, 3, *new_shape])
        self._verify_shape(self.numpy_array_5d_padded, [1, 3, *new_shape])
        self._verify_equal_content(self.torch_tensor_5d_padded, self.numpy_array_5d_padded)

        self._verify_all_slicers()

    def test_padding_to_new_small_shape(self):
        new_shape = (30, 31)
        self._pad_tensors(new_shape=new_shape)
        self._verify_shape(self.torch_tensor_2d_padded, [32, 31])
        self._verify_shape(self.numpy_array_2d_padded, [32, 31])
        self._verify_equal_content(self.torch_tensor_2d_padded, self.numpy_array_2d_padded)

        self._verify_shape(self.torch_tensor_3d_padded, [61, 30, 37])
        self._verify_shape(self.numpy_array_3d_padded, [61, 30, 37])
        self._verify_equal_content(self.torch_tensor_3d_padded, self.numpy_array_3d_padded)

        self._verify_shape(self.torch_tensor_4d_padded, [3, 34, 55, 31])
        self._verify_shape(self.numpy_array_4d_padded, [3, 34, 55, 31])
        self._verify_equal_content(self.torch_tensor_4d_padded, self.numpy_array_4d_padded)

        self._verify_shape(self.torch_tensor_5d_padded, [1, 3, 57, *new_shape])
        self._verify_shape(self.numpy_array_5d_padded, [1, 3, 57, *new_shape])
        self._verify_equal_content(self.torch_tensor_5d_padded, self.numpy_array_5d_padded)

        self._verify_all_slicers()

    def test_global_must_be_divisible_by(self):
        divisible_by = 16
        self._pad_tensors(must_be_divisible_by=divisible_by)
        self._verify_divisible_by(self.torch_tensor_2d_padded, divisible_by, self.torch_tensor_2d)
        self._verify_divisible_by(self.numpy_array_2d_padded, divisible_by, self.numpy_array_2d)
        self._verify_equal_content(self.torch_tensor_2d_padded, self.numpy_array_2d_padded)

        self._verify_divisible_by(self.torch_tensor_3d_padded, divisible_by, self.torch_tensor_3d)
        self._verify_divisible_by(self.numpy_array_3d_padded, divisible_by, self.numpy_array_3d)
        self._verify_equal_content(self.torch_tensor_3d_padded, self.numpy_array_3d_padded)

        self._verify_divisible_by(self.torch_tensor_4d_padded, divisible_by, self.torch_tensor_4d)
        self._verify_divisible_by(self.numpy_array_4d_padded, divisible_by, self.numpy_array_4d)
        self._verify_equal_content(self.torch_tensor_4d_padded, self.numpy_array_4d_padded)

        self._verify_divisible_by(self.torch_tensor_5d_padded, divisible_by, self.torch_tensor_5d)
        self._verify_divisible_by(self.numpy_array_5d_padded, divisible_by, self.numpy_array_5d)
        self._verify_equal_content(self.torch_tensor_5d_padded, self.numpy_array_5d_padded)

        self._verify_all_slicers()

    def test_global_must_be_divisible_by2(self):
        divisible_by = 31
        self._pad_tensors(must_be_divisible_by=divisible_by)
        self._verify_divisible_by(self.torch_tensor_2d_padded, divisible_by, self.torch_tensor_2d)
        self._verify_divisible_by(self.numpy_array_2d_padded, divisible_by, self.numpy_array_2d)
        self._verify_equal_content(self.torch_tensor_2d_padded, self.numpy_array_2d_padded)

        self._verify_divisible_by(self.torch_tensor_3d_padded, divisible_by, self.torch_tensor_3d)
        self._verify_divisible_by(self.numpy_array_3d_padded, divisible_by, self.numpy_array_3d)
        self._verify_equal_content(self.torch_tensor_3d_padded, self.numpy_array_3d_padded)

        self._verify_divisible_by(self.torch_tensor_4d_padded, divisible_by, self.torch_tensor_4d)
        self._verify_divisible_by(self.numpy_array_4d_padded, divisible_by, self.numpy_array_4d)
        self._verify_equal_content(self.torch_tensor_4d_padded, self.numpy_array_4d_padded)

        self._verify_divisible_by(self.torch_tensor_5d_padded, divisible_by, self.torch_tensor_5d)
        self._verify_divisible_by(self.numpy_array_5d_padded, divisible_by, self.numpy_array_5d)
        self._verify_equal_content(self.torch_tensor_5d_padded, self.numpy_array_5d_padded)

        self._verify_all_slicers()

    def test_local_must_be_divisible_by(self):
        divisible_by = (16, 7)
        self.torch_tensor_2d_padded, self.torch_tensor_2d_padded_slicer = \
            pad_nd_image(self.torch_tensor_2d, shape_must_be_divisible_by=divisible_by, return_slicer=True)
        self.numpy_array_2d_padded, self.numpy_array_2d_padded_slicer = \
            pad_nd_image(self.numpy_array_2d, shape_must_be_divisible_by=divisible_by, return_slicer=True)
        self._verify_divisible_by(self.torch_tensor_2d_padded, divisible_by, self.torch_tensor_2d)
        self._verify_divisible_by(self.numpy_array_2d_padded, divisible_by, self.numpy_array_2d)
        self._verify_equal_content(self.torch_tensor_2d_padded, self.numpy_array_2d_padded)

        divisible_by = (3, 19, 64)
        self.torch_tensor_3d_padded, self.torch_tensor_3d_padded_slicer = \
            pad_nd_image(self.torch_tensor_3d, shape_must_be_divisible_by=divisible_by, return_slicer=True)
        self.numpy_array_3d_padded, self.numpy_array_3d_padded_slicer = \
            pad_nd_image(self.numpy_array_3d, shape_must_be_divisible_by=divisible_by, return_slicer=True)
        self._verify_divisible_by(self.torch_tensor_3d_padded, divisible_by, self.torch_tensor_3d)
        self._verify_divisible_by(self.numpy_array_3d_padded, divisible_by, self.numpy_array_3d)
        self._verify_equal_content(self.torch_tensor_3d_padded, self.numpy_array_3d_padded)

        divisible_by = (7, 21, 34, 49)
        self.torch_tensor_4d_padded, self.torch_tensor_4d_padded_slicer = \
            pad_nd_image(self.torch_tensor_4d, shape_must_be_divisible_by=divisible_by, return_slicer=True)
        self.numpy_array_4d_padded, self.numpy_array_4d_padded_slicer = \
            pad_nd_image(self.numpy_array_4d, shape_must_be_divisible_by=divisible_by, return_slicer=True)
        self._verify_divisible_by(self.torch_tensor_4d_padded, divisible_by, self.torch_tensor_4d)
        self._verify_divisible_by(self.numpy_array_4d_padded, divisible_by, self.numpy_array_4d)
        self._verify_equal_content(self.torch_tensor_4d_padded, self.numpy_array_4d_padded)

        divisible_by = (2, 3, 24, 12, 31)
        self.torch_tensor_5d_padded, self.torch_tensor_5d_padded_slicer = \
            pad_nd_image(self.torch_tensor_5d, shape_must_be_divisible_by=divisible_by, return_slicer=True)
        self.numpy_array_5d_padded, self.numpy_array_5d_padded_slicer = \
            pad_nd_image(self.numpy_array_5d, shape_must_be_divisible_by=divisible_by, return_slicer=True)
        self._verify_divisible_by(self.torch_tensor_5d_padded, divisible_by, self.torch_tensor_5d)
        self._verify_divisible_by(self.numpy_array_5d_padded, divisible_by, self.numpy_array_5d)
        self._verify_equal_content(self.torch_tensor_5d_padded, self.numpy_array_5d_padded)

    def test_local_partial_must_be_divisible_by(self):
        divisible_by = (7, )
        self.torch_tensor_2d_padded, self.torch_tensor_2d_padded_slicer = \
            pad_nd_image(self.torch_tensor_2d, shape_must_be_divisible_by=divisible_by, return_slicer=True)
        self.numpy_array_2d_padded, self.numpy_array_2d_padded_slicer = \
            pad_nd_image(self.numpy_array_2d, shape_must_be_divisible_by=divisible_by, return_slicer=True)
        self._verify_divisible_by(self.torch_tensor_2d_padded, divisible_by, self.torch_tensor_2d)
        self._verify_divisible_by(self.numpy_array_2d_padded, divisible_by, self.numpy_array_2d)
        self._verify_equal_content(self.torch_tensor_2d_padded, self.numpy_array_2d_padded)

        divisible_by = (3, 64)
        self.torch_tensor_3d_padded, self.torch_tensor_3d_padded_slicer = \
            pad_nd_image(self.torch_tensor_3d, shape_must_be_divisible_by=divisible_by, return_slicer=True)
        self.numpy_array_3d_padded, self.numpy_array_3d_padded_slicer = \
            pad_nd_image(self.numpy_array_3d, shape_must_be_divisible_by=divisible_by, return_slicer=True)
        self._verify_divisible_by(self.torch_tensor_3d_padded, divisible_by, self.torch_tensor_3d)
        self._verify_divisible_by(self.numpy_array_3d_padded, divisible_by, self.numpy_array_3d)
        self._verify_equal_content(self.torch_tensor_3d_padded, self.numpy_array_3d_padded)

        divisible_by = (1, 3, 64)
        self.torch_tensor_3d_padded, self.torch_tensor_3d_padded_slicer = \
            pad_nd_image(self.torch_tensor_3d, shape_must_be_divisible_by=divisible_by, return_slicer=True)
        self.numpy_array_3d_padded, self.numpy_array_3d_padded_slicer = \
            pad_nd_image(self.numpy_array_3d, shape_must_be_divisible_by=divisible_by, return_slicer=True)
        self._verify_divisible_by(self.torch_tensor_3d_padded, divisible_by, self.torch_tensor_3d)
        self._verify_divisible_by(self.numpy_array_3d_padded, divisible_by, self.numpy_array_3d)
        self._verify_equal_content(self.torch_tensor_3d_padded, self.numpy_array_3d_padded)

        divisible_by = (34, 49)
        self.torch_tensor_4d_padded, self.torch_tensor_4d_padded_slicer = \
            pad_nd_image(self.torch_tensor_4d, shape_must_be_divisible_by=divisible_by, return_slicer=True)
        self.numpy_array_4d_padded, self.numpy_array_4d_padded_slicer = \
            pad_nd_image(self.numpy_array_4d, shape_must_be_divisible_by=divisible_by, return_slicer=True)
        self._verify_divisible_by(self.torch_tensor_4d_padded, divisible_by, self.torch_tensor_4d)
        self._verify_divisible_by(self.numpy_array_4d_padded, divisible_by, self.numpy_array_4d)
        self._verify_equal_content(self.torch_tensor_4d_padded, self.numpy_array_4d_padded)

        divisible_by = (2, 3, 24, 12, 31)
        self.torch_tensor_5d_padded, self.torch_tensor_5d_padded_slicer = \
            pad_nd_image(self.torch_tensor_5d, shape_must_be_divisible_by=divisible_by, return_slicer=True)
        self.numpy_array_5d_padded, self.numpy_array_5d_padded_slicer = \
            pad_nd_image(self.numpy_array_5d, shape_must_be_divisible_by=divisible_by, return_slicer=True)
        self._verify_divisible_by(self.torch_tensor_5d_padded, divisible_by, self.torch_tensor_5d)
        self._verify_divisible_by(self.numpy_array_5d_padded, divisible_by, self.numpy_array_5d)
        self._verify_equal_content(self.torch_tensor_5d_padded, self.numpy_array_5d_padded)

        divisible_by = (1, 1, 1, 1, 1)
        self.torch_tensor_5d_padded, self.torch_tensor_5d_padded_slicer = \
            pad_nd_image(self.torch_tensor_5d, shape_must_be_divisible_by=divisible_by, return_slicer=True)
        self.numpy_array_5d_padded, self.numpy_array_5d_padded_slicer = \
            pad_nd_image(self.numpy_array_5d, shape_must_be_divisible_by=divisible_by, return_slicer=True)
        self._verify_divisible_by(self.torch_tensor_5d_padded, divisible_by, self.torch_tensor_5d)
        self._verify_divisible_by(self.numpy_array_5d_padded, divisible_by, self.numpy_array_5d)
        self._verify_equal_content(self.torch_tensor_5d_padded, self.numpy_array_5d_padded)
        self._verify_shape(self.torch_tensor_5d_padded, self.torch_tensor_5d.shape)
        self._verify_shape(self.numpy_array_5d_padded, self.numpy_array_5d.shape)

    def test_local_partial_must_be_divisible_by_and_newshape(self):
        # _verify_divisible_by cannot handle new_shape AND divisible_by so we check the output shape manually!
        divisible_by = (7, )
        new_shape = (41, 16)
        expected_output_shape = (41, 28)  # 2d tensor has (32, 23) as shape
        self.torch_tensor_2d_padded, self.torch_tensor_2d_padded_slicer = \
            pad_nd_image(self.torch_tensor_2d, shape_must_be_divisible_by=divisible_by, new_shape=new_shape,
                         return_slicer=True)
        self.numpy_array_2d_padded, self.numpy_array_2d_padded_slicer = \
            pad_nd_image(self.numpy_array_2d, shape_must_be_divisible_by=divisible_by, new_shape=new_shape,
                         return_slicer=True)
        self._verify_equal_content(self.torch_tensor_2d_padded, self.numpy_array_2d_padded)
        self._verify_shape(self.torch_tensor_2d_padded, expected_output_shape)
        self._verify_shape(self.numpy_array_2d_padded, expected_output_shape)

        divisible_by = (7, 7)
        new_shape = (41, 16)
        expected_output_shape = (42, 28)  # 2d tensor has (32, 23) as shape
        self.torch_tensor_2d_padded, self.torch_tensor_2d_padded_slicer = \
            pad_nd_image(self.torch_tensor_2d, shape_must_be_divisible_by=divisible_by, new_shape=new_shape,
                         return_slicer=True)
        self.numpy_array_2d_padded, self.numpy_array_2d_padded_slicer = \
            pad_nd_image(self.numpy_array_2d, shape_must_be_divisible_by=divisible_by, new_shape=new_shape,
                         return_slicer=True)
        self._verify_equal_content(self.torch_tensor_2d_padded, self.numpy_array_2d_padded)
        self._verify_shape(self.torch_tensor_2d_padded, expected_output_shape)
        self._verify_shape(self.numpy_array_2d_padded, expected_output_shape)

        divisible_by = (7, 7)
        new_shape = (31,)
        expected_output_shape = (35, 35)  # 2d tensor has (32, 23) as shape
        self.torch_tensor_2d_padded, self.torch_tensor_2d_padded_slicer = \
            pad_nd_image(self.torch_tensor_2d, shape_must_be_divisible_by=divisible_by, new_shape=new_shape,
                         return_slicer=True)
        self.numpy_array_2d_padded, self.numpy_array_2d_padded_slicer = \
            pad_nd_image(self.numpy_array_2d, shape_must_be_divisible_by=divisible_by, new_shape=new_shape,
                         return_slicer=True)
        self._verify_equal_content(self.torch_tensor_2d_padded, self.numpy_array_2d_padded)
        self._verify_shape(self.torch_tensor_2d_padded, expected_output_shape)
        self._verify_shape(self.numpy_array_2d_padded, expected_output_shape)




