import unittest
from acvl_utils.morphology.morphology_helper import remove_components, remove_components_cc3d
import numpy as np


class TestComponentRemoval(unittest.TestCase):
    def setUp(self) -> None:
        self.binary_image_2D = np.array([[1, 0, 1, 1, 0, 0, 1],
                                         [0, 1, 1, 1, 0, 0, 1],
                                         [0, 0, 0, 0, 0, 1, 0],
                                         [1, 0, 0, 1, 0, 0, 0],
                                         [1, 1, 0, 1, 0, 0, 1]]).astype(bool)

        self.binary_image_3D = np.array([[[1, 0, 1, 1, 0, 0, 1],
                                         [0, 1, 1, 1, 0, 0, 1],
                                         [0, 0, 0, 0, 0, 1, 0],
                                         [1, 0, 0, 1, 0, 0, 0],
                                         [1, 1, 0, 1, 0, 0, 1]],
                                         [[1, 0, 1, 1, 0, 0, 1],
                                          [0, 1, 1, 1, 0, 0, 1],
                                          [0, 0, 0, 0, 0, 1, 0],
                                          [1, 0, 0, 1, 0, 0, 0],
                                          [1, 1, 0, 1, 0, 0, 1]],
                                         [[1, 0, 1, 1, 0, 0, 1],
                                          [0, 1, 1, 1, 0, 0, 1],
                                          [0, 0, 0, 0, 0, 1, 0],
                                          [1, 0, 0, 1, 0, 0, 0],
                                          [1, 1, 0, 1, 0, 0, 1]]
                                         ]).astype(bool)

    def test_remove_small_comp_2D(self):
        skimage_res = remove_components(self.binary_image_2D, 3, 'min', verbose=False)
        cc3d_res = remove_components_cc3d(self.binary_image_2D, 3, 'min', verbose=False, connectivity=8)

        self.assertTrue((skimage_res == cc3d_res).all())

        expected_output = np.array([[1, 0, 1, 1, 0, 0, 1],
                                    [0, 1, 1, 1, 0, 0, 1],
                                    [0, 0, 0, 0, 0, 1, 0],
                                    [1, 0, 0, 0, 0, 0, 0],
                                    [1, 1, 0, 0, 0, 0, 0]]).astype(bool)

        self.assertTrue((skimage_res == expected_output).all())

    def test_remove_large_comp_2D(self):
        skimage_res = remove_components(self.binary_image_2D, 4, 'max', verbose=False)
        cc3d_res = remove_components_cc3d(self.binary_image_2D, 4, 'max', verbose=False, connectivity=8)

        self.assertTrue((skimage_res == cc3d_res).all())

        expected_output = np.array([[0, 0, 0, 0, 0, 0, 1],
                                    [0, 0, 0, 0, 0, 0, 1],
                                    [0, 0, 0, 0, 0, 1, 0],
                                    [1, 0, 0, 1, 0, 0, 0],
                                    [1, 1, 0, 1, 0, 0, 1]]).astype(bool)

        self.assertTrue((skimage_res == expected_output).all())

    def test_remove_small_comp_3D(self):
        skimage_res = remove_components(self.binary_image_3D, 9, 'min', verbose=False)
        cc3d_res = remove_components_cc3d(self.binary_image_3D, 9, 'min', verbose=False, connectivity=26)

        self.assertTrue((skimage_res == cc3d_res).all())

        expected_output = np.array([[[1, 0, 1, 1, 0, 0, 1],
                                    [0, 1, 1, 1, 0, 0, 1],
                                    [0, 0, 0, 0, 0, 1, 0],
                                    [1, 0, 0, 0, 0, 0, 0],
                                    [1, 1, 0, 0, 0, 0, 0]],
                                    [[1, 0, 1, 1, 0, 0, 1],
                                     [0, 1, 1, 1, 0, 0, 1],
                                     [0, 0, 0, 0, 0, 1, 0],
                                     [1, 0, 0, 0, 0, 0, 0],
                                     [1, 1, 0, 0, 0, 0, 0]],
                                    [[1, 0, 1, 1, 0, 0, 1],
                                     [0, 1, 1, 1, 0, 0, 1],
                                     [0, 0, 0, 0, 0, 1, 0],
                                     [1, 0, 0, 0, 0, 0, 0],
                                     [1, 1, 0, 0, 0, 0, 0]],
                                    ]).astype(bool)

        self.assertTrue((skimage_res == expected_output).all())

    def test_remove_large_comp_3D(self):
        skimage_res = remove_components(self.binary_image_3D, 12, 'max', verbose=False)
        cc3d_res = remove_components_cc3d(self.binary_image_3D, 12, 'max', verbose=False, connectivity=26)

        self.assertTrue((skimage_res == cc3d_res).all())

        expected_output = np.array([[[0, 0, 0, 0, 0, 0, 1],
                                    [0, 0, 0, 0, 0, 0, 1],
                                    [0, 0, 0, 0, 0, 1, 0],
                                    [1, 0, 0, 1, 0, 0, 0],
                                    [1, 1, 0, 1, 0, 0, 1]],
                                    [[0, 0, 0, 0, 0, 0, 1],
                                     [0, 0, 0, 0, 0, 0, 1],
                                     [0, 0, 0, 0, 0, 1, 0],
                                     [1, 0, 0, 1, 0, 0, 0],
                                     [1, 1, 0, 1, 0, 0, 1]],
                                    [[0, 0, 0, 0, 0, 0, 1],
                                     [0, 0, 0, 0, 0, 0, 1],
                                     [0, 0, 0, 0, 0, 1, 0],
                                     [1, 0, 0, 1, 0, 0, 0],
                                     [1, 1, 0, 1, 0, 0, 1]],
                                    ]).astype(bool)

        self.assertTrue((skimage_res == expected_output).all())
