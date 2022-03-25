from typing import Union, Tuple, List

import numpy as np
from skimage.measure import label
from skimage.morphology import ball
from skimage.transform import resize


def generate_ball(radius: Union[Tuple, List], spacing: Union[Tuple, List] = (1, 1, 1), dtype=np.uint8) -> np.ndarray:
    """
    Returns a ball/ellipsoid corresponding to the specified size (radius = list/tuple of len 3 with one radius per axis)
    If you use spacing, both radius and spacing will be interpreted relative to each other, so a radius of 10 with a
    spacing of 5 will result in a ball with radius 2 pixels.
    """
    radius_in_voxels = np.round(radius / np.array(spacing)).astype(int)
    n = 2 * radius_in_voxels + 1
    ball_iso = ball(max(n) * 2, dtype=np.float64)
    ball_resampled = resize(ball_iso, n, 1, 'constant', 0, clip=True, anti_aliasing=False, preserve_range=True)
    ball_resampled[ball_resampled > 0.5] = 1
    ball_resampled[ball_resampled <= 0.5] = 0
    return ball_resampled.astype(dtype)


def label_with_component_sizes(binary_image: np.ndarray, connectivity=None):
    if not binary_image.dtype == np.bool:
        print('Warning: it would be way faster if your binary image had dtype bool')
    labeled_image, num_components = label(binary_image, return_num=True, connectivity=connectivity)
    component_sizes = {i + 1: j for i, j in enumerate(np.bincount(labeled_image.ravel())[1:])}
    return labeled_image, component_sizes


def remove_small_components(binary_image: np.ndarray, min_size_in_pixels: int, connectivity=None, verbose: bool = False):
    binary_image = np.copy(binary_image)  # let's not change the input array
    labeled_image, component_sizes = label_with_component_sizes(binary_image, connectivity)
    keep = np.array([i for i, j in component_sizes.items() if j >= min_size_in_pixels])

    if verbose:
        print(f'{len(keep)} objects are larger than the minimum size of {min_size_in_pixels}. '
              f'Removing {len(component_sizes) - len(keep)} small objects...')

    keep = np.in1d(labeled_image.ravel(), keep).reshape(labeled_image.shape).astype(binary_image.dtype)
    return keep


if __name__ == '__main__':
    print(generate_ball((10, 10, 10), (5, 5, 5)).shape)
    print(generate_ball((10, 5, 15), (1, 1, 1)).shape)
