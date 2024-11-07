from copy import deepcopy
from typing import Tuple
from typing import Union, List

import blosc2
import numpy as np
import torch
import torch.nn.functional as F


def pad_bbox(bounding_box: Union[List[List[int]], Tuple[Tuple[int, int]]], pad_amount: Union[int, List[int]],
             array_shape: Tuple[int, ...] = None) -> List[List[int]]:
    """

    """
    if isinstance(bounding_box, tuple):
        # convert to list
        bounding_box = [list(i) for i in bounding_box]
    else:
        # needed because otherwise we overwrite input which could have unforseen consequences
        bounding_box = deepcopy(bounding_box)

    if isinstance(pad_amount, int):
        pad_amount = [pad_amount] * len(bounding_box)

    for i in range(len(bounding_box)):
        new_values = [max(0, bounding_box[i][0] - pad_amount[i]), bounding_box[i][1] + pad_amount[i]]
        if array_shape is not None:
            new_values[1] = min(array_shape[i], new_values[1])
        bounding_box[i] = new_values

    return bounding_box


def regionprops_bbox_to_proper_bbox(regionprops_bbox: Tuple[int, ...]) -> List[List[int]]:
    """
    regionprops_bbox is what you get from `from skimage.measure import regionprops`
    """
    dim = len(regionprops_bbox) // 2
    return [[regionprops_bbox[i], regionprops_bbox[i + dim]] for i in range(dim)]


def bounding_box_to_slice(bounding_box: List[List[int]]):
    return tuple([slice(*i) for i in bounding_box])


def get_bbox_from_mask(mask: np.ndarray) -> List[List[int]]:
    """
    this implementation uses less ram than the np.where one and is faster as well IF we expect the bounding box to
    be close to the image size. If it's not it's likely slower!
    
    bbox is returned so that you can just do slice(minzidx, maxzidx) to retrieve the object of interest with nothing cut off

    :param mask:
    :param outside_value:
    :return:
    """
    Z, X, Y = mask.shape
    minzidx, maxzidx, minxidx, maxxidx, minyidx, maxyidx = 0, Z, 0, X, 0, Y
    zidx = list(range(Z))
    for z in zidx:
        if np.any(mask[z]):
            minzidx = z
            break
    for z in zidx[::-1]:
        if np.any(mask[z]):
            maxzidx = z + 1
            break

    xidx = list(range(X))
    for x in xidx:
        if np.any(mask[:, x]):
            minxidx = x
            break
    for x in xidx[::-1]:
        if np.any(mask[:, x]):
            maxxidx = x + 1
            break

    yidx = list(range(Y))
    for y in yidx:
        if np.any(mask[:, :, y]):
            minyidx = y
            break
    for y in yidx[::-1]:
        if np.any(mask[:, :, y]):
            maxyidx = y + 1
            break
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]


def get_bbox_from_mask_npwhere(mask: np.ndarray) -> List[List[int]]:
    where = np.array(np.where(mask))
    mins = np.min(where, 1)
    maxs = np.max(where, 1) + 1
    return [[i, j] for i, j in zip(mins, maxs)]


def crop_and_pad_nd(
        image: Union[torch.Tensor, np.ndarray, blosc2.ndarray.NDArray],
        bbox: List[List[int]],
        pad_value = 0
) -> Union[torch.Tensor, np.ndarray]:
    """
    Crops a bounding box directly specified by bbox, excluding the upper bound.
    If the bounding box extends beyond the image boundaries, the cropped area is padded
    to maintain the desired size. Initial dimensions not included in bbox remain unaffected.

    Parameters:
    - image: N-dimensional torch.Tensor or np.ndarray representing the image
    - bbox: List of [[dim_min, dim_max], ...] defining the bounding box for the last dimensions.

    Returns:
    - Cropped and padded patch of the requested bounding box size, as the same type as `image`.
    """

    # Determine the number of dimensions to crop based on bbox
    crop_dims = len(bbox)
    img_shape = image.shape
    num_dims = len(img_shape)

    # Initialize the crop and pad specifications for each dimension
    slices = []
    padding = []
    output_shape = list(img_shape[:num_dims - crop_dims])  # Initial dimensions remain as in the original image
    target_shape = output_shape + [max_val - min_val for min_val, max_val in bbox]

    # Iterate through dimensions, applying bbox to the last `crop_dims` dimensions
    for i in range(num_dims):
        if i < num_dims - crop_dims:
            # For initial dimensions not covered by bbox, include the entire dimension
            slices.append(slice(None))
            padding.append([0, 0])
            output_shape.append(img_shape[i])  # Keep the initial dimensions as they are
        else:
            # For dimensions specified in bbox, directly use the min and max bounds
            dim_idx = i - (num_dims - crop_dims)  # Index within bbox

            min_val = bbox[dim_idx][0]
            max_val = bbox[dim_idx][1]

            # Check if the bounding box is completely outside the image bounds
            if max_val <= 0 or min_val >= img_shape[i]:
                # If outside bounds, return an empty array or tensor of the target shape
                if isinstance(image, torch.Tensor):
                    return torch.zeros(target_shape, dtype=image.dtype, device=image.device)
                elif isinstance(image, (np.ndarray, blosc2.ndarray.NDArray)):
                    return np.zeros(target_shape, dtype=image.dtype)

            # Calculate valid cropping ranges within image bounds, excluding the upper bound
            valid_min = max(min_val, 0)
            valid_max = min(max_val, img_shape[i])  # Exclude upper bound by using max_val directly
            slices.append(slice(valid_min, valid_max))

            # Calculate padding needed for this dimension
            pad_before = max(0, -min_val)
            pad_after = max(0, max_val - img_shape[i])
            padding.append([pad_before, pad_after])

            # Define the shape based on the bbox range in this dimension
            output_shape.append(max_val - min_val)

    # Crop the valid part of the bounding box
    cropped = image[tuple(slices)]

    # Apply padding to the cropped patch
    if isinstance(image, torch.Tensor):
        flattened_padding = [p for sublist in reversed(padding) for p in sublist]  # Flatten in reverse order for PyTorch
        padded_cropped = F.pad(cropped, flattened_padding, mode="constant", value=pad_value)
    elif isinstance(image, (np.ndarray, blosc2.ndarray.NDArray)):
        pad_width = [(p[0], p[1]) for p in padding]
        padded_cropped = np.pad(cropped, pad_width=pad_width, mode='constant', constant_values=pad_value)
    else:
        raise ValueError(f'Unsupported image type {type(image)}')

    return padded_cropped


def insert_crop_into_image(
        image: Union[torch.Tensor, np.ndarray],
        crop: Union[torch.Tensor, np.ndarray],
        bbox: List[List[int]]
) -> Union[torch.Tensor, np.ndarray]:
    """
    Inserts a cropped patch back into the original image at the position specified by bbox.
    If the bounding box extends beyond the image boundaries, only the valid portions are inserted.
    If the bounding box lies entirely outside the image, the original image is returned.

    Parameters:
    - image: Original N-dimensional torch.Tensor or np.ndarray to which the crop will be inserted.
    - crop: Cropped patch of the image to be reinserted. May have additional dimensions compared to bbox.
    - bbox: List of [[dim_min, dim_max], ...] defining the bounding box for the last dimensions of the crop in the original image.

    Returns:
    - image: The original image with the crop reinserted at the specified location (modified in-place).
    """

    # Ensure that bbox only applies to the last len(bbox) dimensions of crop and image
    num_dims = len(image.shape)
    crop_dims = len(crop.shape)
    bbox_dims = len(bbox)

    if crop_dims < bbox_dims:
        raise ValueError("Bounding box dimensions cannot exceed crop dimensions.")

    # Validate that non-cropped leading dimensions match between image and crop
    leading_dims = num_dims - bbox_dims
    if image.shape[:leading_dims] != crop.shape[:leading_dims]:
        raise ValueError("Leading dimensions of crop and image must match.")

    # Check if the bounding box lies completely outside the image bounds for each cropped dimension
    for i in range(bbox_dims):
        min_val, max_val = bbox[i]
        dim_idx = leading_dims + i  # Corresponding dimension in the image

        if max_val <= 0 or min_val >= image.shape[dim_idx]:
            # If completely out of bounds in any dimension, return the original image
            return image

    # Prepare slices for inserting the crop into the original image
    image_slices = []
    crop_slices = []

    # Iterate over all dimensions, applying bbox only to the last len(bbox) dimensions
    for i in range(num_dims):
        if i < leading_dims:
            # For leading dimensions, use entire dimension (slice(None)) and validate shape
            image_slices.append(slice(None))
            crop_slices.append(slice(None))
        else:
            # For dimensions specified by bbox, calculate the intersection with image bounds
            dim_idx = i - leading_dims
            min_val, max_val = bbox[dim_idx]

            crop_start = max(0, -min_val)  # Start of the crop within the valid area
            image_start = max(0, min_val)  # Start of the image where the crop will be inserted
            image_end = min(max_val, image.shape[i])  # Exclude upper bound by using max_val directly

            # Adjusted range for insertion
            crop_end = crop_start + (image_end - image_start)

            # Append slices for both image and crop insertion ranges
            image_slices.append(slice(image_start, image_end))
            crop_slices.append(slice(crop_start, crop_end))

    # Insert the valid part of the crop back into the original image
    if isinstance(image, torch.Tensor):
        image[tuple(image_slices)] = crop[tuple(crop_slices)]
    elif isinstance(image, np.ndarray):
        image[tuple(image_slices)] = crop[tuple(crop_slices)]
    else:
        raise ValueError(f"Unsupported image type {type(image)}")

    return image


if __name__ == '__main__':
    bbox = [[32, 64], [21, 46]]
    bbox_padded = pad_bbox(bbox, 3)
    slicer = bounding_box_to_slice(bbox_padded)