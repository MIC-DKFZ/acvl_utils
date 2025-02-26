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
    Pads a bounding box by a specified amount and optionally ensures that it stays within the bounds of an array.

    Parameters:
    -----------
    bounding_box : Union[List[List[int]], Tuple[Tuple[int, int]]]
        The input bounding box, represented as a list or tuple of coordinate pairs. Each pair corresponds to
        one dimension and should have the form `[start, end]`, where `start` is inclusive and `end` is exclusive.

    pad_amount : Union[int, List[int]]
        The amount of padding to apply to each dimension of the bounding box.
        - If an integer is provided, the same padding is applied to all dimensions.
        - If a list is provided, each element specifies the padding for the corresponding dimension.

    array_shape : Tuple[int, ...], optional
        The shape of the array to which the bounding box is applied.
        If provided, the padded bounding box will be clipped to stay within these bounds.

    Returns:
    --------
    List[List[int]]
        A new bounding box with the specified padding applied. Each dimension will be represented as a pair `[start, end]`.

    Notes:
    ------
    - The input `bounding_box` is converted to a list to ensure it can be modified without affecting the original data.
    - The padding is applied symmetrically, reducing the `start` coordinate and increasing the `end` coordinate.
    - If `array_shape` is provided, the padded bounding box will not exceed the valid indices of the array.

    Examples:
    ---------
    1. Padding a bounding box without array shape constraints:
       ```python
       bbox = [[2, 10], [5, 15]]
       padded_bbox = pad_bbox(bbox, pad_amount=2)
       # Result: [[0, 12], [3, 17]]
       ```

    2. Padding with array shape constraints:
       ```python
       bbox = [[2, 10], [5, 15]]
       array_shape = (12, 20)
       padded_bbox = pad_bbox(bbox, pad_amount=2, array_shape=array_shape)
       # Result: [[0, 12], [3, 17]]
       ```

    3. Using dimension-specific padding:
       ```python
       bbox = [[2, 10], [5, 15]]
       padded_bbox = pad_bbox(bbox, pad_amount=[3, 1])
       # Result: [[0, 13], [4, 16]]
       ```
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
    """
    ALL bounding boxes in acvl_utils and nnU-Netv2 are half open interval [start, end)!
    - Alignment with Python Slicing
    - Ease of Subdivision
    - Consistency in Multi-Dimensional Arrays
    - Precedent in Computer Graphics
    https://chatgpt.com/share/679203ec-3fbc-8013-a003-13a7adfb1e73
    """
    return tuple([slice(*i) for i in bounding_box])


def crop_to_bbox(array: np.ndarray, bounding_box: List[List[int]]):
    """
    ALL bounding boxes in acvl_utils and nnU-Netv2 are half open interval [start, end)!
    - Alignment with Python Slicing
    - Ease of Subdivision
    - Consistency in Multi-Dimensional Arrays
    - Precedent in Computer Graphics
    https://chatgpt.com/share/679203ec-3fbc-8013-a003-13a7adfb1e73
    """
    assert len(bounding_box) == len(array.shape), f"Dimensionality of bbox and array do not match. bbox has length " \
                                          f"{len(bounding_box)} while array has dimension {len(array.shape)}"
    slicer = bounding_box_to_slice(bounding_box)
    return array[slicer]


def get_bbox_from_mask(mask: np.ndarray) -> List[List[int]]:
    """
    ALL bounding boxes in acvl_utils and nnU-Netv2 are half open interval [start, end)!
    - Alignment with Python Slicing
    - Ease of Subdivision
    - Consistency in Multi-Dimensional Arrays
    - Precedent in Computer Graphics
    https://chatgpt.com/share/679203ec-3fbc-8013-a003-13a7adfb1e73

    this implementation uses less ram than the np.where one and is faster as well IF we expect the bounding box to
    be close to the image size. If it's not it's likely slower!

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


def int_bbox(bbox):
    bbox2 = [[int(i) for i in row] for row in bbox]  # Convert all values to int
    if bbox2 != bbox:  # Directly compare the original and casted versions
        raise RuntimeError(f"Invalid bbox encountered. Cannot be safely cast to int: {bbox}. Casting result: {bbox2}")
    return bbox2


def get_bbox_from_mask_npwhere(mask: np.ndarray) -> List[List[int]]:
    """
    ALL bounding boxes in acvl_utils and nnU-Netv2 are half open interval [start, end)!
    - Alignment with Python Slicing
    - Ease of Subdivision
    - Consistency in Multi-Dimensional Arrays
    - Precedent in Computer Graphics
    https://chatgpt.com/share/679203ec-3fbc-8013-a003-13a7adfb1e73
    """
    where = np.array(np.where(mask))
    mins = np.min(where, 1)
    maxs = np.max(where, 1) + 1
    return [[i, j] for i, j in zip(mins, maxs)]


def crop_and_pad_nd(
        image: Union[torch.Tensor, np.ndarray, blosc2.ndarray.NDArray],
        bbox: List[List[int]],
        pad_value=0,
        pad_mode: str = 'constant',
        allow_hacky_np_workaround_for_reflect: bool = True
) -> Union[torch.Tensor, np.ndarray]:
    """
    Crops a bounding box directly specified by bbox, adhering to the half-open interval [start, end).
    If the bounding box extends beyond the image boundaries, the cropped area is padded
    to maintain the desired size. Initial dimensions not included in bbox remain unaffected.

    CAREFUL: When allow_hacky_np_workaround_for_reflect=True, pad_mode is 'reflect', the input is torch and the
    padding exceeds the dimension of the tensor in the respective dimension (for example paddding with 80 on a dim with
     shape 32) torch will not be able to perform this operation. We fall back to CPU (!!) numpy padding for this!

    ALL bounding boxes in acvl_utils and nnU-Netv2 are half open interval [start, end)!
    - Alignment with Python Slicing
    - Ease of Subdivision
    - Consistency in Multi-Dimensional Arrays
    - Precedent in Computer Graphics
    https://chatgpt.com/share/679203ec-3fbc-8013-a003-13a7adfb1e73

    Parameters:
    - image: N-dimensional torch.Tensor, np.ndarray, or blosc2.ndarray.NDArray representing the image.
    - bbox: List of [[dim_min, dim_max], ...] defining the bounding box for the last dimensions.
            Each dimension follows the half-open interval [start, end).
    - pad_value: Value used for padding when bbox extends beyond image boundaries.
    - pad_mode: Padding mode, one of 'constant', 'reflect', or 'replicate' (alias for 'edge').
    - allow_hacky_np_workaround_for_reflect: If True, will perform numpy workaround if torch.pad doesn't work. Only
    happening if pad_mode is 'reflect'

    Returns:
    - Cropped and padded patch of the requested bounding box size, as the same type as `image`.
    """

    assert pad_mode in ['constant', 'reflect', 'replicate', 'edge'], "Unsupported pad_mode."

    # make sure bounding boxes are int and not uint. Otherwise we may get underflow
    bbox = int_bbox(bbox)

    # Determine the number of dimensions to crop based on bbox
    crop_dims = len(bbox)
    img_shape = image.shape
    num_dims = len(img_shape)

    # Initialize the crop and pad specifications for each dimension
    slices = []
    padding = []
    output_shape = list(img_shape[:num_dims - crop_dims])  # Initial dimensions remain as in the original image
    target_shape = output_shape + [max_val - min_val for min_val, max_val in bbox]  # Half-open interval

    # Iterate through dimensions, applying bbox to the last `crop_dims` dimensions
    for i in range(num_dims):
        if i < num_dims - crop_dims:
            # For initial dimensions not covered by bbox, include the entire dimension
            slices.append(slice(None))
            padding.append([0, 0])
            output_shape.append(img_shape[i])
        else:
            # For dimensions specified in bbox, directly use the min and max bounds
            dim_idx = i - (num_dims - crop_dims)  # Index within bbox

            min_val = bbox[dim_idx][0]
            max_val = bbox[dim_idx][1]  # This is exclusive by definition

            # Check if the bounding box is completely outside the image bounds
            if max_val <= 0 or min_val >= img_shape[i]:
                # If outside bounds, return an empty array or tensor of the target shape
                if isinstance(image, torch.Tensor):
                    return torch.zeros(target_shape, dtype=image.dtype, device=image.device)
                elif isinstance(image, (np.ndarray, blosc2.ndarray.NDArray)):
                    return np.zeros(target_shape, dtype=image.dtype)

            # Calculate valid cropping ranges within image bounds (half-open interval)
            valid_min = max(min_val, 0)
            valid_max = min(max_val, img_shape[i])  # Exclusive upper bound
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
    if np.any(padding):
        # torch will not allow reflection padding if the amount of padding exceeds the shape in that dimension. Numpy will happily do that. I am annoyed. Implement a numpy fallback for that
        was_torch = False
        if allow_hacky_np_workaround_for_reflect and pad_mode == 'reflect' and isinstance(image, torch.Tensor):
            for d in range(-len(padding), 0):
                if max(padding[d]) < cropped.shape[d]:
                    device = cropped.device
                    cropped = cropped.cpu().numpy()
                    was_torch = True
                    print('used numpy workaround')
                    break

        if isinstance(cropped, torch.Tensor):
            if pad_mode == 'edge':
                pad_mode = 'replicate'
            if pad_mode in ['replicate', 'reflect']:
                # pytorch is weird: https://github.com/pytorch/pytorch/issues/147506
                # getting NotImplementedError: Only 2D, 3D, 4D, 5D padding with non-constant padding are supported for now

                # when padding a 2d array because batch/color channel are expected. This is not documented.
                # wonky workaround that may work, or not. Adding fake dimension. This is fiddly because our
                # image/cropped may have more dimensions than we want to pad and may even have too many dimensions for
                # torch to be happy

                # ok so pytorch works most reliably if the length of the padding is 2 less than the ndim of cropped
                while padding[0] == [0, 0] and len(padding) > cropped.ndim - 2:
                    padding = padding[1:]
                # now check whether cropped.ndim is too small
                n_fake_dims = len(padding) + 2 - cropped.ndim
                if n_fake_dims > 0:
                    for i in range(n_fake_dims):
                        cropped.unsqueeze_(0)
                # check if we need to add fake padding
                while len(padding) < cropped.ndim - 2:
                    padding = [0, 0] + padding
                assert cropped.ndim < 6, 'Torch padding with replicate/reflect works with 3D images at most'

            flattened_padding = [p for sublist in reversed(padding) for p in sublist]  # Flatten in reverse order for PyTorch
            try:
                padded_cropped = F.pad(cropped, flattened_padding, mode=pad_mode, value=pad_value)
            except Exception as e:
                print('Failed torch pad')
                print('cropped', cropped.shape)
                print('cropped device', cropped.device)
                print('cropped type', cropped.dtype)
                print('flattened_padding', flattened_padding)
                print('pad mode', pad_mode)
                print('pad value', pad_value)
                print('image shape', img_shape)
                print('bbox', bbox)
                raise e

            if pad_mode in ['replicate', 'reflect'] and n_fake_dims > 0:
                for i in range(n_fake_dims):
                    padded_cropped.squeeze_(0)
        elif isinstance(cropped, (np.ndarray, blosc2.ndarray.NDArray)):
            if pad_mode == 'replicate':
                pad_mode = 'edge'
            if pad_mode == 'edge' or pad_mode == 'reflect':
                kwargs = {}
            else:
                kwargs = {'constant_values': pad_value}
            pad_width = [(p[0], p[1]) for p in padding]
            padded_cropped = np.pad(cropped, pad_width=pad_width, mode=pad_mode, **kwargs)
            if was_torch:
                padded_cropped = torch.from_numpy(padded_cropped).to(device)
        else:
            raise ValueError(f'Unsupported image type {type(image)}')
    else:
        padded_cropped = cropped

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
    # make sure bounding boxes are int and not uint. Otherwise we may get underflow
    bbox = int_bbox(bbox)

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
    # bbox = [[32, 64], [21, 46]]
    # bbox_padded = pad_bbox(bbox, 3)
    # slicer = bounding_box_to_slice(bbox_padded)

    # Failed torch pad
    # cropped torch.Size([1, 1, 112, 100, 149])
    # cropped device cpu
    # cropped type torch.uint8
    # flattened_padding [6, 0, 0, 0]
    # pad mode replicate
    # pad value 0


    # cropped = torch.ones((1, 1, 112, 100, 149), dtype=torch.uint8, device='cpu')
    # pad_mode = 'replicate'
    # flattened_padding = [0, 0, 6, 0, 0, 0]
    # pad_value = 0
    # padded_cropped = F.pad(cropped, flattened_padding, mode=pad_mode, value=pad_value)

    image = torch.rand((1, 192, 192, 192))
    bbox = [[0, 195], [-1, 204], [-1, 192]]
    ret = crop_and_pad_nd(image, bbox, pad_mode='replicate')
    print(ret.shape)