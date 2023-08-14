import numpy as np
from copy import deepcopy
from typing import List, Tuple, Union, Sequence


def pad_bbox(bounding_box: Sequence[Sequence[int]], pad_amount: Union[int, List[int]],
             array_shape: Tuple[int, ...] = None) -> List[Tuple[int]]:
    """

    """
    if isinstance(bounding_box, tuple):
        # convert to list
        bounding_box = [tuple(i) for i in bounding_box]
    else:
        # needed because otherwise we overwrite input which could have unforseen consequences
        bounding_box = deepcopy(bounding_box)

    if isinstance(pad_amount, int):
        pad_amount = [pad_amount] * len(bounding_box)

    if array_shape is not None:
        for i in range(len(bounding_box)):
            new_values = (max(0, bounding_box[i][0] - pad_amount[i]),
                          min(array_shape[i], bounding_box[i][1] + pad_amount[i]))
            bounding_box[i] = new_values
    else:
        for i in range(len(bounding_box)):
            new_values = (max(0, bounding_box[i][0] - pad_amount[i]), bounding_box[i][1] + pad_amount[i])
            bounding_box[i] = new_values

    return bounding_box


def regionprops_bbox_to_proper_bbox(regionprops_bbox: Tuple[int, ...]) -> List[Tuple[int, int]]:
    """
    regionprops_bbox is what you get from `from skimage.measure import regionprops`
    """
    dim = len(regionprops_bbox) // 2
    return [(regionprops_bbox[i], regionprops_bbox[i + dim]) for i in range(dim)]


def bounding_box_to_slice(bounding_box: Sequence[Sequence[int]]):
    return tuple(slice(*i) for i in bounding_box)


def crop_to_bbox(array: np.ndarray, bounding_box: List[List[int]]):
    assert len(bounding_box) == len(array.shape), f"Dimensionality of bbox and array do not match. bbox has length " \
                                          f"{len(bounding_box)} while array has dimension {len(array.shape)}"
    slicer = bounding_box_to_slice(bounding_box)
    return array[slicer]


def get_bbox_from_mask(mask: np.ndarray) -> Tuple[Tuple[int, int], ...]:
    """
    this implementation uses less ram than the np.where one and is faster as well IF we expect the bounding box to
    be close to the image size. If it's not it's likely slower!

    :param mask:
    :return:
    """
    Z, X, Y = mask.shape
    minzidx, maxzidx, minxidx, maxxidx, minyidx, maxyidx = 0, Z, 0, X, 0, Y
    zidx = tuple(range(Z))
    for z in zidx:
        if np.any(mask[z]):
            minzidx = z
            break
    for z in zidx[::-1]:
        if np.any(mask[z]):
            maxzidx = z + 1
            break

    xidx = tuple(range(X))
    for x in xidx:
        if np.any(mask[:, x]):
            minxidx = x
            break
    for x in xidx[::-1]:
        if np.any(mask[:, x]):
            maxxidx = x + 1
            break

    yidx = tuple(range(Y))
    for y in yidx:
        if np.any(mask[:, :, y]):
            minyidx = y
            break
    for y in yidx[::-1]:
        if np.any(mask[:, :, y]):
            maxyidx = y + 1
            break
    return (minzidx, maxzidx), (minxidx, maxxidx), (minyidx, maxyidx)


def get_bbox_from_mask_npwhere(mask: np.ndarray) -> List[Tuple[int, int]]:
    where = np.array(np.where(mask))
    mins = np.min(where, 1)
    maxs = np.max(where, 1) + 1
    return tuple(zip(mins, maxs))


if __name__ == '__main__':
    bbox = [[32, 64], [21, 46]]
    bbox_padded = pad_bbox(bbox, 3)
    slicer = bounding_box_to_slice(bbox_padded)