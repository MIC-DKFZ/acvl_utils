from multiprocessing import Pool
from typing import Union, Tuple, List

import numpy as np
from acvl_utils.bounding_boxes import pad_bbox, bounding_box_to_slice, get_bbox_from_mask_npwhere
from acvl_utils.morphology_helper import generate_ball, label_with_component_sizes
from skimage.measure import label
from skimage.measure import regionprops
from skimage.morphology import ball, erosion, binary_dilation
from skimage.morphology import binary_erosion
from skimage.morphology import dilation

BORDER_LABEL = 1
CENTER_LABEL = 2


def convert_semantic_to_instanceseg(arr: np.ndarray,
                                    spacing: Tuple[float, ...] = (1, 1, 1),
                                    small_center_threshold: float = 30,
                                    isolated_border_as_separate_instance_threshold: int = 15):
    """
    :param arr:
    :param spacing:
    :param small_center_threshold: volume, as dictated by spacing! If your spacing is (2, 2, 2) then a
    small_center_threshold of 16 would correspond to two pixels!
    :param isolated_border_as_separate_instance_threshold: volume, as dictated by spacing! If your spacing is (2, 2, 2) then a
    isolated_border_as_separate_instance_threshold of 16 would correspond to two pixels!
    :param border_label:
    :param core_label:
    :return:
    """
    # TODO make MP variant of this
    spacing = np.array(spacing)
    small_center_threshold_in_pixels = small_center_threshold / np.prod(spacing)
    isolated_border_as_separate_instance_threshold_in_pixels = isolated_border_as_separate_instance_threshold / np.prod(spacing)

    # we first identify centers that are too small and set them to be border. This should remove false positive instances
    labeled_image, component_sizes = label_with_component_sizes((arr == CENTER_LABEL))
    remove = np.array([i for i, j in component_sizes.items() if j < small_center_threshold_in_pixels])
    remove = np.in1d(labeled_image.ravel(), remove).reshape(labeled_image.shape)
    arr[remove] = BORDER_LABEL

    # recompute core labels
    core_instances = label((arr == CENTER_LABEL))

    # prepare empty array for results
    final = np.zeros_like(core_instances, dtype=np.uint16)

    # besides the core instances we will need the borders
    border_mask = arr == BORDER_LABEL

    # we search for connected components and then convert each connected component into instance segmentation. This should
    # prevent bleeding into neighboring instances even if the instances don't touch
    connected_components, num_components = label((arr > 0), return_num=True, connectivity=1)
    max_component_idx = np.max(core_instances)

    for cidx in range(1, num_components+1):
        mask = connected_components == cidx
        bbox = get_bbox_from_mask_npwhere(mask)
        slicer = bounding_box_to_slice(bbox)

        cropped_mask = mask[slicer]
        cropped_core_instances = np.copy(core_instances[slicer])
        cropped_border = np.copy(border_mask[slicer])

        # remove other objects from the current crop, only keep the current connected component
        cropped_core_instances[~cropped_mask] = 0
        cropped_border[~cropped_mask] = 0
        cropped_current = np.copy(cropped_core_instances)

        unique_core_idx = np.unique(cropped_core_instances)
        # do not use len(unique_core_idx) == 1 because there could be one code filling the entire thing
        if np.sum(cropped_core_instances) == 0:
            # special case no core
            if np.sum(cropped_border) > isolated_border_as_separate_instance_threshold_in_pixels:
                final[slicer][cropped_border] = max_component_idx + 1
                max_component_idx += 1
        elif len(unique_core_idx) == 2:
            # special case only one core = one object
            final[slicer][(cropped_core_instances > 0) | cropped_border] = unique_core_idx[1]
        else:
            already_dilated_mm = np.array((0, 0, 0))
            cropped_final = np.copy(cropped_core_instances)
            while np.sum(cropped_border) > 0:
                strel_size = [0, 0, 0]
                maximum_dilation = max(already_dilated_mm)
                for i in range(3):
                    if spacing[i] == min(spacing):
                        strel_size[i] = 1
                        continue
                    if already_dilated_mm[i] + spacing[i] / 2 < maximum_dilation:
                        strel_size[i] = 1
                ball_here = ball(1)

                if strel_size[0] == 0: ball_here = ball_here[1:2]
                if strel_size[1] == 0: ball_here = ball_here[:, 1:2]
                if strel_size[2] == 0: ball_here = ball_here[:, :, 1:2]

                #print(1)
                dilated = dilation(cropped_current, ball_here)
                diff = (cropped_current == 0) & (dilated != cropped_current)
                cropped_final[diff & cropped_border] = dilated[diff & cropped_border]
                cropped_border[diff] = 0
                cropped_current = dilated
                already_dilated_mm = [already_dilated_mm[i] + spacing[i] if strel_size[i] == 1 else already_dilated_mm[i] for i in range(3)]

            # now place result back
            final[slicer][cropped_mask] = cropped_final[cropped_mask]
    return final


def postprocess_instance_segmentation(instance_segmentation: np.ndarray):
    """
    Sometimes the dilation used to convert sem seg back to inst seg can cause fragmented instances. This is more ofd an artifact
    rather than real. This function can fix this by merging all but the largest connected component of each fragment
    with the nearest neighboring instances.

    :param instance_segmentation:
    :return:
    """
    unique_instances = [i for i in np.unique(instance_segmentation) if i != 0]
    strel = ball(1)
    for instance_id in unique_instances:
        instance_mask = instance_segmentation == instance_id

        bbox = get_bbox_from_mask_npwhere(instance_mask)
        bbox = pad_bbox(bbox, 1, instance_segmentation.shape)
        slicer = bounding_box_to_slice(bbox)

        cropped_instance = instance_segmentation[slicer]
        instance_mask = instance_mask[slicer]

        # let's see if this instance is fragmented
        labeled_cropped_instance, fragment_sizes = label_with_component_sizes(instance_mask, connectivity=1)
        if len(fragment_sizes) > 1:
            max_fragment_size = max(fragment_sizes.values())
            print(instance_id, fragment_sizes)
            for f in fragment_sizes.keys():
                if fragment_sizes[f] == max_fragment_size: continue
                fragment_mask = labeled_cropped_instance == f
                neighbor_mask = binary_dilation(fragment_mask, strel) != fragment_mask
                fragment_neighbors = cropped_instance[neighbor_mask]
                # remove background
                fragment_neighbors = fragment_neighbors[fragment_neighbors != 0]
                if len(fragment_neighbors) > 0:
                    counts = np.bincount(fragment_neighbors)
                    # replace fragment with most instance it shares the largest border with
                    instance_segmentation[slicer][fragment_mask] = np.argmax(counts)
    return instance_segmentation


def convert_instanceseg_to_semantic(instance_segmentation: np.ndarray, spacing: Union[Tuple, List] = (1, 1, 1),
                                    border_thickness: float = 2) -> np.ndarray:
    border_semantic = np.zeros_like(instance_segmentation, dtype=np.uint8)
    selem = generate_ball(spacing, [border_thickness] * 3)
    labels = np.unique(instance_segmentation)
    for label in labels:
        if label == 0:
            continue
        mask = instance_segmentation == label
        eroded = erosion(mask, selem)
        border_semantic[(~eroded) & mask] = BORDER_LABEL
        border_semantic[eroded & mask] = CENTER_LABEL
    return border_semantic


def convert_instanceseg_to_semantic_patched(instance_segmentation: np.ndarray, spacing: Union[Tuple, List] = (1, 1, 1),
                                            border_thickness: float = 2) -> np.ndarray:
    border_semantic = np.zeros_like(instance_segmentation, dtype=np.uint8)
    selem = generate_ball(spacing, [border_thickness] * 3)
    pad_amount = 0  # testing purposes only, erosion should not need padding
    instance_properties = regionprops(instance_segmentation)
    for ip in instance_properties:
        bbox = ip['bbox']
        if pad_amount != 0:
            bbox = pad_bbox(bbox, pad_amount, instance_segmentation.shape)
        slicer = bounding_box_to_slice(bbox)
        instance_cropped = instance_segmentation[slicer]
        instance_mask = instance_cropped == ip["label"]
        instance_mask_eroded = binary_erosion(instance_mask, selem)
        border_semantic[slicer][(~instance_mask_eroded) & instance_mask] = BORDER_LABEL
        border_semantic[slicer][instance_mask_eroded & instance_mask] = CENTER_LABEL
    return border_semantic


def _internal_convert_instanceseg_to_semantic_patched_mp(ip, selem: np.ndarray, cropped_is: np.ndarray):
    instance_mask = cropped_is == ip["label"]
    result = np.zeros(instance_mask.shape, dtype=np.uint8)
    instance_mask_eroded = binary_erosion(instance_mask, selem)
    result[(~instance_mask_eroded) & instance_mask] = BORDER_LABEL
    result[instance_mask_eroded & instance_mask] = CENTER_LABEL
    return result, instance_mask


def convert_instanceseg_to_semantic_patched_mp(instance_segmentation: np.ndarray,
                                               spacing: Union[Tuple, List] = (1, 1, 1),
                                               border_thickness: float = 2,
                                               num_processes: int = 8) -> np.ndarray:
    pool = Pool(num_processes)
    border_semantic = np.zeros_like(instance_segmentation, dtype=np.uint8)
    selem = generate_ball(spacing, [border_thickness] * 3)
    pad_amount = 0  # testing purposes only, erosion should not need padding
    instance_properties = regionprops(instance_segmentation)
    results = []
    slicers = []
    for ip in instance_properties:
        bbox = ip['bbox']
        if pad_amount != 0:
            bbox = pad_bbox(bbox, pad_amount, instance_segmentation.shape)
        slicer = bounding_box_to_slice(bbox)
        instance_cropped = instance_segmentation[slicer]
        results.append(
            pool.starmap_async(
                _internal_convert_instanceseg_to_semantic_patched_mp,
                ((ip, selem, instance_cropped),)
            )
        )
        slicers.append(slicer)

    for r, s in zip(results, slicers):
        semseg, instance_mask = r.get()
        border_semantic[s][instance_mask] = semseg[instance_mask]

    pool.close()
    pool.join()
    return border_semantic

