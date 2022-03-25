from skimage.morphology import binary_erosion
import argparse
import copy
import os
from functools import partial

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join
from nnunet.preprocessing.cropping import get_bbox_from_mask
from skimage.measure import label as ski_label
from skimage.measure import regionprops
from skimage.morphology import ball
from skimage.morphology import binary_erosion
from skimage.morphology import dilation
from tqdm import tqdm



def border_semantic2instance_patchify(load_filepath, save_filepath, small_center_threshold=30,
                                      isolated_border_as_separate_instance_threshold: int = 15, border_label=2, core_label=1):
    border_semantic_seg, spacing, affine, header = utils.load_nifti(load_filepath, is_seg=True, return_meta=True)
    border_semantic_seg = border_semantic_seg.astype(np.uint16)
    border_semantic_seg_binary = border_semantic_seg > 0
    pseudo_instance_seg = ski_label(border_semantic_seg_binary)
    instance_seg = np.zeros_like(border_semantic_seg)
    max_instance = 0

    for index, region in enumerate(tqdm(regionprops(pseudo_instance_seg))):
        i_min, j_min, k_min, i_max, j_max, k_max = region.bbox
        filter_mask = pseudo_instance_seg[i_min:i_max, j_min:j_max, k_min:k_max] == region.label
        border_semantic_seg_patch = copy.deepcopy(border_semantic_seg[i_min:i_max, j_min:j_max, k_min:k_max])
        border_semantic_seg_patch[filter_mask != 1] = 0
        instance_seg_patch = border_semantic2instance(border_semantic_seg_patch, spacing, small_center_threshold, isolated_border_as_separate_instance_threshold, border_label, core_label)
        instance_seg_patch[instance_seg_patch > 0] += max_instance
        max_instance = max(max_instance, np.max(instance_seg_patch))
        patch_labels = np.unique(instance_seg_patch)
        index = np.argwhere(patch_labels == 0)
        patch_labels = np.delete(patch_labels, index)
        for patch_label in patch_labels:
            instance_seg[i_min:i_max, j_min:j_max, k_min:k_max][instance_seg_patch == patch_label] = patch_label
    utils.save_nifti(save_filepath, instance_seg, is_seg=True, spacing=spacing, dtype=np.uint16)


def border_semantic2instance(arr, spacing: tuple = (0.2, 0.125, 0.125), small_center_threshold=30,
                             isolated_border_as_separate_instance_threshold: int = 15, border_label=2, core_label=1):
    spacing = np.array(spacing)

    # we first identify centers that are too small and set them to be border. This should remove false positive instances
    core_instances = ski_label((arr == core_label).astype(int), connectivity=1)
    for o in np.unique(core_instances):
        if o > 0 and np.sum(core_instances == o) <= small_center_threshold:
            arr[core_instances == o] = border_label

    # 1 is core, 2 is border
    core_instances = ski_label((arr == core_label).astype(int))

    # prepare empty array for results
    final = np.zeros_like(core_instances)

    # besides the core instances we will need the borders
    border = arr == border_label

    # we search for connected components and then convert each connected component into instance segmentation. This should
    # prevent bleeding into neighboring instances even if the instances don't touch
    connected_components, num_components = ski_label((arr > 0).astype(int), return_num=True, connectivity=1)
    max_component_idx = np.max(core_instances)

    for cidx in range(1, num_components+1):
        mask = connected_components == cidx
        bbox = get_bbox_from_mask(mask)
        slicer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))

        cropped_mask = mask[slicer]
        cropped_core_instances = np.copy(core_instances[slicer])
        cropped_border = np.copy(border[slicer])
        cropped_current = np.copy(cropped_core_instances)

        # remove other objects from the current crop, only keep the current connected component
        cropped_core_instances[~cropped_mask] = 0
        cropped_border[~cropped_mask] = 0
        cropped_current[~cropped_mask] = 0

        unique_core_idx = np.unique(cropped_core_instances)
        if np.sum(cropped_core_instances) == 0:
            # special case no core
            if np.sum(cropped_border) > isolated_border_as_separate_instance_threshold:
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
                already_dilated_mm = [already_dilated_mm[i] + spacing[i] if strel_size[i] == core_label else already_dilated_mm[i] for i in range(3)]

            # now place result back
            final[slicer][cropped_mask] = cropped_final[cropped_mask]

    # now postprocess those pesky floating pixels...
    unique_instances = [i for i in np.unique(final) if i != 0]
    strel = ball(1)
    shape = final.shape
    for u in unique_instances:
        mask = final == u
        bbox = get_bbox_from_mask(mask)
        bbox = [[max(0, i[0] - 1), min(shape[j], i[1] + 1)] for j, i in enumerate(bbox)]
        slicer = tuple([slice(*i) for i in bbox])
        cropped_instance = (final[slicer] == u).astype(int)
        # let's see if this instance is fragmented
        labeled_cropped_instance, num_fragments = ski_label(np.copy(cropped_instance), return_num=True, connectivity=1)
        if num_fragments > 1:
            fragment_sizes = [np.sum(labeled_cropped_instance == i) for i in range(1, num_fragments + 1)]
            print(u, fragment_sizes)
            for f in range(1, num_fragments + 1):
                if fragment_sizes[f - 1] == max(fragment_sizes): continue
                fragment_mask = (labeled_cropped_instance == f).astype(int)
                neighbor_mask = (dilation(fragment_mask, strel) - fragment_mask).astype(bool)
                fragment_neighbors = final[slicer][neighbor_mask]
                # remove background
                fragment_neighbors = fragment_neighbors[fragment_neighbors != 0]
                if len(fragment_neighbors) > 0:  # TODO: This is only temporary
                    # print("fragment_neighbors: ", fragment_neighbors.shape)
                    assert len(fragment_neighbors) > 0, 'this should not happen'
                    counts = np.bincount(fragment_neighbors)
                    # replace fragment with most instance it shares the largest border with
                    final[slicer][fragment_mask.astype(bool)] = np.argmax(counts)

    return final.astype(np.uint32)


def all_instance2border_semantic(load_filepath, save_filepath, border_label=2, core_label=1, border_thickness_in_pixel=5, parallel=0):
    filenames = utils.load_filepaths(load_filepath)
    for filename in tqdm(filenames, desc="Image conversion"):
        single_instance2border_semantic(filename, join(save_filepath, os.path.basename((filename))), border_label, core_label, border_thickness_in_pixel, parallel)


def single_instance2border_semantic(load_filepath, save_filepath, border_label=2, core_label=1, border_thickness_in_pixel=5, parallel=0):
    label_img, spacing, affine, header = utils.load_nifti(load_filepath, is_seg=True, return_meta=True)
    border_semantic = instance2border_semantic_process(label_img, border_label=border_label, core_label=core_label, border_thickness_in_pixel=border_thickness_in_pixel, progress_bar=True, parallel=parallel)
    utils.save_nifti(save_filepath, border_semantic, is_seg=True, spacing=spacing, dtype=np.uint16)


# Works only for isotropic images. generate_ball uses size_conversion_factor, which is a single value -> Same thickness in every dimension.
def instance2border_semantic_process(instance_seg, border_label=2, core_label=1, border_thickness_in_pixel=5, progress_bar=False, parallel=0):
    # start_time = time.time()
    instance_seg = instance_seg.astype(np.uint16)
    selem = ball(border_thickness_in_pixel, dtype=int)
    border_semantic_particles = []

    if parallel == 0:
        for instance in tqdm(regionprops(instance_seg), desc="Instance conversion", disable=not progress_bar):
            border_semantic_particle = instance2border_semantic_particle(instance, instance_seg, selem, border_label, core_label, border_thickness_in_pixel + 3)
            border_semantic_particles.append(border_semantic_particle)
    else:  # parallel is for some reason much slower. Don't use!
        pool, _ = global_mp_pool.get_pool()
        instances = []
        for instance in regionprops(instance_seg):
            instances.append({"bbox": instance["bbox"], "label": instance["label"]})
        pool.map(partial(instance2border_semantic_particle, instance_seg=instance_seg, selem=selem, border_label=border_label, core_label=core_label, roi_padding=border_thickness_in_pixel + 3), instances)

    border_semantic = np.zeros_like(instance_seg, dtype=np.uint8)
    for border_semantic_particle in border_semantic_particles:
        i_start, j_start, k_start, i_end, j_end, k_end = border_semantic_particle["bbox"]
        mask = border_semantic_particle["image"]
        border_semantic[i_start:i_end, j_start:j_end, k_start:k_end][mask == border_label] = border_label
        border_semantic[i_start:i_end, j_start:j_end, k_start:k_end][mask == core_label] = core_label

    # print("Elapsed time: ", time.time() - start_time)
    return border_semantic


def instance2border_semantic_particle(instance, instance_seg, selem, border_label, core_label, roi_padding):
    border_semantic_particle = {}
    i_start, j_start, k_start, i_end, j_end, k_end = instance["bbox"]
    # Pad the roi to improve quality of the erosion
    i_start, j_start, k_start, i_end, j_end, k_end = max(0, i_start - roi_padding), max(0, j_start - roi_padding), max(0, k_start - roi_padding), \
                                                     min(instance_seg.shape[0], i_end + roi_padding), min(instance_seg.shape[1], j_end + roi_padding), min(instance_seg.shape[2], k_end + roi_padding)
    border_semantic_particle["bbox"] = i_start, j_start, k_start, i_end, j_end, k_end
    roi_mask = instance_seg[i_start:i_end, j_start:j_end, k_start:k_end] == instance["label"]
    border_semantic_particle["image"] = roi_mask.astype(np.uint8)
    eroded = binary_erosion(roi_mask, selem)
    border_semantic_particle["image"][(eroded == 0) & (roi_mask == 1)] = border_label
    border_semantic_particle["image"][(eroded == 1) & (roi_mask == 1)] = core_label
    return border_semantic_particle

