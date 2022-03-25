import sys
from pathlib import Path
sys.path.append(str(Path('').absolute().parent))

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join
from nnunet.preprocessing.cropping import get_bbox_from_mask
from skimage.measure import label as ski_label
from skimage.morphology import ball
from skimage.morphology import dilation
from tqdm import tqdm
import os
from helper import utils
from skimage.measure import regionprops
import copy
import argparse
from tqdm import tqdm


def all_border_semantic2instance(load_dir, save_dir, small_center_threshold=30,
                                 isolated_border_as_separate_instance_threshold: int = 15, border_label=2, core_label=1):
    filenames = utils.load_filepaths(load_dir, extensions=".nii.gz")
    for filename in tqdm(filenames, desc="Image conversion"):
        border_semantic2instance_patchify(filename, join(save_dir, os.path.basename(filename)), small_center_threshold, isolated_border_as_separate_instance_threshold, border_label, core_label)


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input", required=True,
                        help="Absolute input path to the folder or file that should be converted to instance segmentation. In case a folder is given, all .nii.gz files will be converted.")
    parser.add_argument('-o', "--output", required=True, help="Absolute output path to the folder or file that should be used for saving the instance segmentation.")
    args = parser.parse_args()

    input = args.input
    output = args.output

    if input.endswith(".nii.gz"):
        border_semantic2instance_patchify(input, output)
    else:
        all_border_semantic2instance(input, output)

