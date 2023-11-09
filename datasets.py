import os
import xmltodict
import numpy as np
from copy import deepcopy
from itertools import product
import xml.etree.ElementTree as et
from skimage import io as skio
from skimage.util.shape import view_as_blocks
from torch.utils.data.dataset import Dataset


''' Utility function for patch creation '''


def centers_to_slice(voxels, patch_half):
    """
    Function to convert a list of indices defining the center of a patch, to
    a real patch defined using slice objects for each dimension.
    :param voxels: List of indices to the center of the slice.
    :param patch_half: List of integer halves (//) of the patch_size.
    """
    slices = [
        tuple(
            [
                slice(idx - p_len, idx + p_len) for idx, p_len in zip(
                    voxel, patch_half
                )
            ]
        ) for voxel in voxels
    ]
    return slices


def get_slices(masks, patch_size, overlap):
    """
    Function to get all the patches with a given patch size and overlap between
    consecutive patches from a given list of masks. We will only take patches
    inside the bounding box of the mask. We could probably just pass the shape
    because the masks should already be the bounding box.
    :param masks: List of masks.
    :param patch_size: Size of the patches.
    :param overlap: Overlap on each dimension between consecutive patches.

    """
    # Init
    # We will compute some intermediate stuff for later.
    patch_half = [p_length // 2 for p_length in patch_size]
    steps = [max(p_length - o, 1) for p_length, o in zip(patch_size, overlap)]
    # indices = [np.where(mask) for mask in masks]
    # min_indices = [np.min(idx) for idx in indices]
    # max_indices = [np.max(idx) for idx in indices]
    # min_bb = np.min(min_indices, axis=0)
    # max_bb = np.max(max_indices)

    # We will need to define the min and max pixel indices. We define the
    # centers for each patch, so the min and max should be defined by the
    # patch halves.
    # min_bb = [patch_half] * len(masks)
    min_bb = [
        [
            max(patch_len, min_i)
            for min_i, patch_len in zip(
                np.min(np.where(mask), axis=-1), patch_half
            )
        ] for mask in masks
    ]
    # max_bb = [
    #     [
    #         max_i - p_len for max_i, p_len in zip(mask.shape, patch_half)
    #     ] for mask in masks
    # ]
    max_bb = [
        [
            min(bound_i - patch_len, max_i)
            for bound_i, max_i, patch_len in zip(
                mask.shape, np.max(np.where(mask), axis=-1), patch_half
            )
        ] for mask in masks
    ]

    # This is just a "pythonic" but complex way of defining all possible
    # indices given a min, max and step values for each dimension.
    dim_ranges = [
        map(
            lambda t: np.concatenate([np.arange(*t), [t[1]]]),
            zip(min_bb_i, max_bb_i, steps)
        ) for min_bb_i, max_bb_i in zip(min_bb, max_bb)
    ]

    # And this is another "pythonic" but not so intuitive way of computing
    # all possible triplets of center voxel indices given the previous
    # indices. I also added the slice computation (which makes the last step
    # of defining the patches).
    patch_slices = [
        centers_to_slice(
            itertools.product(*dim_range), patch_half
        ) for dim_range in dim_ranges
    ]

    return patch_slices


'''
Dataset classes
'''


# Rumex detection dataset
class RumexDataset(Dataset):
    """
    Dataset that uses a preloaded tensor with natural images, including
    classification labels.
    """
    def __init__(self, path, basenames, patch_size, norm=True):
        patch_list = []
        alpha_list = []
        label_list = []
        for base_filename in basenames:
            # parse xml file of 'base_filename' and get image metadata info
            quadrant = base_filename.split("_")[-1]
            imfile = base_filename + '.png'
            xmlfile = base_filename + '.xml'
            root = et.parse(os.path.join(path, xmlfile)).getroot()
            xmlstr = et.tostring(root, encoding='utf-8', method='xml')
            xmldict = dict(xmltodict.parse(xmlstr))

            # actual image size
            im_size = xmldict['annotation']['size']
            im_height = int(im_size['height'])
            im_width = int(im_size['width'])

            # crop image to be a multiple of patch_size
            npatches_w = im_width // patch_size
            npatches_h = im_height // patch_size
            im = skio.imread(os.path.join(path, imfile))

            # split image into patches
            ### create labels for each patch: rumex, other or outside ###
            # step 1: get rumex bbox from xml file
            # step 2: if bbox overlaps with image patch label it as rumex
            # Note: use mask to determine if the patches are within or outside
            # field

            # logic of finding the overlap between patch and ground truth bbox
            # 1. find the patches that the true rumex bbox straddles:
            #    eg: bbox[0] // patch_size
            # 2. label those patches as rumex
            alpha = im[:npatches_h * patch_size, :npatches_w * patch_size, 3]
            im = np.moveaxis(
                im[:npatches_h * patch_size, :npatches_w * patch_size, :3],
                -1, 0
            )
            if norm:
                im_mean = np.mean(im, axis=(1, 2), keepdims=True)
                im_std = np.std(im, axis=(1, 2), keepdims=True)
                im_norm = (im - im_mean) / im_std
                im_norm = im / 255
                im_patches = np.squeeze(
                    view_as_blocks(im_norm, (3, patch_size, patch_size))
                )
            else:
                im_patches = np.squeeze(
                    view_as_blocks(im, (3, patch_size, patch_size))
                )
            alpha_patches = np.squeeze(
                view_as_blocks(alpha, (patch_size, patch_size))
            )
            alpha_mask = np.mean(alpha_patches, axis=(2, 3)).flatten() > 0
            labels = np.zeros(im_patches.shape[:2])

            objects = xmldict['annotation']['object']
            nobjects = len(objects)
            for i in range(nobjects):
                obj_i = objects[i]

                if obj_i['name'] == 'rumex':
                    temp = obj_i['bndbox']
                    bbox = [
                        int(temp['xmin']), int(temp['ymin']),
                        int(temp['xmax']), int(temp['ymax'])
                    ]

                    xmin_r = bbox[0] // patch_size
                    ymin_r = bbox[1] // patch_size
                    xmax_r = bbox[2] // patch_size
                    ymax_r = bbox[3] // patch_size

                    if (xmax_r < npatches_w) & (ymax_r < npatches_h):
                        if xmax_r - xmin_r >= 1:
                            x_patches_with_rumex = list(
                                np.arange(xmin_r, xmax_r + 1)
                            )
                        else:
                            x_patches_with_rumex = [xmin_r]

                        if ymax_r - ymin_r >= 1:
                            y_patches_with_rumex = list(
                                np.arange(ymin_r, ymax_r + 1)
                            )
                        else:
                            y_patches_with_rumex = [ymin_r]

                        for col in x_patches_with_rumex:
                            for row in y_patches_with_rumex:
                                labels[row, col] = 1
            patch_list.append(
                np.reshape(
                    im_patches, (-1, 3, patch_size, patch_size)
                )[alpha_mask, ...]
            )
            label_list.append(labels.flatten()[alpha_mask])
        self.data = np.concatenate(patch_list, axis=0)
        self.labels = np.concatenate(label_list, axis=0)

    def __getitem__(self, index):
        x = self.data[index].astype(np.float32)
        y = self.labels[index].astype(np.uint8)

        return x, y

    def __len__(self):
        return len(self.data)


class BalancedRumexDataset(RumexDataset):
    """
    Dataset that uses a preloaded tensor with natural images, including
    classification labels.
    """
    def __init__(self, path, basenames, patch_size, norm=True):
        super().__init__(path, basenames, patch_size, norm)
        self.plant = np.where(self.labels.astype(bool))[0]
        self.background = np.where(np.logical_not(self.labels.astype(bool)))[0]
        self.current_background = deepcopy(self.background).tolist()

    def __getitem__(self, index):
        if index < len(self.plant):
            true_index = self.plant[index]
        else:
            random_index = np.random.randint(len(self.current_background))
            true_index = self.current_background.pop(random_index)
            if len(self.current_background) == 0:
                self.current_background = deepcopy(self.background).tolist()
        return super().__getitem__(true_index)

    def __len__(self):
        return len(self.plant) * 2


class RumexTestDataset(Dataset):
    """
    Dataset that uses a preloaded tensor with natural images, including
    classification labels.
    """
    def __init__(self, path, basenames, patch_size, norm=True):
        self.patch_size = patch_size
        self.images = []
        self.patches = []
        label_list = []
        for idx_m, base_filename in enumerate(basenames):
            # parse xml file of 'base_filename' and get image metadata info
            imfile = base_filename + '.png'
            xmlfile = base_filename + '.xml'
            root = et.parse(os.path.join(path, xmlfile)).getroot()
            xmlstr = et.tostring(root, encoding='utf-8', method='xml')
            xmldict = dict(xmltodict.parse(xmlstr))

            # actual image size
            im_size = xmldict['annotation']['size']
            im_height = int(im_size['height'])
            im_width = int(im_size['width'])

            # pad the image to be a multiple of patch_size
            npatches_w = int(np.ceil(im_width / patch_size))
            npatches_h = int(np.ceil(im_height / patch_size))
            new_w = npatches_w * patch_size
            new_h = npatches_h * patch_size
            pad_w = new_w - im_width
            pad_h = new_h - im_height

            mosaic = skio.imread(os.path.join(path, imfile))
            im = mosaic[..., :3]
            if norm:
                im_mean = np.mean(im, axis=(1, 2), keepdims=True)
                im_std = np.std(im, axis=(1, 2), keepdims=True)
                im_norm = (im - im_mean) / im_std
                im_norm = im / 255
                self.images.append(
                    np.moveaxis(
                        np.pad(
                            im_norm,
                            ((0, pad_h), (0, pad_w), (0, 0))
                        ), -1, 0
                    )
                )
            else:
                self.images.append(
                    np.moveaxis(
                        np.pad(im, ((0, pad_h), (0, pad_w), (0, 0))), -1, 0
                    )
                )

            alpha = np.pad(mosaic[..., 3], ((0, pad_h), (0, pad_w)))

            im_patches = [
                (
                    idx_m,
                    slice(
                        idx_i * patch_size,
                        idx_i * patch_size + patch_size
                    ),
                    slice(
                        idx_j * patch_size,
                        idx_j * patch_size + patch_size
                    )
                )
                for idx_i, idx_j in product(
                    range(npatches_h), range(npatches_w)
                )
            ]

            labels = np.zeros(len(im_patches))

            objects = xmldict['annotation']['object']
            nobjects = len(objects)
            for i in range(nobjects):
                obj_i = objects[i]

                if obj_i['name'] == 'rumex':
                    temp = obj_i['bndbox']
                    bbox = [
                        int(temp['xmin']), int(temp['ymin']),
                        int(temp['xmax']), int(temp['ymax'])
                    ]

                    xmin_r = bbox[0] // patch_size
                    ymin_r = bbox[1] // patch_size
                    xmax_r = bbox[2] // patch_size
                    ymax_r = bbox[3] // patch_size

                    if (xmax_r < npatches_w) & (ymax_r < npatches_h):
                        if xmax_r - xmin_r >= 1:
                            x_patches_with_rumex = list(
                                np.arange(xmin_r, xmax_r + 1)
                            )
                        else:
                            x_patches_with_rumex = [xmin_r]

                        if ymax_r - ymin_r >= 1:
                            y_patches_with_rumex = list(
                                np.arange(ymin_r, ymax_r + 1)
                            )
                        else:
                            y_patches_with_rumex = [ymin_r]

                        for col in x_patches_with_rumex:
                            for row in y_patches_with_rumex:
                                labels[row * npatches_w + col] = 1

            self.patches.extend([
                (m, x_idx, y_idx) for m, x_idx, y_idx in im_patches
                if np.mean(alpha[x_idx, y_idx]) > 0
            ])
            label_list.append(np.stack([
                l for (_, x_idx, y_idx), l in zip(im_patches, labels)
                if np.mean(alpha[x_idx, y_idx]) > 0
            ]))
        self.labels = np.concatenate(label_list, axis=0)

    def __getitem__(self, index):
        m, slice_i, slice_j = self.patches[index]
        x = self.images[m][:, slice_i, slice_j].astype(np.float32)
        y = self.labels[index].astype(np.uint8)
        patch_coords = (
            (slice_i.start, slice_i.stop),
            (slice_j.start, slice_j.stop)
        )
        return x, y, patch_coords

    def __len__(self):
        return len(self.patches)


# Lythrum dataset
class LythrumDataset(Dataset):
    """
    Dataset that uses a preloaded tensor with natural images, including
    classification labels.
    """
    def __init__(self, mosaic, mask, patch_size, norm=True):
        if norm:
            im_mean = np.mean(mosaic, axis=(1, 2), keepdims=True)
            im_std = np.std(mosaic, axis=(1, 2), keepdims=True)
            self.mosaic = (mosaic - im_mean) / im_std
        else:
            self.mosaic = mosaic
        self.mask = mask
        self.patches = get_slices([mask], patch_size, patch_size // 2)[0]

    def __getitem__(self, index):
        patch = self.patches[index]
        x = self.mosaic[patch].astype(np.float32)
        y = self.mask[patch].astype(np.uint8)

        return x, y

    def __len__(self):
        return len(self.patches)


class BalancedLythrumDataset(Dataset):
    """
    Dataset that uses a preloaded tensor with natural images, including
    classification labels.
    """
    def __init__(self, mosaic, mask, patch_size, norm=True):
        if norm:
            im_mean = np.mean(mosaic, axis=(1, 2), keepdims=True)
            im_std = np.std(mosaic, axis=(1, 2), keepdims=True)
            self.mosaic = (mosaic - im_mean) / im_std
        else:
            self.mosaic = mosaic
        self.mask = mask
        patches = get_slices([mask], patch_size, patch_size // 2)[0]

        patch_slices = [s for s in patches if np.sum(mask[s]) > 0]
        bck_slices = [s for s in patches if np.sum(mask[s]) == 0]
        n_positives = len(patch_slices)
        n_negatives = len(bck_slices)

        positive_imbalance = n_positives > n_negatives

        if positive_imbalance:
            self.majority = patch_slices
            self.majority_label = np.array([1], dtype=np.uint8)

            self.minority = bck_slices
            self.minority_label = np.array([0], dtype=np.uint8)
        else:
            self.majority = bck_slices
            self.majority_label = np.array([0], dtype=np.uint8)

            self.minority = patch_slices
            self.minority_label = np.array([1], dtype=np.uint8)

        self.current_majority = deepcopy(self.majority)
        self.current_minority = deepcopy(self.minority)

    def __getitem__(self, index):
        if index < len(self.minority):
            index = np.random.randint(len(self.current_minority))
            patch = self.current_minority.pop(index)
            if len(self.current_minority) == 0:
                self.current_minority = deepcopy(self.minority)
        else:
            index = np.random.randint(len(self.current_majority))
            patch = self.current_majority.pop(index)
            if len(self.current_majority) == 0:
                self.current_majority = deepcopy(self.majority)
        x = self.mosaic[patch].astype(np.float32)
        y = self.mask[patch].astype(np.uint8)

        return x, y

    def __len__(self):
        return len(self.minority) * 2
