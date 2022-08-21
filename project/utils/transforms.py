import os
import random
from typing import Any, Dict, List, Tuple, Union

import nibabel as nib
import numpy as np
import torch
from monai.config import KeysCollection
from monai.transforms import (
    AddChannel,
    AddChanneld,
    AsChannelFirstd,
    CenterSpatialCrop,
    CenterSpatialCropd,
    Compose,
    CropForegroundd,
    EnsureTyped,
    LoadImage,
    LoadImaged,
    MapTransform,
    NormalizeIntensity,
    Orientation,
    Orientationd,
    RandAffined,
    Resize,
    SaveImaged,
    ScaleIntensity,
    ScaleIntensityd,
    Spacingd,
    SpatialCrop,
    SpatialPad,
    SpatialPadd,
)
from monai.transforms.compose import Transform
from PIL import GifImagePlugin
from torch import set_num_interop_threads
from utils.const import (
    MS_IMG_SIZE,
    MS_IMG_SIZE_NOVEL,
    PATCH_SIZE,
    PATCH_SIZE_2D,
    PATCH_SIZE_3D,
)


############################################################################
#                                                                          #
#                         Crop and Aggregator                              #
#                                                                          #
############################################################################
def compute_3Dpatch_loc(in_size: Tuple, out_size: Tuple = PATCH_SIZE_3D) -> np.array:
    num_patches = 8
    locs = np.zeros([num_patches, 3])

    # dim 1, 2, 3
    locs[1][0], locs[1][1], locs[1][2] = 0, (in_size[1] - out_size[1]), 0
    locs[2][0], locs[2][1], locs[2][2] = 0, 0, (in_size[2] - out_size[2])
    locs[3][0], locs[3][1], locs[3][2] = (
        0,
        (in_size[1] - out_size[1]),
        (in_size[2] - out_size[2]),
    )

    for i in range(4, 8):
        locs[i][0] = in_size[0] - out_size[0]
        for j in range(1, 3):
            locs[i][j] = locs[i - 4][j]

    return locs.astype(int)


def compute_2Dpatch_loc(in_size: Tuple, out_size: Tuple = PATCH_SIZE_2D) -> np.array:
    num_patches = 4
    locs = np.zeros([num_patches, 2])

    locs[1][0], locs[1][1] = (in_size[0] - out_size[0]), 0
    locs[2][0], locs[2][1] = 0, (in_size[1] - out_size[1])
    locs[3][0], locs[3][1] = (in_size[0] - out_size[0]), (in_size[1] - out_size[1])
    return locs.astype(int)


def patch_3D_aggregator(
    patches: np.ndarray,
    orig_shape: Tuple[int],
    patch_loc: np.array,
    count_ndarray: Union[np.array, None] = None,
) -> np.ndarray:
    """
    Aggregate patches to a whole 3D image.

    Args:
        patches: shape is [patch_num, Channel, patch_size, patch_size, patch_size]
        orig_shape: the image shape after aggregating
        patch_loc: the starting position where each patch in the original images
        count_ndarray: using to divide the aggregating image to average the overlapped regions
    """
    NUM_PATCH = 8

    dim_stack = []
    for dim in range(patches.shape[1]):
        orig = np.zeros(orig_shape)
        for idx in range(NUM_PATCH):
            orig[
                patch_loc[idx][0] : patch_loc[idx][0] + PATCH_SIZE,
                patch_loc[idx][1] : patch_loc[idx][1] + PATCH_SIZE,
                patch_loc[idx][2] : patch_loc[idx][2] + PATCH_SIZE,
            ] += patches[idx, dim, :, :, :]
        dim_stack.append(orig)

    orig = np.stack(dim_stack)
    if count_ndarray is not None:
        orig = np.divide(orig, count_ndarray)

    return orig.squeeze()


def patch_2D_aggregator(
    patches: np.ndarray,
    orig_shape: Tuple[int],
    patch_loc: np.array,
    count_ndarray: Union[np.array, None] = None,
) -> np.ndarray:
    """
    Aggregate patches to a whole 2D image.

    Args:
        patches: shape is [patch_num, Channel, patch_size, patch_size]
        orig_shape: the image shape after aggregating
        patch_loc: the starting position where each patch in the original images
        count_ndarray: using to divide the aggregating image to average the overlapped regions
    """
    NUM_PATCH = 4

    dim_stack = []
    for dim in range(patches.shape[1]):
        orig = np.zeros(orig_shape)
        for idx in range(NUM_PATCH):
            orig[
                patch_loc[idx][0] : patch_loc[idx][0] + PATCH_SIZE,
                patch_loc[idx][1] : patch_loc[idx][1] + PATCH_SIZE,
            ] += patches[idx, dim, :, :]
        dim_stack.append(orig)

    orig = np.stack(dim_stack)
    if count_ndarray is not None:
        orig = np.divide(orig, count_ndarray)

    return orig.squeeze()


def generate_patch(
    roi_start: List[int], img: np.ndarray, roi_size: Tuple[int] = PATCH_SIZE_3D
) -> np.ndarray:
    roi_center = [x + PATCH_SIZE // 2 for x in roi_start]
    spatial_crop = SpatialCrop(roi_center=roi_center, roi_size=roi_size)
    img = spatial_crop(img=img)
    return img


def get_synthesis_transforms() -> Compose:
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityd(
                keys=["image", "label"], minv=-1, maxv=1, channel_wise=True
            ),
            CenterSpatialCropd(keys=["image", "label"], roi_size=MS_IMG_SIZE),
        ]
    )


def get_transforms() -> Compose:
    return Compose(
        [
            LoadImage(),
            AddChannel(),
            Orientation(axcodes="RAS"),
            ScaleIntensity(minv=-1, maxv=1, channel_wise=True),
            CenterSpatialCrop(roi_size=MS_IMG_SIZE),
        ]
    )


def get_data_augmentation() -> Compose:
    return Compose(
        [
            RandAffined(
                keys=["image", "label"],
                mode=("bilinear", "nearest"),
                prob=0.5,
                rotate_range=(0, 0, np.pi / 15),
                scale_range=(0.1, 0.1, 0.1),
            )
        ]
    )


def map_to_range_from_zero_to_one(img: np.ndarray) -> np.ndarray:
    img = img - img.min()
    img = img / img.max()
    return img


def get_niigz_filename(file_path: str) -> str:
    return os.path.basename(file_path).split(".")[0]


# Code is from: https://github.com/taozh2017/HiNet/blob/master/funcs/utils.py#L328
class LambdaLR:
    def __init__(self, n_epochs: int, offset: int, decay_start_epoch: int):
        assert (
            n_epochs - decay_start_epoch
        ) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch: int) -> Any:
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (
            self.n_epochs - self.decay_start_epoch
        )
