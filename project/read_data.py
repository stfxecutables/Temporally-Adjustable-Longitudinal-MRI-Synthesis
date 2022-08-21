import random
from typing import Optional, List
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import nibabel as nib
import matplotlib.pyplot as plt
from monai.transforms.intensity.array import ScaleIntensity
from monai.transforms import (
    LoadImaged,
    Spacingd,
    Compose,
    CropForeground,
    Orientationd,
    AddChanneld,
    apply_transform,
    SaveImaged,
    CropForegroundd,
    CenterSpatialCropd,
)
from monai.data import Dataset, NibabelReader

from utils.transforms import get_transforms
from utils.const import DATA_ROOT, MS_IMG_SIZE_NOVEL

exams_4 = set(
    [
        "test01",
        "test03",
        "test04",
        "test05",
        "test06",
        "test07",
        "test08",
        "test09",
        "test12",
        "test13",
        "training01",
        "training02",
        "training04",
        "training05",
    ]
)
exams_5 = set(["test02", "test11", "test14", "training03"])
exams_6 = set(["test10"])


def get_MS_preprocess_transforms() -> Compose:
    return Compose([CropForeground()])


def read_img(path: Path) -> np.ndarray:
    """
    Reads an image from a path.

    Args:
        path: Path to the image.

    Returns:
        The image as a numpy array.
    """
    img = nib.load(str(path))
    return img.get_fdata()


def load_MS_dataset():
    orig_training_subjects = [f"training0{i}" for i in range(1, 6)]
    orig_testing_subjects = [f"test{i:0>2d}" for i in range(1, 15)]
    all_subjects = []
    all_subjects.extend(orig_training_subjects)
    all_subjects.extend(orig_testing_subjects)
    all_subjects_paths = []

    for subject in all_subjects:
        cur_subjects_path = []
        if subject in exams_4:
            for i in range(1, 5):
                cur_subjects_path.append(list(DATA_ROOT.glob(f"**/*{subject}_0{i}_*_pp.nii")))
        elif subject in exams_5:
            for i in range(1, 6):
                cur_subjects_path.append(list(DATA_ROOT.glob(f"**/*{subject}_0{i}_*_pp.nii")))
        elif subject in exams_6:
            for i in range(1, 7):
                cur_subjects_path.append(list(DATA_ROOT.glob(f"**/*{subject}_0{i}_*_pp.nii")))
        all_subjects_paths.append(cur_subjects_path)

    transform = get_transforms()
    dim1_max, dim2_max, dim3_max = 0, 0, 0
    for subject in all_subjects_paths:
        for time_points in subject:
            img = apply_transform(transform, time_points)
            dim1_max, dim2_max, dim3_max = (
                max(dim1_max, img.shape[1]),
                max(dim2_max, img.shape[2]),
                max(dim3_max, img.shape[3]),
            )
    print(f"dim1_max: {dim1_max}, dim2_max: {dim2_max}, dim3_max: {dim3_max}")


def get_resample(output_dir: Path) -> Compose:
    return Compose(
        [
            LoadImaged(keys=["flair", "t1w", "t2w"]),
            AddChanneld(keys=["flair", "t1w", "t2w"]),
            Spacingd(keys=["flair", "t1w", "t2w"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
            # Orientationd(keys=["flair", "t1w", "t2w"], axcodes="RAS"),
            # CenterSpatialCropd(keys=["flair", "t1w", "t2w"], roi_size=MS_IMG_SIZE_NOVEL),
            SaveImaged(
                keys=["flair", "t1w", "t2w"],
                output_dir=output_dir,
                output_postfix="spacing",
                separate_folder=False,
            ),
        ]
    )


def resample_MS_dataset():
    folder = DATA_ROOT / "open_ms_data"
    patients = [f"patient{i:0>2d}" for i in range(1, 21)]

    # dim1_max, dim2_max, dim3_max = 0, 0, 0
    for patient in patients:
        patient_folder = folder / patient
        # brain_mask_path = patient_folder / f"brainmask.nii.gz"
        # brain_mask_array, brain_mask_info = load_img(brain_mask_path)
        for study in range(1, 3):
            flair_path = patient_folder / f"study{study}_FLAIR_skullstripped.nii.gz"
            t1w_path = patient_folder / f"study{study}_T1W_skullstripped.nii.gz"
            t2w_path = patient_folder / f"study{study}_T2W_skullstripped.nii.gz"

            data = {
                "flair": flair_path,
                "t1w": t1w_path,
                "t2w": t2w_path,
            }
            data = apply_transform(get_resample(output_dir=patient_folder), data)

            break
        break
    #         print(f"{patient} study{study}")
    #         print("flair: ", data["flair"].shape)
    #         print("t1w: ", data["t1w"].shape)
    #         print("t2w: ", data["t2w"].shape)

    #         dim1_max, dim2_max, dim3_max = (
    #             max(dim1_max, data["flair"].shape[1]),
    #             max(dim2_max, data["flair"].shape[2]),
    #             max(dim3_max, data["flair"].shape[3]),
    #         )
    # print(f"dim1_max: {dim1_max}, dim2_max: {dim2_max}, dim3_max: {dim3_max}")

    # flair_array, flair_info = load_img(flair_path)
    # t1w_array, t1w_info = load_img(t1w_path)
    # t2w_array, t2w_info = load_img(t2w_path)

    # flair_array_tmp = np.where(brain_mask_array > 0, flair_array, 0)
    # t1w_array_tmp = np.where(brain_mask_array > 0, t1w_array, 0)
    # t2w_array_tmp = np.where(brain_mask_array > 0, t2w_array, 0)

    # flair_array_tmp = np.expand_dims(flair_array_tmp, axis=0)
    # t1w_array_tmp = np.expand_dims(t1w_array_tmp, axis=0)
    # t2w_array_tmp = np.expand_dims(t2w_array_tmp, axis=0)

    # flair, t1w, t2w = data["flair"], data["t1w"], data["t2w"]
    # save_img = SaveImage(
    #     output_dir=patient_folder, output_postfix="skullstripped_resampled", separate_folder=False
    # )
    # save_img(img=flair_array_tmp, meta_data=flair_info)
    # save_img(img=t1w_array_tmp, meta_data=t1w_info)
    # save_img(img=t2w_array_tmp, meta_data=t2w_info)

    # print("Okay!")


if __name__ == "__main__":
    # time_point_1 = read_img(
    #     Path(__file__).resolve().parent.parent
    #     / "data/Multiple Sclerosis/training/training01/preprocessed/training01_01_flair_pp.nii"
    # )
    # time_point_2 = read_img(
    #     Path(__file__).resolve().parent.parent
    #     / "data/Multiple Sclerosis/training/training01/preprocessed/training01_02_flair_pp.nii"
    # )
    # scale_intensity = ScaleIntensity(minv=-1, maxv=1)
    # time_point_1, time_point_2 = scale_intensity(time_point_1), scale_intensity(time_point_2)
    # diff = time_point_2 - time_point_1

    # label_2 = read_img(
    #     Path(__file__).resolve().parent.parent
    #     / "data/Multiple Sclerosis/training/training01/masks/training01_02_mask1.nii"
    # )
    # load_MS_dataset()

    resample_MS_dataset()

    # folder = DATA_ROOT / "open_ms_data"
    # patients = [f"patient{i:0>2d}" for i in range(1, 21)]
    # load_img = LoadImage()

    # for patient in patients:
    #     patient_folder = folder / patient
    #     brain_mask_path = patient_folder / f"brainmask.nii.gz"
    #     brain_mask_array, brain_mask_info = load_img(brain_mask_path)

    #     for study in range(1, 3):
    #         flair_path = patient_folder / f"study{study}_FLAIR.nii.gz"
    #         flair_skull_stripped_path = patient_folder / f"study{study}_FLAIR_skullstripped.nii.gz"
    #         t1w_path = patient_folder / f"study{study}_T1W.nii.gz"
    #         t1w_path_skull_stripped = patient_folder / f"study{study}_T1W_skullstripped.nii.gz"
    #         t2w_path = patient_folder / f"study{study}_T2W.nii.gz"
    #         t2w_path_skull_stripped = patient_folder / f"study{study}_T2W_skullstripped.nii.gz"

    #         flair_array, flair_info = load_img(flair_path)
    #         flair_array_skull_stripped, flair_info_skull_stripped = load_img(flair_skull_stripped_path)
    #         t1w_array, t1w_info = load_img(t1w_path)
    #         t1w_array_skull_stripped, t1w_info_skull_stripped = load_img(t1w_path_skull_stripped)
    #         t2w_array, t2w_info = load_img(t2w_path)
    #         t2w_array_skull_stripped, t2w_info_skull_stripped = load_img(t2w_path_skull_stripped)

    #         print("Okay!")
