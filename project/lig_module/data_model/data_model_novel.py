import random
from typing import Optional, List, Any, Tuple, Union
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from monai.transforms import apply_transform
from monai.utils import set_determinism
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold

from utils.const import DATA_ROOT, NUM_WORKERS, MS_IMG_SIZE_NOVEL
from utils.transforms import (
    get_novel_synthesis_transforms,
    generate_patch,
    get_data_augmentation,
    compute_3Dpatch_loc,
)

days_between_studies = [
    203,
    283,
    671,
    155,
    380,
    473,
    317,
    305,
    116,
    236,
    405,
    81,
    329,
    723,
    168,
    164,
    321,
    524,
    228,
    454,
]


def get_imgs(subject: str, idx: int, idy: int) -> Tuple[List[Path], Path]:
    """
    Get X and y images for one sample

    return:
        X_imgs: list of 4 paths
        y_imgs: path of label image
    """
    X_imgs_path: List[Path] = []
    # X_imgs
    X_imgs_path.append(*list(DATA_ROOT.glob(f"**/*{subject}_0{idx}_flair_pp.nii")))
    X_imgs_path.append(*list(DATA_ROOT.glob(f"**/*{subject}_0{idx}_mprage_pp.nii")))
    X_imgs_path.append(*list(DATA_ROOT.glob(f"**/*{subject}_0{idx}_pd_pp.nii")))
    X_imgs_path.append(*list(DATA_ROOT.glob(f"**/*{subject}_0{idx}_t2_pp.nii")))
    # y_img
    y_img_path = list(DATA_ROOT.glob(f"**/*{subject}_0{idy}_flair_pp.nii"))
    return X_imgs_path, y_img_path[0]


def get_samples(
    subject: str,
    n_scans: int,
    X_paths: List[List[Path]],
    y_paths: List[Path],
    predict_time: List[int],
    sample_ids: List[str],
) -> None:
    """
    Classify subjects from time points and get the samples
    """
    for idx in range(1, n_scans):
        for idy in range(idx + 1, n_scans + 1):
            cur_X_imgs_path, cur_y_img_path = get_imgs(subject, idx, idy)
            cur_predict_time = idy - idx
            X_paths.append(cur_X_imgs_path)
            y_paths.append(cur_y_img_path)
            predict_time.append(cur_predict_time)
            sample_ids.append(f"{subject}_{idx}_{idy}")


def build_ndarray_and_create_idx_to_img(paths: List[Any]) -> np.ndarray:
    """
    to create 8 patches from the same img
    """
    paths = np.array(paths)
    length = len(paths)
    paths = np.repeat(paths, 8, axis=0)
    patches_idx = np.tile(np.arange(0, 8), length)
    return np.column_stack((paths, patches_idx))


class SynthesisDataset(Dataset):
    def __init__(
        self,
        X_paths: Union[List[List[Path]], np.ndarray],
        y_paths: Union[List[Path], np.ndarray],
        predict_times: Union[List[int], np.ndarray],
        sample_ids: Union[List[str], np.ndarray],
        is_val: bool,
        class_for_each_sample: List[int] = None,
    ):
        self.X_paths = X_paths
        self.y_paths = y_paths
        self.predict_times = predict_times
        self.sample_ids = sample_ids
        self.is_val = is_val
        self.class_for_each_sample = class_for_each_sample
        self.transform = get_novel_synthesis_transforms()
        self.patch_loc = compute_3Dpatch_loc(in_size=MS_IMG_SIZE_NOVEL)

    def __len__(self):
        return int(len(self.y_paths))

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        if not self.is_val:
            X_path, y_path, predict_time, sample_id, class_ = (
                self.X_paths[i][:-1].tolist(),
                self.y_paths[i][0],
                self.predict_times[i][0],
                self.sample_ids[i][0],
                self.class_for_each_sample[i][0],
            )
        else:
            X_path, y_path, predict_time, sample_id = (
                self.X_paths[i],
                self.y_paths[i],
                self.predict_times[i],
                self.sample_ids[i],
            )

        data = {"image": [str(path) for path in X_path], "label": str(y_path)}
        data = apply_transform(self.transform, data)
        X_img, y_img = data["image"], data["label"]

        if not self.is_val:
            idx = self.X_paths[i][-1]
            roi_start = self.patch_loc[idx].tolist()
            X_img = generate_patch(roi_start=roi_start, img=X_img)
            y_img = generate_patch(roi_start=roi_start, img=y_img)
            data = {"image": X_img, "label": y_img}
            data = apply_transform(get_data_augmentation(), data)
            X_img, y_img = data["image"], data["label"]
        else:
            X_patches_list = []
            y_patches_list = []
            for idx in range(8):
                roi_start = self.patch_loc[idx].tolist()
                X_patch = generate_patch(roi_start=roi_start, img=X_img)
                y_patch = generate_patch(roi_start=roi_start, img=y_img)
                X_patches_list.append(X_patch)
                y_patches_list.append(y_patch)
            X_img = np.stack(X_patches_list)
            y_img = np.stack(y_patches_list)

        if not self.is_val:
            return (
                torch.from_numpy(X_img).float(),
                torch.from_numpy(y_img).float(),
                torch.tensor([predict_time] * 8).float().view(1, 2, 2, 2),
                sample_id,
                torch.tensor(class_).long(),
            )
        else:
            return (
                torch.from_numpy(X_img).float(),
                torch.from_numpy(y_img).float(),
                torch.tensor([predict_time] * 8).float().view(1, 2, 2, 2),
                sample_id,
            )


class NovelDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, kfold_num: int):
        super().__init__()
        self.batch_size = batch_size
        self.kfold_num = kfold_num
        # self.class_for_each_sample = (
        #     np.array(
        #         [  # From KMeans(n_clusters=5)
        #             2,
        #             3,
        #             5,
        #             1,
        #             3,
        #             4,
        #             3,
        #             3,
        #             1,
        #             2,
        #             4,
        #             1,
        #             3,
        #             5,
        #             1,
        #             1,
        #             3,
        #             4,
        #             2,
        #             4,
        #         ]
        #     )
        #     - 1
        # )
        self.class_for_each_sample = np.array(
            [  # From KMeans(n_clusters=4)
                0,
                1,
                3,
                0,
                2,
                2,
                1,
                1,
                0,
                1,
                2,
                0,
                1,
                3,
                0,
                0,
                1,
                2,
                1,
                2,
            ]
        )

    def prepare_data(self, *args, **kwargs):
        # set deterministic training for reproducibility
        # use the default seed
        set_determinism()
        return super().prepare_data(*args, **kwargs)

    # called on every process in DDP
    def setup(self, stage: Optional[str] = None) -> None:
        years = [tmp / 365 for tmp in days_between_studies]
        random_state = random.randint(0, 100)
        kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

        flair_paths = sorted(list(DATA_ROOT.glob("**/*study1_FLAIR_skullstripped.nii.gz")))
        t1_paths = sorted(list(DATA_ROOT.glob("**/*study1_T1W_skullstripped.nii.gz")))
        t2_paths = sorted(list(DATA_ROOT.glob("**/*study1_T2W_skullstripped.nii.gz")))
        sample_ids = [f"patient{idx:0>2d}" for idx in range(1, 21)]
        y_paths = sorted(list(DATA_ROOT.glob("**/*study2_FLAIR_skullstripped.nii.gz")))
        for idx, (train_index, val_index) in enumerate(kf.split(days_between_studies)):
            if self.kfold_num != (idx + 1):
                continue

            flair_paths, t1_paths, t2_paths, y_paths, years, sample_ids = (
                np.array(flair_paths),
                np.array(t1_paths),
                np.array(t2_paths),
                np.array(y_paths),
                np.array(years),
                np.array(sample_ids),
            )
            train_X_paths = [
                [t1_path, t2_path, flair_path]
                for flair_path, t1_path, t2_path in zip(
                    flair_paths[train_index], t1_paths[train_index], t2_paths[train_index]
                )
            ]
            train_y_paths, train_predict_time, train_sample_ids, train_sample_class = (
                y_paths[train_index],
                years[train_index],
                sample_ids[train_index],
                self.class_for_each_sample[train_index],
            )

            val_X_paths = [
                [t1_path, t2_path, flair_path]
                for flair_path, t1_path, t2_path in zip(
                    flair_paths[val_index], t1_paths[val_index], t2_paths[val_index]
                )
            ]
            val_y_paths, val_predict_time, val_sample_ids = y_paths[val_index], years[val_index], sample_ids[val_index]

            train_X_paths: np.ndarray = build_ndarray_and_create_idx_to_img(train_X_paths)
            train_y_paths: np.ndarray = build_ndarray_and_create_idx_to_img(train_y_paths)
            train_predict_time: np.ndarray = build_ndarray_and_create_idx_to_img(train_predict_time)
            train_sample_ids: np.ndarray = build_ndarray_and_create_idx_to_img(train_sample_ids)
            train_sample_class: np.ndarray = build_ndarray_and_create_idx_to_img(train_sample_class)

            self.train_dataset = SynthesisDataset(
                X_paths=train_X_paths,
                y_paths=train_y_paths,
                predict_times=train_predict_time,
                sample_ids=train_sample_ids,
                is_val=False,
                class_for_each_sample=train_sample_class,
            )
            self.val_dataset = SynthesisDataset(
                X_paths=val_X_paths,
                y_paths=val_y_paths,
                predict_times=val_predict_time,
                sample_ids=val_sample_ids,
                is_val=True,
            )

    def train_dataloader(self) -> DataLoader:
        print(f"get {len(self.train_dataset)} training 3D image!")
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=NUM_WORKERS, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        print(f"get {len(self.val_dataset)} validation 3D image!")
        return DataLoader(self.val_dataset, batch_size=1, num_workers=NUM_WORKERS, shuffle=False)
