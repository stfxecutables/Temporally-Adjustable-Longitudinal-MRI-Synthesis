import random
from collections import defaultdict
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from matplotlib.pyplot import Figure
from monai.transforms import apply_transform
from monai.utils import set_determinism
from skimage.metrics import normalized_root_mse as nmse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.model_selection import KFold
from torch import Tensor

from model.unet.unet import UNet
from model.GAN.discriminator import Discriminator
from model.GAN.ACGAN_discriminator import ACGANDiscriminator
from utils.const import DATA_ROOT, MS_IMG_SIZE, PATCH_SIZE_3D
from utils.transforms import (
    compute_3Dpatch_loc,
    generate_patch,
    get_synthesis_transforms,
    map_to_range_from_zero_to_one,
    patch_3D_aggregator,
)

PATCH_SIZE = 128
PAPER_PLOT_PATH = Path(__file__).resolve().parent.parent / "data" / "Result"

exams_4 = [
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
exams_5 = ["test02", "test11", "test14", "training03"]
exams_6 = ["test10"]


class rUNet(pl.LightningModule):
    def __init__(self):
        super(rUNet, self).__init__()
        self.model = UNet(
            dimensions=3,
            out_classes=1,
            kernel_size=3,
            use_bias=False,
            padding_mode="zeros",
            activation="LeakyReLU",
            downsampling_type="max",
            out_channels_first_layer=16,  # 16
            residual=False,
            use_tanh=True,
            conv_num_in_layer=[2, 2, 2, 2, 2, 2],  # [2, 2, 2, 2, 2, 2]
            in_channels=4,
            normalization="Batch",
            merge_original_patches=False,
        )

    def forward(self, x: Tensor, predict_time: Tensor) -> Tensor:
        return self.model(x, predict_time)


class gtGAN(pl.LightningModule):
    def __init__(self):
        super(gtGAN, self).__init__()
        self.generator = UNet(
            dimensions=3,
            out_classes=1,
            kernel_size=3,
            use_bias=False,
            padding_mode="zeros",
            activation="LeakyReLU",
            downsampling_type="max",
            out_channels_first_layer=16,
            residual=False,
            use_tanh=True,
            conv_num_in_layer=[2, 2, 2, 2, 2, 2],
            in_channels=4,
            normalization="Batch",
            merge_original_patches=False,
        )
        self.discriminator = Discriminator(
            in_channels=5,
            putting_time_into_discriminator=False,
        )  # the output of generator is always 1 channel

    def forward(self, x: Tensor, predict_time: Tensor) -> Tensor:
        return self.generator(x, predict_time)


class dtGAN(pl.LightningModule):
    def __init__(self):
        super(dtGAN, self).__init__()
        self.generator = UNet(
            dimensions=3,
            out_classes=1,
            kernel_size=3,
            use_bias=False,
            padding_mode="zeros",
            activation="LeakyReLU",
            downsampling_type="max",
            out_channels_first_layer=16,
            residual=False,
            use_tanh=True,
            conv_num_in_layer=[2, 2, 2, 2, 2, 2],
            in_channels=4,
            normalization="Batch",
            merge_original_patches=False,
        )
        self.discriminator = Discriminator(
            in_channels=6,
            putting_time_into_discriminator=True,
        )  # the output of generator is always 1 channel

    def forward(self, x: Tensor, predict_time: Tensor) -> Tensor:
        return self.generator(x, predict_time)


class ACGAN(pl.LightningModule):
    def __init__(self):
        super(ACGAN, self).__init__()
        self.generator = UNet(
            dimensions=3,
            out_classes=1,
            kernel_size=3,
            in_channels=4,
            use_tanh=True,
            use_bias=False,
            residual=False,
            padding_mode="zeros",
            normalization="Batch",
            activation="LeakyReLU",
            downsampling_type="max",
            out_channels_first_layer=16,
            merge_original_patches=False,
            conv_num_in_layer=[2, 2, 2, 2, 2, 2],
        )
        self.discriminator = ACGANDiscriminator(
            in_channels=5,
            putting_time_into_discriminator=False,
        )

    def forward(self, x: Tensor, predict_time: Tensor) -> Tensor:
        return self.generator(x, predict_time)


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


def load_data(subject_id: str, num_sample_exam: int) -> Tuple[List, List, List, List]:
    X_paths = []
    y_paths = []
    predict_times = []
    sample_ids = []
    for idy in range(2, num_sample_exam + 1):
        cur_X_imgs_path, cur_y_img_path = get_imgs(subject_id, 1, idy)
        cur_predict_time = idy - 1
        X_paths.append(cur_X_imgs_path)
        y_paths.append(cur_y_img_path)
        predict_times.append(cur_predict_time)
        sample_ids.append(f"{subject_id}_1_{idy}")
    return X_paths, y_paths, predict_times, sample_ids


def get_all_samples(subject_id: str, num_sample_exam: int) -> Tuple[List, List, List, List]:
    X_paths = []
    y_paths = []
    predict_times = []
    sample_ids = []
    for idx in range(1, num_sample_exam):
        for idy in range(idx + 1, num_sample_exam + 1):
            cur_X_imgs_path, cur_y_img_path = get_imgs(subject_id, idx, idy)
            cur_predict_time = idy - idx
            X_paths.append(cur_X_imgs_path)
            y_paths.append(cur_y_img_path)
            predict_times.append(cur_predict_time)
            sample_ids.append(f"{subject_id}_{idx}_{idy}")
    return X_paths, y_paths, predict_times, sample_ids


def inference(
    X_path: List[Path], y_path: Path, predict_time: int, predict_dict: Dict, model: pl.LightningModule
) -> None:
    patch_loc = compute_3Dpatch_loc(in_size=MS_IMG_SIZE)
    ones_patches = np.ones((8, 1, *PATCH_SIZE_3D))
    count_ndarray = patch_3D_aggregator(patches=ones_patches, orig_shape=MS_IMG_SIZE, patch_loc=patch_loc)

    data = {"image": [str(path) for path in X_path], "label": str(y_path)}
    data = apply_transform(get_synthesis_transforms(), data)
    X_img, y_img = data["image"], data["label"]

    X_patches_list = []
    for idx in range(8):
        roi_start = patch_loc[idx].tolist()
        X_patch = generate_patch(roi_start=roi_start, img=X_img)
        X_patches_list.append(X_patch)
    X_img = np.stack(X_patches_list)

    device = torch.device("cuda:0")
    X_img = torch.from_numpy(X_img).float().to(device)
    predict_time_tensor = torch.tensor([predict_time] * 8).float().view(1, 2, 2, 2).to(device)
    X_img, predict_time_tensor = X_img.unsqueeze(0), predict_time_tensor.unsqueeze(0)
    logits_patches: Tensor = torch.zeros((1, 8, 1, 128, 128, 128)).type_as(X_img)

    # create 8 predict_time for input_patches:
    predict_time_tensor = predict_time_tensor.repeat(1, 8, 1, 1, 1, 1)
    for i in range(8):
        logits_patches[:, i, ...] = model(X_img[:, i, ...], predict_time_tensor[:, i, ...])
    input_patches = X_img.squeeze(axis=0)
    logits_patches = logits_patches.squeeze(axis=0)
    input_patches = input_patches.cpu().detach().numpy()
    logits_patches = logits_patches.cpu().detach().numpy()
    shared_options: Dict = dict(
        orig_shape=MS_IMG_SIZE,
        patch_loc=patch_loc,
        count_ndarray=count_ndarray,
    )
    inputs = patch_3D_aggregator(patches=input_patches, **shared_options)
    predicts = patch_3D_aggregator(patches=logits_patches, **shared_options)

    inputs = map_to_range_from_zero_to_one(inputs)
    targets = np.squeeze(map_to_range_from_zero_to_one(y_img))
    predicts = map_to_range_from_zero_to_one(predicts)
    ssim_ = ssim(
        targets,
        predicts,
        win_size=11,
        data_range=1.0,
        gaussian_weights=True,
        use_sample_covariance=False,
    )
    # Code for computing PSNR is adapted from
    # https://github.com/agis85/multimodal_brain_synthesis/blob/master/error_metrics.py#L32
    data_range = np.max([predicts.max(), targets.max()]) - np.min([predicts.min(), targets.min()])
    psnr_ = psnr(targets, predicts, data_range=data_range)
    nmse_ = nmse(targets, predicts)
    predict_dict["input"] = inputs
    predict_dict[f"target{predict_time}"] = targets
    predict_dict[f"predict{predict_time}"] = predicts
    predict_dict[f"SSIM{predict_time}"] = ssim_
    predict_dict[f"PSNR{predict_time}"] = psnr_
    predict_dict[f"NMSE{predict_time}"] = nmse_


def load_model(model: pl.LightningModule, path: Path) -> pl.LightningModule:
    model = model.load_from_checkpoint(checkpoint_path=str(path))
    device = torch.device("cuda:0")
    model.to(device)
    model.eval()
    model.freeze()
    return model


def draw_plot(predict_dict: Dict[str, Any]):
    si, sj, sk = MS_IMG_SIZE
    k = sk // 2
    slices = [np.rot90(predict_dict["input"][0][:, :, k], -1)]
    num_exam = predict_dict["num_exam"]
    title = ["Input"]

    for idx in range(1, num_exam):
        slices.append(np.rot90(predict_dict["predict" + str(idx)][:, :, k, ...], -1))
        title.append(f"Time Point-{idx} Predict-SSIM: {predict_dict[f'SSIM{idx}']}")

    slices.append(np.rot90(predict_dict[f"target{num_exam - 1}"][:, :, k, ...], -1))
    title.append("Target")
    nrows, ncols = 1, num_exam + 1

    fig = plt.figure(figsize=(5 * (num_exam + 1), 5))
    gs = gridspec.GridSpec(nrows, ncols)
    for idx in range(0, num_exam + 1):
        ax = plt.subplot(gs[idx])
        ax.imshow(slices[idx], cmap="gray", alpha=0.8)
        ax.grid(False)
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title[idx])
    plt.tight_layout()
    fig.savefig(
        PAPER_PLOT_PATH / f"fold{predict_dict['fold']}_{predict_dict['sample']}.png",
        format="png",
        dpi=300,
    )


def draw_all_plots():
    model_path_list = [
        Path(__file__).resolve().parent.parent / "log" / "checkpoint" / f"ACGAN_{i}.ckpt" for i in range(1, 6)
    ]

    all_samples = exams_4 + exams_5 + exams_6
    num_samples_exams = [4] * len(exams_4) + [5] * len(exams_5) + [6] * len(exams_6)
    random_state = random.randint(0, 100)
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

    for k_fold in range(0, 5):
        loaded_model = load_model(ACGAN(), model_path_list[k_fold])
        _, test_idx = list(kf.split(all_samples))[k_fold]

        all_samples, num_samples_exams = np.array(all_samples), np.array(num_samples_exams)
        test_samples, test_num_samples_exams = all_samples[test_idx], num_samples_exams[test_idx]

        for sample, num_sample_exam in zip(test_samples, test_num_samples_exams):
            print(f"start processing {sample}")
            predict_dict: Dict[str, Any] = {"fold": k_fold, "sample": sample}
            X_paths, y_paths, predict_times, sample_ids = load_data(sample, num_sample_exam)
            predict_dict["num_exam"] = num_sample_exam
            patch_loc = compute_3Dpatch_loc(in_size=MS_IMG_SIZE)

            for X_path, y_path, predict_time, sample_id in zip(X_paths, y_paths, predict_times, sample_ids):
                inference(X_path, y_path, predict_time, predict_dict, loaded_model)

            draw_plot(predict_dict)


def draw_img(img: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots()
    ax.imshow(np.rot90(img, 1), cmap="gray")
    ax.axis("off")
    fig.savefig(
        PAPER_PLOT_PATH / f"synthesized_{title}.png",
        bbox_inches="tight",
        pad_inches=0,
        format="png",
        dpi=300,
    )


def predict_on_one_sample(model_id: int, subject_id: str, sample_exams: List[int]) -> None:
    model_path_list = [
        Path(__file__).resolve().parent.parent / "log" / "checkpoint" / f"ACGAN_{i}.ckpt" for i in range(1, 6)
    ]

    X_paths = []
    y_paths = []
    predict_times = []
    sample_ids = []

    for idy in sample_exams:
        cur_X_imgs_path, cur_y_img_path = get_imgs(subject_id, 1, idy)
        cur_predict_time = idy - 1
        X_paths.append(cur_X_imgs_path)
        y_paths.append(cur_y_img_path)
        predict_times.append(cur_predict_time)
        sample_ids.append(f"{subject_id}_1_{idy}")

    loaded_model = load_model(ACGAN(), model_path_list[model_id])
    predict_dict: Dict[str, Any] = {"fold": model_id, "sample": subject_id, "predict_exams": sample_exams}
    for X_path, y_path, predict_time in zip(X_paths, y_paths, predict_times):
        print(f"processing {predict_time}")
        inference(X_path, y_path, predict_time, predict_dict, loaded_model)

    si, sj, sk = MS_IMG_SIZE
    k = sk // 2
    draw_img(img=predict_dict["input"][0][:, :, k], title=f"{subject_id}_input")
    draw_img(img=predict_dict[f"target{sample_exams[-1] - 1}"][:, :, k], title=f"{subject_id}_target")
    draw_img(
        img=predict_dict[f"predict{sample_exams[-1] - 1}"][:, :, k], title=f"{subject_id}_predict_{sample_exams[-1]}"
    )
    draw_img(
        img=predict_dict[f"predict{sample_exams[-2] - 1}"][:, :, k], title=f"{subject_id}_predict_{sample_exams[-2]}"
    )


def compute_all_metrics(model_name: str):
    model_path_list = [
        Path(__file__).resolve().parent.parent / "log" / "checkpoint" / f"{model_name}_{i}.ckpt" for i in range(1, 6)
    ]

    all_samples = exams_4 + exams_5 + exams_6
    num_samples_exams = [4] * len(exams_4) + [5] * len(exams_5) + [6] * len(exams_6)
    random_state = random.randint(0, 100)
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    results = defaultdict(list)

    run = "test11_2_3"
    flag = True

    for k_fold in range(0, 5):
        if k_fold != 3:
            continue

        if model_name == "dt-GAN":
            model = dtGAN()
        elif model_name == "gt-GAN":
            model = gtGAN()
        elif model_name == "rUNet":
            model = rUNet()
        elif model_name == "ACGAN":
            model = ACGAN()
        loaded_model = load_model(model, model_path_list[k_fold])
        _, test_idx = list(kf.split(all_samples))[k_fold]

        all_samples, num_samples_exams = np.array(all_samples), np.array(num_samples_exams)
        test_samples, test_num_samples_exams = all_samples[test_idx], num_samples_exams[test_idx]

        for sample, num_sample_exam in zip(test_samples, test_num_samples_exams):
            X_paths, y_paths, predict_times, sample_ids = get_all_samples(sample, num_sample_exam)

            for X_path, y_path, predict_time, sample_id in zip(X_paths, y_paths, predict_times, sample_ids):
                if sample_id == run:
                    flag = False
                if flag:
                    continue

                print(f"start processing {sample_id}")
                predict_dict: Dict[str, Any] = {"fold": k_fold, "sample": sample}
                predict_dict["num_exam"] = num_sample_exam
                results["fold"].append(k_fold)
                results["sample_id"].append(sample_id)
                results["predict_time"].append(predict_time)
                inference(X_path, y_path, predict_time, predict_dict, loaded_model)
                results["SSIM"].append(predict_dict[f"SSIM{predict_time}"])
                results["PSNR"].append(predict_dict[f"PSNR{predict_time}"])
                results["NMSE"].append(predict_dict[f"NMSE{predict_time}"])

            df = pd.DataFrame(results)
            df.to_csv(PAPER_PLOT_PATH / f"{model_name}.csv", index=False)


if __name__ == "__main__":
    pl.seed_everything(42)
    # set deterministic for monai
    set_determinism()
    # draw_all_plots()
    compute_all_metrics(model_name="ACGAN")

    # predict_on_one_sample(model_id=3, subject_id="test11", sample_exams=[2, 5])
