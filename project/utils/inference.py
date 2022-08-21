from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List, Tuple, Dict

import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.utilities.cloud_io import load
import torch
from torch import Tensor
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from skimage.metrics import structural_similarity as ssim
from monai.transforms import apply_transform

from model.unet.unet import UNet
from model.GAN.discriminator import Discriminator
from model.create_structure_data import CreateStructuredData
from utils.const import DATA_ROOT, MS_IMG_SIZE, PATCH_SIZE_3D
from utils.transforms import (
    get_synthesis_transforms,
    generate_patch,
    compute_3Dpatch_loc,
    patch_3D_aggregator,
    map_to_range_from_zero_to_one
)


def add_argument(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--loss", type=str, choices=["l1", "l2", "smoothl1", "SSIM", "MS_SSIM"], default="l1")
    parser.add_argument(
        "--task",
        type=str,
        choices=[
            "Segmentation",
            "LongitudinalSynthesis",
            "LongitudinalSynthesisGAN",
        ],
        default="LongitudinalSynthesis",
    )
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument(
        "--optim",
        type=str,
        choices=["Adam", "AdamW"],
        default="Adam",
    )
    parser.add_argument("--log_mod", type=int, default=15)
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--clip_min", type=int, default=2)
    parser.add_argument("--clip_max", type=int, default=5)
    parser.add_argument("--use_tanh", action="store_true")
    parser.add_argument("--kfold_num", type=int, choices=[1, 2, 3, 4, 5], default=1)
    parser.add_argument("--lambda_l1", type=int, default=300)
    parser.add_argument("--max_epochs", type=int, default=300)
    parser.add_argument("--in_channels", type=int, default=4)
    parser.add_argument("--decay_epoch", type=int, default=25)
    parser.add_argument("--weight_decay", type=float, default=1e-8)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        choices=["Cosine", "LinearlyDecay", "ReduceLROnPlateau"],
        default="LinearlyDecay",
    )
    parser.add_argument("--smooth_label", action="store_true")
    parser.add_argument(
        "--normalization",
        type=str,
        choices=["Batch", "Group", "InstanceNorm3d"],
        default="Batch",
    )
    parser.add_argument(
        "--save_validation_result",
        action="store_true",
        help="save the prediction of validation set in the format of npz and mp4",
    )
    parser.add_argument(
        "--merge_original_patches",
        action="store_true",
        help="whether to merge the original patches",
    )
    parser.add_argument("--putting_time_into_discriminator", action="store_true")


class gGAN(pl.LightningModule):
    def __init__(self):
        super(gGAN, self).__init__()
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


def load_model(model: pl.LightningModule, path: str) -> pl.LightningModule:
    model = model.load_from_checkpoint(checkpoint_path=path)
    device = torch.device('cuda:0')
    model.float()
    model.to(device)
    model.eval()
    model.freeze()
    return model

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

def load_data() -> Tuple[List, List, List, List]:
    X_paths = []
    y_paths = []
    predict_times = []
    sample_ids = []
    for idy in range(2, 7):
        idx = 1
        cur_X_imgs_path, cur_y_img_path = get_imgs("test10", idx, idy)
        cur_predict_time = idy - idx
        X_paths.append(cur_X_imgs_path)
        y_paths.append(cur_y_img_path)
        predict_times.append(cur_predict_time)
        sample_ids.append(f"test10_{idx}_{idy}")
    return X_paths, y_paths, predict_times, sample_ids

def inference(model: pl.LightningModule) -> None:
    X_paths, y_paths, predict_times, sample_ids = load_data()
    patch_loc = compute_3Dpatch_loc(in_size=MS_IMG_SIZE)
    device = torch.device('cuda:0')
    transform = get_synthesis_transforms()
    ones_patches = np.ones((8, 1, *PATCH_SIZE_3D))
    count_ndarray = patch_3D_aggregator(
        patches=ones_patches, orig_shape=MS_IMG_SIZE, patch_loc=patch_loc
    )

    for X_path, y_path, predict_time, sample_id in zip(X_paths, y_paths, predict_times, sample_ids):
        data = {"image": [str(path) for path in X_path], "label": str(y_path)}
        data = apply_transform(transform, data)
        X_img, y_img = data["image"], data["label"]

        X_patches_list = []
        for idx in range(8):
            roi_start = patch_loc[idx].tolist()
            X_patch = generate_patch(roi_start=roi_start, img=X_img)
            X_patches_list.append(X_patch)
        X_img = np.stack(X_patches_list)

        X_img = torch.from_numpy(X_img).float().to(device)
        predict_time = torch.tensor([predict_time] * 8).float().view(1, 2, 2, 2).to(device)
        X_img, predict_time = X_img.unsqueeze(0), predict_time.unsqueeze(0)
        logits_patches: Tensor = torch.zeros((1, 8, 1, 128, 128, 128)).type_as(X_img)
        # create 8 predict_time for input_patches:
        predict_time = predict_time.repeat(1, 8, 1, 1, 1, 1)
        for i in range(8):
            logits_patches[:, i, ...] = model(X_img[:, i, ...], predict_time[:, i, ...])
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
        np.savez(
            f"{sample_id}-SSIM:{ssim_:0.5f}.npz",
            inputs=inputs,
            target=targets,
            predict=predicts,
        )
        print(f"Finish processing {sample_id}!")


if __name__ == '__main__':
    parser = ArgumentParser(description="Trainer args", add_help=False)
    parser = add_argument(parser)
    path = Path(__file__).resolve().parent / "checkpoint" / "epoch=55-val_loss=0.04756-val_MSE=0.01282-val_SSIM=0.86696-val_PSNR=21.16103-val_NMSE=0.54528.ckpt"
    g_gan = gGAN()
    model = load_model(g_gan, str(path))
    inference(model)
