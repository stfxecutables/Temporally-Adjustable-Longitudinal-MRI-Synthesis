from argparse import ArgumentParser
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from model.unet.unet import UNet
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import normalized_root_mse as nmse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch import Tensor
from torch.nn import L1Loss, MSELoss, SmoothL1Loss
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from utils.const import MS_IMG_SIZE, MS_IMG_SIZE_NOVEL
from utils.transforms import (
    PATCH_SIZE_3D,
    LambdaLR,
    compute_3Dpatch_loc,
    map_to_range_from_zero_to_one,
    patch_3D_aggregator,
)
from utils.visualize import BrainSlices
from utils.visualize_6_timepoint_sample import SixTimepointBrainSlices
from utils.visualize_training import log_training_img


class SynthesisLitModel(pl.LightningModule):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(
            "loss",
            "task",
            "optim",
            "log_mod",
            "dataset",
            "residual",
            "clip_min",
            "clip_max",
            "use_tanh",
            "lr_policy",
            "kfold_num",
            "max_epochs",
            "in_channels",
            "decay_epoch",
            "weight_decay",
            "learning_rate",
            "normalization",
            "save_validation_result",
            "merge_original_patches",
        )
        self.model = UNet(
            dimensions=3,
            out_classes=1,
            kernel_size=3,
            use_bias=False,
            padding_mode="zeros",
            activation="LeakyReLU",
            downsampling_type="max",
            out_channels_first_layer=16,
            residual=self.hparams.residual,
            use_tanh=self.hparams.use_tanh,
            conv_num_in_layer=[2, 2, 2, 2, 2, 2],  # [2, 2, 2, 2, 2, 2]
            in_channels=self.hparams.in_channels,
            normalization=self.hparams.normalization,
            merge_original_patches=self.hparams.merge_original_patches,
        )
        if self.hparams.loss == "l2":
            self.criterion = MSELoss()
        elif self.hparams.loss == "l1":
            self.criterion = L1Loss()
        elif self.hparams.loss == "smoothl1":
            self.criterion = SmoothL1Loss()

        if self.hparams.dataset == "ISBI2015":
            self.img_size = MS_IMG_SIZE
        elif self.hparams.dataset == "novel":
            self.img_size = MS_IMG_SIZE_NOVEL
        self.patch_loc = compute_3Dpatch_loc(in_size=self.img_size)
        ones_patches = np.ones((8, 1, *PATCH_SIZE_3D))
        self.count_ndarray = patch_3D_aggregator(
            patches=ones_patches, orig_shape=self.img_size, patch_loc=self.patch_loc
        )

    def forward(self, x: Tensor, predict_time: Tensor) -> Tensor:
        return self.model(x, predict_time)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str], batch_idx: int):
        inputs, targets, predict_time, _, real_aux = batch
        logits = self(inputs, predict_time)
        loss = self.criterion(logits, targets)

        if self.hparams.kfold_num == 1:
            if (self.current_epoch % self.hparams.log_mod == 0 and batch_idx == 1) or (
                (self.current_epoch % self.hparams.max_epochs - 1 == 0) and batch_idx == 1
            ):
                log_training_img(
                    module=self,
                    preb=logits,
                    state="train",
                    target=targets,
                    batch_idx=batch_idx,
                    train_loss=loss.item(),
                )

        self.log("train_loss", loss, sync_dist=True, on_step=True, on_epoch=True)
        return {"loss": loss}

    def get_first_and_second_time_point(self, sample_id: str) -> Tuple[str, str]:
        return sample_id[7], sample_id[9]

    def validation_step(self, batch, batch_idx: int):
        # [batch size = 1, num patches = 8, channels, 128, 128, 128]
        input_patches, target_patches, predict_time, sample_id = batch

        logits_patches: Tensor = torch.zeros_like(target_patches).type_as(target_patches)
        # create 8 predict_time for input_patches:
        predict_time = predict_time.repeat(1, 8, 1, 1, 1, 1)
        for i in range(8):
            # change to (batch size, channels, patch_size, patch_size, patch_size)
            logits_patches[:, i, ...] = self(input_patches[:, i, ...], predict_time[:, i, ...])

        loss = self.criterion(logits_patches, target_patches)
        self.log("val_loss", loss, sync_dist=True, on_step=True, on_epoch=True)

        # I have set batch_size = 1, so this could get rid of batch dimension
        input_patches = input_patches.squeeze(axis=0)
        target_patches = target_patches.squeeze(axis=0)
        logits_patches = logits_patches.squeeze(axis=0)
        input_patches = input_patches.cpu().detach().numpy()
        target_patches = target_patches.cpu().detach().numpy()
        logits_patches = logits_patches.cpu().detach().numpy()

        shared_options: Dict = dict(
            orig_shape=self.img_size,
            patch_loc=self.patch_loc,
            count_ndarray=self.count_ndarray,
        )
        inputs = patch_3D_aggregator(patches=input_patches, **shared_options)
        targets = patch_3D_aggregator(patches=target_patches, **shared_options)
        predicts = patch_3D_aggregator(patches=logits_patches, **shared_options)

        inputs = map_to_range_from_zero_to_one(inputs)
        targets = map_to_range_from_zero_to_one(targets)
        predicts = map_to_range_from_zero_to_one(predicts)

        mse_ = mse(targets, predicts)
        nmse_ = nmse(targets, predicts)
        # Setting all the parameters to get the same result as the original SSIM paper
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
        if self.hparams.dataset != "novel":
            first_time_point, second_time_point = self.get_first_and_second_time_point(sample_id[0])
            if self.hparams.kfold_num == 1:
                if (self.current_epoch % self.hparams.log_mod == 0 and batch_idx == 1) or (
                    (self.current_epoch % self.hparams.max_epochs - 1 == 0) and batch_idx == 1
                ):
                    brain_slices = BrainSlices(
                        lightning=self,
                        img=inputs,
                        target=targets,
                        prediction=predicts,
                        first_time_point=first_time_point,
                        second_time_point=second_time_point,
                    )
                    fig = brain_slices.plot()
                    brain_slices.log("val", fig, ssim_, batch_idx)
        else:
            if self.hparams.kfold_num == 1:
                if (self.current_epoch % self.hparams.log_mod == 0 and batch_idx == 1) or (
                    (self.current_epoch % self.hparams.max_epochs - 1 == 0) and batch_idx == 1
                ):
                    brain_slices = BrainSlices(
                        lightning=self, img=inputs, target=targets, prediction=predicts, sample_id=sample_id
                    )
                    fig = brain_slices.plot()
                    brain_slices.log("val", fig, ssim_, batch_idx)

        return {"MSE": mse_, "SSIM": ssim_, "PSNR": psnr_, "NMSE": nmse_}

    def get_average_on_gpus(self, validation_step_outputs: List[Union[Tensor, Dict[str, Any]]], metric: str) -> None:
        average = np.mean([x[metric] for x in validation_step_outputs])
        self.log(f"val_{metric}", average, sync_dist=True, on_step=False, on_epoch=True)

    def get_image_from_step_outputs(
        self, validation_step_outputs: List[Union[Tensor, Dict[str, Any]]], img_key: str
    ) -> np.ndarray:
        for validation_step_output in validation_step_outputs:
            if img_key in validation_step_output:
                return validation_step_output[img_key]
            else:
                continue

    def validation_epoch_end(self, validation_step_outputs: List[Union[Tensor, Dict[str, Any]]]) -> None:
        metrics = ["MSE", "SSIM", "PSNR", "NMSE"]
        for metric in metrics:
            self.get_average_on_gpus(validation_step_outputs, metric)

    def test_step(self, batch, batch_idx: int) -> Dict[str, float]:
        # [batch size = 1, num patches = 8, channels, 128, 128, 128]
        input_patches, target_patches, predict_time, sample_id = batch

        logits_patches: Tensor = torch.zeros_like(target_patches).type_as(target_patches)
        # create 8 predict_time for input_patches:
        predict_time = predict_time.repeat(1, 8, 1, 1, 1, 1)
        for i in range(8):
            # change to (batch size, channels, patch_size, patch_size, patch_size)
            logits_patches[:, i, ...] = self(input_patches[:, i, ...], predict_time[:, i, ...])

        loss = self.criterion(logits_patches, target_patches)
        self.log("val_loss", loss, sync_dist=True, on_step=True, on_epoch=True)

        # I have set batch_size = 1, so this could get rid of batch dimension
        input_patches = input_patches.squeeze(axis=0)
        target_patches = target_patches.squeeze(axis=0)
        logits_patches = logits_patches.squeeze(axis=0)
        input_patches = input_patches.cpu().detach().numpy()
        target_patches = target_patches.cpu().detach().numpy()
        logits_patches = logits_patches.cpu().detach().numpy()

        shared_options: Dict = dict(
            orig_shape=self.img_size,
            patch_loc=self.patch_loc,
            count_ndarray=self.count_ndarray,
        )
        inputs = patch_3D_aggregator(patches=input_patches, **shared_options)
        targets = patch_3D_aggregator(patches=target_patches, **shared_options)
        predicts = patch_3D_aggregator(patches=logits_patches, **shared_options)

        inputs = map_to_range_from_zero_to_one(inputs)
        targets = map_to_range_from_zero_to_one(targets)
        predicts = map_to_range_from_zero_to_one(predicts)

        mse_ = mse(targets, predicts)
        nmse_ = nmse(targets, predicts)
        # Setting all the parameters to get the same result as the original SSIM paper
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

        first_time_point, second_time_point = self.get_first_and_second_time_point(sample_id[0])
        if first_time_point == "1" and second_time_point == "2":
            return {
                "MSE": mse_,
                "SSIM": ssim_,
                "PSNR": psnr_,
                "NMSE": nmse_,
                "input": inputs[0],
                "predict2": predicts,
                "target2": targets,
                "SSIM2": ssim_,
            }
        elif first_time_point == "1":
            return {
                "MSE": mse_,
                "SSIM": ssim_,
                "PSNR": psnr_,
                "NMSE": nmse_,
                "predict" + second_time_point: predicts,
                "target" + second_time_point: targets,
                "SSIM" + second_time_point: ssim_,
            }
        else:
            return {"MSE": mse_, "SSIM": ssim_, "PSNR": psnr_, "NMSE": nmse_}

    def test_epoch_end(self, test_step_outputs: List[Union[Tensor, Dict[str, Any]]]) -> None:
        metrics = ["MSE", "SSIM", "PSNR", "NMSE"]
        for metric in metrics:
            self.get_average_on_gpus(test_step_outputs, metric)

        ssim = np.mean([x["SSIM"] for x in test_step_outputs])

        visualization_dict = {}
        visualization_dict["input"] = self.get_image_from_step_outputs(test_step_outputs, "input")
        for idx in range(2, 7):
            visualization_dict["predict" + str(idx)] = self.get_image_from_step_outputs(
                test_step_outputs, "predict" + str(idx)
            )
            visualization_dict["target" + str(idx)] = self.get_image_from_step_outputs(
                test_step_outputs, "target" + str(idx)
            )
            visualization_dict["SSIM" + str(idx)] = self.get_image_from_step_outputs(
                test_step_outputs, "SSIM" + str(idx)
            )
        print(f"visualization_dict: {visualization_dict}")
        print(f"validation step outputs: {test_step_outputs}")
        six_timepoint_brain_slices = SixTimepointBrainSlices(lightning=self, img=visualization_dict)
        fig = six_timepoint_brain_slices.plot()
        six_timepoint_brain_slices.log("val", fig, ssim=ssim)

    def configure_optimizers(self):
        if self.hparams.optim == "Adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optim == "AdamW":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )

        if self.hparams.lr_policy == "cosine":
            lr_scheduler = CosineAnnealingLR(optimizer, T_max=300, eta_min=0.000001)
        elif self.hparams.lr_policy == "linear":
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=LambdaLR(self.hparams.max_epochs, 0, self.hparams.decay_epoch).step,
            )

        lr_dict = {
            "scheduler": lr_scheduler,
            "monitor": "val_SSIM",
            "reduce_on_plateau": True,
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [lr_dict]

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("LitModel")
        parser.add_argument("--learning_rate", type=float, default=1e-5)
        parser.add_argument("--loss", type=str, choices=["l1", "l2", "smoothl1", "SSIM", "MS_SSIM"], default="l1")
        parser.add_argument("--activation", type=str, choices=["ReLU", "LeakyReLU"], default="LeakyReLU")
        parser.add_argument(
            "--normalization",
            type=str,
            choices=["Batch", "Group", "InstanceNorm3d"],
            default="Batch",
        )
        parser.add_argument("--weight_decay", type=float, default=1e-8)
        parser.add_argument("--clip_min", type=int, default=2)
        parser.add_argument("--clip_max", type=int, default=5)
        parser.add_argument("--log_mod", type=int, default=10)
        parser.add_argument("--times", type=int, default=5)
        parser.add_argument("--class_num", type=int, default=6)
        parser.add_argument("--in_channels", type=int, default=3)
        parser.add_argument("--use_tanh", action="store_true")
        parser.add_argument("--residual", action="store_true")
        parser.add_argument(
            "--optim",
            type=str,
            choices=["Adam", "AdamW"],
            default="Adam",
        )
        parser.add_argument("--decay_epoch", type=int, default=100)
        # GAN parameters
        parser.add_argument("--beta1", type=float, default=0.5)
        parser.add_argument("--lambda_l1", type=int, default=300)
        parser.add_argument("--smooth_label", action="store_true")
        parser.add_argument("--putting_time_into_discriminator", action="store_true")
        parser.add_argument("--GAN_loss", type=str, choices=["MSE", "BCE"], default="MSE")
        return parent_parser
