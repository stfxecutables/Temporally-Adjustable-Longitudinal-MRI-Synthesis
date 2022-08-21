from argparse import ArgumentParser
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import normalized_root_mse as nmse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch import Tensor
from torch.nn import L1Loss, MSELoss, SmoothL1Loss
from torch.optim.lr_scheduler import CosineAnnealingLR

from model.unet.unet import UNet
from model.GAN.discriminator import Discriminator
from utils.const import MS_IMG_SIZE, PATCH_SIZE_3D, MS_IMG_SIZE_NOVEL
from utils.transforms import compute_3Dpatch_loc, map_to_range_from_zero_to_one, patch_3D_aggregator, LambdaLR
from utils.visualize import BrainSlices
from utils.visualize_6_timepoint_sample import SixTimepointBrainSlices
from utils.visualize_training import log_training_img


class GAN(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters(
            "loss",
            "task",
            "beta1",
            "optim",
            "log_mod",
            "dataset",
            "residual",
            "clip_min",
            "clip_max",
            "use_tanh",
            "kfold_num",
            "lambda_l1",
            "lr_policy",
            "max_epochs",
            "in_channels",
            "decay_epoch",
            "weight_decay",
            "smooth_label",
            "learning_rate",
            "normalization",
            "save_validation_result",
            "merge_original_patches",
            "putting_time_into_discriminator",
        )
        self.generator = UNet(
            dimensions=3,
            out_classes=1,
            kernel_size=3,
            use_bias=False,
            padding_mode="zeros",
            activation="LeakyReLU",
            downsampling_type="max",
            out_channels_first_layer=16,  # 16
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

        self.discriminator = Discriminator(
            in_channels=(self.hparams.in_channels + 2)
            if self.hparams.putting_time_into_discriminator
            else (self.hparams.in_channels + 1),
            putting_time_into_discriminator=self.hparams.putting_time_into_discriminator,
        )  # the output of generator is always 1 channel
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
        return self.generator(x, predict_time)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y)

    # Code is adapted from: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/635aa9531e5646ac731149e55b96305078d036e4/models/networks.py#L240
    def get_target_tensor(self, y_hat: Tensor, target_is_real: bool) -> Tensor:
        """Create label tensors with the same size as the input.
        I also have tested for one side or two side smooth label, two sides work better.
        Parameters:
            prediction: tpyically the prediction from a discriminator
            target_is_real: if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """
        if target_is_real:
            label = torch.ones_like(y_hat).type_as(y_hat)
            target = torch.mul(label, 0.9) + 0.1 * torch.rand(label.size()).type_as(y_hat)
        else:
            label = torch.zeros_like(y_hat).type_as(y_hat)
            target = label + 0.1 * torch.rand(label.size()).type_as(y_hat)
        return target

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs, targets, predict_time, _, real_aux = batch

        # train generator
        if optimizer_idx == 0:
            generated_imgs = self(inputs, predict_time)
            fix_loss = self.criterion(generated_imgs, targets)
            if self.hparams.loss == "SSIM" or self.hparams.loss == "MS_SSIM":
                fix_loss = 1 - fix_loss

            # log sampled images
            if self.hparams.kfold_num == 1:
                if (self.current_epoch % self.hparams.log_mod == 0 and batch_idx == 1) or (
                    (self.current_epoch % self.hparams.max_epochs - 1 == 0) and batch_idx == 1
                ):
                    log_training_img(
                        module=self,
                        state="train",
                        target=targets,
                        preb=generated_imgs,
                        batch_idx=batch_idx,
                        train_loss=fix_loss.item(),
                    )

            # adversarial loss is binary cross-entropy
            fake_imgs = torch.cat((inputs, generated_imgs), 1)
            D_fake_result = self.discriminator(fake_imgs, predict_time)
            # D_fake_result shape: [3, 1, 6, 6, 6]
            D_fake_label = self.get_target_tensor(D_fake_result, target_is_real=True)
            loss_GAN = self.adversarial_loss(D_fake_result, D_fake_label)
            loss_G = self.hparams.lambda_l1 * fix_loss + loss_GAN

            self.log("loss_G", loss_G, sync_dist=True, on_step=True, on_epoch=True)
            return {"loss": loss_G}

        # train discriminator
        if optimizer_idx == 1:
            generated_imgs = self(inputs, predict_time).detach()
            # Real loss
            real_imgs = torch.cat((inputs, targets), 1)
            D_real_result = self.discriminator(real_imgs, predict_time)
            D_real_label = self.get_target_tensor(D_real_result, target_is_real=True)
            real_loss = self.adversarial_loss(D_real_result, D_real_label)
            self.log("real_D_loss", real_loss, sync_dist=True, on_step=True, on_epoch=True)
            # Fake loss
            fake_imgs = torch.cat((inputs, generated_imgs), 1)
            D_fake_result = self.discriminator(fake_imgs, predict_time)
            D_fake_label = self.get_target_tensor(D_fake_result, target_is_real=False)
            fake_loss = self.adversarial_loss(D_fake_result, D_fake_label)
            self.log("fake_D_loss", fake_loss, sync_dist=True, on_step=True, on_epoch=True)
            # Total loss, in pix2pix they also did this!
            loss_D = (real_loss + fake_loss) / 2.0

            self.log("loss_D", loss_D, sync_dist=True, on_step=True, on_epoch=True)
            return {"loss": loss_D}

    def get_first_and_second_time_point(self, sample_id: str) -> Tuple[str, str]:
        return sample_id[7], sample_id[9]

    def validation_step(self, batch, batch_idx: int):
        # [batch size = 1, num patches = 8, channels, 128, 128, 128]
        input_patches, target_patches, predict_time, sample_id = batch
        # (num_patch, channels, patch_size, patch_size, patch_size)

        logits_patches: Tensor = torch.zeros_like(target_patches).type_as(target_patches)
        # create 8 predict_time for input_patches:
        predict_time = predict_time.repeat(1, 8, 1, 1, 1, 1)
        for i in range(8):
            # change to (batch size, channels, patch_size, patch_size, patch_size)
            logits_patches[:, i, ...] = self(input_patches[:, i, ...], predict_time[:, i, ...])

        loss = self.criterion(logits_patches, target_patches)
        if self.hparams.loss == "SSIM" or self.hparams.loss == "MS_SSIM":  # type: ignore
            loss = 1 - loss
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
            if self.current_epoch % self.hparams.log_mod == 0 or self.current_epoch % self.hparams.max_epochs - 1 == 0:
                brain_slices = BrainSlices(
                    lightning=self, img=inputs, target=targets, prediction=predicts, sample_id=sample_id
                )
                fig = brain_slices.plot()
                brain_slices.log("val", fig, ssim_, batch_idx)

        if len(inputs.shape) == 4:
            brain_mask = inputs[0] == inputs[0][0][0][0]
        else:
            brain_mask = inputs == inputs[0][0][0]

        pred_clip = np.clip(predicts, -self.hparams.clip_min, self.hparams.clip_max) - min(
            -self.hparams.clip_min, np.min(predicts)
        )
        targ_clip = np.clip(targets, -self.hparams.clip_min, self.hparams.clip_max) - min(
            -self.hparams.clip_min, np.min(targets)
        )
        pred_255 = np.floor(256 * (pred_clip / (self.hparams.clip_min + self.hparams.clip_max)))
        targ_255 = np.floor(256 * (targ_clip / (self.hparams.clip_min + self.hparams.clip_max)))
        pred_255[brain_mask] = 0
        targ_255[brain_mask] = 0

        diff_255 = np.absolute(pred_255.ravel() - targ_255.ravel())
        mae = np.mean(diff_255)

        return {"MAE": mae, "MSE": mse_, "SSIM": ssim_, "PSNR": psnr_, "NMSE": nmse_}

    def get_average_on_gpus(self, validation_step_outputs: List[Union[Tensor, Dict[str, Any]]], metric: str) -> None:
        average = np.mean([x[metric] for x in validation_step_outputs])
        self.log(f"val_{metric}", average, sync_dist=True, on_step=False, on_epoch=True)

    def validation_epoch_end(self, validation_step_outputs: List[Union[Tensor, Dict[str, Any]]]) -> None:
        metrics = ["MSE", "SSIM", "PSNR", "NMSE"]
        for metric in metrics:
            self.get_average_on_gpus(validation_step_outputs, metric)

    def configure_optimizers(self):
        # Optimizers
        optimizer_G = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(self.hparams.beta1, 0.999),
        )
        optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(self.hparams.beta1, 0.999),
        )

        if self.hparams.lr_policy == "cosine":
            lr_scheduler_G = CosineAnnealingLR(optimizer_G, T_max=300, eta_min=0.000001)
            lr_scheduler_D = CosineAnnealingLR(optimizer_D, T_max=300, eta_min=0.000001)
        elif self.hparams.lr_policy == "linear":
            lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
                optimizer_G,
                lr_lambda=LambdaLR(self.hparams.max_epochs, 0, self.hparams.decay_epoch).step,
            )
            lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(
                optimizer_D,
                lr_lambda=LambdaLR(self.hparams.max_epochs, 0, self.hparams.decay_epoch).step,
            )

        return [optimizer_G, optimizer_D], [lr_scheduler_G, lr_scheduler_D]
