"""Some code is borrowed and adapted from:
https://github.com/DM-Berger/unet-learn/blob/6dc108a9a6f49c6d6a50cd29d30eac4f7275582e/src/lightning/log.py
https://github.com/fepegar/miccai-educational-challenge-2019/blob/master/visualization.py
"""
from pathlib import Path
from typing import Any, List, Tuple, Optional, Union

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure
import numpy as np
import torch
from matplotlib.pyplot import Figure
from numpy import ndarray
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor


class BrainSlices:
    def __init__(
        self,
        lightning: LightningModule,
        img: Union[Tensor, np.ndarray],
        target: Union[Tensor, np.ndarray],
        prediction: Union[Tensor, np.ndarray],
        first_time_point: str = None,
        second_time_point: str = None,
        sample_id: str = None,
    ):
        # Some code is adapted from: https://github.com/fepegar/unet/blob/master/unet/conv.py#L78
        self.lightning = lightning
        self.input_img: ndarray = (
            img.cpu().detach().numpy().squeeze()
            if torch.is_tensor(img)
            else img.squeeze()
        )
        self.target_img: ndarray = (
            target.cpu().detach().numpy().squeeze()
            if torch.is_tensor(target)
            else target.squeeze()
        )
        self.predict_img: ndarray = (
            prediction.cpu().detach().numpy().squeeze()
            if torch.is_tensor(prediction)
            else prediction.squeeze()
        )

        if len(self.input_img.shape) == 3:
            si, sj, sk = self.input_img.shape
            i = si // 2 + 60
            j = sj // 2
            k = sk // 2
            self.slices = [
                self.get_slice(self.input_img, i, j, k),
                self.get_slice(self.target_img, i, j, k),
                self.get_slice(self.predict_img, i, j, k),
            ]
        else:
            si, sj, sk = self.input_img.shape[1:]
            i = si // 2 + 60
            j = sj // 2
            k = sk // 2
            if self.input_img.shape[0] == 2:
                self.slices = [
                    self.get_slice(self.input_img[0], i, j, k),
                    self.get_slice(self.input_img[1], i, j, k),
                    self.get_slice(self.target_img, i, j, k),
                    self.get_slice(self.predict_img, i, j, k),
                ]
            elif self.input_img.shape[0] == 3:
                self.slices = [
                    self.get_slice(self.input_img[0], i, j, k),
                    self.get_slice(self.input_img[1], i, j, k),
                    self.get_slice(self.input_img[2], i, j, k),
                    self.get_slice(self.target_img, i, j, k),
                    self.get_slice(self.predict_img, i, j, k),
                ]
            elif self.input_img.shape[0] == 4:
                self.slices = [
                    self.get_slice(self.input_img[1], i, j, k),
                    self.get_slice(self.input_img[2], i, j, k),
                    self.get_slice(self.input_img[3], i, j, k),
                    self.get_slice(self.input_img[0], i, j, k),
                    self.get_slice(self.target_img, i, j, k),
                    self.get_slice(self.predict_img, i, j, k),
                ]

        self.title = [
            "input image: mprage",
            "input image: pd",
            "input image: t2",
            f"input image {first_time_point} time point: flair",
            f"target image {second_time_point} time point: flair",
            "predict image",
        ]
        self.shape = np.array(self.input_img.shape)

    def get_slice(
        self, input: ndarray, i: int, j: int, k: int
    ) -> List[Tuple[ndarray, ...]]:
        return [input[i, ...], input[:, j, ...], input[:, :, k, ...]]

    def plot(self) -> Figure:
        nrows, ncols = len(self.slices), 3  # one row for each slice position

        if nrows == 6:
            fig = plt.figure(figsize=(12, 30))
        elif nrows == 5:
            fig = plt.figure(figsize=(12, 25))
        elif nrows == 4:
            fig = plt.figure(figsize=(12, 14))
        elif nrows == 3:
            fig = plt.figure(figsize=(15, 14))
        gs = gridspec.GridSpec(nrows, ncols)
        for i in range(0, nrows):
            ax1 = plt.subplot(gs[i * 3])
            ax2 = plt.subplot(gs[i * 3 + 1])
            ax3 = plt.subplot(gs[i * 3 + 2])
            axes = ax1, ax2, ax3
            self.plot_row(self.slices[i], axes)
            for axis in axes:
                if i == 0:
                    axis.set_title(self.title[0])
                elif i == (len(self.slices) - 1):
                    axis.set_title(self.title[-1])
                elif i == (len(self.slices) - 2):
                    axis.set_title(self.title[-2])
                elif i == 1:
                    axis.set_title(self.title[1])
                elif i == 2:
                    axis.set_title(self.title[2])
        plt.tight_layout()
        return fig

    def plot_row(self, slices: List, axes: Tuple[Any, Any, Any]) -> None:
        for (slice_, axis) in zip(slices, axes):
            axis.imshow(slice_, cmap="gray", alpha=0.8)
            axis.grid(False)
            axis.invert_xaxis()
            axis.invert_yaxis()
            axis.set_xticks([])
            axis.set_yticks([])

    def log(
        self,
        state: str,
        fig: Figure,
        ssim: float,
        batch_idx: int,
        gpu_id: Optional[int] = None,
    ) -> None:
        logger = self.lightning.logger
        summary = f"{state}-Epoch:{self.lightning.current_epoch + 1}-batch:{batch_idx}-SSIM:{ssim:0.5f}"
        if gpu_id is not None:
            summary += f"-GPU:{self.lightning.current_epoch}"
        logger.experiment.add_figure(summary, fig, close=True)
        # if you want to manually intervene, look at the code at
        # https://github.com/pytorch/pytorch/blob/master/torch/utils/tensorboard/_utils.py
        # permalink to version:
        # https://github.com/pytorch/pytorch/blob/780fa2b4892512b82c8c0aaba472551bd0ce0fad/torch/utils/tensorboard/_utils.py#L5
        # then use logger.experiment.add_image(summary, image)
