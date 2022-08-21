"""Some code is borrowed and adapted from:
https://github.com/DM-Berger/unet-learn/blob/6dc108a9a6f49c6d6a50cd29d30eac4f7275582e/src/lightning/log.py
https://github.com/fepegar/miccai-educational-challenge-2019/blob/master/visualization.py
"""
from pathlib import Path
from typing import Any, List, Tuple, Optional, Dict, Union

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

from utils.const import MS_IMG_SIZE


class SixTimepointBrainSlices:
    def __init__(
        self,
        lightning: LightningModule,
        img: Dict[str, Union[Tensor, np.ndarray, int]],
    ):
        # Some code is adapted from: https://github.com/fepegar/unet/blob/master/unet/conv.py#L78
        self.lightning = lightning
        self.img: Dict[str, Union[Tensor, np.ndarray, int]] = img

        si, sj, sk = MS_IMG_SIZE
        k = sk // 2
        self.slices = [[self.get_slice(img["input"], k=k)]]

        for idx in range(2, 7):
            self.slices.append([self.get_slice(img["predict" + str(idx)]), self.get_slice(img["target" + str(idx)])])

        self.title = [
            ["Input"],
            [f"Predict-SSIM: {self.img['SSIM2']}", "Target"],
            [f"Predict-SSIM: {self.img['SSIM3']}", "Target"],
            [f"Predict-SSIM: {self.img['SSIM4']}", "Target"],
            [f"Predict-SSIM: {self.img['SSIM5']}", "Target"],
            [f"Predict-SSIM: {self.img['SSIM6']}", "Target"],
        ]
        self.y_title = [
            "Time Point 1",
            "Time Point 2",
            "Time Point 3",
            "Time Point 4",
            "Time Point 5",
            "Time Point 6",
        ]

    def get_slice(self, input_: ndarray, i: Optional[int] = None, j: Optional[int] = None, k: int = None) -> ndarray:
        return input_[:, :, k, ...]

    def plot(self) -> Figure:
        nrows, ncols = 6, 2  # one row for each slice position

        fig = plt.figure(figsize=(8, 30))
        gs = gridspec.GridSpec(nrows, ncols)
        for i in range(0, nrows):
            ax1 = plt.subplot(gs[i * 2])
            ax2 = plt.subplot(gs[i * 2 + 1])
            axes = ax1, ax2
            ax1.set_ylabel(self.y_title[i])
            if i != 0:
                self.plot_row(self.slices[i], axes)
            else:
                # plot first row, because it only has 1 img
                ax1.imshow(self.slices[0][0], cmap="gray", alpha=0.8)
                ax1.grid(False)
                ax1.invert_xaxis()
                ax1.invert_yaxis()

                for ax in axes:
                    ax.set_xticks([])
                    ax.set_yticks([])
            for idx, axis in enumerate(axes):
                if i == 0 and idx == 0:  # Don't set title for the first row and second column
                    axis.set_title(self.title[0][0])
                elif i == 0 and idx == 1:
                    continue
                else:
                    axis.set_title(self.title[i][idx])
        plt.tight_layout()
        return fig

    def plot_row(self, slices: List, axes: Tuple[Any, Any]) -> None:
        for (slice_, axis) in zip(slices, axes):
            axis.imshow(slice_, cmap="gray", alpha=0.8)
            axis.grid(False)
            axis.invert_xaxis()
            axis.invert_yaxis()
            axis.set_xticks([])
            axis.set_yticks([])

    def log(self, state: str, fig: Figure, ssim: float) -> None:
        logger = self.lightning.logger
        summary = f"{state}-Epoch:{self.lightning.current_epoch + 1}-SSIM:{ssim:0.5f}"
        fig.savefig("/home/jueqi/projects/def-jlevman/jueqi/MS/2/sixtimepoint.png")
        logger.experiment.add_figure(summary, fig, close=True)
        # if you want to manually intervene, look at the code at
        # https://github.com/pytorch/pytorch/blob/master/torch/utils/tensorboard/_utils.py
        # permalink to version:
        # https://github.com/pytorch/pytorch/blob/780fa2b4892512b82c8c0aaba472551bd0ce0fad/torch/utils/tensorboard/_utils.py#L5
        # then use logger.experiment.add_image(summary, image)
