"""Some code is borrowed and adapted from:
https://github.com/DM-Berger/unet-learn/blob/6dc108a9a6f49c6d6a50cd29d30eac4f7275582e/src/lightning/log.py
https://github.com/fepegar/miccai-educational-challenge-2019/blob/master/visualization.py
"""
import os
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

RESULT = Path(__file__).resolve().parent.parent / "data" / "Result"
DGAN_RESULT = RESULT / "dGAN" / "plot"
GGAN_RESULT = RESULT / "gGAN" / "plot"
RUNET_RESULT = RESULT / "rUNet" / "plot"

class SixTimepointBrainSlices:
    def __init__(
        self,
        img: Dict[str, Union[Tensor, np.ndarray, int]],
    ):
        self.img: Dict[str, Union[Tensor, np.ndarray, int]] = img

        si, sj, sk = MS_IMG_SIZE
        k = sk // 2
        self.slices = [[img["input"][0][:, :, k]]]

        for idx in range(2, 7):
            tmp = []
            tmp.append(img["predict" + str(idx)][:, :, k, ...])
            tmp.append(img["target" + str(idx)][:, :, k, ...])
            self.slices.append(tmp)

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

    def plot(self) -> Figure:
        nrows, ncols = 6, 2  # one row for each slice position

        fig = plt.figure(figsize=(10, 25))
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

                ax2.imshow(self.slices[0][0], cmap="gray", alpha=0.8)
                ax2.grid(False)
                ax2.invert_xaxis()
                ax2.invert_yaxis()

                for ax in axes:
                    ax.set_xticks([])
                    ax.set_yticks([])
            for idx, axis in enumerate(axes):
                if i == 0 and idx == 1:
                    axis.set_title(self.title[0][0])
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


def get_ssim(file_path: str) -> str:
    return os.path.basename(file_path).split(".")[1]


if __name__ == "__main__":
    folder = Path("/home/jq/Desktop/MS/data/Result/gGAN")
    paths = list(folder.glob("*.npz"))
    paths.sort()
    # img = {}
    # img["input"] = np.load(paths[0])["inputs"]
    si, sj, sk = MS_IMG_SIZE
    k = sk // 2
    # input_img = img["input"][0][:, :, k, ...]
    for idx, path in enumerate(paths):
        idx += 2
        ssim = get_ssim(path)
        data = np.load(path)
        predict_img = data["predict"][:, :, k, ...]
        ssim_value = f"0.{ssim}"
        fig, ax = plt.subplots()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(np.rot90(predict_img, k=1), cmap="gray", alpha=0.8)
        fig.savefig(GGAN_RESULT / f"predict_{idx}_{ssim_value}.pdf", bbox_inches='tight', pad_inches=0)


    # six_timepoint_brain_slices = SixTimepointBrainSlices(img)
    # fig = six_timepoint_brain_slices.plot()
    # fig.savefig("rUNet.png")
