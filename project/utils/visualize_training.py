"""Some code is borrowed and adapted from:
https://github.com/DM-Berger/unet-learn/blob/6dc108a9a6f49c6d6a50cd29d30eac4f7275582e/src/lightning/log.py
https://github.com/fepegar/miccai-educational-challenge-2019/blob/master/visualization.py
"""
from typing import Any, List, Union

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.pyplot import Figure
from numpy import ndarray
from torch import Tensor


class BrainSlices:
    def __init__(
        self,
        lightning,  # : LightningModule
        target: Union[Tensor, np.ndarray],
        prediction: Union[Tensor, np.ndarray],
    ):
        # Some code is adapted from: https://github.com/fepegar/unet/blob/master/unet/conv.py#L78
        self.lightning = lightning
        self.target_img: ndarray = (
            target.cpu().detach().numpy().squeeze() if torch.is_tensor(target) else target.squeeze()
        )
        self.predict_img: ndarray = (
            prediction.cpu().detach().numpy().squeeze() if torch.is_tensor(prediction) else prediction.squeeze()
        )

        if self.target_img.ndim == 3:
            self.slices = [
                [self.target_img[:, :, 64]],
                [self.predict_img[:, :, 64]],
            ]
        elif self.predict_img.ndim != 3:
            self.slices = [
                self.get_slice(self.target_img),
                self.get_slice(self.predict_img),
            ]

        self.title = [
            "target image",
            "predict image",
        ]

    def get_slice(self, input: ndarray) -> List[ndarray]:
        l: list = []
        for i in range(input.shape[0]):
            l.append(input[i, :, :, 64])
        return l

    def plot(self) -> Figure:
        nrows, ncols = len(self.slices), 1  # one row for each slice position

        fig = plt.figure(figsize=(20, 10))
        gs = gridspec.GridSpec(nrows, ncols)
        for i in range(0, nrows):
            axes = []
            for j in range(0, ncols):
                ax = plt.subplot(gs[i * ncols + j])
                axes.append(ax)
            self.plot_row(self.slices[i], axes)
            for axis in axes:
                axis.set_title(self.title[i])
        plt.tight_layout()
        return fig

    def plot_row(self, slices: List, axes: List[Any]) -> None:
        for (slice_, axis) in zip(slices, axes):
            axis.imshow(slice_, cmap="gray")
            axis.grid(False)
            axis.invert_xaxis()
            axis.invert_yaxis()
            axis.set_xticks([])
            axis.set_yticks([])

    def log(self, state: str, fig: Figure, loss: float, batch_idx: int) -> None:
        logger = self.lightning.logger
        summary = f"{state}-Epoch:{self.lightning.current_epoch + 1}-batch:{batch_idx}-train_loss:{loss:0.5e}"
        logger.experiment.add_figure(summary, fig, close=True)
        # if you want to manually intervene, look at the code at
        # https://github.com/pytorch/pytorch/blob/master/torch/utils/tensorboard/_utils.py
        # permalink to version:
        # https://github.com/pytorch/pytorch/blob/780fa2b4892512b82c8c0aaba472551bd0ce0fad/torch/utils/tensorboard/_utils.py#L5
        # then use logger.experiment.add_image(summary, image)


"""
Actual methods on logger.experiment can be found here!!!
https://pytorch.org/docs/stable/tensorboard.html
"""


def log_training_img(
    module,  # : LightningModule
    target: Union[Tensor, ndarray],
    preb: Union[Tensor, ndarray],
    train_loss: float,
    batch_idx: int,
    state: str,
) -> None:
    brainSlice = BrainSlices(module, target, preb)
    fig = brainSlice.plot()
    # fig.savefig("test.png")
    brainSlice.log(state, fig, train_loss, batch_idx)
