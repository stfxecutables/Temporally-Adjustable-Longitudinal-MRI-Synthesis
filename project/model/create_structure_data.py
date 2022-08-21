from typing import Tuple, List

from torch import nn, Tensor
from torch.nn import ConvTranspose3d, BatchNorm3d, LeakyReLU

from utils.const import PATCH_SIZE_3D


class CreateStructuredData(nn.Module):
    def __init__(self, patch_shape: Tuple[int] = PATCH_SIZE_3D):
        super().__init__()
        self.patch_shape = patch_shape

        def block(stride: int, padding: int, normalize: bool = True, kernel_size: int = 5):
            layers: List = [
                ConvTranspose3d(
                    in_channels=1,
                    stride=stride,
                    out_channels=1,
                    padding=padding,
                    output_padding=1,
                    kernel_size=kernel_size,
                )
            ]
            if normalize:
                layers.append(BatchNorm3d(num_features=1))
            layers.append(LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(stride=2, padding=1, normalize=False, kernel_size=3),  # TODO: why the first layer is not normalized?
            *block(stride=4, padding=1, kernel_size=5),
            *block(stride=4, padding=1, kernel_size=5),
            *block(stride=2, padding=2, kernel_size=5),
        )

    def forward(self, x: Tensor) -> Tensor:
        img = self.model(x)
        return img
