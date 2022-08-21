# Code in this file adapted from: https://github.com/fepegar/unet/blob/master/unet/encoding.py
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from model.unet.decoding import CHANNELS_DIMENSION
from .conv import ConvolutionalBlock


class Encoder(nn.Module):
    def __init__(
        self,
        residual: bool,
        dimensions: int,
        in_channels: int,
        kernel_size: int,
        normalization: str,
        downsampling_type: str,
        out_channels_first: int,
        conv_num_in_layer: List[int],
        use_bias: bool = True,
        padding_mode: str = "zeros",
        activation: Optional[str] = "ReLU",
    ):
        super().__init__()

        self.encoding_blocks = nn.ModuleList()
        num_encoding_blocks = len(conv_num_in_layer) - 1
        out_channels = out_channels_first
        for idx in range(num_encoding_blocks):
            encoding_block = EncodingBlock(
                num_block=idx,
                use_bias=use_bias,
                residual=residual,
                dimensions=dimensions,
                in_channels=in_channels,
                out_channels=out_channels,
                conv_num=conv_num_in_layer[idx],
                normalization=normalization,
                kernel_size=kernel_size,
                downsampling_type=downsampling_type,
                padding_mode=padding_mode,
                activation=activation,
            )
            self.encoding_blocks.append(encoding_block)
            if dimensions == 2:
                in_channels = out_channels
                out_channels = in_channels * 2
            elif dimensions == 3:
                in_channels = out_channels
                out_channels = in_channels * 2

            self.out_channels = self.encoding_blocks[-1].out_channels

    def forward(self, x: Tensor, predict_time: Optional[Tensor] = None) -> Tuple[List[Tensor], Tensor]:
        skips = []
        for idx, encoding_block in enumerate(self.encoding_blocks):
            if idx == 0:
                x, skip = encoding_block(x, predict_time)
            else:
                x, skip = encoding_block(x)
            skips.append(skip)
        return skips, x


class EncodingBlock(nn.Module):
    def __init__(
        self,
        conv_num: int,
        num_block: int,
        residual: bool,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        normalization: Optional[str],
        kernel_size: int = 5,
        use_bias: bool = True,
        padding_mode: str = "zeros",
        activation: Optional[str] = "ReLU",
        downsampling_type: Optional[str] = "conv",
    ):
        super().__init__()

        self.num_block = num_block
        self.residual = residual
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dimensions = dimensions

        opts: Dict = dict(
            normalization=normalization,
            kernel_size=kernel_size,
            padding_mode=padding_mode,
            use_bias=use_bias,
        )

        self.first_conv = ConvolutionalBlock(
            dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            activation=activation,
            **opts,
        )

        conv_blocks = []

        for i in range(conv_num - 1):
            if self.residual and i == (conv_num - 2):  # for the last conv, do not use the activation function
                activation = None
            else:
                activation = activation

            if num_block == 0 and i == 0:
                conv_blocks.append(
                    ConvolutionalBlock(
                        dimensions,
                        in_channels=out_channels + 1,
                        out_channels=out_channels,
                        activation=activation,
                        **opts,
                    )
                )
            else:
                conv_blocks.append(
                    ConvolutionalBlock(
                        dimensions,
                        in_channels=out_channels,
                        out_channels=out_channels,
                        activation=activation,
                        **opts,
                    )
                )

        self.downsampling_type = downsampling_type

        self.downsample = None
        if downsampling_type == "max":
            maxpool = getattr(nn, f"MaxPool{dimensions}d")
            self.downsample = maxpool(kernel_size=2)
        elif downsampling_type == "conv":
            self.downsample = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=2, stride=2)

        self.out_channels = out_channels
        self.conv_blocks = nn.Sequential(*conv_blocks)
        if activation == "LeakyReLU":
            self.activation = getattr(nn, activation)(0.2)

    def forward(self, x: Tensor, predict_time: Optional[Tensor] = None) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if self.residual:
            out = self.conv_blocks(x)
            # TODO: might need to change this code
            repeat_num = int(self.out_channels / self.in_channels)
            if self.out_channels % self.in_channels != 0:
                repeat_num += 1
                residual_layer = x.repeat([1, repeat_num, 1, 1, 1][: self.dimensions + 2])
                residual_layer = residual_layer[:, : self.out_channels, :, :, :]
            else:
                residual_layer = x.repeat([1, repeat_num, 1, 1, 1][: self.dimensions + 2])
            x = self.activation(torch.add(out, residual_layer))
        else:
            x = self.first_conv(x)
            if predict_time is not None:
                x = torch.cat([x, predict_time], dim=CHANNELS_DIMENSION)
            x = self.conv_blocks(x)

        if self.downsample is None:
            return x

        skip = x
        return self.downsample(x), skip
