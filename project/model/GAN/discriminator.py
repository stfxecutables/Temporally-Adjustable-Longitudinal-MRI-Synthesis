from torch import nn
from torch import Tensor
import torch

from model.create_structure_data import CreateStructuredData
from utils.const import PATCH_SIZE_3D


# Code is adapted from: https://github.com/taozh2017/HiNet/blob/master/model/syn_model.py#L272
class Discriminator(nn.Module):
    def __init__(self, in_channels=1, putting_time_into_discriminator: bool = False):
        super(Discriminator, self).__init__()
        self.putting_time_into_discriminator = putting_time_into_discriminator

        def discrimintor_block(in_features: int, out_features: int, normalize: bool = True):
            """Discriminator block"""
            layers = [nn.Conv3d(in_features, out_features, 3, stride=2, padding=1)]
            if normalize:
                # the code in HiNet is set eps=0.8, I don't understand why, and I delete it
                layers.append(nn.BatchNorm3d(out_features))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.create_structure_data = CreateStructuredData(patch_shape=PATCH_SIZE_3D)
        self.model = nn.Sequential(
            *discrimintor_block(in_channels, 64, normalize=False),
            *discrimintor_block(64, 128),
            *discrimintor_block(128, 256),
            *discrimintor_block(256, 512),
            nn.Conv3d(512, 1, kernel_size=3)
        )

    def forward(self, img: Tensor, predict_time: Tensor) -> Tensor:
        if self.putting_time_into_discriminator:
            predict_time = self.create_structure_data(predict_time)
            img = torch.cat((img, predict_time), dim=1)
        return self.model(img)
