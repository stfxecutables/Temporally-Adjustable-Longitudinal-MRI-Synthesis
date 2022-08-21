from typing import Tuple

from torch import nn
from torch import Tensor
import torch

from model.create_structure_data import CreateStructuredData
from utils.const import PATCH_SIZE_3D


# Code is adapted from: https://github.com/taozh2017/HiNet/blob/master/model/syn_model.py#L272
class ACGANDiscriminator(nn.Module):
    def __init__(self, in_channels=1, putting_time_into_discriminator: bool = False, class_num: int = 6):
        super(ACGANDiscriminator, self).__init__()
        self.putting_time_into_discriminator = putting_time_into_discriminator
        self.out_channels = 512

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
            # Why u don't use normalization in the first layer?
            *discrimintor_block(in_channels, 64, normalize=False),
            *discrimintor_block(64, 128),
            *discrimintor_block(128, 256),
            *discrimintor_block(256, self.out_channels)
        )

        # Ref: https://github.com/semantic-retina/semantic-retina-generation/blob/main/src/models/acgan/discriminator.py
        self.adv_layers = nn.Conv3d(self.out_channels, 1, kernel_size=3)
        self.aux_layer = nn.Linear(self.out_channels * 512, class_num)

    def forward(self, img: Tensor, predict_time: Tensor) -> Tuple[Tensor, Tensor]:
        if self.putting_time_into_discriminator:
            predict_time = self.create_structure_data(predict_time)
            img = torch.cat((img, predict_time), dim=1)
        out = self.model(img)
        validity = self.adv_layers(out)
        aux_output = self.aux_layer(out.view(out.size(0), -1))

        return validity, aux_output
