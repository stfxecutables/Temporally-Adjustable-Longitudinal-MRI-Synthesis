import pytorch_lightning as pl
from pytorch_lightning.core.memory import ModelSummary
import torch

# from monai.networks.nets import UNet
from model.unet.unet import UNet


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.example_input_array = [torch.zeros(1, 4, 128, 128, 128), torch.zeros(1, 1, 2, 2, 2)]

        self.model = UNet(
            dimensions=3,
            in_channels=4,
            out_classes=1,
            kernel_size=3,
            use_tanh=True,
            use_bias=False,
            residual=False,
            padding_mode="zeros",
            normalization="Batch",
            activation="LeakyReLU",
            downsampling_type="max",
            # TODO: might need to using more conv layers
            out_channels_first_layer=16,
            merge_original_patches=False,
            conv_num_in_layer=[2, 2, 2, 2, 2, 2],  # [1, 2, 2, 2, 2, 2] [1, 2, 3, 3, 3]
        )

    def forward(self, x, predict_time):
        return self.model(x, predict_time)


# class HighResNetModel(pl.LightningModule):
#     def __init__(self):
#         super().__init__()
#         self.example_input_array = torch.zeros(1, 1, 96, 96, 96)

#         self.unet = HighResNet(in_channels=1, out_channels=139, dimen sions=3)

#     def forward(self, x):
#         return self.unet(x)


if __name__ == "__main__":
    # HighResNet = HighResNetModel()
    # print("highResNet Model:")
    # print(ModelSummary(HighResNet, mode="full"))

    Net = Model()
    print(ModelSummary(Net, max_depth=-1))
