import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

from lig_module.data_model.data_model_synthesis import SynthesisDataModule
from lig_module.data_model.data_model_novel import NovelDataModule
from lig_module.lig_model.lig_model_synthesis import SynthesisLitModel
from lig_module.lig_model.lig_model_GAN import GAN
from lig_module.lig_model.lig_model_ACGAN import ACGAN
from utils.const import COMPUTECANADA


def main(hparams: Namespace) -> None:
    # Function that sets seed for pseudo-random number generators in: pytorch, numpy,
    # python.random and sets PYTHONHASHSEED environment variable.
    # To make sure every GPU get the same data
    pl.seed_everything(42)
    if COMPUTECANADA:
        cur_path = Path(__file__).resolve().parent
        default_root_dir = cur_path
        dirpath = Path(__file__).resolve().parent / "checkpoint"
        filename = "{epoch}-{val_loss:0.5f}-{val_MSE:0.5f}-{val_SSIM:0.5f}-{val_PSNR:0.5f}-{val_NMSE:0.5f}"
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
    else:
        default_root_dir = Path("./log")
        if not os.path.exists(default_root_dir):
            os.mkdir(default_root_dir)
        dirpath = Path("./log/checkpoint")
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
        filename = "{epoch}-{val_loss:0.5f}"

    ckpt_path = (
        str(Path(__file__).resolve().parent / "checkpoint" / hparams.checkpoint_file)
        if hparams.checkpoint_file
        else None
    )

    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(
            dirpath=dirpath,
            filename=filename,
            monitor="val_SSIM",
            save_top_k=2,
            mode="max",
            save_weights_only=False,
        ),
    ]

    if hparams.lr_policy == "cosine":
        callbacks.append(EarlyStopping("val_loss", patience=25, mode="min"))

    # training
    trainer = Trainer(
        gpus=hparams.gpus,
        accelerator="ddp",
        deterministic=True,
        callbacks=callbacks,
        checkpoint_callback=True,
        max_epochs=hparams.max_epochs,
        resume_from_checkpoint=ckpt_path,
        fast_dev_run=hparams.fast_dev_run,
        default_root_dir=str(default_root_dir),
        plugins=DDPPlugin(find_unused_parameters=True),
        logger=loggers.TensorBoardLogger(hparams.TensorBoardLogger),
    )

    dict_args = vars(hparams)

    if hparams.dataset == "novel":
        data_module = NovelDataModule(
            batch_size=hparams.batch_size,
            kfold_num=hparams.kfold_num,
        )
    else:
        data_module = SynthesisDataModule(
            batch_size=hparams.batch_size,
            kfold_num=hparams.kfold_num,
        )

    if hparams.task == "LongitudinalSynthesis":
        model = SynthesisLitModel(**dict_args)
    elif hparams.task == "LongitudinalSynthesisGAN":
        model = GAN(**dict_args)
    elif hparams.task == "LongitudinalSynthesisACGAN":
        model = ACGAN(**dict_args)

    trainer.fit(model, data_module)


if __name__ == "__main__":
    parser = ArgumentParser(description="Trainer args", add_help=False)
    parser.add_argument("--gpus", type=int, default=1, help="how many gpus")
    parser.add_argument(
        "--tensor_board_logger",
        dest="TensorBoardLogger",
        default="/home/jq/Desktop/log",
        help="TensorBoardLogger dir",
    )
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        help="whether to run 1 train, val, test batch and program ends",
    )
    parser.add_argument(
        "--save_validation_result",
        action="store_true",
        help="save the prediction of validation set in the format of npz and mp4",
    )
    parser.add_argument(
        "--normalization_on_each_subject",
        action="store_true",
        help="normalization on each subject",
    )
    parser.add_argument(
        "--merge_original_patches",
        action="store_true",
        help="whether to merge the original patches",
    )
    parser.add_argument("--batch_size", type=int, default=3, help="batch size")
    parser.add_argument("--kfold_num", type=int, choices=[1, 2, 3, 4, 5], default=5)
    parser.add_argument("--checkpoint_file", type=str, help="resume from checkpoint file")
    parser.add_argument("--dataset", type=str, choices=["novel", "ISBI2015"], default="ISBI2015")
    parser.add_argument("--max_epochs", type=int, default=150)
    parser.add_argument(
        "--task",
        type=str,
        choices=[
            "Segmentation",
            "LongitudinalSynthesis",
            "LongitudinalSynthesisGAN",
            "LongitudinalSynthesisACGAN",
        ],
        default="LongitudinalSynthesis",
    )
    parser.add_argument(
        "--lr_policy",
        type=str,
        choices=["cosine", "linear"],
        default="linear",
    )
    parser = SynthesisLitModel.add_model_specific_args(parser)
    hparams = parser.parse_args()

    main(hparams)
