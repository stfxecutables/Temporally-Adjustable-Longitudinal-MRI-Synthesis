import pandas as pd
import numpy as np

from utils.const import DATA_ROOT


ELSX_PATH = DATA_ROOT.parent / "Result"


if __name__ == "__main__":
    filenames = ["ACGAN.xlsx", "dt-GAN.xlsx", "gt-GAN.xlsx", "rUNet.xlsx"]

    for filename in filenames:
        cur_file = ELSX_PATH / filename
        df = pd.read_excel(cur_file).to_numpy()

        std_SSIM = []
        std_PSNR = []
        std_NMSR = []
        for i in range(5):
            std_SSIM.append(np.std(df[df[:, 0] == i][:, 3]))
            std_PSNR.append(np.std(df[df[:, 0] == i][:, 4]))
            std_NMSR.append(np.std(df[df[:, 0] == i][:, 5]))

        print(f"{filename} SSIM: {std_SSIM} Mean: {np.mean(std_SSIM)}")
        print(f"{filename} PSNR: {std_PSNR} Mean: {np.mean(std_PSNR)}")
        print(f"{filename} NMSR: {std_NMSR} Mean: {np.mean(std_NMSR)}")
