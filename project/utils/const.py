import os
from pathlib import Path

# flake8: noqa
# Use environment variables to auto-detect whether we are running an a Compute Canada cluster:
# Thanks to https://github.com/DM-Berger/unet-learn/blob/master/src/train/load.py for this trick.
COMPUTECANADA = False
IN_COMPUTE_CAN_JOB = False

TMP = os.environ.get("SLURM_TMPDIR")
ACT = os.environ.get("SLURM_ACCOUNT")


if ACT:  # If only ACT is True, we are just in a login node
    COMPUTECANADA = True
if TMP:  # If there is a SLURM_TMPDIR we are (probably) running on a non-login node, i.e. in a job
    COMPUTECANADA = True
    IN_COMPUTE_CAN_JOB = True

# fmt: off
if COMPUTECANADA:
    DATA_ROOT = Path(str(TMP)).resolve() / "work"
else:
    DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "MS"

PATCH_SIZE     = 128
PATCH_SIZE_3D  = (PATCH_SIZE, PATCH_SIZE, PATCH_SIZE)
PATCH_SIZE_2D  = (PATCH_SIZE, PATCH_SIZE)
MS_IMG_SIZE    = (150, 190, 150)
NUM_WORKERS    = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
MS_IMG_SIZE_NOVEL = (160, 200, 160)
