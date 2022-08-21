#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=510000M                      # memory
#SBATCH --cpus-per-task=48
#SBATCH --output=gt-GAN-%j.out             # %N for node name, %j for jobID
#SBATCH --time=00-08:00                    # time (DD-HH:MM)
#SBATCH --mail-user=x2019cwn@stfx.ca       # used to send email
#SBATCH --mail-type=ALL

SOURCEDIR=/home/coladog/projects/def-jlevman/coladog/

module load python/3.8.10 scipy-stack

# debugging flags (optional)
export NCCL_DEBUG=INFO
export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1

# set NCCL parameters to speed up
# export NCCL_NSOCKS_PERTHREAD=4
# export NCCL_SOCKET_NTHREADS=2
# export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# force to synchronization, can pinpoint the exact number of lines of error code where our memory operation is observed
export CUDA_LAUNCH_BLOCKING=1

# virtualenv --no-download $SLURM_TMPDIR/ENV
# source $SLURM_TMPDIR/env/bin/activate

echo -e '\n'
cd $SLURM_TMPDIR
mkdir work
echo "$(date +"%T"):  Copying data"
tar -xf /home/jueqi/projects/def-jlevman/jueqi/Data/MS.tar -C work && echo "$(date +"%T"):  Copied data"

cd work

GPUS=4
LOSS=l1                         # l1 l2 smoothl1 SSIM MS_SSIM
BETA1=0.5
CLIP_MIN=2
CLIP_MAX=3
LOG_MOD=15
OPTIM=Adam                      # Adam AdamW
KFOLD_NUM=1
LAMBDA_L1=300                   # 100 300
MAX_EPOCHS=200
DECAY_EPOCH=150
BATCH_SIZE=3
GAN_LOSS=BCE                    # BCE MSE
IN_CHANNELS=4
LR_POLICY=linear                # cosine, linear, plateau
WEIGHT_DECAY=7e-8
LEARNING_RATE=2e-4              # 7e-5
NORMALIZATION=Batch             # Batch Group InstanceNorm3d
ACTIVATION=LeakyReLU            # LeakyReLU ReLU
TASK=LongitudinalSynthesisGAN   # LongitudinalSynthesisGAN LongitudinalSynthesis

LOG_DIR=/home/jueqi/projects/def-jlevman/jueqi/MS_Result


# run script
echo -e '\n\n\n'
echo "$(date +"%T"):  start running model!"
tensorboard --logdir="$LOG_DIR" --host 0.0.0.0 & python3 /home/jueqi/projects/def-jlevman/jueqi/MS/2_1/project/main.py \
       --use_tanh \
       --gpus=$GPUS \
       --loss="$LOSS" \
       --optim=$OPTIM \
       --task="$TASK" \
       --beta1=$BETA1 \
       --log_mod=$LOG_MOD \
       --clip_min=$CLIP_MIN \
       --clip_max=$CLIP_MAX \
       --lambda_l1=$LAMBDA_L1 \
       --GAN_loss="$GAN_LOSS" \
       --kfold_num=$KFOLD_NUM \
       --batch_size=$BATCH_SIZE \
       --max_epochs=$MAX_EPOCHS \
       --lr_policy="$LR_POLICY" \
       --activation="$ACTIVATION" \
       --in_channels=$IN_CHANNELS \
       --decay_epoch=$DECAY_EPOCH \
       --weight_decay=$WEIGHT_DECAY \
       --learning_rate=$LEARNING_RATE \
       --normalization="$NORMALIZATION" \
       --tensor_board_logger="$LOG_DIR" && echo "$(date +"%T"):  Finished running!"


    #    --putting_time_into_discriminator \
#       --use_multichannel_input \
#       --fast_dev_run \
# --residual
# --merge_original_patches \
#        --smooth_label \
       # --train_with_randomly_cropping \
#       --predict_with_sliding_window \
#       --normalization_on_each_subject \
       # --save_validation_result \
       # --fine_tune \
#       --checkpoint_file="epoch=40-val_loss=0.00178-val_MAE=0.32242.ckpt" \

# zip /home/coladog/projects/def-jlevman/coladog/Data/flair-AdamW-npz.zip *.npz
# zip /home/coladog/projects/def-jlevman/coladog/Data/flair-AdamW-mp4.zip *.mp4