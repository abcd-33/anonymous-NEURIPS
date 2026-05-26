#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree


#SBATCH --job-name=sphere-encoder-cifar10
#SBATCH --output=slurm/sphere-encoder-cifar10_%j.log
#SBATCH --error=slurm/sphere-encoder-cifar10_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1

module purge
module load 2024
module load Anaconda3/2024.06-1

source activate sphere_hyper

# 5000 was the original number of epochs, but it is too long for testing purposes


./run.sh train.py \
  --dataset_name cifar-10 \
  --image_size 32 \
  --batch_size_per_rank 128 \
  --data_dir /scratch-local/scur0199/datasets/cifar-10 \
  --warmup_epochs 10 \
  --epochs 1000 \
  --early_stop_patience 15 \
  --compression_ratio 3 \
  --noise_sigma_max_angle 80 \
  --vit_enc_model_size small \
  --vit_dec_model_size small \
  --vit_enc_latent_mlp_mixer_depth 2 \
  --vit_dec_latent_mlp_mixer_depth 2 \
  --affine_latent_mlp_mixer True \
  --pixel_head_type conv \
  --ckpt_save_interval 100 \
  --out_dir experiments \
  --lat_con_loss_weight 0.1 \
  --pix_recon_dist_loss_weight 1.0 \
  --pix_recon_perc_loss_weight 1.0 \
  --pix_con_dist_loss_weight 0.5 \
  --pix_con_perc_loss_weight 0.5
