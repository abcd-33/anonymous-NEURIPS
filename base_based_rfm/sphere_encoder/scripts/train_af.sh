#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree


#SBATCH --job-name=sphere-encoder-af
#SBATCH --output=slurm/sphere-encoder-af%j.log
#SBATCH --error=slurm/sphere-encoder-af%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1

module purge
module load 2024
module load Anaconda3/2024.06-1

source activate sphere_hyper

./run.sh train.py \
  --dataset_name animal-faces \
  --image_size 256 \
  --batch_size_per_rank 32 \
  --warmup_epochs 10 \
  --epochs 1000 \
  --early_stop_patience 15 \
  --compression_ratio 1.5 \
  --noise_sigma_max_angle 85 \
  --cond_generator True \
  --vit_enc_model_size small \
  --vit_dec_model_size small \
  --vit_enc_latent_mlp_mixer_depth 4 \
  --vit_dec_latent_mlp_mixer_depth 4 \
  --affine_latent_mlp_mixer True \
  --pixel_head_type linear \
  --ckpt_save_interval 100 \
  --out_dir experiments \
  --lat_con_loss_weight 0.1 \
  --pix_recon_dist_loss_weight 25.0 \
  --pix_recon_perc_loss_weight 1.0 \
  --pix_con_dist_loss_weight 1.0 \
  --pix_con_perc_loss_weight 1.0
