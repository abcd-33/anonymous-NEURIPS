#!/bin/bash
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

python train.py \
    --dataset_name "cifar-10" \
    --data_dir "../../../sphere-encoder-main/workspace/datasets" \
    --num_classes 10 \
    --image_size 32 \
    --latent_dim 16 \
    --hidden_dim 768 \
    --cond_dim 256 \
    --num_layers 12 \
    --patch_size 4 \
    --batch_size 256 \
    --lr 5e-4 \
    --epochs 300 \
    --warmup_epochs 5 \
    --dropout 0.1 \
    --null_chance 0.1 \
    --noise_level 0.1 \
    --num_noisy_tokens 8 \
    --patience 20 \
    --lambda_mse 1.0 \
    --lambda_l1 0.2 \
    --lambda_perc 0.2 \
    --lambda_latent 0.2 \
