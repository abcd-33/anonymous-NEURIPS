#!/bin/bash

python encode_dataset.py \
  --data_path workspace/datasets/cifar-10 \
  --checkpoint workspace/experiments/sphere-small-small-cifar-10-32px/ckpt \
  --output_path workspace/experiments/sphere-small-small-cifar-10-32px/encoding \
  --output_name encoded_dataset.npz \
  --dataset_name cifar-10 \
  --batch_size 128 \
  --num_workers 8 \
  --seed 42 \
  --deterministic False \
  --save_dtype bfloat16 \
  --use_ema True \
  --compile_model True