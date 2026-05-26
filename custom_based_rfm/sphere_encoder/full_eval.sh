# #!/bin/bash

python3 full_eval.py \
  --encoding_path workspace/experiments/custom/encoding/output_encodings.npz \
  --checkpoint workspace/experiments/custom/encoding/best_spherical_autoencoder.pt \
  --output_dir workspace/experiments/custom/decoded_eval \
  --dataset_name cifar-10 \
  --batch_size 8 \
  --dtype bfloat16 \
  --normalize_latents True \
  --image_size 32 \
  --eval_per_class False \
  --save_images False \
  --seed 42 \
  --deterministic False