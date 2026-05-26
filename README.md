# Spherical Flow Matching

> Investigating Riemannian Flow Matching on hyperspherical latent spaces - diagnosing dimensionality bottlenecks and introducing a multi-sphere encoder architecture for improved generative transport.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Platform](https://img.shields.io/badge/platform-Linux-lightgrey)
![GPU](https://img.shields.io/badge/GPU-H100%20recommended-green)

---

## Overview

This repository accompanies the paper:

> **Generative Transport on the Hypersphere: Diagnosing Dimensionality Bottlenecks in Spherical Flow Matching**  

We investigate whether [Riemannian Flow Matching (RFM)](https://arxiv.org/abs/2302.03660) can learn meaningful transport dynamics on hyperspherical latent spaces produced by the [Sphere Encoder](https://arxiv.org/abs/2602.15030). Our key findings:

- **RFM on the original Sphere Encoder yields no meaningful gains.** Global RMS normalization produces a near-uniform latent distribution with weak density gradients, giving RFM little geometric structure to exploit. The single massive hypersphere (L = h·w·d) also causes distance concentration, making transport learning ill-conditioned.
- **A multi-sphere architecture resolves this.** Replacing the single large hypersphere with smaller per-token hyperspheres restores localized geometric structure, allowing RFM to produce substantial FID improvements (129.96 → 22.64 on CIFAR-10).

This repository provides two implementations:
- **Meta-based RFM** - RFM applied to the original Sphere Encoder (Meta).
- **Custom Sphere Encoder (SAE)** - our novel Transformer-based multi-sphere autoencoder, paired with a modified RFM.

---

## Results

### Meta-based Sphere Encoder (CIFAR-10)
    
| Sampling | Precision ↑ | Recall ↑ | FID ↓ |
|---|---|---|---|
| RFM | 0.8299 | 0.1312 | 37.66 |
| Uniform (no RFM) | 0.8110 | 0.1820 | 35.55 |

RFM marginally improves precision but slightly hurts recall and FID - consistent with our hypothesis that near-uniform latent occupancy leaves little for the flow to learn.

### Custom Spherical Autoencoder (CIFAR-10)

| Sampling | FID ↓ |
|---|---|
| RFM | **22.64** |
| Uniform (no RFM) | 129.96 |

The multi-sphere architecture enables RFM to reduce FID by over 5× compared to uniform sampling, demonstrating that latent geometry is the key bottleneck.

See the paper appendix for full per-class breakdowns.

### Compute

All experiments were run on a single H100 GPU node (Snellius).

| Component | Training | Inference |
|---|---|---|
| Meta SAE | 36 hours | 3 mins |
| Our SAE | 3 hours | 1 min |
| RFM (Meta SAE) | 3 mins | 3 mins |
| Our RFM (Our SAE) | 2 hours | 10 mins |

---

## Getting Started

### Prerequisites

- Linux
- Python 3.10+
- Conda
- NVIDIA H100 or A100 GPU (strongly recommended - see [Hardware](#hardware))

```bash
conda env create -f environment.yml
conda activate sphere_hyper
```

### Workspace Structure

Both pipelines expect the following layout. Set up this structure before running any scripts:

```text
workspace/
├── datasets/
│   └── cifar-10/
│       └── cifar-10-python.tar.gz        # https://www.cs.toronto.edu/~kriz/cifar.html
├── experiments/
│   └── sphere-small-small-cifar-10-32px/
│       ├── ckpt/                          # Saved checkpoints
│       └── encoding/                      # Precomputed latent encodings
└── pretrained/
    └── lpips/
        └── vgg.pth                        # VGG-16 weights for perceptual loss
```

> Update `config/config.yaml` to point to your actual workspace root before running.

---

## Usage Guide

### 1. Meta-based RFM (`meta_based_rfm`)

**Train**

```bash
cd meta_based_rfm
bash train.sh
```

Preprocesses CIFAR-10 and trains the Riemannian flow matching model on the Sphere Encoder latent space. Review `config/config.yaml` and verify all paths before running.

**Sample**

```bash
bash sample.sh
```

Control sampling mode via `direct_sampling` in your config:

| `direct_sampling` | Behavior |
|---|---|
| `False` | Sample using learned Flow Matching (recommended) |
| `True` | Sample directly from the encoder (uniform baseline) |

**Evaluate**

Run the evaluation scripts in the `sphere_encoder` directory. Update file paths to match your workspace first.

---

### 2. Custom Sphere Encoder (`custom_sae`)

The workflow mirrors the Meta-based pipeline above, with two differences:

**Localized config:** Paths are set inside individual script files rather than a central config, check each file before running.

**Expected training exit:** The RFM training loop will appear to terminate abruptly. This is intentional. Model weights are saved correctly before this happens and all downstream steps work normally.

---

## Architecture: Custom Spherical Autoencoder (SAE)

Our SAE is a conditional Transformer-based autoencoder (~200M parameters) that constrains each latent token to its own unit hypersphere, rather than one global sphere.

**Key design choices:**

- **Backbone:** Symmetric ViT - 24 layers, hidden dim 768, 4×4 patch tokenization
- **FFN:** SwiGLU variants instead of standard MLPs
- **Normalization:** RMSNorm throughout
- **Conditioning:** AdaLN-Single for class conditioning across all layers
- **Latent constraint:** L2 normalization per token, enforcing ‖z‖₂ = 1.0
- **Decoder head:** PixelShuffle to avoid checkerboard artifacts

**Training objective:**

```
L_total = λ_mse · L_MSE + λ_l1 · L_L1 + λ_perc · L_perc + λ_latent · L_latent
```

with λ_mse=1.0, λ_l1=0.2, λ_perc=0.2, λ_latent=0.2. The perceptual loss uses a pretrained VGG-16 up to `relu3_3`. The latent loss penalizes squared geodesic distance on the sphere between the original and re-encoded latents.

Spherifying noise injection perturbs clean encoder tokens via exponential map (`ε = 0.10`), with 4 random tokens replaced by isotropic noise per sequence.

**RFM modifications:** Self-attention over the sequence dimension gives the flow model context across tokens; the model outputs a per-token velocity vector on each local hypersphere.

---

## Hardware

Training is computationally intensive. Consumer GPUs may work for small-scale experiments but are not recommended for full runs.

**Recommended:** NVIDIA H100 or A100 (40GB+). All reported results used a single H100 on the Snellius cluster.

---


## License

This project is licensed under the [MIT License](LICENSE).