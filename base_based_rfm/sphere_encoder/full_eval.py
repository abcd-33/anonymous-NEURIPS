import os
import os.path as osp
import json
import argparse
import logging
import shutil
import random
from types import SimpleNamespace
from cli_utils import str2bool

import numpy as np
import torch
import torch.nn.functional as F
import torch_fidelity
from tqdm import tqdm
from PIL import Image
import torchvision
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader

from sphere.model import G
from sphere.ema import SimpleEMA
from sphere.utils import load_ckpt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClassSubsetDataset(Dataset):
    """Wraps a standard dataset and filters it to a single target class label."""
    def __init__(self, base_dataset, target_class):
        self.samples = []
        for img, label in base_dataset:
            if label == target_class:
                tensor_img = TF.pil_to_tensor(img)
                self.samples.append(tensor_img)

        if len(self.samples) == 0:
            raise ValueError(f"No samples found for class {target_class} in the base dataset.")

        logger.info(f"Created real reference subset for class {target_class} "
                    f"containing {len(self.samples)} images.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class InMemoryImageDataset(Dataset):
    """Holds decoded images completely in RAM as torch.Tensors for torch_fidelity."""
    def __init__(self, images=None):
        self.images = images if images is not None else []

    def append(self, img_tensor):
        self.images.append(img_tensor)

    def extend(self, other_dataset):
        self.images.extend(other_dataset.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]


def make_inception_extractor(device):
    """
    Builds the torch_fidelity-compatible Inception-v3 extractor once and returns it.
    Reuse this across real and fake feature extraction to avoid loading weights twice.
    """
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
    extractor = FeatureExtractorInceptionV3(
        name="inception-v3-compat",
        features_list=["2048"],  # pool3; valid keys: 64, 192, 768, 2048, logits_unbiased, logits
    ).to(device)
    extractor.eval()
    return extractor


def extract_inception_features(dataset, batch_size, device, extractor=None, desc="Extracting features"):
    """
    Extracts Inception-v3 pool3 features (2048-d) using torch_fidelity's internal
    "inception-v3-compat" extractor — the exact same model and preprocessing used
    for FID/Precision/Recall inside torch_fidelity.  This guarantees all four
    metrics (Precision, Recall, Density, Coverage) live in the same feature space.

    Pass a pre-built extractor to avoid reloading weights for each call.
    Images must be uint8 tensors of shape (C, H, W) in range [0, 255].
    Returns: float32 numpy array of shape (N, 2048).
    """
    if extractor is None:
        extractor = make_inception_extractor(device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=False)  # pin_memory=False saves RAM

    all_features = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            feats = extractor(batch.to(device))[0].squeeze(-1).squeeze(-1)  # tuple[0] = "2048" (pool3)
            all_features.append(feats.cpu().float())
            torch.cuda.empty_cache()

    return torch.cat(all_features, dim=0).numpy()


# -------------------------------------------------
# Density & Coverage  (Naeem et al., 2020)
# -------------------------------------------------
def compute_density_coverage(real_feats, fake_feats, nearest_k, chunk_size=2000):
    """
    Computes Density and Coverage from Inception feature arrays.

    Density  (quality proxy):
        For each fake feature, count how many real k-NN balls it falls inside,
        normalised by k.  Higher → generated images are more concentrated in
        densely-populated real regions.

    Coverage (diversity proxy):
        Fraction of real images whose k-NN ball contains at least one generated
        image.  Higher → generated images spread across more real modes.

    Args:
        real_feats : np.ndarray  (N_real, D)
        fake_feats : np.ndarray  (N_fake, D)
        nearest_k  : int         neighbourhood size
        chunk_size : int         rows of real processed at a time to cap RAM usage.
                                 At 2000 rows × 10k fake × 4 bytes = ~80 MB per chunk.

    Returns:
        density  : float
        coverage : float
    """
    real_t = torch.from_numpy(real_feats)   # (N_r, D)  — CPU
    fake_t = torch.from_numpy(fake_feats)   # (N_f, D)  — CPU

    # Clamp k so it never exceeds dataset size - 1 (guards against tiny per-class splits)
    N_real = real_t.shape[0]
    effective_k = max(1, min(nearest_k, N_real - 1))
    if effective_k != nearest_k:
        logger.warning(f"nearest_k={nearest_k} clamped to {effective_k} (only {N_real} real samples)")

    # ── Step 1: k-NN radius for every real point ──────────────────────────
    # Build real–real distances in chunks to avoid a full (N_r × N_r) matrix.
    # For CIFAR-10 train (50k): 50k×50k×4 bytes = 10 GB — way too large unsharded.
    logger.info("Computing real k-NN radii (chunked)...")
    real_knn_radius = torch.empty(N_real, dtype=torch.float32)
    for start in tqdm(range(0, N_real, chunk_size), desc="kNN radii"):
        end     = min(start + chunk_size, N_real)
        chunk   = real_t[start:end]                            # (C, D)
        dists   = torch.cdist(chunk, real_t)                   # (C, N_r)
        # Mask self-distances: set the diagonal block to inf
        for i in range(end - start):
            dists[i, start + i] = float("inf")
        topk, _ = torch.topk(dists, k=effective_k, dim=1, largest=False)
        real_knn_radius[start:end] = topk[:, -1] + 1e-8       # epsilon guards zero-radius

    # ── Step 2: Density & Coverage via chunked real–fake distances ─────────
    # Full (N_r × N_f) float32 at 50k×10k = 2 GB — processed in row-chunks instead.
    logger.info("Computing Density & Coverage (chunked)...")
    density_count = 0
    covered       = torch.zeros(N_real, dtype=torch.bool)

    for start in tqdm(range(0, N_real, chunk_size), desc="Density/Coverage"):
        end    = min(start + chunk_size, N_real)
        chunk  = real_t[start:end]                             # (C, D)
        rf     = torch.cdist(chunk, fake_t)                    # (C, N_f)
        radius = real_knn_radius[start:end].unsqueeze(1)       # (C, 1)
        inside = rf <= radius                                  # (C, N_f) bool
        density_count       += inside.sum().item()
        covered[start:end]   = inside.any(dim=1)

    density  = density_count / (effective_k * len(fake_feats))
    coverage = covered.float().mean().item()

    return density, coverage


# -------------------------------------------------
# CLI
# -------------------------------------------------
parser = argparse.ArgumentParser(description="Decode encodings + eval (split-aware)")

parser.add_argument("--encoding_path",   type=str, required=True)
parser.add_argument("--checkpoint",      type=str, required=True)
parser.add_argument("--output_dir",      type=str, default="decoded_eval")
parser.add_argument("--dataset_name",    type=str, default="cifar-10")
parser.add_argument("--batch_size",      type=int, default=128)
parser.add_argument("--use_ema",         type=bool, default=True)
parser.add_argument("--compile_model",   type=str2bool, default=True)
parser.add_argument("--dtype",           type=str, default="bfloat16",
                    choices=["float32", "bfloat16", "float16"])
parser.add_argument("--normalize_latents", type=bool, default=True)
parser.add_argument("--image_size",      type=int, default=32)

parser.add_argument("--eval_per_class",  type=str2bool, default=True,
                    help="Compute metrics per individual class label.")
parser.add_argument("--save_images",     type=str2bool, default=False,
                    help="Save decoded images to disk.")

# Metric options
parser.add_argument("--nearest_k",      type=int, default=5,
                    help="k for k-NN manifold estimation used by all four metrics. "
                         "Default 5 (Naeem et al. recommendation). "
                         "Kynkäänniemi et al. use k=3 for Precision/Recall.")
parser.add_argument("--use_isc",        type=str2bool, default=False,
                    help="Also compute Inception Score (optional).")
parser.add_argument("--metrics_output", type=str, default="metrics_summary.json")

parser.add_argument("--seed",           type=int, default=42)
parser.add_argument("--deterministic",  type=str2bool, default=False)

cli_args = parser.parse_args()


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_data(path):
    data = np.load(path, allow_pickle=False)
    z_input    = torch.from_numpy(data["encodings"]).float()
    labels     = torch.from_numpy(data["labels"]).long()
    split_ids  = torch.from_numpy(data["split_ids"]).long()
    split_names = data["split_names"].tolist()
    return z_input, labels, split_ids, split_names


def find_checkpoint(path):
    if os.path.isfile(path):
        return path
    elif os.path.isdir(path):
        files = sorted([f for f in os.listdir(path) if f.endswith(".pth")])
        if not files:
            raise FileNotFoundError(f"No .pth files in {path}")
        return os.path.join(path, files[-1])
    else:
        raise FileNotFoundError(path)


def get_split_indices(split_ids, target_id):
    return (split_ids == target_id).nonzero(as_tuple=True)[0]


def decode_from_latents(model, z, y=None):
    x = model.decoder(z, y)
    return torch.clamp(x * 0.5 + 0.5, 0, 1)


@torch.inference_mode()
def decode_and_collect(model, z, y, save_dir, args, ptdtype, device):
    """Decodes latents and returns an InMemoryImageDataset."""
    if args.save_images:
        os.makedirs(save_dir, exist_ok=True)

    memory_ds   = InMemoryImageDataset()
    num_samples = z.shape[0]
    num_batches = int(np.ceil(num_samples / args.batch_size))
    image_size  = getattr(args, "image_size", 32)

    logger.info(f"Decoding {num_samples} samples...")
    for batch_idx in tqdm(range(num_batches)):
        start   = batch_idx * args.batch_size
        end     = min(start + args.batch_size, num_samples)
        z_batch = z[start:end].to(device)
        y_batch = y[start:end].to(device) if y is not None else None

        if args.normalize_latents:
            z_batch = model.spherify(z_batch)

        with torch.autocast(device_type="cuda", dtype=ptdtype):
            x_rec = decode_from_latents(model, z_batch, y_batch)

        x_rec  = (x_rec * 255.0).clamp(0, 255).to(torch.uint8).cpu()
        y_cpu  = y_batch.cpu().numpy() if y_batch is not None else [None] * len(x_rec)

        for i, (img_tensor, label) in enumerate(zip(x_rec, y_cpu)):
            if args.save_images:
                from sphere.loader import resize_arr
                img_np     = img_tensor.permute(1, 2, 0).numpy()
                image_name = f"label={label}_ord={batch_idx:05d}_idx={i:05d}.png"
                image      = Image.fromarray(img_np)
                # if image_size > 0:
                #     image = resize_arr(image, image_size=image_size)
                image.save(osp.join(save_dir, image_name), format="PNG", compress_level=0)
            memory_ds.append(img_tensor)

        torch.cuda.empty_cache()

    return memory_ds


# -------------------------------------------------
# Metric Runner
# -------------------------------------------------
def run_metrics(fake_ds, real_ds_or_str, args, eval_name, device,
                cache_input2_name=None, inception_extractor=None,
                cached_real_feats=None):
    """
    Computes Precision, Recall, Density, Coverage, and FID using the original 
    torch_fidelity workflow for FID while keeping RAM footprint clean.
    """
    logger.info(f"\nRunning metrics for: {eval_name}")

    if inception_extractor is None:
        inception_extractor = make_inception_extractor(device)

    # 1. Safely extract or load Real Features for Density / Coverage
    if cached_real_feats is not None:
        real_feats = cached_real_feats
        logger.info("Using cached real Inception features.")
    else:
        real_src = real_ds_or_str if not isinstance(real_ds_or_str, str) else _load_tf_builtin(real_ds_or_str)
        real_feats = extract_inception_features(
            real_src, batch_size=args.batch_size, device=device,
            extractor=inception_extractor, desc="Extracting real features"
        )

    # 2. Safely extract Fake Features for Density / Coverage
    fake_src = fake_ds if not isinstance(fake_ds, str) else _load_tf_builtin(fake_ds)
    fake_feats = extract_inception_features(
        fake_src, batch_size=args.batch_size, device=device,
        extractor=inception_extractor, desc="Extracting fake features"
    )

    # 3. Pure PyTorch Memory-Safe Precision & Recall Implementation
    logger.info("Computing Precision & Recall via chunked matrix distances...")
    real_t = torch.from_numpy(real_feats)
    fake_t = torch.from_numpy(fake_feats)

    N_real = real_t.shape[0]
    N_fake = fake_t.shape[0]
    effective_k = max(1, min(args.nearest_k, N_real - 1))
    effective_k_fake = max(1, min(args.nearest_k, N_fake - 1))

    # --- Step 3a: Get k-NN radii for Real Manifold ---
    real_knn_radius = torch.empty(N_real, dtype=torch.float32)
    chunk_size = 2000
    for start in range(0, N_real, chunk_size):
        end = min(start + chunk_size, N_real)
        dists = torch.cdist(real_t[start:end], real_t)
        for i in range(end - start):
            dists[i, start + i] = float("inf")  # Mask self-distance
        topk, _ = torch.topk(dists, k=effective_k, dim=1, largest=False)
        real_knn_radius[start:end] = topk[:, -1] + 1e-8

    # --- Step 3b: Get k-NN radii for Fake Manifold ---
    fake_knn_radius = torch.empty(N_fake, dtype=torch.float32)
    for start in range(0, N_fake, chunk_size):
        end = min(start + chunk_size, N_fake)
        dists = torch.cdist(fake_t[start:end], fake_t)
        for i in range(end - start):
            dists[i, start + i] = float("inf")  # Mask self-distance
        topk, _ = torch.topk(dists, k=effective_k_fake, dim=1, largest=False)
        fake_knn_radius[start:end] = topk[:, -1] + 1e-8

    # --- Step 3c: Evaluate Precision and Recall ---
    fake_inside_real = torch.zeros(N_fake, dtype=torch.bool)
    for start in range(0, N_real, chunk_size):
        end = min(start + chunk_size, N_real)
        rf_dists = torch.cdist(real_t[start:end], fake_t)
        inside = rf_dists <= real_knn_radius[start:end].unsqueeze(1)
        fake_inside_real |= inside.any(dim=0)
    precision = fake_inside_real.float().mean().item()

    real_inside_fake = torch.zeros(N_real, dtype=torch.bool)
    for start in range(0, N_fake, chunk_size):
        end = min(start + chunk_size, N_fake)
        fr_dists = torch.cdist(fake_t[start:end], real_t)
        inside = fr_dists <= fake_knn_radius[start:end].unsqueeze(1)
        real_inside_fake |= inside.any(dim=0)
    recall = real_inside_fake.float().mean().item()

    # 4. Standard FID Calculation using torch_fidelity high-level entrypoint
    # By using the native dataset type, torch_fidelity stays happy.
    logger.info("Computing FID via torch_fidelity public API...")
    
    tf_kwargs = {
        "input1":        fake_ds,
        "input2":        real_ds_or_str,
        "cuda":          torch.cuda.is_available(),
        "batch_size":    args.batch_size,
        "isc":           False,
        "fid":           True,
        "kid":           False,
        "prc":           False,
        "ppl":           False,
        "verbose":       False,
    }
    if isinstance(fake_ds, str):
        tf_kwargs["samples_find_deep"] = True
    if cache_input2_name is not None:
        tf_kwargs["cache_input2_name"] = cache_input2_name

    tf_metrics = torch_fidelity.calculate_metrics(**tf_kwargs)
    fid = tf_metrics.get("frechet_inception_distance", float("nan"))

    # CRITICAL RAM SAFETY: Instantly purge heavy raw images AFTER torch_fidelity finishes
    if isinstance(fake_src, InMemoryImageDataset):
        fake_src.images.clear()
    if isinstance(fake_ds, InMemoryImageDataset):
        fake_ds.images.clear()
    import gc; gc.collect()

    # 5. Compute Density & Coverage using your chunked implementation
    density, coverage = compute_density_coverage(real_feats, fake_feats, args.nearest_k)

    result = {
        "quality_precision": precision,
        "quality_density":   density,
        "diversity_recall":  recall,
        "diversity_coverage": coverage,
        "fid": fid,
    }

    # Pretty log
    sep = "=" * 62
    logger.info(f"\n{sep}")
    logger.info(f"  {eval_name}")
    logger.info(sep)
    logger.info(f"  {'Metric':<30}  {'Score':>8}  {'Meaning'}")
    logger.info(f"  {'-'*58}")
    logger.info(f"  {'QUALITY  — Precision':<30}  {precision:>8.4f}  (↑ = more realistic)")
    logger.info(f"  {'QUALITY  — Density':<30}  {density:>8.4f}  (↑ = denser in real manifold)")
    logger.info(f"  {'DIVERSITY — Recall':<30}  {recall:>8.4f}  (↑ = more modes covered)")
    logger.info(f"  {'DIVERSITY — Coverage':<30}  {coverage:>8.4f}  (↑ = broader real coverage)")
    logger.info(f"  {'FID':<30}  {fid:>8.4f}  (↓ = better overall)")
    logger.info(sep)

    return result

def _load_tf_builtin(name, root="./workspace/datasets/cifar-10"):
    """
    Loads a torch_fidelity built-in dataset by name (e.g. 'cifar10-train')
    as an InMemoryImageDataset so we can run our own feature extractor on it.
    """
    import torch_fidelity.datasets as tfd
    ds = tfd.Cifar10_RGB(root=root, train=("train" in name), download=True)
    mem = InMemoryImageDataset()
    for img in ds:
        mem.append(TF.pil_to_tensor(img))
    return mem


# -------------------------------------------------
# Main
# -------------------------------------------------
@torch.inference_mode()
def main(cli_args):
    exp_dir  = osp.dirname(cli_args.checkpoint)
    cfg_path = osp.join(exp_dir, "cfg.json")
    with open(cfg_path, "r") as f:
        cfg_args = json.load(f)
    cfg_args.update(vars(cli_args))
    args   = SimpleNamespace(**cfg_args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed, args.deterministic)

    ptdtype = {"float32": torch.float32,
               "bfloat16": torch.bfloat16,
               "float16": torch.float16}[args.dtype]

    z, y, split_ids, split_names = load_data(args.encoding_path)

    real_cifar10_train = None
    if args.dataset_name == "cifar-10":
        logger.info("Loading real CIFAR-10 train set...")
        real_cifar10_train = torchvision.datasets.CIFAR10(
            root="./workspace/datasets/cifar-10", train=True, download=True)

    model = G(
        input_size=args.image_size, patch_size=args.patch_size,
        vit_enc_model_size=args.vit_enc_model_size,
        vit_dec_model_size=args.vit_dec_model_size,
        token_channels=args.token_channels,
        num_classes=args.num_classes if args.cond_generator else 0,
        halve_model_size=args.halve_model_size,
        spherify_model=args.spherify_model,
        pixel_head_type=args.pixel_head_type,
        in_context_size=args.in_context_size,
        noise_sigma_max_angle=args.noise_sigma_max_angle,
        vit_enc_latent_mlp_mixer_depth=args.vit_enc_latent_mlp_mixer_depth,
        vit_dec_latent_mlp_mixer_depth=args.vit_dec_latent_mlp_mixer_depth,
        affine_latent_mlp_mixer=args.affine_latent_mlp_mixer,
    ).to(device=device, memory_format=torch.channels_last)

    ema_model = SimpleEMA(model)
    load_ckpt(model, find_checkpoint(args.checkpoint), ema_model=ema_model,
              strict=True, override_model_with_ema=args.use_ema)
    if args.compile_model:
        model.compile()
    model.eval().requires_grad_(False)

    splits_to_eval = ([(i, name) for i, name in enumerate(split_names)]
                      if split_ids is not None else [(None, "all")])
    all_metrics = {}

    # Build Inception extractor once — reused across all splits/classes
    logger.info("Building Inception extractor (loaded once for all evaluations)...")
    inception_extractor = make_inception_extractor(device)

    # Pre-extract real features once for the overall reference set.
    # This avoids re-running Inception over 50k CIFAR images for every split.
    cached_overall_real_feats = None
    if args.dataset_name == "cifar-10":
        logger.info("Pre-extracting real CIFAR-10 train Inception features (one-time cost)...")
        real_ref_ds = _load_tf_builtin("cifar10-train")
        cached_overall_real_feats = extract_inception_features(
            real_ref_ds, batch_size=args.batch_size, device=device,
            extractor=inception_extractor, desc="Real CIFAR-10 features"
        )
        del real_ref_ds
        import gc; gc.collect()
        logger.info(f"Real features cached: {cached_overall_real_feats.shape}")

    for split_id, split_name in splits_to_eval:
        logger.info(f"\n{'#'*60}\n  Split: {split_name}\n{'#'*60}")

        if split_id is not None:
            idx     = get_split_indices(split_ids, split_id)
            z_split = z[idx]
            y_split = y[idx] if y is not None else None
        else:
            z_split, y_split = z, y

        split_dir = osp.join(args.output_dir, f"decoded_{split_name}")
        if args.save_images and osp.exists(split_dir):
            shutil.rmtree(split_dir)

        overall_split_ds = InMemoryImageDataset()

        # 1. Per-class evaluation
        if args.eval_per_class and y_split is not None:
            unique_classes = torch.unique(y_split)
            for cls_id in unique_classes:
                cls_id  = cls_id.item()
                cls_idx = (y_split == cls_id).nonzero(as_tuple=True)[0]
                z_cls, y_cls = z_split[cls_idx], y_split[cls_idx]
                cls_dir = osp.join(split_dir, f"class_{cls_id}")
                cls_ds  = decode_and_collect(model, z_cls, y_cls, cls_dir, args, ptdtype, device)
                overall_split_ds.extend(cls_ds)

                if real_cifar10_train is not None:
                    input2_ref  = ClassSubsetDataset(real_cifar10_train, target_class=cls_id)
                    cache_name  = f"{args.dataset_name}_train_class_{cls_id}"
                else:
                    input2_ref = None
                    cache_name = None

                metric_input = cls_dir if args.save_images else cls_ds

                cls_key      = f"{split_name}_class_{cls_id}"
                # Per-class real features are small — no pre-caching needed
                all_metrics[cls_key] = run_metrics(
                    metric_input, input2_ref, args, cls_key, device,
                    cache_input2_name=cache_name,
                    inception_extractor=inception_extractor,
                )
        else:
            overall_split_ds = decode_and_collect(
                model, z_split, y_split, split_dir, args, ptdtype, device)

        # 2. Overall split evaluation
        overall_input2   = "cifar10-train" if args.dataset_name == "cifar-10" else None
        metric_input_ovr = split_dir if args.save_images else overall_split_ds
        all_metrics[split_name] = run_metrics(
            metric_input_ovr, overall_input2, args, split_name, device,
            inception_extractor=inception_extractor,
            cached_real_feats=cached_overall_real_feats,  # skip re-extracting 50k real images
        )
        # Free the decoded split images — features already extracted inside run_metrics
        overall_split_ds.images.clear()
        import gc; gc.collect()

    # -------------------------------------------------
    # Final summary table
    # -------------------------------------------------
    sep = "=" * 78
    logger.info(f"\n{sep}")
    logger.info("  FINAL SUMMARY")
    logger.info(f"  {'Eval Set':<32}  {'Precision':>9}  {'Density':>9}  "
                f"{'Recall':>9}  {'Coverage':>9}  {'FID':>9}")
    logger.info(f"  {'':─<32}  {'(quality)':>9}  {'(quality)':>9}  "
                f"{'(divers.)':>9}  {'(divers.)':>9}  {'(↓ better)':>9}")
    logger.info(f"  {'-'*84}")
    for key, m in all_metrics.items():
        p  = m.get("quality_precision",  float("nan"))
        d  = m.get("quality_density",    float("nan"))
        r  = m.get("diversity_recall",   float("nan"))
        c  = m.get("diversity_coverage", float("nan"))
        fi = m.get("fid",                float("nan"))
        logger.info(f"  {key:<32}  {p:>9.4f}  {d:>9.4f}  {r:>9.4f}  {c:>9.4f}  {fi:>9.4f}")
    logger.info(sep)

    # Save JSON
    out_path = args.metrics_output
    os.makedirs(osp.dirname(out_path) if osp.dirname(out_path) else ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"\nMetrics saved to: {out_path}")


if __name__ == "__main__":
    main(cli_args)