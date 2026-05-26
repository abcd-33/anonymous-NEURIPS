import os
import os.path as osp
import json
import argparse
import logging
import shutil
import random
from types import SimpleNamespace

import numpy as np
import torch
import torch_fidelity
import torchvision
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from models import SphericalAutoencoder

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

        logger.info(f"Created real reference subset for class {target_class} containing {len(self.samples)} images.")

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


parser = argparse.ArgumentParser(description="Decode Spherical Latents + Split-Aware Per-Class Evaluation")

parser.add_argument("--encoding_path", type=str, default="workspace/output_encodings.npz", help="Path to .npz file containing latent codes")
parser.add_argument("--checkpoint", type=str, default="workspace/best_spherical_autoencoder.pt", help="Path to trained autoencoder checkpoint")
parser.add_argument("--output_dir", type=str, default="./workspace/decoded_eval")
parser.add_argument("--dataset_name", type=str, default="cifar-10")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"])
parser.add_argument("--normalize_latents", type=bool, default=True)
parser.add_argument("--image_size", type=int, default=32, help="Image resolution dimension.")

parser.add_argument("--eval_per_class", type=bool, default=False,
                    help="If true, computes evaluation metrics for each individual class label separately.")

parser.add_argument("--save_images", type=bool, default=True,
                    help="If true, saves images to disk in split/class/ directories. Otherwise, runs purely in-memory.")

parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--deterministic", type=bool, default=False)

cli_args = parser.parse_args()


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
    z_input = torch.from_numpy(data["encodings"]).float()
    labels = torch.from_numpy(data["labels"]).long() if "labels" in data else None
    split_ids = torch.from_numpy(data["split_ids"]).long() if "split_ids" in data else None
    split_names = data["split_names"].tolist() if "split_names" in data else ["all"]
    return z_input, labels, split_ids, split_names


def get_split_indices(split_ids, target_id):
    return (split_ids == target_id).nonzero(as_tuple=True)[0]


@torch.inference_mode()
def decode_and_collect(decoder, z, y, save_dir, args, ptdtype, device, H_grid: int, W_grid: int): # <-- Add grid variables here
    if args.save_images:
        os.makedirs(save_dir, exist_ok=True)

    memory_ds = InMemoryImageDataset()
    num_samples = z.shape[0]
    num_batches = int(np.ceil(num_samples / args.batch_size))

    logger.info(f"Decoding {num_samples} samples through spherical manifold pathways...")
    pbar = tqdm(range(num_batches), desc="Decoding Pipeline")

    for batch_idx in pbar:
        start = batch_idx * args.batch_size
        end = min(start + args.batch_size, num_samples)

        z_batch = z[start:end].to(device)
        y_batch = y[start:end].to(device) if y is not None else torch.zeros(end - start, dtype=torch.long, device=device)

        if args.normalize_latents:
            z_batch = torch.nn.functional.normalize(z_batch, p=2, dim=-1)

        with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=ptdtype):
            x_rec = decoder(z_batch, class_ids=y_batch, H_grid=H_grid, W_grid=W_grid, force_drop=False)
            x_rec = torch.clamp(x_rec * 0.5 + 0.5, 0.0, 1.0)

        x_rec = (x_rec * 255.0).clamp(0, 255).to(torch.uint8).cpu()
        y_cpu = y_batch.cpu().numpy() if y is not None else [None] * len(x_rec)

        for i, (img_tensor, label) in enumerate(zip(x_rec, y_cpu)):
            if args.save_images:
                img_np = img_tensor.permute(1, 2, 0).numpy()
                image_name = f"label={label}_ord={batch_idx:05d}_idx={i:05d}.png"
                image_path = os.path.join(save_dir, image_name)
                image = Image.fromarray(img_np)
                image.save(image_path, format="PNG", compress_level=0)

            memory_ds.append(img_tensor)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return memory_ds


def run_metrics(input1_src, args, eval_name, reference_input2, cache_input2_name=None):
    """Invokes torch_fidelity validation matrices over RAM datasets or disk pathways."""
    logger.info(f"Running torch_fidelity verification matrix for: {eval_name}")
    
    kwargs = {
        "input1": input1_src,
        "input2": reference_input2,
        "cuda": torch.cuda.is_available(),
        "batch_size": args.batch_size,
        "isc": True,   # Inception Score
        "fid": True,   # Frechet Inception Distance
        "kid": False,
        "prc": False,
        "ppl": False,
        "verbose": True,
    }

    if isinstance(input1_src, str):
        kwargs["samples_find_deep"] = True

    if cache_input2_name is not None:
        kwargs["cache_input2_name"] = cache_input2_name

    metrics = torch_fidelity.calculate_metrics(**kwargs)
    logger.info(f"[{eval_name}] Metrics Computed: {metrics}")
    return metrics


@torch.inference_mode()
def main(cli_args):
    logger.info(f"Loading checkpoint metadata from {cli_args.checkpoint}")
    ckpt = torch.load(cli_args.checkpoint, map_location="cpu")
    args_ae = ckpt["args"]

    cfg_args = vars(args_ae)
    cfg_args.update(vars(cli_args))
    args = SimpleNamespace(**cfg_args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed, args.deterministic)
    torch.set_float32_matmul_precision('high')

    ptdtype = torch.bfloat16 if args.dtype == "bfloat16" else (torch.float16 if args.dtype == "float16" else torch.float32)
    z, y, split_ids, split_names = load_data(args.encoding_path)

    real_reference_dataset = None
    if args.dataset_name == "cifar-10":
        logger.info("Caching real CIFAR-10 train distribution for conditional subset matching...")
        real_reference_dataset = torchvision.datasets.CIFAR10(root="../../../sphere-encoder-main/workspace/datasets/cifar-10", train=True, download=True)

    autoencoder = SphericalAutoencoder(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        cond_dim=args.cond_dim,
        num_classes=args.num_classes,
        num_layers=args.num_layers,
        patch_size=args.patch_size,
        dropout=args.dropout,
        null_chance=args.null_chance
    ).to(device=device, dtype=ptdtype)
    
    autoencoder.load_state_dict(ckpt["autoencoder"])
    decoder = autoencoder.decoder
    decoder.eval().requires_grad_(False)

    H_grid = args.image_size // args.patch_size
    W_grid = args.image_size // args.patch_size

    splits_to_eval = [(i, name) for i, name in enumerate(split_names)] if split_ids is not None else [(None, "all")]
    metrics_log_book = {}

    for split_id, split_name in splits_to_eval:
        logger.info(f"\n==== Evaluating Data Split: {split_name} ====")

        if split_id is not None:
            idx = get_split_indices(split_ids, split_id)
            z_split, y_split = z[idx], y[idx] if y is not None else None
        else:
            z_split, y_split = z, y

        split_dir = osp.join(args.output_dir, f"decoded_{split_name}")
        if args.save_images and osp.exists(split_dir):
            shutil.rmtree(split_dir)

        overall_split_ds = InMemoryImageDataset()

        if args.eval_per_class and y_split is not None:
            unique_classes = torch.unique(y_split)
            for cls_id in unique_classes:
                cls_id = cls_id.item()
                cls_idx = (y_split == cls_id).nonzero(as_tuple=True)[0]

                z_cls, y_cls = z_split[cls_idx], y_split[cls_idx]
                cls_dir = osp.join(split_dir, f"class_{cls_id}")
                
                cls_ds = decode_and_collect(decoder, z_cls, y_cls, cls_dir, args, ptdtype, device, H_grid, W_grid)
                overall_split_ds.extend(cls_ds)

                cache_name = None
                if real_reference_dataset is not None:
                    input2_ref = ClassSubsetDataset(real_reference_dataset, target_class=cls_id)
                    cache_name = f"{args.dataset_name}_train_class_{cls_id}"
                else:
                    input2_ref = None

                cls_metric_key = f"{split_name}_class_{cls_id}"
                metric_input = cls_dir if args.save_images else cls_ds

                metrics_log_book[cls_metric_key] = run_metrics(
                    metric_input, args, cls_metric_key, input2_ref, cache_input2_name=cache_name
                )
        else:
            overall_split_ds = decode_and_collect(decoder, z_split, y_split, split_dir, args, ptdtype, device, H_grid, W_grid)

        overall_input2 = "cifar10-train" if args.dataset_name == "cifar-10" else None
        metric_input_overall = split_dir if args.save_images else overall_split_ds
        
        metrics_log_book[split_name] = run_metrics(metric_input_overall, args, split_name, overall_input2)

    print("\n" + "="*50 + "\n COMPREHENSIVE GENERATION METRICS REPORT\n" + "="*50)
    for eval_key, split_metrics in metrics_log_book.items():
        print(f"\n Run Target Category: [{eval_key.upper()}]")
        print("-" * 40)
        for metric_name, val in split_metrics.items():
            print(f" {metric_name:<30} : {val:.6f}")
    print("\n" + "="*50)


if __name__ == "__main__":
    main(cli_args)
