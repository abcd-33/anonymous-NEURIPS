import os
import os.path as osp
import gc
import json
import argparse
import logging
import random
import math
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import datasets, transforms

from models import SphericalAutoencoder
from train import ListDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Streaming Dataset Encoder to Spherical Manifold NPZ")

parser.add_argument("--data_path", type=str, default="workspace/datasets/cifar-10", help="Path to raw dataset storage location")
parser.add_argument("--checkpoint", type=str, default="workspace/experiments/custom/encoding/best_spherical_autoencoder.pt", help="Path to trained autoencoder checkpoint file")
parser.add_argument("--output_path", type=str, default="./workspace/experiments/custom/encoding/", help="Output directory path (defaults to workspace folder)")
parser.add_argument("--output_name", type=str, default="encoded_dataset.npz")

parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_workers", type=int, default=4)

parser.add_argument("--dataset_name", type=str, default="cifar-10")
parser.add_argument("--split", type=str, default="all", choices=["train", "test", "all"])

parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--deterministic", type=bool, default=False)
parser.add_argument("--save_dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"])

cli_args = parser.parse_args()


def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def append_to_temp_file(tmp_path, enc, labels=None, splits=None):
    """Streams data arrays continuously down to the temporary storage cache file on disk."""
    if os.path.exists(tmp_path):
        old = torch.load(tmp_path, weights_only=False)
        enc = torch.cat([old["enc"], enc], dim=0)
        if labels is not None:
            labels = torch.cat([old["labels"], labels], dim=0)
        if splits is not None:
            splits = torch.cat([old["splits"], splits], dim=0)

    torch.save(
        {
            "enc": enc,
            "labels": labels,
            "splits": splits,
        },
        tmp_path,
        _use_new_zipfile_serialization=True
    )


@torch.inference_mode()
def main(cli_args):
    logger.info(f"Loading checkpoint metadata from {cli_args.checkpoint}")
    ckpt = torch.load(cli_args.checkpoint, map_location="cpu")
    args_ae = ckpt["args"]

    cfg_args = vars(args_ae)
    cfg_args.update(vars(cli_args))
    args = SimpleNamespace(**cfg_args)

    set_seed(args.seed, args.deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.set_float32_matmul_precision('high')

    ptdtype = torch.bfloat16 if args.save_dtype == "bfloat16" else (torch.float16 if args.save_dtype == "float16" else torch.float32)
    target_dtype = ptdtype

    os.makedirs(args.output_path, exist_ok=True)
    output_file = osp.join(args.output_path, args.output_name)

    tmp_pt = output_file.replace(".npz", "_stream.pt")
    if os.path.exists(tmp_pt):
        os.remove(tmp_pt)

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
    encoder = autoencoder.encoder
    encoder.eval().requires_grad_(False)

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    is_cifar = args.dataset_name in ["cifar-10", "cifar-100"]
    is_animal_faces = (args.dataset_name == "animal-faces")

    if is_cifar:
        dataset_cls = datasets.CIFAR10 if args.dataset_name == "cifar-10" else datasets.CIFAR100
    elif is_animal_faces:
        dataset_cls = datasets.ImageFolder
    else:
        dataset_cls = ListDataset

    if is_cifar:
        if args.split == "all":
            splits = [("train", True), ("test", False)]
        else:
            splits = [(args.split, args.split == "train")]
    elif is_animal_faces:
        if args.split == "all":
            splits = [("train", True), ("val", False)]
        elif args.split == "train":
            splits = [("train", True)]
        else:
            splits = [("val", False)]
    else:
        splits = [("data", True)]

    chunk_size = 50  
    enc_list = []
    label_list = []
    split_list = []
    total_samples = 0

    for split_id, (split_name, train_flag) in enumerate(splits):
        logger.info(f"Targeting Matrix Conversion for Split: {split_name}")

        if is_cifar:
            dataset = dataset_cls(root=args.data_path, train=train_flag, transform=transform, download=True)
        elif is_animal_faces:
            dataset = dataset_cls(root=osp.join(args.data_path, split_name), transform=transform)
        else:
            image_paths, labels = [], []
            root_dir = osp.join(args.data_path, args.dataset_name)
            if os.path.exists(root_dir):
                for idx, c_dir in enumerate(sorted(os.listdir(root_dir))):
                    c_path = osp.join(root_dir, c_dir)
                    if osp.isdir(c_path):
                        for img in os.listdir(c_path):
                            if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                                image_paths.append(osp.join(c_path, img))
                                labels.append(idx)
            dataset = dataset_cls(image_paths, labels, transform)

        logger.info(f"Parsed {split_name} distribution profile dimension length: {len(dataset)}")

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        for batch in tqdm(loader, desc=split_name):
            if isinstance(batch, (list, tuple)):
                x, y = batch
                if y is not None:
                    y = y.to(device, non_blocking=True)
            else:
                x = batch
                y = torch.zeros(x.shape[0], dtype=torch.long, device=device) # Fallback if no target labels given

            x = x.to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=ptdtype):
                z, _, _ = encoder(x, class_ids=y, force_drop=False) 
                
                z = F.normalize(z.float(), p=2, dim=-1).to(ptdtype)

            z = z.detach().cpu().to(target_dtype)
            enc_list.append(z)

            batch_size_actual = z.shape[0]
            total_samples += batch_size_actual

            y_cpu = y.cpu()
            label_list.append(y_cpu)
            split_list.append(torch.full((batch_size_actual,), split_id, dtype=torch.long))

            if len(enc_list) >= chunk_size:
                enc_chunk = torch.cat(enc_list, dim=0)
                label_chunk = torch.cat(label_list, dim=0)
                split_chunk = torch.cat(split_list, dim=0)

                append_to_temp_file(tmp_pt, enc_chunk, label_chunk, split_chunk)
                logger.info(f"Safely flushed {total_samples} serialized tokens blocks to Workspace Disk Cache.")

                enc_list = []
                label_list = []
                split_list = []

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    if len(enc_list) > 0:
        enc_chunk = torch.cat(enc_list, dim=0)
        label_chunk = torch.cat(label_list, dim=0)
        split_chunk = torch.cat(split_list, dim=0)
        append_to_temp_file(tmp_pt, enc_chunk, label_chunk, split_chunk)

    logger.info("Assembling and pulling streaming tensor files back into Memory block...")
    final_data = torch.load(tmp_pt, weights_only=False)
    final_enc = final_data["enc"].float() # Force to float32 immediately to recover precision
    final_labels = final_data["labels"]
    final_splits = final_data["splits"]

    final_enc = F.normalize(final_enc, p=2, dim=-1)

    logger.info(f"Verifying final compressed geometry matrix layout dimensions: {final_enc.shape}")

    magnitudes = torch.norm(final_enc, p=2, dim=-1)
    assert torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=1e-5), \
        f"Unit Hypersphere norm validation failed! Min: {magnitudes.min().item():.6f}, Max: {magnitudes.max().item():.6f}"

    logger.info(f"Writing compressed NPZ archive mapping targets directly to: {output_file}")
    np.savez_compressed(
        output_file,
        allow_pickle=False,
        encodings=final_enc.numpy(), 
        labels=final_labels.numpy(),
        split_ids=final_splits.numpy(),
        split_names=np.array([s[0] for s in splits], dtype=str),
    )
    
    if os.path.exists(tmp_pt):
        os.remove(tmp_pt)
        logger.info("Temporary workspace streaming serialization array storage files deleted cleanly.")

    del autoencoder, encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Dataset sequence encoding workspace loop concluded successfully.")


if __name__ == "__main__":
    main(cli_args)
