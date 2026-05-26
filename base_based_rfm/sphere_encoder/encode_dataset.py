import os
import os.path as osp
import gc
import json
import argparse
import logging
import random
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm
from torchvision import datasets, transforms

from cli_utils import str2bool
from sphere.model import G
from sphere.ema import SimpleEMA
from sphere.loader import ListDataset
from sphere.utils import load_ckpt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -------------------------------------------------
# CLI
# -------------------------------------------------
parser = argparse.ArgumentParser(description="Encode dataset")

parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--output_path", type=str, default=None)
parser.add_argument("--output_name", type=str, default="encoded_dataset.npz")

parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_workers", type=int, default=8)

parser.add_argument("--dataset_name", type=str, default="cifar-10")
parser.add_argument("--split", type=str, default="all", choices=["train", "test", "all"])

parser.add_argument("--max_samples", type=int, default=-1)
parser.add_argument("--load_from_zip", type=str2bool, default=False)

# reproducibility
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--deterministic", type=str2bool, default=False)

# precision
parser.add_argument("--save_dtype", type=str, default="bfloat16",
                    choices=["float32", "bfloat16", "float16"])

# model behavior
parser.add_argument("--use_ema", type=str2bool, default=True)
parser.add_argument("--compile_model", type=str2bool, default=True)

cli_args = parser.parse_args()


# -------------------------------------------------
# SEEDING
# -------------------------------------------------
def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# -------------------------------------------------
# CHECKPOINT HELPER
# -------------------------------------------------
def find_checkpoint(path):
    if os.path.isfile(path):
        return path

    elif os.path.isdir(path):
        files = sorted([
            f for f in os.listdir(path)
            if f.endswith(".pth")
        ])

        if not files:
            raise FileNotFoundError(f"No .pth files in {path}")

        return os.path.join(path, files[-1])

    else:
        raise FileNotFoundError(path)


# -------------------------------------------------
# APPEND TO TEMP FILE
# -------------------------------------------------
def append_to_temp_file(tmp_path, enc, labels=None, splits=None):

    if os.path.exists(tmp_path):

        old = torch.load(tmp_path)

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


# -------------------------------------------------
# MAIN
# -------------------------------------------------
@torch.inference_mode()
def main(cli_args):

    exp_dir = osp.dirname(cli_args.checkpoint)
    cfg_path = osp.join(exp_dir, "cfg.json")

    logger.info(f"Loading config from {cfg_path}")

    with open(cfg_path, "r") as f:
        cfg_args = json.load(f)

    cfg_args.update(vars(cli_args))
    args = SimpleNamespace(**cfg_args)

    set_seed(args.seed, args.deterministic)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }.get(
        getattr(args, "dtype", "bfloat16"),
        torch.bfloat16
    )

    target_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.save_dtype]

    # -------------------------------------------------
    # OUTPUT
    # -------------------------------------------------
    output_dir = args.output_path or args.data_path

    os.makedirs(output_dir, exist_ok=True)

    output_file = osp.join(
        output_dir,
        args.output_name
    )

    tmp_pt = output_file.replace(
        ".npz",
        "_stream.pt"
    )

    # remove stale temp file
    if os.path.exists(tmp_pt):
        os.remove(tmp_pt)

    # -------------------------------------------------
    # MODEL
    # -------------------------------------------------
    model = G(
        input_size=args.image_size,
        patch_size=args.patch_size,
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
    ).to(
        device=device,
        dtype=ptdtype,
        memory_format=torch.channels_last
    )

    ema_model = SimpleEMA(model)

    ckpt_path = find_checkpoint(args.checkpoint)

    logger.info(f"Loading checkpoint: {ckpt_path}")

    load_ckpt(
        model,
        ckpt_path,
        ema_model=ema_model,
        strict=True,
        override_model_with_ema=args.use_ema,
        verbose=True,
    )

    if args.compile_model:
        model.compile()

    model.eval().requires_grad_(False)

    # -------------------------------------------------
    # DATASET
    # -------------------------------------------------
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    is_cifar = args.dataset_name in [
        "cifar-10",
        "cifar-100"
    ]

    is_animal_faces = (
        args.dataset_name == "animal-faces"
    )

    if is_cifar:

        dataset_cls = datasets.__dict__[
            args.dataset_name.upper().replace("-", "")
        ]

    elif is_animal_faces:

        dataset_cls = datasets.ImageFolder

    else:

        dataset_cls = ListDataset

    # -------------------------------------------------
    # SPLITS
    # -------------------------------------------------
    splits = []

    if is_cifar:

        if args.split == "all":
            splits = [
                ("train", True),
                ("test", False)
            ]

        else:
            splits = [
                (args.split, args.split == "train")
            ]

    elif is_animal_faces:

        if args.split == "all":
            splits = [
                ("train", True),
                ("val", False)
            ]

        elif args.split == "train":
            splits = [("train", True)]

        else:
            splits = [("val", False)]

    else:

        splits = [("data", True)]

    # -------------------------------------------------
    # STREAMING ENCODE
    # -------------------------------------------------
    chunk_size = 50

    enc_list = []
    label_list = []
    split_list = []

    total_samples = 0

    for split_id, (split_name, train_flag) in enumerate(splits):

        logger.info(f"Encoding split: {split_name}")

        # -------------------------------------------------
        # DATASET
        # -------------------------------------------------
        if is_cifar:

            dataset = dataset_cls(
                root=args.data_path,
                train=train_flag,
                transform=transform,
                download=False,
            )

        elif is_animal_faces:

            dataset = dataset_cls(
                root=osp.join(
                    args.data_path,
                    split_name
                ),
                transform=transform,
            )

        else:

            dataset = dataset_cls(
                root=args.data_path,
                transform=transform,
                max_samples=args.max_samples,
                load_from_zip=args.load_from_zip,
            )

        logger.info(f"Dataset size: {len(dataset)}")

        # -------------------------------------------------
        # LOADER
        # -------------------------------------------------
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        # -------------------------------------------------
        # ENCODE
        # -------------------------------------------------
        for batch in tqdm(loader, desc=split_name):

            if isinstance(batch, (list, tuple)):

                x, y = batch

                if y is not None:
                    y = y.to(
                        device,
                        non_blocking=True
                    )

            else:

                x = batch
                y = None

            x = x.to(
                device,
                non_blocking=True
            )

            with torch.autocast(
                device_type="cuda",
                dtype=ptdtype
            ):

                z = model.encoder(x, y)
                z = model.spherify(
                    z,
                    sampling=False
                )

            z = (
                z.detach()
                .cpu()
                .to(target_dtype)
            )

            enc_list.append(z)

            batch_size_actual = z.shape[0]
            total_samples += batch_size_actual

            if y is not None:

                y_cpu = y.cpu()

                label_list.append(y_cpu)

                split_list.append(
                    torch.full(
                        (batch_size_actual,),
                        split_id,
                        dtype=torch.long
                    )
                )

            # -------------------------------------------------
            # FLUSH TO DISK
            # -------------------------------------------------
            if len(enc_list) >= chunk_size:

                enc_chunk = torch.cat(
                    enc_list,
                    dim=0
                )

                label_chunk = (
                    torch.cat(label_list, dim=0)
                    if label_list else None
                )

                split_chunk = (
                    torch.cat(split_list, dim=0)
                    if split_list else None
                )

                append_to_temp_file(
                    tmp_pt,
                    enc_chunk,
                    label_chunk,
                    split_chunk
                )

                logger.info(
                    f"Flushed {total_samples} samples"
                )

                enc_list = []
                label_list = []
                split_list = []

                gc.collect()
                torch.cuda.empty_cache()

    # -------------------------------------------------
    # SAVE REMAINING
    # -------------------------------------------------
    if len(enc_list) > 0:

        enc_chunk = torch.cat(
            enc_list,
            dim=0
        )

        label_chunk = (
            torch.cat(label_list, dim=0)
            if label_list else None
        )

        split_chunk = (
            torch.cat(split_list, dim=0)
            if split_list else None
        )

        append_to_temp_file(
            tmp_pt,
            enc_chunk,
            label_chunk,
            split_chunk
        )

    # -------------------------------------------------
    # FINALIZE
    # -------------------------------------------------
    logger.info("Loading final tensor file...")

    final_data = torch.load(tmp_pt)

    final_enc = final_data["enc"]
    final_labels = final_data["labels"]
    final_splits = final_data["splits"]

    logger.info(
        f"Final tensor shape: {final_enc.shape}"
    )

    # -------------------------------------------------
    # SAVE NPZ
    # -------------------------------------------------
    logger.info("Converting to NPZ...")

    np.savez_compressed(
        output_file,
        allow_pickle=False,
        encodings=final_enc.float().numpy(),
        labels=(
            final_labels.numpy()
            if final_labels is not None else None
        ),
        split_ids=(
            final_splits.numpy()
            if final_splits is not None else None
        ),
        split_names=np.array(
            [s[0] for s in splits],
            dtype=str
        ),
    )

    # -------------------------------------------------
    # CLEANUP
    # -------------------------------------------------
    if os.path.exists(tmp_pt):

        os.remove(tmp_pt)

        logger.info(
            f"Removed temp file: {tmp_pt}"
        )

    del model
    del ema_model

    gc.collect()
    torch.cuda.empty_cache()

    logger.info(f"Saved to {output_file}")
    logger.info(f"Shape: {final_enc.shape}")


if __name__ == "__main__":
    main(cli_args)