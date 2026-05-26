import gc
import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from manifm.manifolds import Sphere
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
manifold = Sphere()

def normalize(z, sphere_dims):
    N, T, D = z.shape
    assert (T, D) == tuple(sphere_dims), f"Expected shape [N, {sphere_dims[0]}, {sphere_dims[1]}], got {z.shape}"
    z = z.reshape(N, T * D)
    z = z / z.norm(dim=-1, keepdim=True)
    return z

@torch.no_grad()
def manifold_squeeze(z, labels, class_means=None, alpha=0.2, reverse=False):
    z_out = z.clone()
    unique_labels = torch.unique(labels)
    if class_means is None: class_means = {label.item(): None for label in unique_labels}
    scale = (1.0 - alpha)
    if reverse: scale = 1.0 / scale

    for label in unique_labels:
        mask = (labels == label)
        z_class = z[mask]
        if z_class.shape[0] == 0: continue
        if class_means[label.item()] is not None:
            centroid = class_means[label.item()]
        else:
            centroid = z_class.mean(dim=0, keepdim=True)
            centroid = centroid / centroid.norm()
            class_means[label.item()] = centroid
        tangent_v = manifold.logmap(centroid.expand_as(z_class), z_class)
        scaled_v = tangent_v * scale
        z_out[mask] = manifold.expmap(centroid.expand_as(z_class), scaled_v)
    return manifold.projx(z_out), class_means

@torch.no_grad()
def remove_manifold_outliers(z, labels, std_devs=2.0):
    z = z.to(DEVICE)
    labels = labels.to(DEVICE)
    final_selection = torch.zeros_like(labels, dtype=bool)
    unique_labels = torch.unique(labels)
    for label in unique_labels:
        class_indices = (labels == label)
        z_class = z[class_indices]
        centroid = z_class.mean(dim=0, keepdim=True)
        centroid = centroid / centroid.norm(dim=-1, keepdim=True)
        distances = manifold.dist(z_class, centroid.expand_as(z_class))
        m, s = distances.mean(), distances.std()
        threshold = m + (std_devs * s)
        mask = torch.zeros_like(class_indices, dtype=bool)
        mask[class_indices == 1] = distances <= threshold
        final_selection |= mask
    return z[final_selection], labels[final_selection]

def process_splits(z_input, labels, split_ids, split_names, data_cfg):
    splits = {split_name: None for split_name in split_names}
    for split_id, split_name in zip(split_ids.unique(), split_names):
        mask = split_ids == split_id
        splits[split_name] = {"encodings": z_input[mask], "labels": labels[mask], "split_ids": split_ids[mask]}

    class_means = None
    train = splits.pop("train")
    if data_cfg.squeeze_data: 
        z_train, train_labels = remove_manifold_outliers(train["encodings"], train["labels"], std_devs=data_cfg.std_devs)
        z_train, class_means = manifold_squeeze(z_train, train_labels, alpha=data_cfg.squeeze_alpha, reverse=False)
        for split_name, split in splits.items():
            if split_name == "train": continue
            z_split, _ = manifold_squeeze(split["encodings"], split["labels"], alpha=data_cfg.squeeze_alpha, reverse=False)
            splits[split_name]["encodings"] = z_split
    else:
        z_train, train_labels = train["encodings"], train["labels"]

    z_all = torch.cat([z_train] + [splits[split_name]["encodings"] for split_name in splits], dim=0)
    labels_all = torch.cat([train_labels] + [splits[split_name]["labels"] for split_name in splits], dim=0)
    split_ids_all = torch.cat([torch.zeros_like(train_labels)] + [splits[split_name]["split_ids"] for split_name in splits], dim=0)
    return z_all, labels_all, split_ids_all, class_means


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    d = cfg.data
    raw_path = Path(d.raw_data_path).resolve()
    proc_path = Path(d.proc_data_path).resolve()


    data = np.load(raw_path, allow_pickle=False)
    z_input = torch.from_numpy(data["encodings"]).float().to(DEVICE)
    labels = torch.from_numpy(data["labels"]).long().to(DEVICE)
    split_ids = torch.from_numpy(data["split_ids"]).long().to(DEVICE)
    split_names = data["split_names"].tolist()

    print("Processing...")
    z_input = normalize(z_input, d.sphere_dims)
    z_all, labels_all, split_ids_all, class_means = process_splits(z_input, labels, split_ids, split_names, d)
    
    del z_input, labels, split_ids
    torch.cuda.empty_cache()
    gc.collect()

    output = {
        "encodings": z_all.cpu().numpy(),
        "labels": labels_all.cpu().numpy(),
        "split_ids": split_ids_all.cpu().numpy(),
        "split_names": np.array(split_names, dtype=str),
    }
    if class_means is not None:
        output["class_means"] = np.array([mean.cpu().numpy() for mean in class_means.values()])

    np.savez_compressed(proc_path, allow_pickle=False, **output)
    print("Processed dataset saved to:", proc_path)

if __name__ == "__main__":
    main()