import gc
import numpy as np
import torch
from manifm.manifolds import Sphere
from configs.config import RAW_DATA_PATH, PROC_DATA_PATH, SPHERE_DIMS, SQUEEZE_DATA, SQUEEZE_ALPHA

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_PATH = RAW_DATA_PATH
OUTPUT_PATH = PROC_DATA_PATH

STD_DEVS = 2.0

manifold = Sphere()


def normalize(z):
    N, T, D = z.shape
    assert (T, D) == tuple(SPHERE_DIMS), f"Expected shape [N, {SPHERE_DIMS[0]}, {SPHERE_DIMS[1]}], got {z.shape}"
    z = z.reshape(N, T * D)
    z = z / z.norm(dim=-1, keepdim=True)
    return z


@torch.no_grad()
def manifold_squeeze(z, labels, class_means=None, alpha=0.2, reverse=False):
    """
    Squeezes or expands points along the geodesic toward/away from class centroids.
    alpha: 0.0 to 1.0.
           0.2 means points move 20% closer to the mean.
    reverse: If True, moves points AWAY from the mean to restore variance.
    """
    z_out = z.clone()
    unique_labels = torch.unique(labels)

    if class_means is None:
        class_means = {label.item(): None for label in unique_labels}

    scale = (1.0 - alpha)
    if reverse:
        scale = 1.0 / scale

    for label in unique_labels:
        mask = (labels == label)
        z_class = z[mask]
        if z_class.shape[0] == 0:
            continue

        if class_means[label.item()] is not None:
            centroid = class_means[label.item()]
        else:
            centroid = z_class.mean(dim=0, keepdim=True)
            centroid = centroid / centroid.norm()  # Project to unit sphere
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

        m = distances.mean()
        s = distances.std()
        threshold = m + (std_devs * s)

        mask = torch.zeros_like(class_indices, dtype=bool)
        mask[class_indices == 1] = distances <= threshold

        final_selection |= mask

        print(f"Label {label.item()}: Mean Dist {m:.4f}, Threshold {threshold:.4f}. "
              f"Keeping {len(mask.nonzero())}/{len(z_class)}")

    return z[final_selection], labels[final_selection]


def process_splits(z_input, labels, split_ids, split_names):
    splits = {split_name: None for split_name in split_names}
    for split_id, split_name in zip(split_ids.unique(), split_names):
        mask = split_ids == split_id
        splits[split_name] = {"encodings": z_input[mask], "labels": labels[mask], "split_ids": split_ids[mask]}

    class_means = None
    train = splits.pop("train")
    if SQUEEZE_DATA:
        z_train, train_labels = remove_manifold_outliers(train["encodings"], train["labels"], std_devs=STD_DEVS)
        z_train, class_means = manifold_squeeze(z_train, train_labels, alpha=SQUEEZE_ALPHA, reverse=False)
        for split_name, split in splits.items():
            if split_name == "train":
                continue

            z_split, _ = manifold_squeeze(split["encodings"], split["labels"], alpha=SQUEEZE_ALPHA, reverse=False)
            splits[split_name]["encodings"] = z_split
    else:
        z_train, train_labels = train["encodings"], train["labels"]

    z_all = torch.cat([z_train] + [splits[split_name]["encodings"] for split_name in splits], dim=0)
    labels_all = torch.cat([train_labels] + [splits[split_name]["labels"] for split_name in splits], dim=0)
    split_ids_all = torch.cat([torch.zeros_like(train_labels)] + [splits[split_name]["split_ids"] for split_name in splits], dim=0)
    return z_all, labels_all, split_ids_all, class_means


def main():
    data = np.load(INPUT_PATH, allow_pickle=False)

    z_input = torch.from_numpy(data["encodings"]).float().to(DEVICE)
    labels = torch.from_numpy(data["labels"]).long().to(DEVICE)
    split_ids = torch.from_numpy(data["split_ids"]).long().to(DEVICE)
    split_names = data["split_names"].tolist()

    print("Processing...")

    z_all, labels_all, split_ids_all, class_means = process_splits(z_input, labels, split_ids, split_names)
    del z_input, labels, split_ids
    torch.cuda.empty_cache()
    gc.collect()

    print("Saving...")

    output = dict(
        encodings=z_all.cpu().numpy(),
        labels=labels_all.cpu().numpy(),
        split_ids=split_ids_all.cpu().numpy(),
        split_names=np.array(split_names, dtype=str),
    )

    if class_means is not None:
        output["class_means"] = np.array([mean.cpu().numpy() for mean in class_means.values()])

    np.savez_compressed(
        OUTPUT_PATH,
        allow_pickle=False,
        **output,
    )
    print("Processed dataset saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
