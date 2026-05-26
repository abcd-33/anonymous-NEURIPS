import torch

Z = []
labels = []
split_ids = torch.zeros(50000, dtype=torch.long)  # dummy split ids (not used)
split_names = ["generated"]
for i in range(10):
    z = torch.load(f"workspace/experiments/sphere-small-small-cifar-10-32px/encoding/output_class_{i}_encodings.pt")
    Z.append(z["encodings"])
    labels.append(torch.full((z["encodings"].shape[0],), i, dtype=torch.long))

Z_concat = torch.cat(Z, dim=0)
labels = torch.cat(labels, dim=0)

torch.save({
    "encodings": Z_concat,
    "labels": labels if labels is not None else None,
    "split_ids": split_ids,
    "split_names": split_names,
}, "workspace/experiments/sphere-small-small-cifar-10-32px/encoding/output_encodings.pt")