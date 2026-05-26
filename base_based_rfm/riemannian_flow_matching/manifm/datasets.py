"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import os
from csv import reader
import random
import numpy as np
import pandas as pd
import igl
import torch
from torch.utils.data import Dataset, DataLoader

from manifm.manifolds import Sphere, FlatTorus, Mesh, SPD, PoincareBall, Euclidean
from manifm.manifolds.mesh import Metric
from manifm.utils import cartesian_from_latlon
from manifm.manifolds.poincare import PoincareBallManifold


def load_csv(filename):
    file = open(filename, "r")
    lines = reader(file)
    dataset = np.array(list(lines)[1:]).astype(np.float64)
    return dataset


class SphereEncodingDataset(Dataset):
    def __init__(self, data_path):
        data = np.load(data_path, allow_pickle=False)
        self.data = torch.from_numpy(data["encodings"]).float()
        self.labels = torch.from_numpy(data["labels"]).long()

        split_ids = torch.from_numpy(data["split_ids"]).long()

        train_mask = split_ids == 0
        self.data = self.data[train_mask].clone()
        self.labels = self.labels[train_mask].clone()

        self.manifold = Sphere()
        self.dim = self.data.shape[1]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x1 = self.data[idx]
        y = self.labels[idx]

        x0 = self.manifold.random_base(1, self.dim).squeeze(0)

        return {"x0": x0, "x1": x1, "y": y}


class EarthData(Dataset):
    manifold = Sphere()
    dim = 3

    def __init__(self, dirname, filename):
        filename = os.path.join(dirname, filename)
        dataset = load_csv(filename)
        dataset = torch.Tensor(dataset)
        self.latlon = dataset
        self.data = cartesian_from_latlon(dataset / 180 * np.pi)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Volcano(EarthData):
    def __init__(self, dirname):
        super().__init__(dirname, "volcano.csv")


class Earthquake(EarthData):
    def __init__(self, dirname):
        super().__init__(dirname, "earthquake.csv")


class Fire(EarthData):
    def __init__(self, dirname):
        super().__init__(dirname, "fire.csv")


class Flood(EarthData):
    def __init__(self, dirname):
        super().__init__(dirname, "flood.csv")


class Top500(Dataset):
    manifold = FlatTorus()
    dim = 2

    def __init__(self, root="data/top500", amino="General"):
        data = pd.read_csv(
            f"{root}/aggregated_angles.tsv",
            delimiter="\t",
            names=["source", "phi", "psi", "amino"],
        )

        amino_types = ["General", "Glycine", "Proline", "Pre-Pro"]
        assert amino in amino_types, f"amino type {amino} not implemented"

        data = data[data["amino"] == amino][["phi", "psi"]].values.astype("float32")
        self.data = torch.tensor(data % 360 * np.pi / 180)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class RNA(Dataset):
    manifold = FlatTorus()
    dim = 7

    def __init__(self, root="data/rna"):
        data = pd.read_csv(
            f"{root}/aggregated_angles.tsv",
            delimiter="\t",
            names=[
                "source",
                "base",
                "alpha",
                "beta",
                "gamma",
                "delta",
                "epsilon",
                "zeta",
                "chi",
            ],
        )

        data = data[
            ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi"]
        ].values.astype("float32")
        self.data = torch.tensor(data % 360 * np.pi / 180)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class MeshDataset(Dataset):
    dim = 3

    def __init__(self, root: str, data_file: str, obj_file: str, scale=1 / 250):
        with open(os.path.join(root, data_file), "rb") as f:
            data = np.load(f)

        v, f = igl.read_triangle_mesh(os.path.join(root, obj_file))

        self.v = torch.tensor(v).float() * scale
        self.f = torch.tensor(f).long()
        self.data = torch.tensor(data).float() * scale

    def manifold(self, *args, **kwargs):
        return Mesh(self.v, self.f, *args, **kwargs)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class SimpleBunny(MeshDataset):
    def __init__(self, root="data/mesh"):
        super().__init__(
            root=root,
            data_file="bunny_simple.npy",
            obj_file="bunny_simp.obj",
            scale=1 / 250,
        )


class Bunny10(MeshDataset):
    def __init__(self, root="data/mesh"):
        super().__init__(
            root=root,
            data_file="bunny_eigfn009.npy",
            obj_file="bunny_simp.obj",
            scale=1 / 250,
        )


class Bunny50(MeshDataset):
    def __init__(self, root="data/mesh"):
        super().__init__(
            root=root,
            data_file="bunny_eigfn049.npy",
            obj_file="bunny_simp.obj",
            scale=1 / 250,
        )


class Bunny100(MeshDataset):
    def __init__(self, root="data/mesh"):
        super().__init__(
            root=root,
            data_file="bunny_eigfn099.npy",
            obj_file="bunny_simp.obj",
            scale=1 / 250,
        )


class Spot10(MeshDataset):
    def __init__(self, root="data/mesh"):
        super().__init__(
            root=root,
            data_file="spot_eigfn009.npy",
            obj_file="spot_simp.obj",
            scale=1.0,
        )


class Spot50(MeshDataset):
    def __init__(self, root="data/mesh"):
        super().__init__(
            root=root,
            data_file="spot_eigfn049.npy",
            obj_file="spot_simp.obj",
            scale=1.0,
        )


class Spot100(MeshDataset):
    def __init__(self, root="data/mesh"):
        super().__init__(
            root=root,
            data_file="spot_eigfn099.npy",
            obj_file="spot_simp.obj",
            scale=1.0,
        )


class SphericalDatasetPair(Dataset):
    def __init__(self, dataset, n_samples=20000):
        self.dataset = dataset
        self.n_samples = n_samples

        self.manifold = dataset.manifold
        self.dim = dataset.dim

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x1 = self.dataset[idx % len(self.dataset)]

        x0 = self.manifold.random_uniform(1, self.dim).squeeze(0)

        return {"x0": x0, "x1": x1}
    
class HyperbolicDatasetPair(Dataset):
    manifold = PoincareBall()
    dim = 2

    def __init__(self, distance=0.6, std=0.7):
        self.distance = distance
        self.std = std

    def __len__(self):
        return 20000

    def __getitem__(self, idx):
        mean0 = torch.tensor([0.0,0.0])
        mean1 = torch.tensor([-self.distance, -self.distance])

        x0 = PoincareBall().wrapped_normal(2, mean=mean0, std=100.0)
        x1 = PoincareBall().wrapped_normal(2, mean=mean1, std=self.std)
        

        return {"x0": x0, "x1": x1}
    


#Old hyperbolic images class, before it was matching from uniform random noise to images
class HyperbolicImages(Dataset):
    dim = 512

    """
    Dataset for real hyperbolic embeddings on a Poincaré ball of curvature -c.
    """

    def __init__(self, emb_path, label_path=None, pair_mode="self"):
        """
        Args:
            emb_path: path to saved embeddings (torch.save)
            label_path: optional label file
            pair_mode:
                "self" → x0 is tangent noise, x1 is embedding
                "paired" → sample two different classes
                "none" → return only x1 
        """
        self.emb = torch.tensor(torch.load(emb_path)).float()

        if self.emb.ndim == 1:
            self.emb = self.emb.unsqueeze(0)
        elif self.emb.ndim > 2:
            self.emb = self.emb.reshape(self.emb.shape[0], -1)

        self.labels = None
        if label_path is not None:
            self.labels = torch.tensor(torch.load(label_path))

        self.manifold = PoincareBall()
        self.dim = self.emb.shape[1]
        self.pair_mode = pair_mode

    def __len__(self):
        return len(self.emb)
    
    
    def __getitem__(self, idx, dim=512):

        x1 = self.emb[idx].reshape(-1)

        if self.pair_mode == "none":
            return {"x1": x1}

        if self.pair_mode == "self":
            x0 = self.manifold.wrapped_normal(self.dim, mean=torch.zeros(self.dim), std=0.3)
            return {"x0": x0, "x1": x1}

        if self.pair_mode == "paired":
            j = torch.randint(0, len(self.emb), (1,)).item()
            x0 = self.emb[j]
            return {"x0": x0, "x1": x1}


class EuclideanImages(Dataset):
    dim = 9216 #512x18

    """
    Dataset for real Euclidean embeddings.
    """

    def __init__(self, emb_path, label_path=None):
        """
        Args:
            emb_path: path to saved embeddings (torch.save)
            label_path: optional label file
            pair_mode:
                "self" → x0 is tangent noise, x1 is embedding
                "paired" → sample two different classes
                "none" → return only x1 
        """
        self.emb = torch.tensor(torch.load(emb_path)).float()

        if self.emb.ndim == 1:
            self.emb = self.emb.unsqueeze(0)

        self.labels = None
        if label_path is not None:
            self.labels = torch.tensor(torch.load(label_path))

        self.manifold = Euclidean()
        self.dim = self.emb.shape[1]

    def __len__(self):
        return len(self.emb)
    
    
    def __getitem__(self, idx, dim=512):

        x1 = self.emb[idx]
        x0 = self.manifold.random_normal(self.dim, mean=torch.zeros(self.dim), std=1.0)
        return {"x0": x0, "x1": x1}
            


class HyperbolicUniformToGaussian(Dataset):
    """
    Synthetic dataset for learning a flow from
    Uniform(Poincaré Ball) → Wrapped Gaussian(Poincaré Ball)
    """
    
    def __init__(self, dim=2, mean=None, std=0.3, n_samples=20000):
        #super().__init__()
        self.dim = dim
        self.n_samples = n_samples
        self.std = std
        if mean is None:
            mean = torch.zeros(dim)
        self.mean = mean.float()
        self._manifold = PoincareBall()
    @property
    def manifold(self):
        return self._manifold

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):

        ### 1. Sample x0 ~ Uniform on ball - using VRFM implementation
        x0 = self._manifold.random_base(batch_size=1, dim=self.dim).squeeze(0)
        device = x0.device     
        mean = self.mean.to(device)
        ### 2. Sample x1 ~ Wrapped Normal
        x1 = self._manifold.wrapped_normal(self.dim, mean=mean, std=self.std)
        return {"x0": x0, "x1": x1}


class MeshDatasetPair(Dataset):
    dim = 3

    def __init__(self, root: str, data_file: str, obj_file: str, scale: float):
        data = np.load(os.path.join(root, data_file), "rb")
        x0 = data["x0"]
        x1 = data["x1"]

        self.Z0 = float(data["Z0"])
        self.Z1 = float(data["Z1"])

        if "std" in data:
            self.std = float(data["std"])
        else:
            # previous default
            self.std = 1 / 9.5

        v, f = igl.read_triangle_mesh(os.path.join(root, obj_file))

        self.v = torch.tensor(v).float() * scale
        self.f = torch.tensor(f).long()

        self.x0 = torch.tensor(x0).float() * scale
        self.x1 = torch.tensor(x1).float() * scale

    def manifold(self, *args, **kwargs):
        def base_logprob(x):
            x = (x[..., :2] - 0.5) / self.std
            logZ = -0.5 * np.log(2 * np.pi)
            logprob = logZ - x.pow(2) / 2
            logprob = logprob - np.log(self.std)
            return logprob.sum(-1) - np.log(self.Z0)

        mesh = Mesh(self.v, self.f, *args, **kwargs)
        mesh.base_logprob = base_logprob
        return mesh

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, idx):
        idx0 = int(len(self.x0) * random.random())
        return {"x0": self.x0[idx0], "x1": self.x1[idx]}


class Maze3v2(MeshDatasetPair):
    def __init__(self, root="data/mesh"):
        super().__init__(
            root=root, data_file="maze_3x3v2.npz", obj_file="maze_3x3.obj", scale=1 / 3
        )


class Maze4v2(MeshDatasetPair):
    def __init__(self, root="data/mesh"):
        super().__init__(
            root=root, data_file="maze_4x4v2.npz", obj_file="maze_4x4.obj", scale=1 / 4
        )


class Wrapped(Dataset):
    def __init__(
        self,
        manifold,
        dim,
        n_mixtures=1,
        scale=0.2,
        centers=None,
        dataset_size=200000,
    ):
        self.manifold = manifold
        self.dim = dim
        self.n_mixtures = n_mixtures
        if centers is None:
            self.centers = self.manifold.random_uniform(n_mixtures, dim)
        else:
            self.centers = centers
        self.scale = scale
        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        del idx

        idx = torch.randint(self.n_mixtures, (1,)).to(self.centers.device)
        mean = self.centers[idx].squeeze(0)

        tangent_vec = torch.randn(self.dim).to(self.centers)
        tangent_vec = self.manifold.proju(mean, tangent_vec)
        tangent_vec = self.scale * tangent_vec
        sample = self.manifold.expmap(mean, tangent_vec)
        return sample


class ExpandDataset(Dataset):
    def __init__(self, dset, expand_factor=1):
        self.dset = dset
        self.expand_factor = expand_factor

    def __len__(self):
        return len(self.dset) * self.expand_factor

    def __getitem__(self, idx):
        return self.dset[idx % len(self.dset)]


def _get_dataset(cfg):
    expand_factor = 1
    # Extract the string type from the dictionary
    dataset_type = cfg.data.type 
    
    if dataset_type == "sphere_encodings":
        dataset = SphereEncodingDataset(cfg.data.proc_data_path) # Use the path from the dict
        expand_factor = 100
    else:
        raise ValueError(f"Unknown dataset option '{dataset_type}'")
    return dataset, expand_factor

def get_loaders(cfg):
    dataset, expand_factor = _get_dataset(cfg)

    N = len(dataset)
    N_val = N_test = N // 10
    N_train = N - N_val - N_test

    data_seed = cfg.seed if cfg.data.data_seed is None else cfg.data.data_seed
    if data_seed is None:
        raise ValueError("seed for data generation must be provided")
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset,
        [N_train, N_val, N_test],
        generator=torch.Generator().manual_seed(data_seed),
    )

    # Expand the training set (we optimize based on number of iterations anyway).
    train_set = ExpandDataset(train_set, expand_factor=expand_factor)

    train_loader = DataLoader(
        train_set, cfg.optim.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=15, prefetch_factor=2
    )
    val_loader = DataLoader(
        val_set, cfg.optim.val_batch_size, shuffle=False, pin_memory=True, num_workers=15, prefetch_factor=2
    )
    test_loader = DataLoader(
        test_set, cfg.optim.val_batch_size, shuffle=False, pin_memory=True, num_workers=15, prefetch_factor=2
    )

    return train_loader, val_loader, test_loader


def get_manifold(cfg):
    dataset, _ = _get_dataset(cfg)

    if isinstance(dataset, MeshDataset) or isinstance(dataset, MeshDatasetPair):
        manifold = dataset.manifold(
            numeigs=cfg.mesh.numeigs, metric=Metric(cfg.mesh.metric), temp=cfg.mesh.temp
        )
        return manifold, dataset.dim
    else:
        return dataset.manifold, dataset.dim
