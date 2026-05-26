import tqdm
import torch
import numpy as np
import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from manifm.model_pl import ManifoldFMLitModule
from preprocess_data import manifold_squeeze

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def dummy_labels(num_classes, n_samples=50000):
    labels = []
    for i in range(num_classes):
        class_i = torch.full((n_samples // num_classes,), i, dtype=torch.long)
        labels.append(class_i)
    return torch.cat(labels, dim=0)


def get_v(model, t_val, x, y=None, chunk_size=8192):
    """Chunked vector field evaluation to avoid CUDA kernel limits on large batches."""
    outputs = []
    for x_chunk, y_chunk in zip(x.split(chunk_size), y.split(chunk_size)):
        t_chunk = torch.full((x_chunk.shape[0], 1), t_val, device=DEVICE)
        outputs.append(model.vecfield(t_chunk, x_chunk, y=y_chunk))
    return torch.cat(outputs, dim=0)


@torch.inference_mode()
def integrate_flow(model, manifold, z_start, labels, steps=100, start_t=0.0, guidance_scale=1.5,
                   runtime_stats=False, null_label=-1, sde_noise=0.01):
    """Euler integrator for the manifold flow with SDE noise and old CFG schedule."""
    z_start = z_start.to(DEVICE)
    labels = labels.to(DEVICE)
    null_labels = torch.full_like(labels, null_label)

    z_prev = z_start.clone().to(DEVICE)
    dt = (1.0 - start_t) / steps

    for i in tqdm.tqdm(range(steps), desc="Integrating flow"):
        current_t = start_t + dt * i
        progress = i / steps

        init_cfg = 5.0      
        target_cfg = 1.1    
        current_cfg = target_cfg + (init_cfg - target_cfg) * (1.0 - current_t)**2

        if guidance_scale != 1.0 and current_cfg > 1.0:
            z_in = torch.cat([z_prev, z_prev], dim=0)
            y_in = torch.cat([null_labels, labels], dim=0)
            v_all = get_v(model, current_t, z_in, y=y_in)
            v_uncond, v_cond = v_all.chunk(2, dim=0)
            v = v_uncond + guidance_scale * (v_cond - v_uncond)
        else:
            v = get_v(model, current_t, z_prev, y=labels)

        u = dt * v
        z_next = manifold.expmap(z_prev, u)
                
        if i < (steps - 1):
            noise = torch.randn_like(z_next) * sde_noise * torch.sqrt(torch.tensor(dt))
            noise = manifold.proju(z_next, noise)
            z_next = manifold.expmap(z_next, noise)

        z_next = manifold.projx(z_next)

        if runtime_stats:
            measure_manifold_distance(manifold, z_next, z_prev, "Step Distance")

        z_prev = z_next

    measure_manifold_distance(manifold, z_prev, z_start, "Start vs End")
    return z_prev


def improve_encodings(
    cfg: DictConfig,
    model,
    manifold,
    input_path: Path,
):
    """Loads existing .npz file and pushes encodings through the flow."""
    s = cfg.sampling
    data_cfg = cfg.data

    print(f"Loading existing encodings from {input_path}")
    data = np.load(input_path, allow_pickle=False)
    z_input = torch.from_numpy(data["encodings"]).float()
    labels_input = torch.from_numpy(data["labels"]).long()
    split_ids = torch.from_numpy(data["split_ids"]).long()
    split_names = data["split_names"].tolist()
    class_means = data.get("class_means", None)

    if class_means is not None:
        class_means = torch.from_numpy(class_means).squeeze(1).float()
        print("Loaded class means.")

    if not s.generation and s.n_samples != -1 and z_input.shape[0] > s.n_samples:
        permutation = torch.randperm(z_input.shape[0])
        subsample_indices = permutation[: s.n_samples]
        z_input = z_input[subsample_indices]
        labels_input = labels_input[subsample_indices]
        split_ids = split_ids[subsample_indices]

    if s.check_uniformity:
        for label in labels_input.unique():
            mask = labels_input == label
            z_label = z_input[mask]
            check_hypersphere_uniformity(z_label, text=f"Label {label}")
            # z_label = normalize(z_label, data_cfg.sphere_dims)
            z_label = manifold.projx(z_label)
            check_uniformity(z_label)
        check_hypersphere_uniformity(z_input)

    z_input = manifold.projx(z_input)

    if data_cfg.squeeze_data:
        z_input, _ = manifold_squeeze(
            z_input, labels_input, class_means=class_means, alpha=data_cfg.squeeze_alpha, reverse=False
        )

    if s.check_uniformity:
        check_uniformity(z_input)

    if s.generation:
        labels = dummy_labels(s.num_classes, s.n_samples)
    else:
        labels = labels_input

    z_init = manifold.random_base(
        s.n_samples if s.generation else len(z_input),
        z_input.shape[-1],
    )
    z_init = z_init / z_init.norm(dim=-1, keepdim=True)
    z_init = manifold.projx(z_init)

    if s.generation:
        z_noise = z_init
    else:
        noise = torch.randn_like(z_input) * s.noise_std
        z_noise = z_input + noise
        z_noise = manifold.projx(z_noise)

    if s.direct_sampling:
        print("Using direct sampling without flow integration...")
        z_final = z_noise
    else:
        print("Integrating flow to improve encodings...")
        z_final = integrate_flow(
            model,
            manifold,
            z_noise,
            labels,
            steps=s.steps,
            start_t=s.start_t,
            guidance_scale=s.guidance_scale,
            runtime_stats=s.runtime_stats,
            sde_noise=getattr(s, "sde_noise", 0.01),
        )

    if data_cfg.squeeze_data:
        z_final, _ = manifold_squeeze(
            z_final, labels, class_means=class_means, alpha=data_cfg.squeeze_alpha, reverse=True
        )

    return z_final, labels, split_ids, split_names


@torch.no_grad()
def measure_manifold_distance(manifold, z_noisy, z_original, text="Manifold Distance"):
    z_noisy = z_noisy.to(DEVICE)
    z_original = z_original.to(DEVICE)
    distances = manifold.dist(z_noisy, z_original)
    print(
        f"{text} - Avg: {distances.mean().item():.4f}, "
        f"Std: {distances.std().item():.4f}, "
        f"Max: {distances.max().item():.4f}, "
        f"Min: {distances.min().item():.4f}"
    )


@torch.no_grad()
def check_uniformity(z):
    z = z.float()
    N, D = z.shape
    mean_vec_norm = z.mean(dim=0).norm().item()
    _, S, _ = torch.svd(z)
    S_normalized = S / S.max()
    entropy = -torch.sum(S_normalized * torch.log(S_normalized + 1e-8)).item()
    idx1 = torch.randint(0, N, (1000,))
    idx2 = torch.randint(0, N, (1000,))
    avg_cos = torch.nn.functional.cosine_similarity(z[idx1], z[idx2]).abs().mean().item()
    print(f"--- Uniformity Report ---")
    print(f"Mean Vector Norm: {mean_vec_norm:.4f} (Target: 0.0)")
    print(f"SVD Entropy:      {entropy:.4f} (Higher = More Uniform)")
    print(f"Avg Abs CosSim:   {avg_cos:.4f} (Target for high-D uniform: ~0.0)")


@torch.no_grad()
def check_hypersphere_uniformity(z, text="Raw Encodings"):
    z = z.float().to(DEVICE)
    N, T, D = z.shape
    total_dim = T * D
    z_flat = z.reshape(N, total_dim)
    norms = torch.norm(z_flat, p=2, dim=-1)
    avg_norm = norms.mean().item()
    expected_norm = total_dim ** 0.5
    mean_vec_norm = z_flat.mean(dim=0).norm().item()
    _, S, _ = torch.svd(z_flat)
    S_norm = S / S.max()
    entropy = -torch.sum(S_norm * torch.log(S_norm + 1e-8)).item()
    idx1 = torch.randint(0, N, (1000,))
    idx2 = torch.randint(0, N, (1000,))
    cos_sim = torch.nn.functional.cosine_similarity(z_flat[idx1], z_flat[idx2])
    avg_abs_cos = cos_sim.abs().mean().item()
    target_cos = (1 / total_dim) ** 0.5
    print(f"--- {text} Report ---")
    print(f"Shape:             {N} samples x {total_dim} dims")
    print(f"Avg L2 Norm:       {avg_norm:.4f} (Paper Expected: {expected_norm:.4f})")
    print(f"Mean Vector Norm:  {mean_vec_norm:.4f} (Closer to 0 is more uniform)")
    print(f"SVD Entropy:       {entropy:.4f}")
    print(f"Avg Abs CosSim:    {avg_abs_cos:.4f} (Closer to 0 is more uniform)")
    print(f"Theoretical Random CosSim: ~{target_cos:.4f}")


@torch.no_grad()
def check_class_clustering(manifold, z, labels, text="Class Clustering"):
    z = z.to(DEVICE)
    labels = labels.to(DEVICE)
    unique_labels = torch.unique(labels)
    print(f"\n=== {text} Report ===")
    print(f"{'Label':<10} | {'Mean Norm':<10} | {'Avg Dist':<10} | {'Std Dist':<10} | {'Max Dist':<10}")
    print("-" * 65)
    all_stats = []
    for label in unique_labels:
        mask = labels == label
        z_class = z[mask]
        if z_class.shape[0] == 0:
            continue
        mean_vec = z_class.mean(dim=0, keepdim=True)
        centroid = manifold.projx(mean_vec)
        mean_norm = mean_vec.norm().item()
        centroid_batch = centroid.expand(z_class.shape[0], -1)
        distances = manifold.dist(z_class, centroid_batch)
        avg_d = distances.mean().item()
        std_d = distances.std().item()
        max_d = distances.max().item()
        print(f"{label.item():<10} | {mean_norm:<10.4f} | {avg_d:<10.4f} | {std_d:<10.4f} | {max_d:<10.4f}")
        all_stats.append({"label": label.item(), "mean_norm": mean_norm, "avg_dist": avg_d})
    avg_class_tightness = sum(s["avg_dist"] for s in all_stats) / len(all_stats)
    print("-" * 65)
    print(f"Global Average Class Tightness (Distance to Mean): {avg_class_tightness:.4f}")
    return all_stats


def resolve_run_dir(cfg: DictConfig) -> Path:
    """Resolve the checkpoint run directory from sampling config."""
    pc = cfg.sampling.path_control
    runs_root = Path(pc.runs_root)
    if pc.use_latest:
        candidates = sorted(
            (p for p in runs_root.glob("*/*") if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(f"No run directories found under {runs_root}")
        run_dir = candidates[0]
        print(f"Using latest run: {run_dir}")
    else:
        run_dir = runs_root / pc.manual_path
        print(f"Using manual run: {run_dir}")
    return run_dir


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_dir = resolve_run_dir(cfg)
    ckpt_path = run_dir / "checkpoints" / "last.ckpt"
    run_cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")

    model = ManifoldFMLitModule.load_from_checkpoint(str(ckpt_path), cfg=run_cfg).to(DEVICE)
    model.eval()
    model.compile()

    manifold = model.manifold
    input_path = Path(cfg.data.proc_data_path)

    s = cfg.sampling
    if s.generation:
        z_final, labels, *_ = improve_encodings(cfg, model, manifold, input_path)
        split_ids = torch.zeros(s.n_samples, dtype=torch.long)
        split_names = ["generated"]
    else:
        z_final, labels, split_ids, split_names = improve_encodings(cfg, model, manifold, input_path)

    z_final = z_final.cpu()

    if s.save_output:
        T, D = cfg.data.sphere_dims
        
        z_final_to_save = z_final.view(z_final.shape[0], T, D)
        
        output_path = Path(cfg.data.data_dir) / "output_encodings.npz"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            allow_pickle=False,
            encodings=z_final_to_save.numpy(), # Now guaranteed to be [B, T, D]
            labels=labels.cpu().numpy(),
            split_ids=split_ids.cpu().numpy(),
            split_names=np.array(split_names, dtype=str),
        )
        print(f"Done. Saved shape: {z_final_to_save.shape} -> {Path(output_path).resolve()}")

if __name__ == "__main__":
    main()
