import random
import tqdm
import torch
import numpy as np
from omegaconf import OmegaConf
from manifm.model_pl import ManifoldFMLitModule
from preprocess_data import manifold_squeeze
from configs.config import PROC_DATA_PATH, OUTPUT_DATA_PATH, SPHERE_DIMS, SQUEEZE_DATA, SQUEEZE_ALPHA

# --- Precision Configurations ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    torch.set_float32_matmul_precision("high")
# --------------------------------

CHECK_UNIFORMITY = False
RUNTIME_STATS = False

N_SAMPLES = 50000
NUM_CLASSES = 10
STEPS = 100
NOISE_STD = 0.0
START_T = 0.0
BATCH_SIZE = 5000  # Updated to reflect full sequence batches [B, S, D] instead of flat elements

GUIDANCE_SCALE = 1.5

SAVE_OUTPUT = True
GENERATION = True
INPUT_PATH = PROC_DATA_PATH
OUTPUT_PATH = OUTPUT_DATA_PATH

RUN_DIR = "outputs/runs/sphere_encodings/fm/2026.05.23/194308"

cfg = OmegaConf.load(f"{RUN_DIR}/.hydra/config.yaml")
ckpt_path = f"{RUN_DIR}/checkpoints/last.ckpt"

print(ckpt_path, "lanmlerjnsjkj")
model = ManifoldFMLitModule.load_from_checkpoint(ckpt_path, cfg=cfg).to(DEVICE)

model.eval()
model.compile()

manifold = model.manifold
dim = model.dim


def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def dummy_labels(n_samples=50000):
    labels = []
    for i in range(NUM_CLASSES):
        class_i = torch.full((n_samples // NUM_CLASSES,), i, dtype=torch.long)
        labels.append(class_i)
    return torch.cat(labels, dim=0)


def get_v(t_val, x, y=None):
    # Match the sequence batch size dimension of x [B, S, D]
    t = torch.full((x.shape[0],), t_val, device=DEVICE)
    if DEVICE == "cuda":
        t = t.to(dtype=torch.bfloat16)
    return model.vecfield(t, x, y=y)


@torch.inference_mode()
def integrate_flow(z_start, labels, steps=100, start_t=0.0, guidance_scale=GUIDANCE_SCALE, null_label=-1):
    """Generic Euler integrator for the unflattened manifold flow."""
    z_start = z_start.to(DEVICE)
    labels = labels.to(DEVICE)
    null_labels = torch.full_like(labels, null_label)

    z_prev = z_start.clone().to(DEVICE)
    dt = (1.0 - start_t) / steps

    for i in range(steps):
        current_t = start_t + dt * i
        init_cfg = 5.0      
        target_cfg = 1.1    

        current_cfg = target_cfg + (init_cfg - target_cfg) * (1.0 - current_t)**2

        with torch.amp.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
            if guidance_scale != 1.0 and current_cfg > 1.0:
                z_in = torch.cat([z_prev, z_prev], dim=0).to(DEVICE)
                y_in = torch.cat([null_labels, labels], dim=0).to(DEVICE)

                v_all = get_v(current_t, z_in, y=y_in)
                v_uncond, v_cond = v_all.chunk(2, dim=0)

                v = v_uncond + current_cfg * (v_cond - v_uncond)
            else:
                v = get_v(current_t, z_prev, y=labels)
        
        v = v.to(torch.float32)

        u = dt * v
        z_next = manifold.expmap(z_prev, u)
        z_next = manifold.projx(z_next)

        if RUNTIME_STATS:
            print(f"Step {i} | Drift: {u.norm().mean():.4f}")
            measure_manifold_distance(z_next, z_prev, "Step Distance")
        z_prev = z_next

    return z_prev


@torch.inference_mode()
def improve_encodings(
    input_path,
    noise_std=NOISE_STD,
    start_t=START_T,
    generation=False,
    generation_samples=N_SAMPLES,
    batch_size=BATCH_SIZE,
):
    """Loads existing .pt file and pushes unflattened sequence encodings through the flow."""
    print(f"Loading existing encodings from {input_path}")
    data = np.load(input_path, allow_pickle=False)
    z_input = torch.from_numpy(data["encodings"]).float()  # Now natively expected as [B, S, D]
    labels_input = torch.from_numpy(data["labels"]).long()  # Natively [B]
    split_ids = torch.from_numpy(data["split_ids"]).long()
    split_names = data["split_names"].tolist()
    class_means = data.get("class_means", None)

    T, D = SPHERE_DIMS  # 64, 3

    if class_means is not None:
        class_means = torch.from_numpy(class_means).squeeze(1).float()
        print("Loaded class means.")

    if not generation and N_SAMPLES != -1 and z_input.shape[0] > N_SAMPLES:
        total_samples = z_input.shape[0]
        permutation = torch.randperm(total_samples)
        subsample_indices = permutation[:N_SAMPLES]

        z_input = z_input[subsample_indices]
        labels_input = labels_input[subsample_indices]
        split_ids = split_ids[subsample_indices]

    z_input = z_input / z_input.norm(dim=-1, keepdim=True)
    z_input = manifold.projx(z_input)

    if SQUEEZE_DATA:
        z_input, _ = manifold_squeeze(z_input, labels_input, class_means=class_means, alpha=SQUEEZE_ALPHA, reverse=False)

    if generation:
        labels = dummy_labels(generation_samples)
        # Directly build random base vectors on the hypersphere with shape [B, T, D]
        z_init = manifold.random_base(generation_samples * T, D).reshape(generation_samples, T, D)
        z_init = z_init / z_init.norm(dim=-1, keepdim=True)
        z_noise = z_init
    else:
        labels = labels_input
        noise = torch.randn_like(z_input) * noise_std
        z_noise = manifold.expmap(z_input, noise)

    print(f"Refining {z_noise.shape[0]} sequences in chunks of {batch_size}...")
    z_final_list = []
    num_samples = z_noise.shape[0]

    # for i in tqdm.tqdm(range(0, num_samples, batch_size), desc="Processing Batches"):
    #     z_batch = z_noise[i : i + batch_size]
    #     labels_batch = labels[i : i + batch_size]
        
    #     z_final_batch = integrate_flow(
    #         z_batch, 
    #         labels_batch, 
    #         steps=STEPS, 
    #         start_t=start_t
    #     )
    #     z_final_list.append(z_final_batch.cpu())
        
    # z_final = torch.cat(z_final_list, dim=0)
    z_final = z_noise

    print("Clustering report:")
    check_class_clustering(z_noise, labels, text="Noise Class Clustering")
    check_class_clustering(z_input, labels_input, text="Original Class Clustering")
    check_class_clustering(z_final, labels, text="Final Class Clustering")

    if not generation:
        print("Manifold distance stats:")
        measure_manifold_distance(z_noise, z_input, text="Noise vs Original")
        measure_manifold_distance(z_noise, z_final, text="Noise vs Improved")
        measure_manifold_distance(z_final, z_input, text="Improved vs Original")

    if SQUEEZE_DATA:
        z_final, _ = manifold_squeeze(z_final, labels, class_means=class_means, alpha=SQUEEZE_ALPHA, reverse=True)

    return z_final, labels, split_ids, split_names


@torch.no_grad()
def measure_manifold_distance(z_noisy, z_original, text="Manifold Distance"):
    """Measures the geodesic distance over unflattened dimension structures."""
    z_noisy_split = torch.split(z_noisy.to(DEVICE), BATCH_SIZE)
    z_original_split = torch.split(z_original.to(DEVICE), BATCH_SIZE)
    
    distances_list = []
    for zn, zo in zip(z_noisy_split, z_original_split):
        distances_list.append(manifold.dist(zn, zo).cpu())
        
    distances = torch.cat(distances_list, dim=0)

    avg_dist = distances.mean().item()
    std_dist = distances.std().item()
    max_dist = distances.max().item()
    min_dist = distances.min().item()

    print(f"{text} - Avg: {avg_dist:.4f}, Std: {std_dist:.4f}, Max: {max_dist:.4f}, Min: {min_dist:.4f}")


@torch.no_grad()
def check_class_clustering(z, labels, text="Class Clustering"):
    """Evaluates class clustering tracking across the sequential [B, S, D] structure."""
    z = z.to(DEVICE)
    labels = labels.to(DEVICE)
    unique_labels = torch.unique(labels)

    print(f"\n=== {text} Report ===")
    print(f"{'Label':<10} | {'Mean Norm':<10} | {'Avg Dist':<10} | {'Std Dist':<10} | {'Max Dist':<10}")
    print("-" * 65)

    all_stats = []

    for label in unique_labels:
        mask = (labels == label)
        z_class = z[mask]

        if z_class.shape[0] == 0:
            continue

        mean_vec = z_class.mean(dim=0, keepdim=True)
        centroid = manifold.projx(mean_vec)
        mean_norm = mean_vec.norm().item()

        centroid_batch = centroid.expand(z_class.shape[0], -1, -1)
        
        distances_list = []
        for z_chunk, c_chunk in zip(torch.split(z_class, BATCH_SIZE), torch.split(centroid_batch, BATCH_SIZE)):
            distances_list.append(manifold.dist(z_chunk, c_chunk).cpu())
        distances = torch.cat(distances_list, dim=0)

        avg_d = distances.mean().item()
        std_d = distances.std().item()
        max_d = distances.max().item()

        print(f"{label.item():<10} | {mean_norm:<10.4f} | {avg_d:<10.4f} | {std_d:<10.4f} | {max_d:<10.4f}")

        all_stats.append({
            'label': label.item(),
            'mean_norm': mean_norm,
            'avg_dist': avg_d
        })

    avg_class_tightness = sum(s['avg_dist'] for s in all_stats) / len(all_stats) if all_stats else 0
    print("-" * 65)
    print(f"Global Average Class Tightness (Distance to Mean): {avg_class_tightness:.4f}")
    return all_stats


set_seed(42)

if GENERATION:
    z_final, labels, *_ = improve_encodings(INPUT_PATH, generation=True, generation_samples=N_SAMPLES)
    split_ids = torch.zeros(N_SAMPLES, dtype=torch.long)
    split_names = ["generated"]
else:
    z_final, labels, split_ids, split_names = improve_encodings(INPUT_PATH)

z_final = z_final.cpu()
labels_output = labels.cpu().numpy()
split_ids_output = split_ids.cpu().numpy()

assert z_final.shape[0] == labels_output.shape[0] == split_ids_output.shape[0], \
    f"Shape mismatch error! Encodings: {z_final.shape[0]}, Labels: {labels_output.shape[0]}, Split IDs: {split_ids_output.shape[0]}"

if SAVE_OUTPUT:
    print(f"Writing corrected NPZ archive straight to: {OUTPUT_PATH}")
    np.savez_compressed(
        OUTPUT_PATH,
        allow_pickle=False,
        encodings=z_final.numpy(),
        labels=labels_output,
        split_ids=split_ids_output,
        split_names=np.array(split_names, dtype=str),
    )

print(f"Done successfully. Perfectly formatted sequence shape saved: {z_final.shape}")