import os
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image
from tqdm import tqdm
import geoopt 



class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features.eval().to(device)
        
        for param in vgg.parameters():
            param.requires_grad = False
            
        self.slice1 = nn.Sequential(*vgg[:4])  # relu1_2
        self.slice2 = nn.Sequential(*vgg[4:9])  # relu2_2
        self.slice3 = nn.Sequential(*vgg[9:16]) # relu3_3
        
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def forward(self, recon, target):
        x = (recon + 1.0) / 2.0
        y = (target + 1.0) / 2.0
        
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        
        loss = 0.0
        for slice_net in [self.slice1, self.slice2, self.slice3]:
            x = slice_net(x)
            y = slice_net(y)
            loss += F.l1_loss(x, y)
            
        return loss



def sample_uniform_sphere(batch_size, latent_dim, device, dtype=torch.float32):
    """
    Samples vectors uniformly distributed across an (n-1)-dimensional hypersphere surface
    via standard isotropic Gaussian projection normalization.
    """
    z = torch.randn(batch_size, latent_dim, device=device, dtype=dtype)
    return F.normalize(z, p=2, dim=-1)


class ListDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self): 
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform: 
            img = self.transform(img)
        return img, self.labels[idx]

class TransformedSubset(Dataset):
    """Isolates data transformation domains to prevent leakage between training and validation."""
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
        
    def __len__(self): 
        return len(self.subset)
        
    def __getitem__(self, idx):
        base_dataset = self.subset.dataset
        actual_idx = self.subset.indices[idx]
        if isinstance(base_dataset, ListDataset):
            img = Image.open(base_dataset.image_paths[actual_idx]).convert("RGB")
            label = base_dataset.labels[actual_idx]
        else:
            img, label = base_dataset[actual_idx]
            if isinstance(img, torch.Tensor): 
                img = transforms.ToPILImage()(img)
        if self.transform: 
            img = self.transform(img)
        return img, label

def get_datasets(args, train_transform, val_transform):
    args.num_classes = {"cifar-10": 10, "cifar-100": 100, "imagenet": 1000}[args.dataset_name]
    root = os.path.join(args.data_dir, args.dataset_name)

    if "cifar" in args.dataset_name:
        ds_cls = datasets.CIFAR10 if args.dataset_name == "cifar-10" else datasets.CIFAR100
        train_set = ds_cls(root=root, train=True, download=True, transform=None)
        test_set = ds_cls(root=root, train=False, download=True, transform=val_transform)
    else:
        image_paths, labels = [], []
        for idx, c_dir in enumerate(sorted(os.listdir(root))):
            c_path = os.path.join(root, c_dir)
            if os.path.isdir(c_path):
                for img in os.listdir(c_path):
                    if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_paths.append(os.path.join(c_path, img))
                        labels.append(idx)
        full_dataset = ListDataset(image_paths, labels, transform=None)
        generator = torch.Generator().manual_seed(42)
        train_len = int(0.85 * len(full_dataset))
        train_set, test_subset = random_split(full_dataset, [train_len, len(full_dataset)-train_len], generator=generator)
        test_set = TransformedSubset(test_subset, val_transform)

    generator = torch.Generator().manual_seed(42)
    v_len = int(0.10 * len(train_set))
    t_split_raw, v_split_raw = random_split(train_set, [len(train_set)-v_len, v_len], generator=generator)
    return TransformedSubset(t_split_raw, train_transform), TransformedSubset(v_split_raw, val_transform), test_set


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="cifar-10")
    parser.add_argument("--data_dir", type=str, default="./datasets")
    parser.add_argument("--num_classes", type=str, default=10)
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--latent_dim", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--cond_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--null_chance", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_noisy_tokens", type=int, default=4, help="Number of noisy tokens to inject.")
    parser.add_argument("--noise_level", type=float, default=0.10, help="Spherifying perturbation variance magnitude.")
    parser.add_argument("--lambda_mse", type=float, default=1.0, help="Weight factor for MSE loss.")
    parser.add_argument("--lambda_l1", type=float, default=0.2, help="Weight factor for L1 loss.")
    parser.add_argument("--lambda_perc", type=float, default=0.2, help="Weight factor for perceptual loss.")
    parser.add_argument("--lambda_latent", type=float, default=0.1, help="Weight factor for latent consistency loss.")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping evaluation epochs window threshold limit.")
    args = parser.parse_args()

    torch.set_float32_matmul_precision('high')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    val_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set, val_set, test_set = get_datasets(args, train_transform, val_transform)
    loader_kwargs = {"batch_size": args.batch_size, "num_workers": 4, "pin_memory": True, "drop_last": True}
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **{**loader_kwargs, "drop_last": False})
    test_loader = DataLoader(test_set, shuffle=False, **{**loader_kwargs, "drop_last": False})

    from models import SphericalAutoencoder

    autoencoder = SphericalAutoencoder(
        latent_dim=args.latent_dim, hidden_dim=args.hidden_dim, cond_dim=args.cond_dim,
        num_classes=args.num_classes, num_layers=args.num_layers, patch_size=args.patch_size,
        dropout=args.dropout, null_chance=args.null_chance, image_size=args.image_size
    ).to(device=device, memory_format=torch.channels_last)

    perceptual_criterion = PerceptualLoss(device)

    manifold = geoopt.manifolds.Sphere()

    autoencoder.compile()
    optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=args.lr, weight_decay=0.05)
    scaler = torch.amp.GradScaler(device="cuda", enabled=torch.cuda.is_available())

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    best_val_loss = float("inf")
    patience_counter = 0
    checkpoint_filename = "best_spherical_autoencoder.pt"
    
    for epoch in range(args.epochs):
        autoencoder.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for imgs, class_ids in pbar:
            imgs = imgs.to(device, memory_format=torch.channels_last, non_blocking=True)
            class_ids = class_ids.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                encoder_module = autoencoder.module.encoder if hasattr(autoencoder, 'module') else autoencoder.encoder
                decoder_module = autoencoder.module.decoder if hasattr(autoencoder, 'module') else autoencoder.decoder

                z_clean, H_grid, W_grid = encoder_module(imgs, class_ids)
                B, T, D = z_clean.shape  

                rand_indices = torch.argsort(torch.rand(B, T, device=z_clean.device), dim=1)
                mask = rand_indices < args.num_noisy_tokens  
                mask = mask.unsqueeze(-1)               

                pure_noise = torch.randn_like(z_clean)

                z_perturbed = manifold.expmap(z_clean, args.noise_level * torch.randn_like(z_clean))
                z_perturbed = torch.where(mask, pure_noise, z_perturbed)

                z_spherified = F.normalize(z_perturbed, p=2, dim=-1)

                rec_images = decoder_module(z_spherified, class_ids, H_grid=H_grid, W_grid=W_grid)
                
                z_recon, _, _ = encoder_module(rec_images, class_ids)
                
                loss_mse = F.mse_loss(rec_images, imgs)
                loss_l1  = F.l1_loss(rec_images, imgs)
                loss_perceptual = perceptual_criterion(rec_images, imgs.to(dtype=rec_images.dtype))
                
                loss_latent = manifold.dist(z_clean, z_recon).pow(2).mean()
                
                total_loss = (args.lambda_mse * loss_mse + 
                              args.lambda_l1 * loss_l1 + 
                              args.lambda_perc * loss_perceptual + 
                              args.lambda_latent * loss_latent)

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()
            
            current_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix({
                "lr": f"{current_lr:.2e}",
                "mse": f"{loss_mse.item():.4f}",
                "l1": f"{loss_l1.item():.4f}",
                "perc": f"{loss_perceptual.item():.4f}",
                "lat": f"{loss_latent.item():.4f}"
            })

        autoencoder.eval()
        val_loss_cumulative = 0.0
        val_steps = 0
        
        print(f"Executing Validation Loop Tracker diagnostics...")
        with torch.inference_mode():
            for v_imgs, v_class_ids in val_loader:
                v_imgs = v_imgs.to(device, memory_format=torch.channels_last, non_blocking=True)
                v_class_ids = v_class_ids.to(device, non_blocking=True)
                
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    encoder_mod = autoencoder.module.encoder if hasattr(autoencoder, 'module') else autoencoder.encoder
                    decoder_mod = autoencoder.module.decoder if hasattr(autoencoder, 'module') else autoencoder.decoder
                    
                    v_z_clean, v_H, v_W = encoder_mod(v_imgs, v_class_ids)
                    v_rec = decoder_mod(v_z_clean, v_class_ids, H_grid=v_H, W_grid=v_W)
                    v_z_recon, _, _ = encoder_mod(v_rec, v_class_ids)
                    
                    v_loss_mse = F.mse_loss(v_rec, v_imgs)
                    v_loss_l1 = F.l1_loss(v_rec, v_imgs)
                    v_loss_perc = perceptual_criterion(v_rec, v_imgs.to(dtype=v_rec.dtype))
                    
                    v_loss_latent = manifold.dist(v_z_clean, v_z_recon).pow(2).mean()
                    
                    v_loss = (args.lambda_mse * v_loss_mse + 
                              args.lambda_l1 * v_loss_l1 + 
                              args.lambda_perc * v_loss_perc + 
                              args.lambda_latent * v_loss_latent)
                    val_loss_cumulative += v_loss.item()
                    val_steps += 1
                    
        epoch_val_loss = val_loss_cumulative / val_steps
        print(f">> Epoch {epoch+1} Complete. Validation Loss Target: {epoch_val_loss:.5f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            state_dict = autoencoder.module.state_dict() if hasattr(autoencoder, 'module') else autoencoder.state_dict()
            torch.save({"autoencoder": state_dict, "args": args}, checkpoint_filename)
            print(f" -> Validation loss decreased. Snapshot saved to {checkpoint_filename}")
        else:
            patience_counter += 1
            print(f" -> Validation loss did not improve. Early stopping tracker: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print(f" Early stopping criteria window reached. Breaking optimization sequence loop.")
                break

    print("\nLoading best weights snapshot for final holdout testing performance run...")
    if os.path.exists(checkpoint_filename):
        if hasattr(autoencoder, 'module'):
            autoencoder.module.load_state_dict(torch.load(checkpoint_filename)["autoencoder"])
        else:
            autoencoder.load_state_dict(torch.load(checkpoint_filename)["autoencoder"])

    autoencoder.eval()
    test_loss_cumulative = 0.0
    test_steps = 0
    
    with torch.inference_mode():
        for t_imgs, t_class_ids in test_loader:
            t_imgs = t_imgs.to(device, memory_format=torch.channels_last, non_blocking=True)
            t_class_ids = t_class_ids.to(device, non_blocking=True)
            
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                encoder_mod = autoencoder.module.encoder if hasattr(autoencoder, 'module') else autoencoder.encoder
                decoder_mod = autoencoder.module.decoder if hasattr(autoencoder, 'module') else autoencoder.decoder
                
                t_z_clean, t_H, t_W = encoder_mod(t_imgs, t_class_ids)
                t_rec = decoder_mod(t_z_clean, t_class_ids, H_grid=t_H, W_grid=t_W)
                t_z_recon, _, _ = encoder_mod(t_rec, t_class_ids)
                
                t_loss_mse = F.mse_loss(t_rec, t_imgs)
                t_loss_l1 = F.l1_loss(t_rec, t_imgs)
                t_loss_perc = perceptual_criterion(t_rec, t_imgs.to(dtype=t_rec.dtype))
                
                t_loss_latent = manifold.dist(t_z_clean, t_z_recon).pow(2).mean()
                
                t_loss = (args.lambda_mse * t_loss_mse + 
                          args.lambda_l1 * t_loss_l1 + 
                          args.lambda_perc * t_loss_perc + 
                          args.lambda_latent * t_loss_latent)
                test_loss_cumulative += t_loss.item()
                test_steps += 1


    H_grid = args.image_size // args.patch_size
    W_grid = args.image_size // args.patch_size

    with torch.inference_mode():
        eval_labels = torch.randint(0, 10, (16,), device=device)
        raw_samples = torch.randn(16, H_grid * W_grid, args.latent_dim, device=device)
        z_direct = F.normalize(raw_samples, p=2, dim=-1)
        
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            decoder_module = autoencoder.module.decoder if hasattr(autoencoder, 'module') else autoencoder.decoder
            images_direct = decoder_module(z_direct, eval_labels, H_grid=H_grid, W_grid=W_grid)
            
    