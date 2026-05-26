import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from PIL import Image
from tqdm import tqdm

from train import get_dataset

class FolderDataset(torch.utils.data.Dataset):
    """A clean dataset to read all generated/decoded images from a target folder."""
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.img_names = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
    def __len__(self):
        return len(self.img_names)
        
    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.img_names[idx])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Generation Quality using Torchmetrics FID")
    parser.add_argument("--dataset_name", type=str, default="cifar-10")
    parser.add_argument("--data_dir", type=str, default="../../../spherical-flow-matching/sphere-encoder-main/workspace/datasets")
    parser.add_argument("--gen_dir", type=str, required=True, help="Path to folder containing your RFM generated & decoded images")
    parser.add_argument("--image_size", type=int, default=32, help="Resolution matching your dataset configurations")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"FID Evaluation Engine Active on: {device}")

    fid_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.PILToTensor() 
    ])

    print(f"Loading reference real distribution ('{args.dataset_name}')...")
    real_dataset = get_dataset(args, transform=fid_transform)
    real_loader = DataLoader(real_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"Loading generated distribution from '{args.gen_dir}'...")
    if not os.path.exists(args.gen_dir) or len(os.listdir(args.gen_dir)) == 0:
        raise ValueError(f"Generation directory '{args.gen_dir}' is empty or does not exist!")
        
    gen_dataset = FolderDataset(args.gen_dir, transform=fid_transform)
    gen_loader = DataLoader(gen_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    fid_metric = FrechetInceptionDistance(feature=2048).to(device)

    print("\nProcessing Real Images through InceptionV3 network...")
    for real_imgs, _ in tqdm(real_loader, desc="Real Distribution"):
        real_imgs = real_imgs.to(device)
        fid_metric.update(real_imgs, real=True)

    print("\nProcessing Generated Images through InceptionV3 network...")
    for gen_imgs in tqdm(gen_loader, desc="Generated Distribution"):
        gen_imgs = gen_imgs.to(device)
        fid_metric.update(gen_imgs, real=False)

    print("\nCalculating Final Covariance and Mean Differences...")
    final_fid_score = fid_metric.compute().item()
    
    print("-" * 50)
    print(f" SUCCESSFUL EVALUATION FOR {args.dataset_name.upper()}")
    print(f"   Total Real Reference Samples: {len(real_dataset)}")
    print(f"   Total Generated Test Samples: {len(gen_dataset)}")
    print(f"    FINAL FID SCORE: {final_fid_score:.4f} ")
    print("-" * 50)