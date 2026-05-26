import os
import argparse
import torch
import numpy as np
import torchvision.utils as vutils
from tqdm import tqdm
from models import ScalableSphereDecoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode Spherical Latents from NPZ into Individual Image Files")
    parser.add_argument("--input_vectors", type=str, required=True, help="Path to .npz file containing latent codes")
    parser.add_argument("--output_dir", type=str, default="workspace/generated_outputs", help="Directory where individual images will be written")
    args_cli = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ckpt = torch.load("sphere_encoder/autoencoder.pt", map_location=device)
    args_ae = ckpt["args"]

    decoder = ScalableSphereDecoder(latent_dim=args_ae.latent_dim, target_resolution=args_ae.image_size).to(device)
    decoder.load_state_dict(ckpt["decoder"])
    decoder.eval()

    data_source = np.load(args_cli.input_vectors, allow_pickle=False)
    
    if "encodings" in data_source:
        raw_vectors = data_source["encodings"]
    elif "x1" in data_source:
        raw_vectors = data_source["x1"]
    else:
        first_key = list(data_source.keys())[0]
        raw_vectors = data_source[first_key]

    z_vectors = torch.from_numpy(raw_vectors).float().to(device)
    
    print(f"Loaded source NPZ matrix. Processing pool dimension: {z_vectors.shape}")
    
    os.makedirs(args_cli.output_dir, exist_ok=True)

    print(f"--- Decoding Tensors to Folder: '{args_cli.output_dir}' ---")
    
    with torch.no_grad():
        z_vectors = torch.nn.functional.normalize(z_vectors, p=2, dim=-1)
        
        for idx in tqdm(range(z_vectors.shape[0]), desc="Saving Images"):
            single_latent = z_vectors[idx].unsqueeze(0)
            decoded_pixel = decoder(single_latent)
            
            decoded_pixel = (decoded_pixel + 1.0) / 2.0
            
            img_filename = os.path.join(args_cli.output_dir, f"sample_{idx:06d}.png")
            
            vutils.save_image(decoded_pixel, img_filename, normalize=False)

    print(f"\nDecoding complete! Total individual files written: {z_vectors.shape[0]}")
    print(f"Output directory path target: {os.path.abspath(args_cli.output_dir)}")