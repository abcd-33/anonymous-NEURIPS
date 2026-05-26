from PIL import Image
import os
from pathlib import Path

SRC = "workspace/datasets/animal-faces/train"
DST = "workspace/fid_refs/ref_images_animal-faces_256px/images"

os.makedirs(DST, exist_ok=True)

exts = [".jpg", ".jpeg", ".png", ".webp"]

idx = 0

for path in Path(SRC).rglob("*"):
    if path.suffix.lower() not in exts:
        continue

    try:
        img = Image.open(path).convert("RGB")
        img = img.resize((256, 256), Image.LANCZOS)

        save_path = os.path.join(DST, f"{idx:06d}.png")
        img.save(save_path)

        idx += 1

        if idx % 1000 == 0:
            print(f"processed {idx}")

    except Exception as e:
        print("failed:", path, e)

print("done:", idx)