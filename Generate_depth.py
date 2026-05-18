import os
import cv2
import torch
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2

# -----------------------------
# Configuration
# -----------------------------
INPUT_DIR = "img_RGB"
OUTPUT_DIR = "img_Depth"

ENCODER = "vits"  # small model
MODEL_PATH = "checkpoints/depth_anything_v2_vits.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Create output folder
# -----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Model configuration
# -----------------------------
model_configs = {
    "vits": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
    }
}

# -----------------------------
# Load model
# -----------------------------
depth_anything = DepthAnythingV2(**model_configs[ENCODER])

depth_anything.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

depth_anything = depth_anything.to(DEVICE).eval()

# -----------------------------
# Process all images
# -----------------------------
image_extensions = [".png", ".jpg", ".jpeg", ".bmp"]

image_files = [
    f for f in os.listdir(INPUT_DIR)
    if os.path.splitext(f)[1].lower() in image_extensions
]

print(f"Found {len(image_files)} images")

for idx, filename in enumerate(image_files):

    input_path = os.path.join(INPUT_DIR, filename)

    print(f"[{idx+1}/{len(image_files)}] Processing: {filename}")

    # Read image
    raw_image = cv2.imread(input_path)

    if raw_image is None:
        print(f"Failed to read: {filename}")
        continue

    # Infer depth
    depth = depth_anything.infer_image(raw_image)

    # Normalize depth to 0-255
    depth_norm = cv2.normalize(
        depth,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    # Save output
    output_path = os.path.join(OUTPUT_DIR, filename)

    cv2.imwrite(output_path, depth_norm)

print("Done!")