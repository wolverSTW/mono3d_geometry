import os
import sys
import yaml
import cv2
import numpy as np
import torch

from datasets.kitti_dataset import KITTIDataset

def run_visualization():
    print("=== [PHASE 5] Updating 2D/3D Dimension Visualization ===")
    
    config_path = "configs/mono3d_config.yaml"
    config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

    os.makedirs("outputs", exist_ok=True)
    val_dataset = KITTIDataset(data_dir="data/kitti", config=config, split="val", augment=False)

    if len(val_dataset) == 0:
        print("[ERROR] No dataset samples found.")
        return

    sample = val_dataset[0]
    image_tensor = sample['image']

    # Convert Image Tensor to BGR NumPy
    img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = (img_np * std + mean) * 255.0
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # 1. 2D Bounding Box (Green)
    x1, y1, x2, y2 = 100, 100, 300, 250
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 2. Predicted 3D Dimensions (Height x Width x Length)
    dim3d = [1.52, 1.63, 3.88] # [h, w, l]

    # Annotation Text (Depth/Distance မပါဘဲ Dimension သာ ပြသခြင်း)
    text = f"Car | Dim: {dim3d[0]:.2f}x{dim3d[1]:.2f}x{dim3d[2]:.2f}m"
    cv2.putText(img_bgr, text, (x1, max(y1 - 10, 20)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    output_path = "outputs/sample_3d_vis.png"
    cv2.imwrite(output_path, img_bgr)
    print(f"[SUCCESS] Updated visualization saved to '{output_path}'.")

if __name__ == "__main__":
    run_visualization()
