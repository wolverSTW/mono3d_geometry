import os
import yaml
import cv2
import torch
import numpy as np

from datasets.kitti_dataset import KITTIDataset
from models.mono3d_network import Mono3DNetwork

def compute_3d_box_cam2(h, w, l, x, y, z, ry):
    """
    Computes 8 corners of 3D bounding box in Camera Coordinates.
    """
    R = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    
    # 3D bounding box corners relative to center
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] += x
    corners_3d[1, :] += y
    corners_3d[2, :] += z
    
    return corners_3d

def draw_projected_box3d(image, corners_3d, P2, color=(0, 255, 0), thickness=2):
    """
    Projects 3D bounding box corners onto 2D image plane using P2 matrix.
    """
    pts_3d_homo = np.vstack((corners_3d, np.ones((1, 8))))
    pts_2d_homo = np.dot(P2, pts_3d_homo)
    
    # Normalize with depth Z
    z_vals = pts_2d_homo[2, :]
    z_vals[z_vals == 0] = 1e-5
    pts_2d = pts_2d_homo[:2, :] / z_vals
    pts_2d = pts_2d.T.astype(np.int32)
    
    # Draw 12 edges of 3D box
    for k in range(4):
        i, j = k, (k + 1) % 4
        cv2.line(image, tuple(pts_2d[i]), tuple(pts_2d[j]), color, thickness)
        
        i_top, j_top = k + 4, ((k + 1) % 4) + 4
        cv2.line(image, tuple(pts_2d[i_top]), tuple(pts_2d[j_top]), color, thickness)
        
        cv2.line(image, tuple(pts_2d[k]), tuple(pts_2d[k + 4]), color, thickness)
        
    return image

def main():
    print("=" * 70)
    print("=== [INFERENCE & VISUALIZATION] Mono3D Box Projection ===")
    print("=" * 70)
    
    config_path = "configs/mono3d_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_path = "weights/mono3d_best.pth"
    
    if not os.path.exists(weight_path):
        print(f"[ERROR] Weight file not found: {weight_path}")
        return

    print(f"[MODEL] Loading model weights from '{weight_path}'...")
    model = Mono3DNetwork(config=config).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    val_dataset = KITTIDataset(data_dir="data/kitti", config=config, split="val", augment=False)
    output_dir = "visualization_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[DATASET] Running visualization for top 5 samples in Validation Set...\n")

    for i in range(min(5, len(val_dataset))):
        sample = val_dataset[i]
        file_id = sample['file_id']
        P2 = sample['calib_p2'].numpy() if isinstance(sample['calib_p2'], torch.Tensor) else sample['calib_p2']
        
        # Search original image
        img_path = os.path.join("data/kitti/training/image_2", f"{file_id}.png")
        if not os.path.exists(img_path):
            img_path = os.path.join("data/kitti/training/image_2", f"{file_id}.jpg")
            
        vis_img = cv2.imread(img_path)
        if vis_img is None:
            print(f" -> Skipping {file_id}: Image not found at {img_path}")
            continue

        labels = sample['labels']
        boxes_drawn = 0

        # Handle list/tensor structure for labels safely
        if isinstance(labels, (torch.Tensor, np.ndarray)):
            for lbl in labels:
                # Assuming KITTI tensor order: [cls, h, w, l, x, y, z, ry, ...]
                if len(lbl) >= 8:
                    h, w, l = float(lbl[1]), float(lbl[2]), float(lbl[3])
                    x, y, z = float(lbl[4]), float(lbl[5]), float(lbl[6])
                    ry = float(lbl[7])
                    corners_3d = compute_3d_box_cam2(h, w, l, x, y, z, ry)
                    vis_img = draw_projected_box3d(vis_img, corners_3d, P2, color=(0, 255, 0), thickness=2)
                    boxes_drawn += 1
        elif isinstance(labels, list):
            for lbl in labels:
                if isinstance(lbl, dict):
                    h, w, l = lbl.get('dimensions', (1.5, 1.6, 3.8))
                    x, y, z = lbl.get('location', (0, 0, 10))
                    ry = lbl.get('rotation_y', 0.0)
                elif isinstance(lbl, (list, tuple, np.ndarray, torch.Tensor)) and len(lbl) >= 7:
                    h, w, l = float(lbl[0]), float(lbl[1]), float(lbl[2])
                    x, y, z = float(lbl[3]), float(lbl[4]), float(lbl[5])
                    ry = float(lbl[6])
                else:
                    continue

                corners_3d = compute_3d_box_cam2(h, w, l, x, y, z, ry)
                vis_img = draw_projected_box3d(vis_img, corners_3d, P2, color=(0, 255, 0), thickness=2)
                boxes_drawn += 1

        out_file = os.path.join(output_dir, f"vis_{file_id}.png")
        cv2.imwrite(out_file, vis_img)
        print(f" -> Saved visualization: {out_file} ({boxes_drawn} Ground-Truth boxes drawn)")

    print(f"\n[SUCCESS] Visualizations saved to '{output_dir}/' directory.")

if __name__ == "__main__":
    main()
