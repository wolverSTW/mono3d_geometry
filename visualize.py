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
    
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] += x
    corners_3d[1, :] += y
    corners_3d[2, :] += z
    
    return corners_3d

def draw_projected_box3d(image, corners_3d, P2, color=(0, 0, 255), thickness=2, label_text=None):
    """
    Projects 3D box onto image and draws Class Name & Confidence score.
    """
    pts_3d_homo = np.vstack((corners_3d, np.ones((1, 8))))
    pts_2d_homo = np.dot(P2, pts_3d_homo)
    
    z_vals = pts_2d_homo[2, :]
    z_vals[z_vals <= 0] = 1e-5
    pts_2d = pts_2d_homo[:2, :] / z_vals
    pts_2d = pts_2d.T.astype(np.int32)
    
    # Draw 12 edges
    for k in range(4):
        i, j = k, (k + 1) % 4
        cv2.line(image, tuple(pts_2d[i]), tuple(pts_2d[j]), color, thickness)
        i_top, j_top = k + 4, ((k + 1) % 4) + 4
        cv2.line(image, tuple(pts_2d[i_top]), tuple(pts_2d[j_top]), color, thickness)
        cv2.line(image, tuple(pts_2d[k]), tuple(pts_2d[k + 4]), color, thickness)

    # Draw Class Name & Confidence Score Text above box
    if label_text:
        min_x = int(np.min(pts_2d[:, 0]))
        min_y = int(np.min(pts_2d[:, 1]))
        cv2.putText(image, label_text, (max(0, min_x), max(20, min_y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    return image

def load_kitti_gt_boxes(label_path):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15 or parts[0] == 'DontCare':
                continue
            cls_type = parts[0]
            h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
            x, y, z = float(parts[11]), float(parts[12]), float(parts[13])
            ry = float(parts[14])
            boxes.append((cls_type, h, w, l, x, y, z, ry))
    return boxes

def decode_predictions(preds, conf_threshold=0.2):
    """
    Decodes raw model output logits into 3D bounding box predictions.
    Adjust key names according to model output structure if needed.
    """
    decoded = []
    # If model has decode method builtin
    if hasattr(preds, 'decode'):
        return preds.decode(conf_threshold=conf_threshold)
        
    # Placeholder for predicted tensors parsing
    if isinstance(preds, dict):
        cls_scores = preds.get('cls_scores', None)
        bbox_3d = preds.get('bbox_3d', None)
        if cls_scores is not None and bbox_3d is not None:
            scores, labels = torch.max(cls_scores.sigmoid(), dim=-1)
            mask = scores > conf_threshold
            # Filter and append detections
            # Output format: (class_name, score, h, w, l, x, y, z, ry)
    return decoded

def main():
    print("=" * 70)
    print("=== [INFERENCE & VISUALIZATION] Mono3D Predictions vs Ground Truth ===")
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
    
    print(f"[DATASET] Running Inference & Visualization on Validation Samples...\n")

    class_names = config.get('class_names', ['Car', 'Pedestrian', 'Cyclist'])

    for i in range(min(5, len(val_dataset))):
        sample = val_dataset[i]
        file_id = sample['file_id']
        P2 = sample['calib_p2'].numpy() if isinstance(sample['calib_p2'], torch.Tensor) else sample['calib_p2']
        
        img_path = os.path.join("data/kitti/training/image_2", f"{file_id}.png")
        if not os.path.exists(img_path):
            img_path = os.path.join("data/kitti/training/image_2", f"{file_id}.jpg")
            
        vis_img = cv2.imread(img_path)
        if vis_img is None:
            continue

        # 1. Draw Ground Truth Boxes (GREEN Color)
        txt_label_path = os.path.join("data/kitti/training/label_2", f"{file_id}.txt")
        gt_boxes = load_kitti_gt_boxes(txt_label_path)
        for cls_type, h, w, l, x, y, z, ry in gt_boxes:
            corners_3d = compute_3d_box_cam2(h, w, l, x, y, z, ry)
            vis_img = draw_projected_box3d(vis_img, corners_3d, P2, color=(0, 255, 0), thickness=2, label_text=f"GT: {cls_type}")

        # 2. Run Model Inference
        img_tensor = sample['image'].unsqueeze(0).to(device)
        with torch.no_grad():
            preds = model(img_tensor)

        # 3. Draw Model Predictions (RED Color)
        predictions = decode_predictions(preds, conf_threshold=0.3)
        pred_count = 0
        for pred in predictions:
            cls_id, score, h, w, l, x, y, z, ry = pred
            cls_name = class_names[cls_id] if cls_id < len(class_names) else 'Obj'
            corners_3d = compute_3d_box_cam2(h, w, l, x, y, z, ry)
            label_text = f"Pred: {cls_name} {score:.2f}"
            vis_img = draw_projected_box3d(vis_img, corners_3d, P2, color=(0, 0, 255), thickness=2, label_text=label_text)
            pred_count += 1

        out_file = os.path.join(output_dir, f"vis_{file_id}.png")
        cv2.imwrite(out_file, vis_img)
        print(f" -> Saved: {out_file} (GT Boxes: {len(gt_boxes)}, Predictions: {pred_count})")

    print(f"\n[SUCCESS] Visualizations saved in '{output_dir}/'. Green = Ground Truth, Red = Predictions")

if __name__ == "__main__":
    main()
