import os
import yaml
import cv2
import torch
import numpy as np

def compute_3d_box_cam2(h, w, l, x, y, z, ry):
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

def project_to_image(pts_3d, P2):
    pts_3d_homo = np.vstack((pts_3d, np.ones((1, pts_3d.shape[1]))))
    pts_2d_homo = np.dot(P2, pts_3d_homo)
    pts_2d = pts_2d_homo[:2, :] / pts_2d_homo[2, :]
    return pts_2d.T

def draw_projected_box3d(image, corners_3d, P2, color=(0, 255, 0), thickness=2, label=None):
    pts_2d = project_to_image(corners_3d, P2).astype(np.int32)
    
    if np.any(corners_3d[2, :] <= 0.1):
        return image

    lines = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    
    h_img, w_img, _ = image.shape
    for p1, p2 in lines:
        pt1, pt2 = tuple(pts_2d[p1]), tuple(pts_2d[p2])
        cv2.line(image, pt1, pt2, color, thickness)

    if label:
        min_x = np.clip(np.min(pts_2d[:, 0]), 0, w_img - 10)
        min_y = np.clip(np.min(pts_2d[:, 1]), 15, h_img - 10)
        cv2.putText(image, label, (int(min_x), int(min_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    return image

def main():
    config_path = "configs/mono3d_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    output_dir = "visualization_results"
    os.makedirs(output_dir, exist_ok=True)
    
    label_files = sorted(os.listdir("data/kitti/training/label_2"))[:5]

    for label_file in label_files:
        file_id = label_file.replace('.txt', '')
        img_path = os.path.join("data/kitti/training/image_2", f"{file_id}.png")
        calib_path = os.path.join("data/kitti/training/calib", f"{file_id}.txt")
        
        if not os.path.exists(img_path) or not os.path.exists(calib_path):
            continue

        image = cv2.imread(img_path)
        
        P2 = None
        with open(calib_path, 'r') as f:
            for line in f:
                if line.startswith('P2:'):
                    P2 = np.array([float(x) for x in line.strip().split()[1:]]).reshape(3, 4)
                    break

        if P2 is None:
            continue

        boxes_drawn = 0
        with open(os.path.join("data/kitti/training/label_2", label_file), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 15 or parts[0] == 'DontCare':
                    continue
                cls_name = parts[0]
                h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
                x, y, z = float(parts[11]), float(parts[12]), float(parts[13])
                ry = float(parts[14])

                corners_3d = compute_3d_box_cam2(h, w, l, x, y, z, ry)
                image = draw_projected_box3d(image, corners_3d, P2, color=(0, 255, 0), label=cls_name)
                boxes_drawn += 1

        out_path = os.path.join(output_dir, f"gt_vis_{file_id}.png")
        cv2.imwrite(out_path, image)
        print(f"Saved: {out_path} ({boxes_drawn} boxes)")

if __name__ == "__main__":
    main()
