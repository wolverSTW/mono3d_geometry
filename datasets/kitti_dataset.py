import os
import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class KITTIDataset(Dataset):
    """
    PyTorch Dataset for KITTI 3D Object Detection.
    Includes 3D Geometry-Aware Data Augmentation and Train/Val Split Support.
    """
    def __init__(self, data_dir="data/kitti", config=None, split="train", augment=True):
        self.data_dir = data_dir
        self.config = config
        self.split = split
        self.augment = augment and (split == "train")
        
        self.img_dir = os.path.join(data_dir, "image_2")
        self.label_dir = os.path.join(data_dir, "label_2")
        self.calib_dir = os.path.join(data_dir, "calib")
        
        # Load classes and mapping from config or fallback defaults
        if config and 'names' in config:
            self.class_names = config['names']
            self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
            self.class_map = config.get('class_map', {})
        else:
            self.class_names = ['Car', 'Van', 'Truck', 'Pedestrian', 'Cyclist']
            self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
            self.class_map = {name: name for name in self.class_names}

        # Target image resolution
        self.img_size = config.get('img_size', [384, 1280]) if config else [384, 1280]
        
        # --- Stage 7: Train / Val Split Handling ---
        split_file = os.path.join(data_dir, f"{split}.txt")
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                self.file_ids = [line.strip() for line in f.readlines() if line.strip()]
        else:
            # Fallback to loading all available images if split file is not present
            all_files = sorted(glob.glob(os.path.join(self.img_dir, "*.png")))
            self.file_ids = [os.path.splitext(os.path.basename(f))[0] for f in all_files]

        # PyTorch Image Normalization Transform
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def parse_calibration(self, calib_path):
        """
        Parses P2 intrinsic calibration matrix from KITTI calibration file.
        """
        calib = {}
        if not os.path.exists(calib_path):
            return np.eye(3, 4)
            
        with open(calib_path, 'r') as f:
            for line in f:
                if line.startswith("P2:"):
                    p2 = np.array([float(x) for x in line.strip().split(' ')[1:]])
                    calib['P2'] = p2.reshape(3, 4)
                    break
        return calib.get('P2', np.eye(3, 4))

    def parse_label(self, label_path):
        """
        Parses annotations and maps raw classes to configured target classes.
        """
        targets = []
        if not os.path.exists(label_path):
            return targets

        with open(label_path, 'r') as f:
            for line in f:
                data = line.strip().split(' ')
                if not data or len(data) < 15:
                    continue
                
                raw_cls = data[0]
                mapped_cls = self.class_map.get(raw_cls, raw_cls)
                if mapped_cls not in self.class_to_idx:
                    continue

                cls_idx = self.class_to_idx[mapped_cls]
                truncation = float(data[1])
                occlusion = int(data[2])
                alpha = float(data[3])
                bbox_2d = [float(x) for x in data[4:8]]         # [x1, y1, x2, y2]
                dimensions_3d = [float(x) for x in data[8:11]]   # [height, width, length]
                location_3d = [float(x) for x in data[11:14]]   # [x, y, z]
                rotation_y = float(data[14])

                targets.append({
                    'class_idx': cls_idx,
                    'class_name': mapped_cls,
                    'truncation': truncation,
                    'occlusion': occlusion,
                    'alpha': alpha,
                    'bbox_2d': bbox_2d,
                    'dimensions_3d': dimensions_3d,
                    'location_3d': location_3d,
                    'rotation_y': rotation_y
                })
        return targets

    def apply_geometry_aware_flip(self, image, labels, calib_p2, orig_w):
        """
        --- Stage 6: 3D Geometry-Aware Data Augmentation ---
        Applies Horizontal Flip to image while correctly adjusting 2D Box, 
        3D Angles (Alpha & Rotation_y), 3D Center X, and Calibration P2 Matrix.
        """
        # 1. Flip Image Horizontally
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        # 2. Update Calibration Matrix P2 (Adjust Principal Center X: cx' = orig_w - cx)
        calib_p2[0, 2] = orig_w - calib_p2[0, 2]

        # 3. Adjust Bounding Boxes and 3D Geometry
        for target in labels:
            # Flip 2D Bounding Box: [x1, y1, x2, y2] -> [orig_w - x2, y1, orig_w - x1, y2]
            x1, y1, x2, y2 = target['bbox_2d']
            target['bbox_2d'] = [orig_w - x2, y1, orig_w - x1, y2]
            
            # Mirror Orientation Angles
            target['alpha'] = -target['alpha']
            target['rotation_y'] = -target['rotation_y']
            
            # Mirror 3D Center Location X-coordinate
            target['location_3d'][0] = -target['location_3d'][0]

        return image, labels, calib_p2

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        
        img_path = os.path.join(self.img_dir, f"{file_id}.png")
        label_path = os.path.join(self.label_dir, f"{file_id}.txt")
        calib_path = os.path.join(self.calib_dir, f"{file_id}.txt")

        # 1. Load Image and Calibration Matrix
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size
        p2_matrix = self.parse_calibration(calib_path)
        labels = self.parse_label(label_path)

        # 2. Apply Stage 6 Data Augmentation (Random Horizontal Flip with 50% probability)
        if self.augment and np.random.rand() > 0.5:
            image, labels, p2_matrix = self.apply_geometry_aware_flip(image, labels, p2_matrix, orig_w)

        # 3. Resize and Normalize Image Tensor
        image_resized = image.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
        img_tensor = transforms.ToTensor()(image_resized)
        img_tensor = self.normalize(img_tensor)

        return {
            'image': img_tensor,
            'labels': labels,
            'calib_p2': torch.tensor(p2_matrix, dtype=torch.float32),
            'orig_size': (orig_h, orig_w),
            'file_id': file_id
        }
