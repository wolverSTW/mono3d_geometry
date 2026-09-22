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
    Handles image loading, label parsing, class mapping, and 3D bounding box geometry extraction.
    """
    def __init__(self, data_dir="data/kitti", config=None, is_train=True):
        self.data_dir = data_dir
        self.config = config
        self.is_train = is_train
        
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
        
        # Collect image file list
        self.image_files = sorted(glob.glob(os.path.join(self.img_dir, "*.png")))
        
        # PyTorch Image Preprocessing Transforms
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size[0], self.img_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def parse_calibration(self, calib_path):
        """
        Parses P2 intrinsic calibration matrix from KITTI calibration file.
        """
        calib = {}
        if not os.path.exists(calib_path):
            # Return identity-like default if missing
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
        Parses annotations and maps raw classes to configured 5 classes.
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
                # Apply class mapping or check if class is target
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

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        file_id = os.path.splitext(os.path.basename(img_path))[0]
        
        label_path = os.path.join(self.label_dir, f"{file_id}.txt")
        calib_path = os.path.join(self.calib_dir, f"{file_id}.txt")

        # 1. Load and Transform Image
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size
        img_tensor = self.transform(image)

        # 2. Parse Calibration and Labels
        p2_matrix = self.parse_calibration(calib_path)
        labels = self.parse_label(label_path)

        return {
            'image': img_tensor,
            'labels': labels,
            'calib_p2': torch.tensor(p2_matrix, dtype=torch.float32),
            'orig_size': (orig_h, orig_w),
            'file_id': file_id
        }
