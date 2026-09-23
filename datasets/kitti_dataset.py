import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class KITTIDataset(Dataset):
    def __init__(self, data_dir="data/kitti", config=None, split="train", augment=False):
        self.data_dir = data_dir
        self.split = split
        self.augment = augment
        self.config = config or {}

        split_file = os.path.join(self.data_dir, f"{split}.txt")
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                self.file_ids = [line.strip() for line in f.readlines() if line.strip()]
        else:
            # Fallback for local testing without split files
            self.file_ids = [f"{i:06d}" for i in range(10)]

        # Determine target directory: 'training' or 'testing'
        subfolder = "testing" if split == "test" else "training"
        self.base_dir = os.path.join(self.data_dir, subfolder)

        self.img_dir = os.path.join(self.base_dir, "image_2")
        self.calib_dir = os.path.join(self.base_dir, "calib")
        self.label_dir = os.path.join(self.base_dir, "label_2")

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        
        # Look for existing image file
        img_path = None
        for ext in ['.png', '.PNG', '.jpg', '.JPG', '.jpeg']:
            possible_path = os.path.join(self.img_dir, f"{file_id}{ext}")
            if os.path.exists(possible_path):
                img_path = possible_path
                break

        # If running locally without downloaded dataset, return Mock Tensor
        if img_path is None or not os.path.exists(img_path):
            image_tensor = torch.zeros((3, 375, 1242), dtype=torch.float32)
            h, w = 375, 1242
            labels = []
        else:
            image = Image.open(img_path).convert('RGB')
            w, h = image.size
            import torchvision.transforms.functional as TF
            image_tensor = TF.to_tensor(image)

            labels = []
            label_path = os.path.join(self.label_dir, f"{file_id}.txt")
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    labels = [line.strip().split() for line in f.readlines()]

        calib_p2 = torch.eye(4)

        return {
            'image': image_tensor,
            'calib_p2': calib_p2,
            'labels': labels,
            'file_id': file_id,
            'orig_size': (h, w)
        }
