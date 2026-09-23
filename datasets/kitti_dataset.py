import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF

class KITTIDataset(Dataset):
    def __init__(self, data_dir="data/kitti", config=None, split="train", augment=False, img_size=(384, 1248)):
        self.data_dir = data_dir
        self.split = split
        self.augment = augment
        self.config = config or {}
        self.img_size = img_size

        split_file = os.path.join(self.data_dir, f"{split}.txt")
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                self.file_ids = [line.strip() for line in f.readlines() if line.strip()]
        else:
            # Check training folder directly if split text file is missing
            train_img_dir = os.path.join(self.data_dir, "training", "image_2")
            if os.path.exists(train_img_dir):
                all_files = sorted(os.listdir(train_img_dir))
                all_ids = [os.path.splitext(f)[0] for f in all_files if f.endswith(('.png', '.jpg'))]
                if split == "val":
                    self.file_ids = all_ids[::5] # Use 20% for validation
                else:
                    self.file_ids = [i for idx, i in enumerate(all_ids) if idx % 5 != 0]
            else:
                self.file_ids = [f"{i:06d}" for i in range(10)]

        subfolder = "testing" if split == "test" else "training"
        self.base_dir = os.path.join(self.data_dir, subfolder)

        self.img_dir = os.path.join(self.base_dir, "image_2")
        self.calib_dir = os.path.join(self.base_dir, "calib")
        self.label_dir = os.path.join(self.base_dir, "label_2")

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        
        img_path = None
        for ext in ['.png', '.PNG', '.jpg', '.JPG', '.jpeg']:
            possible_path = os.path.join(self.img_dir, f"{file_id}{ext}")
            if os.path.exists(possible_path):
                img_path = possible_path
                break

        if img_path is None or not os.path.exists(img_path):
            image_tensor = torch.zeros((3, self.img_size[0], self.img_size[1]), dtype=torch.float32)
            h, w = self.img_size
            labels = []
        else:
            image = Image.open(img_path).convert('RGB')
            w, h = image.size
            image = image.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
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
