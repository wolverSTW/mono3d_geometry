import sys
import os

# Add root directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_parser import load_config
from datasets.kitti_dataset import KITTIDataset
from torch.utils.data import DataLoader

if __name__ == "__main__":
    config = load_config("configs/mono3d_config.yaml")
    
    dataset = KITTIDataset(data_dir="data/kitti", config=config, is_train=True)
    print(f"[DATASET CHECK] Total Loaded Samples: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"[SAMPLE CHECK] Image Tensor Shape: {sample['image'].shape}")
        print(f"[SAMPLE CHECK] Calibration P2 Shape: {sample['calib_p2'].shape}")
        print(f"[SAMPLE CHECK] Total Annotations in Sample 0: {len(sample['labels'])}")
        if len(sample['labels']) > 0:
            print(f"[SAMPLE CHECK] First Object Class: {sample['labels'][0]['class_name']} (Index: {sample['labels'][0]['class_idx']})")
        print("[SUCCESS] Dataset Loader verification passed!")
    else:
        print("[WARNING] No dataset samples found. Please run scripts/create_dummy_data.py first.")
