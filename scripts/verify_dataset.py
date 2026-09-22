import sys
import os
import yaml

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.kitti_dataset import KITTIDataset

def verify():
    config_path = "configs/mono3d_config.yaml"
    config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

    print("[DATASET CHECK] Initializing KITTIDataset (split='train')...")
    dataset = KITTIDataset(data_dir="data/kitti", config=config, split="train", augment=True)
    
    total_samples = len(dataset)
    print(f"[DATASET CHECK] Total Loaded Samples: {total_samples}")
    
    if total_samples == 0:
        print("[WARNING] No dataset samples found. Please run scripts/create_dummy_data.py and scripts/create_splits.py first.")
        return

    sample = dataset[0]
    print("\n--- First Sample Verification ---")
    print(f"File ID      : {sample['file_id']}")
    print(f"Image Tensor : {sample['image'].shape} (Channels x Height x Width)")
    print(f"Calib P2     : {sample['calib_p2'].shape}")
    print(f"Labels Count : {len(sample['labels'])}")
    if len(sample['labels']) > 0:
        print(f"Sample Label : {sample['labels'][0]}")

    print("\n[SUCCESS] Dataset Loader verification passed!")

if __name__ == "__main__":
    verify()
