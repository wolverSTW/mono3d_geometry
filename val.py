import os
import torch
import yaml
from torch.utils.data import DataLoader
from datasets.kitti_dataset import KITTIDataset
from models.mono3d_network import Mono3DNetwork
from losses.loss3d import MultiTaskLoss3D

def collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    calib_p2 = torch.stack([item['calib_p2'] for item in batch])
    labels = [item['labels'] for item in batch]
    file_ids = [item['file_id'] for item in batch]
    orig_sizes = [item['orig_size'] for item in batch]
    return {'image': images, 'calib_p2': calib_p2, 'labels': labels, 'file_id': file_ids, 'orig_size': orig_sizes}

def main():
    print("=" * 65)
    print("=== [EVALUATION] Running Mono3D Model Checkpoint Validation ===")
    print("=" * 65)

    config_path = "configs/mono3d_config.yaml"
    config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    val_dataset = KITTIDataset(data_dir="data/kitti", config=config, split="val", augment=False)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=4)

    model = Mono3DNetwork(config=config).to(device)
    checkpoint_path = "weights/mono3d_phase4_latest.pth"
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"[CHECKPOINT] Loaded weights from '{checkpoint_path}'.")
    else:
        print(f"[WARNING] Checkpoint '{checkpoint_path}' not found!")

    criterion = MultiTaskLoss3D(num_classes=config.get('model', {}).get('num_classes', 5)).to(device)
    model.eval()

    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            targets = batch['labels']
            predictions = model(images)
            loss_dict = criterion(predictions, targets)
            val_loss += loss_dict['loss'].item()

    avg_val_loss = val_loss / max(len(val_loader), 1)
    print(f"[RESULT] Final Validation Loss across {len(val_dataset)} samples: {avg_val_loss:.6f}")
    print("=" * 65)

if __name__ == "__main__":
    main()
