import os
import sys
import yaml
import torch
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
    
    return {
        'image': images,
        'calib_p2': calib_p2,
        'labels': labels,
        'file_id': file_ids,
        'orig_size': orig_sizes
    }

def main():
    print("=== [PHASE 4] Starting Config-Driven Mono3D Training ===")
    
    config_path = "configs/mono3d_config.yaml"
    config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

    train_cfg = config.get('training', {})
    epochs = train_cfg.get('epochs', 50)
    batch_size = train_cfg.get('batch_size', 16)
    lr = train_cfg.get('lr', 1e-3)
    weight_decay = train_cfg.get('weight_decay', 1e-4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using Device: {device}")

    train_dataset = KITTIDataset(data_dir="data/kitti", config=config, split="train", augment=True)
    val_dataset = KITTIDataset(data_dir="data/kitti", config=config, split="val", augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Config-driven Model initialization
    model = Mono3DNetwork(config=config).to(device)
    criterion = MultiTaskLoss3D(num_classes=config.get('model', {}).get('num_classes', 5)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    os.makedirs("weights", exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            targets = batch['labels']

            optimizer.zero_grad()
            predictions = model(images)
            loss_dict = criterion(predictions, targets)
            
            loss = loss_dict['loss']
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / max(len(train_loader), 1)

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

        print(f"Epoch [{epoch}/{epochs}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    checkpoint_path = "weights/mono3d_phase4_latest.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"[SUCCESS] Model Checkpoint saved at '{checkpoint_path}'.")

if __name__ == "__main__":
    main()
