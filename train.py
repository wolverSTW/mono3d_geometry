import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader

from datasets.kitti_dataset import KITTIDataset
from models.mono3d_network import Mono3DNetwork
from losses.loss3d import MultiTaskLoss3D

def collate_fn(batch):
    """ Custom collate function to handle variable number of label annotations per image """
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
    print("=== [PHASE 4] Starting Monocular 3D Detection Training Pipeline ===")
    
    # 1. Configuration Setup
    config_path = "configs/mono3d_config.yaml"
    config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using Device: {device}")

    # 2. Datasets and DataLoaders
    train_dataset = KITTIDataset(data_dir="data/kitti", config=config, split="train", augment=True)
    val_dataset = KITTIDataset(data_dir="data/kitti", config=config, split="val", augment=False)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    print(f"[DATA] Train Samples: {len(train_dataset)} | Val Samples: {len(val_dataset)}")

    # 3. Model, Loss Function, Optimizer
    model = Mono3DNetwork(num_classes=5).to(device)
    criterion = MultiTaskLoss3D(num_classes=5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    os.makedirs("weights", exist_ok=True)
    epochs = 2  # Demonstration test epochs

    # 4. Training Loop
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

        # Validation Phase
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

    # Save Checkpoint
    checkpoint_path = "weights/mono3d_phase4_latest.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"[SUCCESS] Model Checkpoint successfully saved at '{checkpoint_path}'.")

if __name__ == "__main__":
    main()
