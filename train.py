import os
import sys
import yaml
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

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
    print("=" * 75, flush=True)
    print("=== [PHASE 4] Executing Config-Driven Mono3D Model Training ===", flush=True)
    print("=" * 75, flush=True)
    
    config_path = "configs/mono3d_config.yaml"
    config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"[CONFIG] Loaded configuration from '{config_path}'.", flush=True)

    train_cfg = config.get('training', {})
    model_cfg = config.get('model', {})
    head_cfg = model_cfg.get('head', {})
    
    # Check for override env variable or default from config
    epochs = int(os.environ.get("EPOCHS", train_cfg.get('epochs', 5)))
    batch_size = train_cfg.get('batch_size', 16)
    lr = float(train_cfg.get('lr', 1e-3))
    weight_decay = float(train_cfg.get('weight_decay', 1e-4))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[HARDWARE] Compute Accelerator: {device.type.upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    print("\n[DATASET] Loading KITTI Mono3D Dataset...", flush=True)
    t0 = time.time()
    train_dataset = KITTIDataset(data_dir="data/kitti", config=config, split="train", augment=True)
    val_dataset = KITTIDataset(data_dir="data/kitti", config=config, split="val", augment=False)
    print(f"[DATASET] Loaded {len(train_dataset)} Train samples, {len(val_dataset)} Val samples ({time.time()-t0:.2f}s).", flush=True)

    num_workers = min(8, os.cpu_count() or 4)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    print(f"[PIPELINE] DataLoaders initialized with Batch Size={batch_size}, Workers={num_workers}.", flush=True)

    print("\n[MODEL] Initializing Mono3D Model Architecture & Loss Heads...", flush=True)
    model = Mono3DNetwork(config=config).to(device)
    
    # Updated MultiTaskLoss3D initialization with configs
    loss_weights = train_cfg.get('loss_weights', None)
    criterion = MultiTaskLoss3D(
        loss_weights=loss_weights,
        num_classes=model_cfg.get('num_classes', 5),
        num_bins=head_cfg.get('num_bins', 12)
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Cosine Annealing Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    os.makedirs("weights", exist_ok=True)
    
    best_val_loss = float('inf')
    best_checkpoint_path = "weights/mono3d_best.pth"
    latest_checkpoint_path = "weights/mono3d_phase4_latest.pth"

    print("\n" + "-" * 75, flush=True)
    print(f"  STARTING MODEL TRAINING PIPELINE ({epochs} EPOCHS)", flush=True)
    print("-" * 75, flush=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        epoch_start = time.time()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        pbar = tqdm(
            train_loader, 
            desc=f"Epoch [{epoch:02d}/{epochs:02d}] Train", 
            leave=True, 
            dynamic_ncols=True,
            bar_format="{l_bar}{bar:25}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        )
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(device, non_blocking=True)
            targets = batch['labels']

            # Move target tensors to GPU/Device if applicable
            if isinstance(targets, dict):
                targets = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in targets.items()}

            optimizer.zero_grad()
            predictions = model(images)
            loss_dict = criterion(predictions, targets)
            
            loss = loss_dict['loss']
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
            # Detailed loss tracking in progress bar
            depth_l = loss_dict.get('loss_depth', torch.tensor(0.0)).item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'depth_l': f"{depth_l:.4f}", 'lr': f"{current_lr:.6f}"})

        avg_train_loss = train_loss / max(len(train_loader), 1)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device, non_blocking=True)
                targets = batch['labels']
                if isinstance(targets, dict):
                    targets = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in targets.items()}
                
                predictions = model(images)
                loss_dict = criterion(predictions, targets)
                val_loss += loss_dict['loss'].item()

        avg_val_loss = val_loss / max(len(val_loader), 1)
        epoch_time = time.time() - epoch_start
        
        # Scheduler Step
        scheduler.step()

        # Save Best Model Checkpoint
        saved_best_tag = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_checkpoint_path)
            saved_best_tag = f" -> [SAVED BEST: {best_checkpoint_path}]"

        print(f" [SUMMARY] Epoch {epoch:02d}/{epochs:02d} | LR: {current_lr:.6f} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} ({epoch_time:.1f}s){saved_best_tag}", flush=True)
        print("-" * 75, flush=True)

    torch.save(model.state_dict(), latest_checkpoint_path)
    print(f"\n[SUCCESS] Final Checkpoint saved at '{latest_checkpoint_path}'.", flush=True)
    print(f"[SUCCESS] Best Checkpoint saved at '{best_checkpoint_path}' (Val Loss: {best_val_loss:.4f}).", flush=True)

if __name__ == "__main__":
    main()