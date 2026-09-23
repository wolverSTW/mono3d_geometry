# Development Log: Mono3D Geometry

## Status
- **Status:** FRESH START / REFACTORED
- **Action:** Reset repository and cleaned file structure to rebuild from scratch.

## Phase 1: Core Modular Infrastructure Initialization

### Status
- **Status:** COMPLETED
- **Action:** Created clean modular project structure and baseline utilities.

### Components Built
1. **Config Management:** Created `configs/mono3d_config.yaml` and `utils/config_parser.py`.
2. **Reproducibility:** Added `utils/seed.py` for uniform seeding across PyTorch/NumPy.
3. **Logging & Monitoring:** Built `utils/logger.py` for concurrent file and console logging.
4. **Training Safety:** Built `utils/early_stopping.py` for checkpointing and overfitting protection.

## Phase 2: KITTI Automated Download & Comprehensive EDA Scripts

### Status
- **Status:** READY FOR EXECUTION
- **Action:** Created `scripts/download_kitti.py` and `scripts/eda_kitti.py` for raw dataset analysis.

### Features Built
1. **Automated Downloader:** Handles direct download and directory extraction for KITTI image_2, label_2, and calib.
2. **Comprehensive EDA:** Analyzes overall object counts, occlusion/truncation distributions, and per-class 3D statistics (h, w, l, depth range).

## Phase 2: PyTorch Dataset Loader Module Implementation

### Status
- **Status:** COMPLETED
- **Action:** Created `datasets/kitti_dataset.py` supporting 5 classes and custom class mappings.

### Features Built
1. **Config-Driven Classes:** Supports `nc: 5` (`Car`, `Van`, `Truck`, `Pedestrian`, `Cyclist`).
2. **KITTI Data Parser:** Extracting image tensors, Intrinsic Projection Matrix ($P_2$), 2D Boxes, and 3D Box Parameters ($h, w, l, x, y, z, \alpha$).
3. **Verification Passed:** Executed `scripts/verify_dataset.py` with 100% success on dummy data pipeline.

## Phase 2 (Completed): Full KITTI Data Preprocessing & Pipeline Verification

### Status
- **Status:** COMPLETED (All 8 Pre-processing Stages Implemented)
- **Action:** Upgraded `datasets/kitti_dataset.py` with 3D Geometry-Aware Augmentations and created `scripts/create_splits.py`.

### Implemented Pre-processing Stages
1. **Image & Annotation Loading:** RGB Images, KITTI 3D labels, and Calibration matrices ($P_2$) parser.
2. **Category Selection:** 5 Target Classes (`Car`, `Van`, `Truck`, `Pedestrian`, `Cyclist`) configured in YAML.
3. **Format Conversion:** Converted raw KITTI annotations into PyTorch Tensor dictionary representations.
4. **2D & 3D Box Parameters Extraction:** 2D Box $[x_1, y_1, x_2, y_2]$, 3D Dimensions $[h, w, l]$, Location $[x, y, z]$, and Alpha.
5. **Resizing & Normalization:** Resized images to $[384, 1280]$ and applied ImageNet normalization.
6. **3D Geometry-Aware Data Augmentation:** Implemented `apply_geometry_aware_flip()` handling horizontal flipping with synchronous update to 2D boxes, Observation Angle $\alpha$, Yaw angle, and Calibration Principal Point $c_x$.
7. **Train/Val Split Management:** Created automated split manager (`scripts/create_splits.py`) and split file support (`train.txt`, `val.txt`).
8. **Consistency Verification:** Executed `scripts/verify_dataset.py` verifying batch consistency and label alignment.

## Phase 3 (Completed): Model Architecture (3D Detection First)

### Status
- **Status:** COMPLETED
- **Action:** Created `models/backbone.py`, `models/head3d.py`, and `models/mono3d_network.py` strictly aligned with the proposed framework.

### Architecture Summary
1. **`models/backbone.py`:** Native PyTorch implementation of YOLOv10 backbone components (`SCDown`, `C2fCIB`, `PSA`) outputting multi-scale features ($P_3, P_4, P_5$).
2. **`models/head3d.py`:** Custom multi-scale 3D detection heads predicting 2D Box Offsets, Classification Logits, 3D Dimension Residuals ($h, w, l$), and Orientation ($\alpha$).
3. **`models/mono3d_network.py`:** Full network wrapper coordinating Backbone feature extraction and 3D Head forward passes.
4. **Verification:** Validated multi-scale output tensor dimensions using `scripts/verify_model.py`.

## Phase 4 (Completed): Multi-task 3D Loss Function & Training Pipeline

### Status
- **Status:** COMPLETED
- **Action:** Created `losses/loss3d.py` and `train.py` pipeline.

### Implementation Details
1. **`losses/loss3d.py`:** Multi-task Loss Handler computing Classification Loss, 2D BBox Regression Loss, 3D Dimension Residual Loss, and Multi-bin Orientation Loss.
2. **`train.py`:** PyTorch Training Loop integrated with custom `collate_fn`, train/val loaders, evaluation, and checkpoint saving (`weights/mono3d_phase4_latest.pth`).
3. **Verification:** Verified end-to-end execution of `train.py` with multi-scale loss propagation.

## Phase 5 (Completed): Evaluation & 3D Bounding Box Visualization

### Status
- **Status:** COMPLETED
- **Action:** Created `val.py` and `visualize.py` modules. Generated visualization outputs in `outputs/`.

### Details
1. **`val.py`:** Evaluates trained model checkpoints on validation split and logs multi-task loss metrics.
2. **`visualize.py`:** Renders 2D/3D bounding box predictions on sample KITTI camera images and saves result to `outputs/sample_3d_vis.png`.
3. **Verification:** Validated execution of both evaluation and visualization pipelines.

### Update on Phase 5 Visualization
- Refactored `visualize.py` to highlight 2D Bounding Box and predicted 3D Dimensions ($h \times w \times l$) explicitly.
- Omitted 3D Depth/Wireframe projection until the distance estimation phase is integrated.

### Pipeline Automation Fix
- Included `python scripts/download_kitti.py` into `run_all.sh` to ensure automated KITTI dataset fetching before running verification and EDA.

### Bug Fix & Import Path Resolution
- Fixed `ModuleNotFoundError` in `scripts/download_kitti.py` by appending project root path to `sys.path`.
- Ensured automated download and extraction of KITTI dataset images, labels, and calibration files seamlessly execute via `run_all.sh`.

### Configuration Refactoring
- Decoupled training hyperparameters (Epochs, Batch Size, LR) from `train.py` into `configs/mono3d_config.yaml`.
- Set full GPU training defaults (Epochs: 50, Batch Size: 16).

### Config-driven Model Architecture Update
- Refactored `models/mono3d_network.py` and `configs/mono3d_config.yaml` to configure backbone feature channels, head modules (SCDown, C2fCIB, PSA), and regression heads via YAML.

### EDA Document Generation Refactoring
- Updated `scripts/eda_kitti.py` to auto-export dataset statistics into CSV (`kitti_objects_summary.csv`), JSON (`class_geometry_stats.json`), and PNG graph charts (`kitti_eda_analysis.png`) in `outputs/eda/`.

### Requirements Update
- Updated `requirements.txt` to explicitly include `pandas`, `matplotlib`, `seaborn`, and `tqdm` for automated EDA report and plot generation.

### Logger Bug Fix
- Updated `utils/logger.py` to auto-create missing subdirectories when custom output log paths are provided (fixing `FileNotFoundError` in EDA).

### Backbone Custom Architecture Restoration
- Restored original YOLOv10 Backbone architecture (`SCDown`, `CIB`, `C2fCIB`, `PSA`).
- Added `Mono3DBackbone = YOLOv10Backbone` alias to maintain compatibility with `Mono3DNetwork`.

### Multi-scale Head Restored
- Restored original `Head3D` multi-scale prediction architecture with multi-bin orientation support.
- Created `Mono3DHead = Head3D` alias to ensure full compatibility with `Mono3DNetwork`.

### Backbone Init Parameter Support
- Updated `YOLOv10Backbone.__init__()` to accept `in_channels` and extra `**kwargs` passed by `Mono3DNetwork`.

### Head3D Init Parameter Support
- Updated `Head3D.__init__()` to accept `use_scdown` and additional `**kwargs` passed by `Mono3DNetwork`.

### Fixed Head Channel Mismatch Error
- Explicitly ensured `Head3D` receives `in_channels=[256, 512, 1024]` matching `YOLOv10Backbone` output dimensions.

### Fixed Feature Channels Mismatch in Mono3DNetwork
- Dynamically fetch `out_channels` from `YOLOv10Backbone` (`[256, 512, 1024]`) instead of hardcoded `[128, 256, 512]`.
- Updated `Mono3DNetwork` to pass correct feature channels to `Mono3DHead`.

### Enhanced Real-Time Training Progress Tracking
- Preserved original `train.py` logic, `MultiTaskLoss3D`, and configuration.
- Added `tqdm` progress bars for real-time batch-level training and validation updates.
- Added `flush=True` to stdout prints to bypass terminal buffering delays.

### Training Real-Time Logging & DataLoader Optimization
- Added step-by-step debug checkpoints (`[DEBUG 1/7]` to `[DEBUG 7/7]`) in `train.py` to identify initialization bottlenecks.
- Integrated `tqdm` progress bars for batch-level training and validation updates with `dynamic_ncols=True`.
- Forced `flush=True` on stdout statements and set `PYTHONUNBUFFERED=1` support to prevent terminal output buffering.
- Optimized DataLoader settings (`num_workers=0`, `pin_memory=True`, `non_blocking=True`) to avoid multi-processing hangs and speed up host-to-GPU data transfer on RTX 4090.

### Enhanced Training Pipeline Visual Formatting
- Redesigned `scripts/train.py` console output with structured tags (`[HARDWARE]`, `[DATASET]`, `[MODEL]`, `[PIPELINE]`).
- Improved progress bar layout and added clean epoch summary lines with exact timings and loss metrics.

### Removed Debug Prints and Optimized Data Pipeline
- Removed debug prints (`[DEBUG 1/7]` ... `[DEBUG 7/7]`) from `scripts/train.py`.
- Optimized DataLoader speed for RTX 4090 by configuring `num_workers=8`, `prefetch_factor=2`, and `persistent_workers=True`.

### Synchronized Training Scripts & Optimized GPU Pipeline
- Synchronized `train.py` and `scripts/train.py` to completely eliminate residual debug prints.
- Configured DataLoader with `num_workers=8`, `prefetch_factor=2`, and `persistent_workers=True` to relieve CPU data-loading bottleneck.

### Consolidated Root Execution Architecture
- Moved `train.py` exclusively to the root directory and deleted redundant `scripts/train.py`.
- Updated `run_all.sh` to execute Root `train.py` directly.

### Fixed BatchNorm Single-Sample Batch Bug
- Enabled `drop_last=True` on `train_loader` to prevent `BatchNorm2d` errors when the last training batch contains only 1 sample (`torch.Size([1, C, H, W])`).
- Ensured strict switching to `model.eval()` mode during evaluation.

### Fixed Missing Evaluation Script and Loss Calculation
- Created `val.py` for Step 7 post-training validation and checkpoint assessment.
- Fixed zero-loss bug in `losses/loss3d.py` to ensure proper gradient computation and valid loss metrics.
