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
