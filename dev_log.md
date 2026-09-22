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
