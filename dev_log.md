# Development Log - Mono3D Geometry Project

## [Phase 1: Visualization & Projection Fixes]
- **Fixed 3D Projection Logic**: Fixed `compute_3d_box_cam2` and `draw_projected_box3d` to accurately project 3D bounding box coordinates onto the image plane using the KITTI $P_2$ calibration matrix.
- **Label Parsing**: Improved label reading logic to handle text labels directly and safely parse class names and numeric parameters.

## [Phase 2: Initial Training & Analysis (Epochs 1 - 10)]
- **Observation**: Training for 10 epochs showed heavy overfitting (Train Loss dropped from `0.0628` to `0.0003`, while Val Loss rose to `0.2697` - `0.4511`).
- **Diagnosis**: Lack of proper depth loss weighting, simple backbone feature extraction, and missing regularization led to memorization.

## [Phase 3: Architecture & Configuration Upgrade]
- **YOLOv10 Backbone Integration**: Updated `configs/mono3d_config.yaml` to utilize YOLOv10 multi-scale feature maps (P3, P4, P5).
- **Head Modules**: Enabled SCDown, C2fCIB, and PSA (Partial Self-Attention) modules to improve spatial and multi-scale contextual awareness.
- **Training Strategy**: Extended training schedule to **50 Epochs**, added `weight_decay=0.0001`, and increased depth loss weighting (`depth: 2.0`) to constrain depth regression errors.
