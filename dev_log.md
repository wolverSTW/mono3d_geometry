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

## [Phase 4: Quantitative Evaluation & Ablation Framework]
- **Evaluation Pipeline Integration**: Built `evaluate.py` to systematically measure $AP_{3D}$, $AP_{BEV}$, Metric Distance Error (MAE, RMSE), and Computational Efficiency (FPS, Latency).
- **Metric Definitions**:
  - **Detection Accuracy**: $AP_{3D}$ and $AP_{BEV}$ evaluated across Easy, Moderate, and Hard KITTI difficulty levels.
  - **Distance Error**: MAE and RMSE calculated on predicted depth $z$ vs ground truth depth $z$.
  - **Efficiency**: Latency (ms/frame) and FPS measured on target GPU hardware.
- **Comparative Baseline Setup**:
  - Baseline: Standard YOLOv10 2D/3D Regression Head.
  - Geometry-Guided Proposed Method: YOLOv10 + Multi-Scale Feature Map (P3-P5) + Depth Loss Weighting ($2.0$).
- **Ablation Strategy**:
  - Test 1: Pure Backbone vs Multi-scale FPN.
  - Test 2: Standard L1 Depth Loss vs Geometry-guided Loss Weighting.

## [Phase 5: Per-Class Evaluation & CSV/Excel Export Integration]
- **Per-Class Metrics Added**: Enhanced `evaluate.py` to calculate $AP_{3D}$, $AP_{BEV}$, MAE, and RMSE broken down across target classes (`Car`, `Pedestrian`, `Cyclist`, `Truck`, `Bus`).
- **Summary Row Added**: Computes Mean / Overall performance metrics across all evaluation categories.
- **Export Capabilities**: Automatically generates and saves structured report files (`eval_results.csv` and `eval_results.xlsx`) into the `evaluation_results/` directory for thesis documentation and comparative plotting.

## [Phase 5: Per-Class & Summary Metric Exporters]
- **Per-Class Breakdown**: Configured `evaluate.py` to calculate AP3D, APBEV, MAE, and RMSE separately for each KITTI class (`Car`, `Pedestrian`, `Cyclist`, `Truck`, `Bus`).
- **Overall Aggregation**: Included automated Mean calculation row to summarize global performance across all target classes.
- **Export Multi-format**: Integrated automated file generator that outputs structured evaluation sheets to `evaluation_results/eval_results.csv` and multi-tab `evaluation_results/eval_results.xlsx`.

## [Phase 6: GFLOPs & Computational Efficiency Metrics]
- **GFLOPs Tracking**: Added GFLOPs column to measure computational complexity alongside Latency (ms) and Frames Per Second (FPS).
- **Comprehensive Evaluation Matrix**: Updated evaluation tables to display AP3D, APBEV, Distance MAE/RMSE, GFLOPs, and FPS both per-class and as overall averages.

## [Phase 7: Dynamic Automated Evaluation Pipeline]
- **Automated Metric Computation**: Replaced manual hardcoded metrics with dynamic calculation functions (`compute_3d_iou` and `calculate_ap_and_errors`).
- **Dynamic Per-Class Evaluation**: Metrics for $AP_{3D}$, $AP_{BEV}$, MAE, and RMSE are computed dynamically from evaluation predictions and dataset ground truths.

## [Phase 8: Added Model Parameters Metric (Params M)]
- **Params (M) Integration**: Added `Params (M)` column to `evaluate.py` evaluation table to directly align with lightweight M3D benchmarks like LeAD-M3D.
- **Complete Benchmark Alignment**: Evaluation pipeline now tracks `Params (M)`, `GFLOPs`, `Latency (ms)`, and `FPS` along with 3D Detection & BEV Accuracy metrics.
