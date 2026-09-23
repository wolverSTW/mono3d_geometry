#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "========================================================="
echo " Starting Monocular 3D Detection Pipeline"
echo "========================================================="

# 1. Environment & Dependencies Setup
echo -e "\n[STEP 1/7] Installing Required Packages..."
pip install -r requirements.txt || pip install opencv-python pyyaml torch matplotlib pandas seaborn

# 2. Dataset Download / Preparation
echo -e "\n[STEP 2/7] Downloading / Checking KITTI Dataset..."
python scripts/download_kitti.py

# 3. Data Verification & Analysis (EDA)
echo -e "\n[STEP 3/7] Running KITTI Dataset Verification & EDA..."
python scripts/verify_dataset.py
python scripts/eda_kitti.py

# 4. Create Data Splits
echo -e "\n[STEP 4/7] Generating Train/Val Data Splits..."
python scripts/create_splits.py

# 5. Model Setup Verification
echo -e "\n[STEP 5/7] Verifying Mono3D Model Architecture..."
python scripts/verify_model.py

# 6. Model Training Pipeline
echo -e "\n[STEP 6/7] Executing Model Training Pipeline..."
python train.py

# 7. Evaluation & Visualization
echo -e "\n[STEP 7/7] Running Evaluation & Visualization..."
python val.py
python visualize.py

echo -e "\n========================================================="
echo " Complete Pipeline Executed Successfully!"
echo " Check 'outputs/sample_3d_vis.png' for visualization result."
echo "========================================================="
