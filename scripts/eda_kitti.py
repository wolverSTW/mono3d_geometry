import sys
import os

# Root Directory ကို Python Path သို့ ပေါင်းထည့်ခြင်း
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import numpy as np
from collections import defaultdict
from utils.logger import setup_logger

def parse_kitti_label_file(file_path):
    objects = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = line.strip().split(' ')
            if not data or len(data) < 15:
                continue
            
            obj = {
                'type': data[0],
                'truncated': float(data[1]),
                'occluded': int(data[2]),
                'alpha': float(data[3]),
                'bbox_2d': [float(x) for x in data[4:8]],
                'dimensions_3d': [float(x) for x in data[8:11]],  # h, w, l
                'location_3d': [float(x) for x in data[11:14]],  # x, y, z (depth)
                'rotation_y': float(data[14])
            }
            objects.append(obj)
    return objects

def run_comprehensive_eda(data_dir="data/kitti/label_2"):
    logger = setup_logger(log_filename="eda_report.log")
    logger.info("=== STARTING COMPREHENSIVE KITTI EDA ANALYSIS ===")

    label_files = glob.glob(os.path.join(data_dir, "*.txt"))
    if not label_files:
        logger.error(f"No label files found in '{data_dir}'. Please run create_dummy_data.py or download_kitti.py first.")
        return

    logger.info(f"Total Label Files Analyzed: {len(label_files)}")

    class_counts = defaultdict(int)
    occlusion_counts = defaultdict(int)
    truncation_stats = []

    class_dims = defaultdict(list)    # h, w, l
    class_depths = defaultdict(list)  # z location
    class_alphas = defaultdict(list)  # observation angle

    total_objects = 0

    for filepath in label_files:
        objs = parse_kitti_label_file(filepath)
        for obj in objs:
            cls_type = obj['type']
            class_counts[cls_type] += 1
            occlusion_counts[obj['occluded']] += 1
            truncation_stats.append(obj['truncated'])

            class_dims[cls_type].append(obj['dimensions_3d'])
            class_depths[cls_type].append(obj['location_3d'][2])
            class_alphas[cls_type].append(obj['alpha'])
            total_objects += 1

    # 1. OVERALL STATISTICAL REPORT
    logger.info("\n" + "="*50)
    logger.info("1. OVERALL DATASET SUMMARY")
    logger.info("="*50)
    logger.info(f"Total Annotated Objects: {total_objects}")
    logger.info("\n--- Class Distribution ---")
    for cls_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_objects) * 100
        logger.info(f"  - {cls_name:<15}: {count:>6} objects ({percentage:>5.2f}%)")

    logger.info("\n--- Occlusion Breakdown ---")
    occlusion_labels = {0: "Fully Visible", 1: "Partly Occluded", 2: "Largely Occluded", 3: "Unknown"}
    for occ_k, count in occlusion_counts.items():
        perc = (count / total_objects) * 100
        logger.info(f"  - {occlusion_labels.get(occ_k, 'Other'):<18}: {count:>6} ({perc:>5.2f}%)")

    # 2. PER-CLASS STATISTICAL ANALYSIS
    logger.info("\n" + "="*50)
    logger.info("2. PER-CLASS 3D GEOMETRY STATISTICS")
    logger.info("="*50)

    for cls_name in sorted(class_counts.keys()):
        dims = np.array(class_dims[cls_name])
        depths = np.array(class_depths[cls_name])
        
        logger.info(f"\n[CLASS: {cls_name}] (Count: {class_counts[cls_name]})")
        
        mean_h, mean_w, mean_l = np.mean(dims, axis=0)
        std_h, std_w, std_l = np.std(dims, axis=0)
        logger.info(f"  3D Dimensions (Height, Width, Length) in meters:")
        logger.info(f"    - Mean : H={mean_h:.2f}m, W={mean_w:.2f}m, L={mean_l:.2f}m")
        logger.info(f"    - Std  : H={std_h:.2f}m, W={std_w:.2f}m, L={std_l:.2f}m")
        
        mean_z, std_z = np.mean(depths), np.std(depths)
        min_z, max_z = np.min(depths), np.max(depths)
        logger.info(f"  Depth Distribution (Z-Distance):")
        logger.info(f"    - Range: {min_z:.2f}m to {max_z:.2f}m | Mean: {mean_z:.2f}m (±{std_z:.2f}m)")

    logger.info("\n=== COMPREHENSIVE EDA COMPLETED ===")

if __name__ == "__main__":
    run_comprehensive_eda()
