import sys
import os

# Root Directory ကို Python Path သို့ ပေါင်းထည့်ခြင်း
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

def run_comprehensive_eda(data_dir="data/kitti/label_2", output_dir="outputs/eda"):
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger(log_filename=os.path.join(output_dir, "eda_report.log"))
    logger.info("=== STARTING COMPREHENSIVE KITTI EDA ANALYSIS ===")

    label_files = glob.glob(os.path.join(data_dir, "*.txt"))
    if not label_files:
        logger.error(f"No label files found in '{data_dir}'.")
        return

    logger.info(f"Total Label Files Analyzed: {len(label_files)}")

    raw_records = []
    class_dims = defaultdict(list)
    class_depths = defaultdict(list)

    for filepath in label_files:
        objs = parse_kitti_label_file(filepath)
        for obj in objs:
            cls_type = obj['type']
            h, w, l = obj['dimensions_3d']
            x, y, z = obj['location_3d']
            
            raw_records.append({
                'class': cls_type,
                'truncated': obj['truncated'],
                'occluded': obj['occluded'],
                'alpha': obj['alpha'],
                'height': h,
                'width': w,
                'length': l,
                'depth_z': z
            })
            class_dims[cls_type].append([h, w, l])
            class_depths[cls_type].append(z)

    df = pd.DataFrame(raw_records)

    # 1. Save Structured CSV Document Report
    csv_path = os.path.join(output_dir, "kitti_objects_summary.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"[DOC SAVED] Full Objects DataFrame saved to '{csv_path}'.")

    # 2. Compute Per-Class Aggregated Summary Table & Save JSON Document
    class_summary = {}
    for cls_name, dims in class_dims.items():
        dims_arr = np.array(dims)
        depths_arr = np.array(class_depths[cls_name])
        
        mean_h, mean_w, mean_l = np.mean(dims_arr, axis=0)
        std_h, std_w, std_l = np.std(dims_arr, axis=0)
        
        class_summary[cls_name] = {
            "count": len(dims),
            "mean_dimensions_hwl": [round(mean_h, 2), round(mean_w, 2), round(mean_l, 2)],
            "std_dimensions_hwl": [round(std_h, 2), round(std_w, 2), round(std_l, 2)],
            "mean_depth_z": round(float(np.mean(depths_arr)), 2),
            "min_depth_z": round(float(np.min(depths_arr)), 2),
            "max_depth_z": round(float(np.max(depths_arr)), 2)
        }

    json_path = os.path.join(output_dir, "class_geometry_stats.json")
    with open(json_path, 'w') as f:
        json.dump(class_summary, f, indent=4)
    logger.info(f"[DOC SAVED] Class Geometry Summary JSON saved to '{json_path}'.")

    # 3. Generate and Save Graphical EDA Plots Document (PNG)
    plt.figure(figsize=(12, 5))
    
    # Plot A: Class Distribution
    plt.subplot(1, 2, 1)
    sns.countplot(data=df, x='class', palette='viridis')
    plt.title("Class Distribution")
    plt.xticks(rotation=45)

    # Plot B: Depth Distribution per Class
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, x='class', y='depth_z', palette='magma')
    plt.title("Depth (Z) Distribution by Class")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "kitti_eda_analysis.png")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"[PLOT SAVED] EDA Analysis Chart saved to '{plot_path}'.")

    logger.info("=== COMPREHENSIVE EDA COMPLETED AND DOCUMENTS STORED ===")

if __name__ == "__main__":
    run_comprehensive_eda()
