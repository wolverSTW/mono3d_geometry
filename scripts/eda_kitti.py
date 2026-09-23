import os
import glob
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_eda(data_dir="data/kitti", output_dir="eda_results"):
    logging.info("=== STARTING COMPREHENSIVE KITTI EDA ANALYSIS & VISUALIZATION ===")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Flexible Label Directory Resolution
    possible_label_dirs = [
        os.path.join(data_dir, "training", "label_2"),
        os.path.join(data_dir, "label_2")
    ]
    
    label_dir = None
    for p in possible_label_dirs:
        if os.path.exists(p) and len(os.listdir(p)) > 0:
            label_dir = p
            break

    if label_dir is None:
        logging.error(f"No label files found in any expected directory: {possible_label_dirs}")
        return

    logging.info(f"[EDA] Found label directory at: '{label_dir}'")
    label_files = glob.glob(os.path.join(label_dir, "*.txt"))
    logging.info(f"[EDA] Total label files found: {len(label_files)}")

    # 2. Parse Label Data
    parsed_objects = []
    for lf in label_files:
        file_id = os.path.splitext(os.path.basename(lf))[0]
        with open(lf, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 15:
                    cls_type = parts[0]
                    truncated = float(parts[1])
                    occluded = int(parts[2])
                    alpha = float(parts[3])
                    bbox_2d = [float(x) for x in parts[4:8]]  # x1, y1, x2, y2
                    dimensions = [float(x) for x in parts[8:11]]  # h, w, l
                    location = [float(x) for x in parts[11:14]]  # x, y, z
                    rotation_y = float(parts[14])

                    box_w = bbox_2d[2] - bbox_2d[0]
                    box_h = bbox_2d[3] - bbox_2d[1]

                    parsed_objects.append({
                        'file_id': file_id,
                        'class': cls_type,
                        'truncated': truncated,
                        'occluded': occluded,
                        'alpha': alpha,
                        'bbox_w': box_w,
                        'bbox_h': box_h,
                        'dim_h': dimensions[0],
                        'dim_w': dimensions[1],
                        'dim_l': dimensions[2],
                        'loc_x': location[0],
                        'loc_y': location[1],
                        'loc_z': location[2],
                        'rotation_y': rotation_y
                    })

    df = pd.DataFrame(parsed_objects)

    # 3. Print Summary Tables
    print("\n" + "="*50)
    print("           KITTI EDA SUMMARY TABLE           ")
    print("="*50)
    print(f"Total Images Analyzed : {len(label_files)}")
    print(f"Total 3D Objects     : {len(df)}")
    print("="*50)

    class_stats = df['class'].value_counts().reset_index()
    class_stats.columns = ['Class Category', 'Object Count']
    class_stats['Percentage (%)'] = (class_stats['Object Count'] / len(df) * 100).round(2)
    
    print("\n--- OBJECT CLASS DISTRIBUTION TABLE ---")
    print(class_stats.to_string(index=False))
    print("-" * 50 + "\n")

    # 4. Generate Visual Plots
    sns.set_theme(style="whitegrid")
    
    # Chart 1: Object Class Frequency
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=class_stats, x='Class Category', y='Object Count', palette='viridis')
    plt.title('KITTI Dataset - Class Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45)
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='baseline', fontsize=10, color='black', xytext=(0, 3),
                    textcoords='offset points')
    plt.tight_layout()
    plot_class_path = os.path.join(output_dir, "class_distribution.png")
    plt.savefig(plot_class_path, dpi=300)
    plt.close()

    # Chart 2: 2D Bounding Box Dimensions
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(df['bbox_w'], bins=40, kde=True, color='skyblue')
    plt.title('2D Bounding Box Width Distribution')
    plt.xlabel('Width (pixels)')

    plt.subplot(1, 2, 2)
    sns.histplot(df['bbox_h'], bins=40, kde=True, color='salmon')
    plt.title('2D Bounding Box Height Distribution')
    plt.xlabel('Height (pixels)')

    plt.tight_layout()
    plot_bbox_path = os.path.join(output_dir, "bbox_dimensions.png")
    plt.savefig(plot_bbox_path, dpi=300)
    plt.close()

    # Chart 3: 3D Depth (Z-Distance) Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df[df['loc_z'] < 100]['loc_z'], bins=50, kde=True, color='purple')
    plt.title('3D Object Depth (Z Location) Distribution (< 100m)')
    plt.xlabel('Distance Z (meters)')
    plt.ylabel('Count')
    plt.tight_layout()
    plot_depth_path = os.path.join(output_dir, "depth_z_distribution.png")
    plt.savefig(plot_depth_path, dpi=300)
    plt.close()

    logging.info(f"[SUCCESS] EDA Summary Tables & Visual Charts generated under '{output_dir}/'!")

if __name__ == "__main__":
    run_eda()
