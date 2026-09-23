import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_eda(data_dir="data/kitti"):
    logging.info("=== STARTING COMPREHENSIVE KITTI EDA ANALYSIS ===")
    
    # Check both split structure (data/kitti/training/label_2) and root structure (data/kitti/label_2)
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
    
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    logging.info(f"[EDA] Total label files found: {len(label_files)}")
    
    class_counts = {}
    for lf in label_files:
        with open(os.path.join(label_dir, lf), 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) > 0:
                    cls_name = parts[0]
                    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

    logging.info("=== CLASS DISTRIBUTION ===")
    for cls_name, count in class_counts.items():
        logging.info(f"  - {cls_name}: {count}")

if __name__ == "__main__":
    run_eda()
