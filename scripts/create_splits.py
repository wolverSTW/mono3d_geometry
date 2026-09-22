import os
import glob
import random

def generate_kitti_splits(data_dir="data/kitti", val_ratio=0.2, seed=42):
    random.seed(seed)
    img_dir = os.path.join(data_dir, "image_2")
    
    if not os.path.exists(img_dir):
        print(f"[ERROR] Image directory '{img_dir}' does not exist.")
        return

    image_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    file_ids = [os.path.splitext(os.path.basename(f))[0] for f in image_files]

    if not file_ids:
        print("[WARNING] No images found to generate splits.")
        return

    random.shuffle(file_ids)
    val_size = int(len(file_ids) * val_ratio)
    
    val_ids = file_ids[:val_size]
    train_ids = file_ids[val_size:]

    with open(os.path.join(data_dir, "train.txt"), 'w') as f:
        f.write("\n".join(train_ids))

    with open(os.path.join(data_dir, "val.txt"), 'w') as f:
        f.write("\n".join(val_ids))

    print(f"[SUCCESS] Generated Splits: {len(train_ids)} Train samples, {len(val_ids)} Validation samples.")

if __name__ == "__main__":
    generate_kitti_splits()
