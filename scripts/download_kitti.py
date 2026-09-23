import os
import zipfile
import urllib.request
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def is_valid_zip(zip_path):
    if not os.path.exists(zip_path):
        return False
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            return zip_ref.testzip() is None
    except Exception:
        return False

def download_and_setup_kitti():
    base_dir = "data/kitti"
    os.makedirs(base_dir, exist_ok=True)

    # Check if dataset is ALREADY extracted and ready
    required_dirs = [
        os.path.join(base_dir, "training", "image_2"),
        os.path.join(base_dir, "training", "label_2"),
        os.path.join(base_dir, "training", "calib")
    ]
    
    # Also check legacy root level directories just in case
    legacy_dirs = [
        os.path.join(base_dir, "image_2"),
        os.path.join(base_dir, "label_2"),
        os.path.join(base_dir, "calib")
    ]

    has_extracted_data = all(os.path.exists(d) for d in required_dirs) or all(os.path.exists(d) for d in legacy_dirs)

    if has_extracted_data:
        logging.info("[SKIP DOWNLOAD] KITTI dataset directories already exist and are set up. Proceeding to pipeline...")
        return

    # If directories do not exist, handle zip downloads
    files = {
        "data_object_image_2.zip": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip",
        "data_object_label_2.zip": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip",
        "data_object_calib.zip": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip"
    }

    for filename, url in files.items():
        zip_path = os.path.join(base_dir, filename)

        if os.path.exists(zip_path) and not is_valid_zip(zip_path):
            logging.warning(f"[CORRUPTED FILE] Removing corrupted zip: {zip_path}")
            os.remove(zip_path)

        if not os.path.exists(zip_path):
            logging.info(f"[DOWNLOADING] {filename} from {url}...")
            try:
                urllib.request.urlretrieve(url, zip_path)
            except Exception as e:
                logging.error(f"[DOWNLOAD FAILED] {filename}: {e}")
                continue

        if is_valid_zip(zip_path):
            logging.info(f"[EXTRACTING] {filename}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(base_dir)
            logging.info(f"[SUCCESS] Extracted {filename}")

if __name__ == "__main__":
    download_and_setup_kitti()
