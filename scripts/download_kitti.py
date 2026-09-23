import os
import zipfile
import urllib.request
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def is_valid_zip(zip_path):
    """Check if the file is a valid zip archive."""
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

    # KITTI URLs or Fallback links
    files = {
        "data_object_image_2.zip": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip",
        "data_object_label_2.zip": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip",
        "data_object_calib.zip": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip"
    }

    for filename, url in files.items():
        zip_path = os.path.join(base_dir, filename)
        
        # If file exists but is corrupted, remove it
        if os.path.exists(zip_path) and not is_valid_zip(zip_path):
            logging.warning(f"[CORRUPTED FILE] Removing invalid zip: {zip_path}")
            os.remove(zip_path)

        # Download if not present
        if not os.path.exists(zip_path):
            logging.info(f"[DOWNLOADING] {filename} from {url}...")
            try:
                urllib.request.urlretrieve(url, zip_path)
            except Exception as e:
                logging.error(f"[DOWNLOAD FAILED] {filename}: {e}")
                continue

        # Extract valid zip
        if is_valid_zip(zip_path):
            logging.info(f"[EXTRACTING] {filename}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(base_dir)
            logging.info(f"[SUCCESS] Extracted {filename}")
        else:
            logging.error(f"[ERROR] Failed to obtain valid zip for {filename}")

if __name__ == "__main__":
    download_and_setup_kitti()
