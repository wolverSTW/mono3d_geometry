import os
import zipfile
import urllib.request
from tqdm import tqdm
from utils.logger import setup_logger

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def download_and_setup_kitti(target_dir="data/kitti"):
    logger = setup_logger(log_filename="dataset_download.log")
    
    req_dirs = [
        os.path.join(target_dir, "image_2"),
        os.path.join(target_dir, "label_2"),
        os.path.join(target_dir, "calib")
    ]
    
    if all(os.path.exists(d) and len(os.listdir(d)) > 0 for d in req_dirs):
        logger.info(f"[DATASET CHECK] KITTI Dataset verified at '{target_dir}'. Ready for EDA.")
        return

    logger.info(f"[DATASET CHECK] Downloading KITTI Dataset to '{target_dir}'...")
    os.makedirs(target_dir, exist_ok=True)

    urls = {
        "data_object_image_2.zip": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip",
        "data_object_label_2.zip": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip",
        "data_object_calib.zip": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip"
    }

    for filename, url in urls.items():
        zip_path = os.path.join(target_dir, filename)
        folder_name = filename.replace("data_object_", "").replace(".zip", "")
        target_folder = os.path.join(target_dir, folder_name)

        if os.path.exists(target_folder) and len(os.listdir(target_folder)) > 0:
            logger.info(f"[SKIP] '{folder_name}' already extracted.")
            continue

        if not os.path.exists(zip_path):
            logger.info(f"[DOWNLOADING] {filename}...")
            download_url(url, zip_path)

        logger.info(f"[EXTRACTING] {filename}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)

        if os.path.exists(zip_path):
            os.remove(zip_path)

    training_dir = os.path.join(target_dir, "training")
    if os.path.exists(training_dir):
        for subfolder in ["image_2", "label_2", "calib"]:
            src = os.path.join(training_dir, subfolder)
            dst = os.path.join(target_dir, subfolder)
            if os.path.exists(src) and not os.path.exists(dst):
                os.rename(src, dst)

    logger.info("[SUCCESS] KITTI Dataset Download & Extraction Complete!")

if __name__ == "__main__":
    download_and_setup_kitti()
