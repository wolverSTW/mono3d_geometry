import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_parser import load_config
from utils.seed import set_seed
from utils.logger import setup_logger

if __name__ == "__main__":
    logger = setup_logger()
    set_seed(42)
    cfg = load_config("configs/mono3d_config.yaml")
    logger.info("[SUCCESS] Core Utilities and Config System verified successfully!")
