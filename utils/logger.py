import logging
import os
import sys

def setup_logger(log_dir="logs", log_filename="training.log"):
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(log_dir, log_filename)
    
    logger = logging.getLogger("Mono3D")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    c_handler = logging.StreamHandler(sys.stdout)
    f_handler = logging.FileHandler(log_filepath)

    format_str = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    c_handler.setFormatter(format_str)
    f_handler.setFormatter(format_str)

    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger
