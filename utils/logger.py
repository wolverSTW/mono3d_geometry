import os
import logging

def setup_logger(log_filename="training.log", name="mono3d"):
    """
    Sets up logger and automatically creates non-existent log directories.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # Prevent duplicate handlers

    # Determine full log path
    if os.path.isabs(log_filename) or "/" in log_filename:
        log_filepath = log_filename
    else:
        log_filepath = os.path.join("logs", log_filename)

    # Ensure output directory exists before creating FileHandler
    os.makedirs(os.path.dirname(log_filepath), exist_ok=True)

    # Handlers
    f_handler = logging.FileHandler(log_filepath)
    c_handler = logging.StreamHandler()

    f_handler.setLevel(logging.INFO)
    c_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    f_handler.setFormatter(formatter)
    c_handler.setFormatter(formatter)

    logger.addHandler(f_handler)
    logger.addHandler(c_handler)

    return logger
