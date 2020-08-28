"""Package dev config."""
import os
import time

import logging


# config
try:
    # inside try to be able to easily run stuff on ipython
    BASE_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..")
except NameError:
    BASE_DIR = "."

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOG_DIR = os.path.join(BASE_DIR, "log")
OUTPUT_PREFIX = os.path.join(BASE_DIR, "output")

PARAMS_FILENAME = "params.yml"
LOG_FILENAME = "log.log"


def configure_logger(output_file: str = LOG_FILENAME):
    """
    Configure logger.
    Defines two handlers, one for console and one for persistent logging.

    Parameters
    ----------
    output_file : str
        Filename to store logs

    Returns
    -------
    logger
    """
    logfmt = "%(asctime)s - %(levelname)s - %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # Create a custom logger
    logger = logging.getLogger(__name__)

    # Create handlers
    # console handler
    c_handler = logging.StreamHandler()
    # file handler
    f_handler = logging.FileHandler(output_file)

    log_handlers = [c_handler, f_handler]

    # Create formatters and add it to handlers
    formatter = logging.Formatter(fmt=logfmt, datefmt=datefmt)

    for handler in log_handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Set level
    logger.setLevel(logging.DEBUG)
    return logger


def generate_file_name() -> str:
    """
    Create unique filename for both the logfile and model related output.
    Returns
    -------
    str
        Unique filename.
    """
    return time.strftime("%Y%m%d-%H%M%S")


output_dir = os.path.join(OUTPUT_PREFIX, "output_" + generate_file_name())
# output_dir = os.path.join(BASE_DIR, "static_output")
if not os.path.exists(output_dir):
    try:
        os.makedirs(output_dir)
    except FileExistsError:
        pass

logger = configure_logger(output_file=os.path.join(output_dir, LOG_FILENAME))
# needed to avoid duplicate logging (tensorflow related issue)
logger.propagate = False
