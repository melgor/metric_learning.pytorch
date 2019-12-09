import logging
import sys
from os import path
import torch
from logging.handlers import WatchedFileHandler


def find_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def setup_logger(cfg):
    logger = logging.getLogger(cfg.LOGGER.NAME)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(message)s')
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = WatchedFileHandler(path.join(cfg.SAVE_DIR, "task.log"))
    fh.setFormatter(formatter)
    logger.addHandler(fh)
