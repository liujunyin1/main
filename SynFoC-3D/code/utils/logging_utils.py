# -*- coding: utf-8 -*-
import logging, sys, os

def create_logger(log_path: str):
    logger = logging.getLogger("SynFoC-3D")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fh = logging.FileHandler(log_path); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger
