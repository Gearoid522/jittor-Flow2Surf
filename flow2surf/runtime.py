"""Process-level logging, reproducibility, and run metadata."""

import logging
import os
import random
import subprocess
from datetime import datetime

import jittor as jt
import numpy as np


def configure_logging(log_file=None):
    """Configure the Flow2Surf logger for one command-line process."""
    logger = logging.getLogger("flow2surf")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file is not None:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def seed_runtime(seed):
    """Seed Python, NumPy, and Jittor with one integer seed."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    jt.set_global_seed(seed)


def git_revision(repo=None):
    """Return the short Git revision, marked dirty when the worktree differs."""
    repo = repo or os.getcwd()
    try:
        commit = (
            subprocess.check_output(
                ["git", "-C", repo, "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        status = subprocess.check_output(
            ["git", "-C", repo, "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        )
        return f"{commit}{'-dirty' if status.strip() else ''}"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def run_stamp():
    """Return a sortable local timestamp for run artifacts."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
