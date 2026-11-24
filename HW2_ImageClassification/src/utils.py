"""
It contains the functions used from all the scripts
- set_seed() makes the results repeatable
- seed_worker() and make_generator makes the reproducibility
- get_device() call the CPU on Mac
- collect_versions() contains the packages versions of PyTorch and TorchVision
"""

import json
import os
import platform
import random
import sys

import numpy as np
import torch

from config import RESULTS_DIR, SEED

# Set the seed makes the output always the same to obtain the same sequence 
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Ask to PyTorch to use always deterministic algorithms and warn if some operation isn't deterministic without blocking the program
    torch.use_deterministic_algorithms(True, warn_only=True)

    #PYTHONHASHSEED influences the hasing in Python if setted before the run to the interpreter
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

# Fixes the seed in the worker processes of DataLoader
# If NUM_WORKERS>0 DataLoader creates separates processes with each one with an own random generator to align the global seed
def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# Random generator for the DataLoader 
def make_generator(seed: int = SEED) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g

# Return the device because PyTorch can execute on CPU or GPU Nvidia like CUDA
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# Return the versions of the environment in a dictionary
def collect_versions() -> dict:
    import sklearn
    import torchvision

    return{
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "seed": SEED,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "device": str(get_device()),
    }

# Save versions in results/environment.json
def save_versions(path=None) -> dict:
    path = path or RESULTS_DIR/"environment.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    info = collect_versions()
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
    return info