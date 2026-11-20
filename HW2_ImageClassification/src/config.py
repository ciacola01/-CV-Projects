"""
In this file we keep fixed data, split and evaluation protocol for all the approaches. 
The comparison is fair and with the same values for all. 
"""

from pathlib import Path

# Reproducibility: seed is the entry point for random generator numbers and the split train/val is identical every time
SEED = 42

# Data split: trainval is divided in train (80% on that learn) and val(20% where we choose iperparameters), while test remain the same
VAL_FRACTION = 0.2
NUM_CLASSES = 37 # All the breeds and not the binary cat/dog

# Paths has ROOT that is the main folder of the project and then all the folders use the relative path
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SPLITS_DIR = ROOT / "splits" # train/val indexes
RESULTS_DIR = ROOT / "results" # metrics, tables and figures

# Loading images parameters
RESIZE_SIZE = 256 # short side equals to 256
IMAGE_SIZE = 224 # final cut 224x224, the expected dimension by MobileNetV3
BATCH_SIZE = 32 # how many images the network sees with every step
NUM_WORKERS = 2 # it is reasonable and a good compromise to avoid to do all in the main process