"""
Every image goes through the pretrained MobileNetV3-Small without training it and we stop after the pooling ahere each image beomes a vector of 576 numbers saved to the disk
"""

import json
import time

import numpy as np
import torch

from config import FEATURES_DIR, RESULTS_DIR
from data import get_dataloaders, get_train_eval_loader
from model import build_feature_extractor
from utils import get_device, set_seed

# Pass all the images of the loader through the network and returns as numpy arrays
def extract(extractor, loader, device):
    feats, labels = [], []
    for i, (x,y) in enumerate (loader):
        feats.append(extractor(x.to(device)).cpu().numpy())
        labels.append(y.numpy())
        if (i +1) % 20 == 0:
            print(f" batch {i+1}/{len(loader)}")
    return np.concatenate(feats), np.concatenate(labels)

def main() -> None:
    set_seed()
    device = get_device()
    extractor = build_feature_extractor().to(device)

    _, val_loader, test_loader, class_names = get_dataloaders()
    train_loader = get_train_eval_loader()

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    out, times = {}, {}
    for name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        print(f"Extracting {name} features")
        t0 = time.perf_counter()
        X, y = extract(extractor, loader, device)
        times[name] = time.perf_counter() - t0
        out[f"X_{name}"], out[f"y_{name}"] = X, y
        print(f" {name}: features {X.shape}, time {times[name]:.1f}s")

    np.savez(FEATURES_DIR / "features.npz", class_names=np.array(class_names), **out)

    times["total"] = sum(times.values())
    with open(RESULTS_DIR / "approach3_extraction_time.json", "w") as f:
        json.dump(times, f, indent=2)

    print(f"\n Done, Total extraction time: {times['total']:.1f}s")

if __name__ == "__main__":
    main()