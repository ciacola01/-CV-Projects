"""
All that regard the data and it is imported from all the approaches and the methods sees all the same datas, splits and transformations. 

Functions:
- get_transforms() / get_mean_std(): how to prepare images
- load_dataset() / get_labels(): load and download Oxford-IIITPet
- make_split(): divides trainval in train/val
- save_split() / load_split(): saves indexes on file (reproducibility)
- class_counts(): table "images per breeds in every split"
- get_dataloaders(): train/val/test loader
- get_train_eval_loader(): train without augmentation
"""

import json
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torchvision.models import MobileNet_V3_Small_Weights

from config import (BATCH_SIZE, DATA_DIR, IMAGE_SIZE, NUM_WORKERS, RESIZE_SIZE, SEED, SPLITS_DIR, VAL_FRACTION)
from utils import make_generator, seed_worker

SPLIT_FILE = SPLITS_DIR / "split.json"

""" Transformations """ 
# return transform_eval with short side 256 and central cut 224x224, tensor and normalization ImageNet
# return transform_train same steps with augmentation and random fix
def get_transforms():
    weights = MobileNet_V3_Small_Weights.DEFAULT
    transform_eval = weights.transforms() # resize + crop + normalize

    mean, std = transform_eval.mean, transform_eval.std # Mean and standard deviation from weights

    transform_train = transforms.Compose([
        transforms.Resize(RESIZE_SIZE),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5), # Reflex in 50% of cases
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return transform_train, transform_eval

# Mean and standard deviation to visualize images
def get_mean_std():
    t = MobileNet_V3_Small_Weights.DEFAULT.transforms()
    return list(t.mean), list(t.std)

""" Datasets """
# Load an official partition of Oxfors-IIITPet
def load_dataset(split: str, transform=None, download: bool = True):
    return OxfordIIITPet(
        root=str(DATA_DIR),
        split=split,
        target_types="category",
        transform=transform,
        download=download,
    )

# List of labels for every image in the dataset
def get_labels(dataset) -> list:
    return list(dataset._labels)

""" Stratified split"""

# make split dividing indexes in train and val; stratifying by breeds in the same proportions
def make_split(labels: list, val_fraction: float = VAL_FRACTION, seed: int = SEED):
    indexes = list(range(len(labels)))
    train_idx, val_idx = train_test_split(
        indexes,
        test_size=val_fraction,
        stratify=labels,
        random_state=seed,
        shuffle=True,
    )
    return sorted(train_idx), sorted(val_idx)

""" Save and load the split"""
# Save indexes in splits/split.json
def save_split(train_idx, val_idx, path=SPLIT_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"seed": SEED, "val_fraction": VAL_FRACTION, "train_idx": train_idx, "val_idx": val_idx}, f)

def load_split(path=SPLIT_FILE):
    with open(path) as f:
        d = json.load(f)
        return d["train_idx"], d["val_idx"]

""" Count by breed """
# How many images by breed in train, val and test
def class_counts( class_names, train_labels, val_labels, test_labels) -> pd.DataFrame:
    def count(labels):
        return pd.Series(labels).value_counts().reindex(range(len(class_names)), fill_value=0)

    df = pd.DataFrame({
        "class_id": range(len(class_names)),
        "breed": class_names,
        "train": count(train_labels).values,
        "val": count(val_labels).values,
        "test": count(test_labels).values,
    })

    df["total"] = df[["train", "val", "test"]].sum(axis=1)
    return df

""" Dataloader """
# Returns train_loader, val_loader, test_loader, class_names where Dataset call an image and Dataloader group this in a batch and for the train shuffle all
def get_dataloaders(batch_size: int = BATCH_SIZE):
    t_train, t_eval = get_transforms()
    train_idx, val_idx = load_split()

    full_train_aug = load_dataset("trainval", transform=t_train)
    full_train_eval = load_dataset("trainval", transform=t_eval)
    test_set = load_dataset("test", transform=t_eval)

    train_set = Subset(full_train_aug, train_idx)
    val_set = Subset(full_train_eval, val_idx)

    common = dict(batch_size=batch_size, num_workers=NUM_WORKERS, worker_init_fn=seed_worker)
    train_loader = DataLoader(train_set, shuffle=True, generator=make_generator(), **common)
    val_loader = DataLoader(val_set, shuffle=False, **common)
    test_loader = DataLoader(test_set, shuffle=False, **common)

    return train_loader, val_loader, test_loader, full_train_eval.classes

# Train images with evaluation transformation (without augmentation)
def get_train_eval_loader(batch_size: int = BATCH_SIZE):
    _, t_eval = get_transforms()
    train_idx, _ = load_split()
    subset = Subset(load_dataset("trainval", transform=t_eval), train_idx)
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, worker_init_fn=seed_worker)