"""
    Download the dataset, create the split train/val and verify that
"""

from config import RESULTS_DIR, SEED
from data import (SPLIT_FILE, class_counts, get_dataloaders, get_labels, load_dataset, make_split, save_split)
from utils import set_seed

def main() -> None:
    set_seed(SEED)

    # Download 
    print("Load/Download dataset...")
    trainval = load_dataset("trainval")
    test = load_dataset("test")
    class_names = trainval.classes
    print(f"trainval: {len(trainval)} images | test: {len(test)} images | classes: {len(class_names)}")

    # Stratified split only on the labels
    trainval_labels = get_labels(trainval)
    test_labels = get_labels(test)
    train_idx, val_idx = make_split(trainval_labels)

    # Save split
    save_split(train_idx, val_idx)
    print(f"Split saved in {SPLIT_FILE}: train={len(train_idx)}, val={len(val_idx)}")

    # Table count by breed
    train_labels = [trainval_labels[i] for i in train_idx]
    val_labels = [trainval_labels[i] for i in val_idx]
    df = class_counts(class_names, train_labels, val_labels, test_labels)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "class_counts.csv", index=False)
    print("\nFirst rows of count tables:")
    print(df.head(8).to_string(index=False))
    print(f"\nMin/Max images for every breed in the train: {df.train.min()} / {df.train.max()}")
    print(f"\nMin/Max images for every breed in the val: {df.val.min()} / {df.val.max()}")
    print(f"\nMin/Max images for every breed in the test: {df.test.min()} / {df.test.max()}")

    # Some check
    assert set(train_idx).isdisjoint(val_idx), "LEAKAGE: train and val overwrapped"
    assert len(train_idx) + len(val_idx) == len(trainval), "Miss or much images"
    assert df.val.min() >= 1, "Miss some breed"

    #Check DataLoader
    train_loader, val_loader, test_loader, _ = get_dataloaders()
    images, labels = next(iter(train_loader))
    print(f"\nCheck batch: images {tuple(images.shape)}, labels {tuple(labels.shape)}")
    print(f"Batch in train/val/test loader: {len(train_loader)}/{len(val_loader)}/{len(test_loader)}")
    print(f"Pixel values after normalization: min={images.min():.2f}, max={images.max():.2f}")
    print("\n Data and split verified")

if __name__ == "__main__":
    main()