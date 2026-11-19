"""
    The main function is to divide the official division trainval in train/val (80%/20%) with stratified division to guarantee that the proportion of every breed is coherent

    Output:
    - train.csv
    - val.csv
    - test.csv

    Every CSV has two columns: image_path and label (breed's ID)

    Fixed seed guarantees that if you execute multiple times you obtain the same division
"""

import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision.datasets import OxfordIIITPet

# Paths and seed
DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
SPLITS_DIR = os.path.join(os.path.dirname(__file__), "..", "splits")

SEED = 42 # Chosen for the reproducibility

# Set random seed guaranteering that the divisions are reproducible
def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def main():
    set_seed(SEED)
    os.makedirs(SPLITS_DIR, exist_ok=True)

    # Metadata of trainval to obtain the paths and labels
    trainval = OxfordIIITPet(root=DATA_ROOT, split="trainval", download=False)

    # Build a Dataframe with all the trainval's samples
    # _images: paths of the images
    # _labels: labels of breeds (0-36)
    df_trainval = pd.DataFrame({
        "image_path": trainval._images,
        "label": trainval._labels
    })

    print(f"Total samples trainval: {len(df_trainval)}")
    print(f"Number of classes: {df_trainval['label'].nunique()}")

    # Stratified division guarantees that the proportion of every breed in the training and validation set are the same of the first division
    train_df, val_df = train_test_split(
        df_trainval,
        test_size=0.2, #20% vlaidation
        stratify=df_trainval["label"],
        random_state=SEED
    )

    print(f"After the division - train: {len(train_df)}, val: {len(val_df)}")

    # Load the test metadata
    test_set = OxfordIIITPet(root=DATA_ROOT, split="test", download=False)
    test_df = pd.DataFrame({
        "image_path": test_set._images,
        "label": test_set._labels
    })
    print(f"Test samples: {len(test_df)}")

    # Save the CSV
    train_df.to_csv(os.path.join(SPLITS_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(SPLITS_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(SPLITS_DIR, "test.csv"), index=False)
    print(f"\nDivision files saved in: {os.path.abspath(SPLITS_DIR)}")

    # Print the distribution of classes
    print(f"\n=== Samples number for breed in every division ===")
    train_counts = train_df["label"].value_counts().sort_index()
    val_counts = val_df["label"].value_counts().sort_index()
    test_counts = test_df["label"].value_counts().sort_index()

    # Summary table
    summary = pd.DataFrame({
        "breed_id": train_counts.index,
        "train": train_counts.values,
        "val": val_counts.values,
        "test": test_counts.values
    })

    # Add breeds name
    class_names = trainval.classes
    summary["breed_name"] = summary["breed_id"].map(lambda x: class_names[x])
    print(summary.to_string(index=False))

    # Save the table
    summary.to_csv(os.path.join(SPLITS_DIR, "class_distribution.csv"), index=False)

if __name__ == "__main__":
    main()