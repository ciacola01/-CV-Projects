"""
After the preparation of the data we check all:
- printing how many batch there are in train/val /test
- checking the form and the interval of the pixel's values of a batch
- save results/sample_images.png
    - upper row: train images with augmentation with breed's name
    - below row: val images without augmentation with breed's name
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import IMAGE_SIZE, RESULTS_DIR, SEED
from data import get_dataloaders, get_mean_std
from utils import denormalize, set_seed

N_SHOW = 4 # images per row

def main() -> None:
    set_seed(SEED)
    train_loader, val_loader, test_loader, class_names = get_dataloaders(batch_size=8)

    print("=== Dataloader informations ===")
    print(f"Batch in the train: {len(train_loader)}")
    print(f"Batch in the val: {len(val_loader)}")
    print(f"Batch in the test: {len(test_loader)}")

    train_images, train_labels = next(iter(train_loader))
    val_images, val_labels = next(iter(val_loader))

    print(f"\n Batch form train: {tuple(train_images.shape)}")
    print(f"Pixel interval (train): [{train_images.min():.2f}, {train_images.max():.2f}]")
    print(f"labels interval: [{train_labels.min()}, {train_labels.max()}]")

    # Check after normalization
    assert tuple(train_images.shape) == (8, 3, IMAGE_SIZE, IMAGE_SIZE), "Wrong shape"
    assert train_images.min() >= -3.0 and train_images.max() <= 3.0, "Pixel out of the range"
    assert 0 <= int(train_labels.min()) and int(train_labels.max()) <= 36, "Labels out of the range"

    # Figure row in train/val
    mean, std = get_mean_std()
    fig, axes = plt.subplots(2, N_SHOW, figsize=(3 * N_SHOW, 6.5))
    for row, (images, labels, name) in enumerate ([(train_images, train_labels, "train (augmentation)"), (val_images, val_labels, "val (without augmentation)")]):
        for col in range(N_SHOW):
            image = denormalize(images[col], mean, std).permute(1, 2, 0).numpy()
            ax = axes[row, col]
            ax.imshow(image)
            ax.set_title(class_names[int(labels[col])], fontsize=9)
            ax.axis("off")
            axes[row, 0].text(-0.05, 0.5, name, transform=axes[row, 0].transAxes, rotation=90, va="center", ha="right", fontsize=9)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "sample_images.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi = 100, bbox_inches="tight")
    print(f"\nSaved figure in {out_path}")
    print("Pipeline verified")

if __name__ == "__main__":
    main()