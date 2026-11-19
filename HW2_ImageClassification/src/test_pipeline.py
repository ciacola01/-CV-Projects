"""
    Function: verify that the data pipeline works correctly
    After the execution we'll have
    - the batch form of train/val/test
    - Pixel's value interval (-2 and 2 after the normalization)
    - Sample image saved in results/
"""

import os
import torch
from dataset import get_dataloaders
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Create DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=8, num_workers=0)

    print("=== Dataloader infos ===")
    print(f"Number of train batch: {len(train_loader)}")
    print(f"Number of val batch: {len(val_loader)}")
    print(f"Number of test batch: {len(test_loader)}")

    # Verify the training set picking up a batch
    images, labels = next(iter(train_loader))
    print(f"\nTrain batch form: {images.shape}") # It will print [8, 3, 224, 224]
    print(f"\nTrain labels batch form: {labels.shape}") # it will be [8]
    print(f"Interval pixel values: [{images.min():.2f}, {images.max():.2f}]")
    print(f"Labels interval: [{labels.min()}, {labels.max()}]")

    # Verify correctness about dimensions
    assert images.shape == (8, 3, 224, 224), f"Wrong form: {images.shape}"
    assert images.max() <= 3.0 and images.min() >= -3.0, "Pixel values out of the boundaries"

    # Visualize an image: we have to denormalize to visualize correfctly with pixel = pixel * std + mean
    img = images[0].clone()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img * std + mean
    img = img.clamp(0, 1)
    img = img.permute(1, 2, 0).numpy() # from [C, H, W] to [H, W, C] for matplotlib

    # Save image
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.title(f"Label: {labels[0].item()}")
    plt.axis("off")
    plt.savefig(os.path.join(results_dir, "sample_image.png"), dpi=100, bbox_inches="tight")
    print(f"\nImage saved in {results_dir}")

    print("\n Pipeline verified with success")

if __name__ == "__main__":
    main()