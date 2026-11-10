import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from methods import apply_methods

DEGRADED_DIR = "data/degraded"
ORIGINAL_DIR = "data/original"
OUTPUT_DIR = "results/images"

def plot_comparison(original_img, degraded_img, results, filename):
    # Plots the original, the degraded and processed images side by side
    # Convert to RGB for matplotlib
    original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    # Columns: original, degraded and number of methods
    num_columns = 1 + len(results)

    fig, axes = plt.subplots(1, num_columns, figsize=(5 * num_columns, 5))

    # Plot original
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original", fontsize=14)
    axes[0].axis('off')

    # Plot methods
    for i, (method_name, img) in enumerate(results.items()):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i+1].imshow(img_rgb)
        title = f"{method_name} (Degraded)" if method_name == 'Baseline' else method_name
        axes[i+1].set_title(title, fontsize=14)
        axes[i+1].axis('off')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"comparison_{filename}")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Process each image
    valid_extensions = ('.jpg', '.jpeg', '.png')
    for filename in os.listdir(ORIGINAL_DIR):
        if filename.lower().endswith(valid_extensions):
            originalPath = os.path.join(ORIGINAL_DIR, filename)
            degradedPath = os.path.join(DEGRADED_DIR, filename)

            original_img = cv2.imread(originalPath)
            degraded_img = cv2.imread(degradedPath)

            if original_img is None or degraded_img is None:
                print(f"[SKIP] Could not load {filename}")
                continue

            # Apply methods
            results = apply_methods(degraded_img)

            # Plot and save
            base_name = os.path.splitext(filename)[0]
            plot_comparison(original_img, degraded_img, results, base_name)


if __name__ == "__main__":
    main()