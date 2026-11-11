import os
import csv
import cv2
import numpy as np
from methods import apply_methods

ORIGINAL_DIR = "data/original"
OUTPUT_DIR = "results/metrics"
CSV_PATH = os.path.join(OUTPUT_DIR, "metrics.csv")


'''
    MSE is the Mean Square Error between two images with same shape and type and is analyzed. 
'''
def calculate_mse(img1, img2):
    img1_float = img1.astype(np.float32)
    img2_float = img2.astype(np.float32)

    # Calculate MSE: mean of squared differences across all pixels and channels
    mse = np.mean((img1_float - img2_float) ** 2)
    return mse

'''
    PSNR computes the Peak Signal-to-Noise Ratio in dB
    Formula: 10 * log10(MAX^2 / MSE)
'''
def calculate_psnr(img1, img2, max_pixel=255.0):

    # We start calculating the MSE and if 0 is infinite and original equals to degraded
    mse = calculate_mse(img1, img2)
    if mse == 0:
        return float('inf')

    # Apply the formula of PSNR
    psnr = 10 * np.log10((max_pixel ** 2)/mse)
    return psnr

'''
    The main function write and save the CSV file
'''

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Preparing CSV file
    csv_rows = []
    csv_rows.append(["Image", "Method", "MSE", "PSNR (dB)"])

    # Verifying if exists and if it has a valid extension
    valid_extensions = ('.jpg', '.png', '.jpeg')
    for filename in os.listdir(ORIGINAL_DIR):
        if filename.lower().endswith(valid_extensions):
            original_path = os.path.join(ORIGINAL_DIR, filename)
            degraded_path = os.path.join("data/degraded", filename)

            original_img = cv2.imread(original_path)
            degraded_img = cv2.imread(degraded_path)

            if original_img is None or degraded_img is None:
                print(f"[SKIP] Could not load {filename}")
                continue

            # Apply methods
            results = apply_methods(degraded_img)

            # Calculate metrics for each method against the original image
            for method_name, result_img in results.items():
                mse_val = calculate_mse(original_img, result_img)
                psnr_val = calculate_psnr(original_img, result_img)

                # Append to CSV rows
                csv_rows.append([filename, method_name, f"{mse_val:.4f}", f"{psnr_val:.2f}"])
                print(f"{filename} | {method_name:25s} | MSE: {mse_val:8.4f} | PSNR: {psnr_val:6.2f} dB")

    # Write on CSV file and save
    with open(CSV_PATH, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)

    print(f"\nMetrics successfully saved to: {CSV_PATH}")


if __name__ == "__main__":
    main()