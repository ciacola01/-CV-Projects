import cv2
import numpy as np
import os

''' 
    Configuration of fixed parameters
'''
INPUT_DIR = "data/original"
OUTPUT_DIR = "data/degraded"
ALPHA = 0.5     # Contrast reduction factor of 50%
BETA = 30       # Brightness offset
SIGMA = 10.0    # Standard deviation for Gaussian noise
SEED = 42       # Fixed seed for reproducibility

'''
    Degrade the image reducing the contrast and shifting the brightness
    Formula: newPixel = alpha * oldPixel + beta
'''

def degrade_image(image):
    degraded = cv2.convertScaleAbs(image, alpha = ALPHA, beta = BETA)

    # Add Gaussian noise
    np.random.seed(SEED)
    noise = np.random.normal(0, SIGMA, degraded.shape).astype(np.float32)

    # Add noise and clip values to valid range [0, 255]
    noisy_degraded = np.clip(degraded.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy_degraded

'''
    The main has these jobs:
    - make directory data/degraded if doesn't exist
    - process all images in the input directory
    - degrade every image
    - save the degraded image
'''

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Process all images in the input directory
    valid_extensions = ('.jpg', '.jpeg', '.png')
    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith(valid_extensions):
            input_path = os.path.join(INPUT_DIR, filename)
            output_path = os.path.join(OUTPUT_DIR, filename)

            # Read image
            img = cv2.imread(input_path)
            if img is None:
                print(f"[ERROR] Error loading {filename}")
                continue

            # Apply degradation
            degraded_img = degrade_image(img)

            # Save
            cv2.imwrite(output_path, degraded_img)
            print(f"Processed and saved: {output_path}")

if __name__ == "__main__":
    main()