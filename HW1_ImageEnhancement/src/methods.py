import cv2
import numpy as np

def linear_contrast_stretching_luminance(image):
    # Applies linear contrast stretching on the luminance channel, preserving color information
    # Convert CIELab color space
    # L: 0 to 100, a: -128 to 127, b:-128 to 127
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply linear stretching: (L - L_min) / (L_max - L_min) * 255
    # cv2.normalize used for efficient vectorized computation
    l_stretched = cv2.normalize(l, None, 0, 255, cv2.NORM_MINMAX)

    # Merge back and convert to BGR
    lab_stretched = cv2.merge((l_stretched, a, b))
    result = cv2.cvtColor(lab_stretched, cv2.COLOR_LAB2BGR)
    return result


def histogram_equalization_cielab(image):
    # Apply Histogram Equalization on the L channel of CIELab distributing the intensity based on their frequency. 
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply standard histogram equalization on the L channel
    l_equalization = cv2.equalizeHist(l)

    lab_equalization = cv2.merge((l_equalization, a, b))
    result = cv2.cvtColor(lab_equalization, cv2.COLOR_LAB2BGR)
    return result


def apply_methods(degraded_img):
    # Apply all the methods and returns a dictionary of results
    results = {}
    results['Baseline'] = degraded_img
    results['Linear Stretching'] = linear_contrast_stretching_luminance(degraded_img)
    results['Histogram Equalization'] = histogram_equalization_cielab(degraded_img)
    return results