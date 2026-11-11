import cv2
import numpy as np

'''
    Linear contrast stretching on the luminance channel preserves the color information and change only the luminance. 
    Formula: (L - L_min) / (L_Max - L_min) * 255
'''
def linear_contrast_stretching_luminance(image):
    # L: 0 to 100, a: -128 to 127, b:-128 to 127
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply linear stretching: (L - L_min) / (L_max - L_min) * 255
    l_stretched = cv2.normalize(l, None, 0, 255, cv2.NORM_MINMAX) # The cv2.normalize is used for efficient vectorize computation

    # Merge back and convert to BGR
    lab_stretched = cv2.merge((l_stretched, a, b))
    result = cv2.cvtColor(lab_stretched, cv2.COLOR_LAB2BGR)
    return result


'''
    This equalization works on L channel (the luminance) distributing the intensity base on their frequency. 
    This follows a logaritmic scale that imitates the non-linear response of the human eye
'''
def histogram_equalization_cielab(image):
    # Convert and split the colors to CIE Lab
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply standard histogram equalization on the L channel
    l_equalization = cv2.equalizeHist(l)

    lab_equalization = cv2.merge((l_equalization, a, b))
    result = cv2.cvtColor(lab_equalization, cv2.COLOR_LAB2BGR)
    return result

'''
    The channel V is the max(R, G, B) and it is equalized returning an innatural image because the saturation is unbound
'''
def histogram_equalization_hsv(image):
    # Apply Histogram Equalization on the V channel
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Apply standard histogram equalization on the V channel
    v_equalization = cv2.equalizeHist(v)

    hsv_equalization = cv2.merge((h, s, v_equalization))
    result = cv2.cvtColor(hsv_equalization, cv2.COLOR_HSV2BGR)
    return result

'''
    Applying the bilateral filter reduce the noise and preserve the edges but in the middle you have a flat image
    Then you apply the linear stretching on the denoised image to see more details
'''
def bilateral_denoise_and_stretch(image):
    # d: diameter of each pixel neighborhood (larger -> risk of oversmoothing)
    # sigmaColor: filter sigma in the color space (larger = mixes more distant colors, below 10 has no effect, above 150 is a cartoon)
    # sigmaSpace: filter sigma in the coordinate space (larger = mixes more distant colors)
    denoised_img = cv2.bilateralFilter(image, d=7, sigmaColor=75, sigmaSpace=75) 

    # Apply Linear Stretching on the denoised image to see more details
    result = linear_contrast_stretching_luminance(denoised_img)
    return result

'''
    Apply all the methods and returns a dictionary of results
'''
def apply_methods(degraded_img):
    results = {}
    results['Baseline'] = degraded_img
    results['Linear Stretching'] = linear_contrast_stretching_luminance(degraded_img)
    results['Histogram Equalization CIE Lab'] = histogram_equalization_cielab(degraded_img)
    results['Histogram Equalization HSV'] = histogram_equalization_hsv(degraded_img)
    results['Bilateral + Stretch'] = bilateral_denoise_and_stretch(degraded_img)
    return results