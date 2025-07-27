import numpy as np
#import cv2
def bgr_to_gray(im):
    
    B = im[:, :, 0]
    G = im[:, :, 1]
    R = im[:, :, 2]
    
    gray_scale = 0.114 * B + 0.587 * G + 0.299 * R
    
    return gray_scale.astype(np.uint8)

def rgb_to_gray(im):
    
    R = im[:, :, 0]
    G = im[:, :, 1]
    B = im[:, :, 2]
    
    gray_scale = 0.299 * R + 0.587 * G + 0.114 * B
    
    return gray_scale.astype(np.uint8)

# def image_preprocessing1(im):

#     h, w = im.shape
#     cos_window = np.outer(
#         np.hanning(h), np.hanning(w)
#     ) 
#     epsilon = 1e-6
    
#     # Log transform
#     image = np.log(im.astype(np.float32) + 1)
    
#     # Apply window
#     image = image * cos_window
    
#     # Normalize
#     image = (image - np.mean(image)) / (np.std(image) + epsilon)
    
#     # Transform to frequency domain
#     return np.fft.fft2(image)


def image_preprocessing(im, use_log=True, window_strength=0.5):
    h, w = im.shape
    cos_window = np.outer(np.hanning(h), np.hanning(w))
    
    # Blend between windowed and non-windowed
    windowed = im * (window_strength * cos_window + (1 - window_strength))

    if use_log:
        image = np.log(windowed.astype(np.float32) + 1)
    else:
        image = windowed.astype(np.float32)

    mean_val = np.mean(image)
    std_val = np.std(image)
    epsilon = 1e-6
    
    normalized = (image - mean_val) / (std_val + epsilon)
    return normalized
