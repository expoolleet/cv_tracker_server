import numpy as np
import numba

FLOAT_TYPE = np.float32

@numba.njit(fastmath=True, cache=True)
def image_preprocessing(im: np.ndarray, use_log : bool = True, window_strength: FLOAT_TYPE = 0.5) -> np.ndarray:

    h, w = im.shape[0], im.shape[1]
    cos_window = np.outer(np.hanning(h), np.hanning(w))
    
    # Blend between windowed and non-windowed
    windowed = im * (window_strength * cos_window + (1 - window_strength))

    if use_log:
        image = np.log(windowed.astype(FLOAT_TYPE) + 1)
    else:
        image = windowed.astype(FLOAT_TYPE)

    mean_val = np.mean(image)
    std_val = np.std(image)
    epsilon = 1e-6
    
    normalized = (image - mean_val) / (std_val + epsilon)
    return normalized.astype(FLOAT_TYPE)
