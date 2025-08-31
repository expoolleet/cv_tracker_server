cimport numpy as np
import numpy as np
np.import_array()

#cython: language_level=3, boundscheck=False, wraparound=False

FLOAT_TYPE = np.float32
ctypedef np.float32_t FLOAT_TYPE_T

cpdef np.ndarray[FLOAT_TYPE_T, ndim=2] make_synthetic_with_regularization(
    int height, 
    int width, 
    np.ndarray[FLOAT_TYPE_T, ndim=2] target_positions, 
    FLOAT_TYPE_T regularization_lambda=0.01, 
    FLOAT_TYPE_T output_sigma_factor=0.05):
    cdef:
        np.ndarray[FLOAT_TYPE_T, ndim=2] y_grid, x_grid, L2, uniform_component, g
        FLOAT_TYPE_T y_j, x_j, sigma
        int i
        int num_targets = target_positions.shape[0]

    y_grid, x_grid = np.mgrid[0:height, 0:width].astype(FLOAT_TYPE)
    
    sigma = np.sqrt(height * width) * output_sigma_factor
    g = np.zeros((height, width), dtype=FLOAT_TYPE)
    
    for i in range(num_targets):
        
        y_j = target_positions[i, 0]
        x_j = target_positions[i, 1]

        L2 = (y_grid - y_j)**2 + (x_grid - x_j)**2
        g += np.exp(-L2 / (2 * sigma**2))
    
    # Normalize
    g = g / (np.max(g) + 1e-6)
    
    # Add regularization to prevent overfitting
    if regularization_lambda > 0:
        # Add small amount of uniform distribution
        uniform_component = regularization_lambda * np.ones_like(g) / (height * width)
        g = (1 - regularization_lambda) * g + uniform_component
    return g


    
                