import numpy as np

def create_set_of_shifted_templates_from_an_image(im, x_crop_min, x_crop_max, y_crop_min, y_crop_max, xy_translations):
    shifted_templates = []
    for dx in xy_translations:
        for dy in xy_translations:
            x_start = x_crop_min + dx
            y_start = y_crop_min + dy
            x_end = x_crop_max + dx
            y_end = y_crop_max + dy
            
            shifted_template = im[y_start:y_end, x_start:x_end]
            shifted_templates.append(shifted_template)
    return shifted_templates

def rotate_and_shift_coordinates(theta, tx, ty, x, y):
    T = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ], dtype=float)
    
    T_inv = np.array([
        [1, 0, -tx],
        [0, 1, -ty],
        [0, 0, 1]
    ], dtype=float)
    
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ],dtype=float)
    
    M = T @ R @ T_inv 
    point = np.array([x, y, 1], dtype=float)    
    uv = M @ point
    return np.array((uv[0], uv[1]), dtype=int)

def rotate_image(im, angle):
    rotated_image = np.zeros(im.shape)
    theta = np.radians(angle)
    cx = im.shape[1] // 2
    cy = im.shape[0] // 2
    for y in range(im.shape[0]):
        for x in range(im.shape[1]):
            uv = rotate_and_shift_coordinates(-theta, cx, cy, x, y)
            if 0 <= uv[1] < im.shape[0] and 0 <= uv[0] < im.shape[1]:
                rotated_image[y, x] = im[uv[1], uv[0]]
    return rotated_image

def create_set_of_rotated_templates_from_an_image(template, angles, iterations):
    if iterations == 0:
        return []
    rotated_images = []
    angle = angles[0]
    h = (angles[1] - angles[0]) / (iterations - 1 if iterations > 1 else iterations)
    
    for iteration in range(iterations):
        rotated_images.append(rotate_image(template, angle))
        angle += h
    return rotated_images

def scale_image(im, scale, tx, ty):

    scaled_image = np.zeros(im.shape)
    
    T = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ], dtype=float)
    

    S = np.array([
        [scale, 0, 0],
        [0, scale, 0],
        [0, 0, 1]
    ], dtype=float)
    
    
    T_inv = np.array([
        [1, 0, -tx],
        [0, 1, -ty],
        [0, 0, 1]
    ], dtype=float)
    

    H, W = im.shape
    for y in range(H):
        for x in range(W):
            point = np.array([x, y, 1], dtype=float)
            uv = T @ S @ T_inv @ point
            uv = (int(uv[0]), int(uv[1]))
            if 0 <= uv[1] < im.shape[0] and 0 <= uv[0] < im.shape[1]:
                scaled_image[y, x] = im[uv[1], uv[0]]
    return scaled_image
    
def create_set_of_scaled_templates_from_an_image(template, scales, iterations):
    if iterations == 0:
        return []
    scaled_images = []
    scale = scales[0]

    h = (scales[1] - scales[0]) / (iterations - 1 if iterations > 1 else iterations)
    cx = template.shape[1] // 2
    cy = template.shape[0] // 2
    for iteration in range(iterations):
        scaled_images.append(scale_image(template, scale, cx, cy))
        scale += h
    return scaled_images