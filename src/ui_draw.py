import cv2
import numpy as np

line_thickness = 2
crosshair_length = 8
draw_color = (255, 255, 255)

def draw_crosshair(args_list)-> np.ndarray:
    args_list = list(args_list)
    frame = args_list[0]
    center_x = frame.shape[1] // 2
    center_y = frame.shape[0] // 2
    cv2.line(frame, (center_x, center_y - crosshair_length), (center_x, center_y + crosshair_length), draw_color, line_thickness)
    cv2.line(frame, (center_x - crosshair_length, center_y), (center_x + crosshair_length, center_y), draw_color, line_thickness)
    args_list[0] = frame
    return args_list


def draw_roi(args_list) -> np.ndarray:
    args_list = list(args_list)
    frame = args_list[0]
    roi = args_list[1]
    if roi is None:
        return args_list
    color = args_list[2] if len(args_list) > 2 else draw_color
    p1 = (int(roi[0]), int(roi[1]))
    p2 = (int(roi[0] + roi[2]), int(roi[1] + roi[3]))
    cv2.rectangle(frame, p1, p2, color, line_thickness)
    args_list[0] = frame
    return args_list