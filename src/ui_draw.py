import cv2
import numpy as np

line_thickness = 2
crosshair_length = 8
draw_color = (255, 255, 255)

def draw_crosshair(args_dict)-> np.ndarray:
    frame = args_dict["frame"]
    center_x = args_dict["crosshair_center"][0]
    center_y = args_dict["crosshair_center"][1]
    cv2.line(frame, (center_x, center_y - crosshair_length), (center_x, center_y + crosshair_length), draw_color, line_thickness)
    cv2.line(frame, (center_x - crosshair_length, center_y), (center_x + crosshair_length, center_y), draw_color, line_thickness)
    args_dict["frame"] = frame
    return args_dict

def draw_roi(args_dict) -> np.ndarray:
    frame = args_dict["frame"]
    if "roi" not in args_dict:
        return args_dict
    roi = args_dict["roi"]
    p1 = (int(roi[0]), int(roi[1]))
    p2 = (int(roi[0] + roi[2]), int(roi[1] + roi[3]))
    cv2.rectangle(frame, p1, p2, draw_color, line_thickness)
    args_dict["frame"] = frame
    return args_dict

def draw_text(args_dict):
    frame = args_dict["frame"]
    if "text_data_list" not in args_dict:
        return args_dict
    text_data_list = args_dict["text_data_list"]
    for text_data in text_data_list:
        text = text_data[0]
        point = text_data[1]
        cv2.putText(frame, text, point, 2, 0.8, draw_color, 2)
    args_dict["frame"] = frame
    return args_dict
    