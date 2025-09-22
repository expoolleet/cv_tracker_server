import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
import subprocess

class DisplayData(ctypes.Structure):
    _fields_ = [
        ("framebuffer_p", ctypes.POINTER(ctypes.c_uint8)),
        ("framebuffer_d", ctypes.c_int),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("x_offset", ctypes.c_int),
        ("y_offset", ctypes.c_int),
        ("pixel_step", ctypes.c_int),
        ("line_length", ctypes.c_int),
        ("screen_size", ctypes.c_int)
    ]

libfb = ctypes.CDLL("./framebuffer_handler.so")

libfb.rgb888_to_rgb565.argtypes = [ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8]
libfb.rgb888_to_rgb565.restype = ctypes.c_uint16

libfb.draw_rgb_frame.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS")]
libfb.draw_rgb_frame.restype = None

libfb.draw_bgr_frame.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS")]
libfb.draw_bgr_frame.restype = None

libfb.get_display_data.argtypes = [ctypes.POINTER(DisplayData)]
libfb.get_display_data.restype = None

libfb.clean_up.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int]
libfb.clean_up.restype = None

def fbcon_bind(enable: bool = True) -> None:
    subprocess.run(["sudo", "sh", "-c", f"echo {int(enable)} > /sys/class/vtconsole/vtcon1/bind"], check=True)

class FrameBufferHandler:
    def __init__(self):
        self.display_data = DisplayData()
        libfb.get_display_data(ctypes.byref(self.display_data))
        fbcon_bind(False)    
        
    def display_rgb_frame(self, frame: np.ndarray) -> None:
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        libfb.draw_rgb_frame(
            self.display_data.framebuffer_p, 
            frame.shape[1], 
            frame.shape[0], 
            self.display_data.width, 
            self.display_data.height, 
            self.display_data.x_offset, 
            self.display_data.y_offset, 
            self.display_data.line_length,
            self.display_data.pixel_step,
            frame)
        
    def display_bgr_frame(self, frame: np.ndarray) -> None:
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        libfb.draw_bgr_frame(
            self.display_data.framebuffer_p, 
            frame.shape[1], 
            frame.shape[0], 
            self.display_data.width, 
            self.display_data.height, 
            self.display_data.x_offset, 
            self.display_data.y_offset, 
            self.display_data.line_length,
            self.display_data.pixel_step,
            frame)   
        
    def clean_up(self) -> None:
        libfb.clean_up(self.display_data.framebuffer_p, self.display_data.framebuffer_d, self.display_data.screen_size)
        fbcon_bind()