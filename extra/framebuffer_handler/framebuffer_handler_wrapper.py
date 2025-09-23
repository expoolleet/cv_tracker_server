import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
import subprocess

class DisplayData(ctypes.Structure):
    _fields_ = [
        ("framebuffer_desc", ctypes.c_int),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("pixel_step", ctypes.c_int),
        ("line_length", ctypes.c_int),
        ("screen_size", ctypes.c_int),
        ("mmap_size", ctypes.c_int)
    ]

libfb = ctypes.CDLL("./framebuffer_handler.so")

libfb.rgb888_to_rgb565.argtypes = [ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8]
libfb.rgb888_to_rgb565.restype = ctypes.c_uint16

libfb.draw_rgb_frame.argtypes = [ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS")]
libfb.draw_rgb_frame.restype = None

libfb.draw_bgr_frame.argtypes = [ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS")]
libfb.draw_bgr_frame.restype = None

libfb.get_display_data.argtypes = [ctypes.POINTER(DisplayData)]
libfb.get_display_data.restype = ctypes.c_int

libfb.clean_up.argtypes = [ctypes.c_int]
libfb.clean_up.restype = None

libfb.wait_for_vsync.argtypes = [ctypes.c_int]
libfb.wait_for_vsync.restype = ctypes.c_int

libfb.display_buffer.argtypes = []
libfb.display_buffer.restype = None

def fbcon_bind(enable: bool = True) -> None:
    subprocess.run(["sudo", "sh", "-c", f"echo {int(enable)} > /sys/class/vtconsole/vtcon1/bind"], check=True)

class FrameBufferHandler:
    def __init__(self, vsync: bool = True):
        self.display_data = DisplayData()
        self.vsync = vsync
        ret = libfb.get_display_data(ctypes.byref(self.display_data))
        if ret == -1:
            raise Exception("Can not initialize frame buffer handler")
        fbcon_bind(False)
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            print(exc_type, exc_val)
        self.clean_up()
        
    def display_rgb_frame(self, frame: np.ndarray) -> None:
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        libfb.draw_rgb_frame(frame.shape[1], frame.shape[0], frame)
        libfb.display_buffer()
        if self.vsync:
            libfb.wait_for_vsync(self.display_data.framebuffer_desc)
        
    def display_bgr_frame(self, frame: np.ndarray) -> None:
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        libfb.draw_bgr_frame(frame.shape[1], frame.shape[0], frame)
        libfb.display_buffer()
        if self.vsync:
            libfb.wait_for_vsync(self.display_data.framebuffer_desc)

        
    def clean_up(self) -> None:
        libfb.clean_up(self.display_data.framebuffer_desc)
        fbcon_bind()