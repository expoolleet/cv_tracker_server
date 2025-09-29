import os
import signal
import cv2
import time
import threading
import numpy as np
import struct
import queue
import multiprocessing as mp
import json
from pathlib import Path

from src.frame_memory_share_handler import FrameMemoryShareHandler, FrameMemoryShareClient
from src.event import subscribe, post_event
from src.event_types import *
from src.command import Command
from src.zeroconf import register_zeroconf
from src.picamera import setup_camera
from src.server import Server
from src.ffmpeg import FFmpeg, StreamProtocol
from src.uart_transmition import serial_transmit_binary, serial_receive_loop, open_serial, close_serial
from src.fps_counter import FPSCounter
from src.screen_stream import start_ffmpeg_screen_stream, stop_ffmpeg_procces, stream_height, stream_width, stream_lock
from src.pipeline import WrapperPipeline
from src.ui_draw import draw_crosshair, draw_roi, draw_text
from src.data_handler import CSVHandler
from src.video_writer import VideoWriter
from src.opengl_renderer import OpenGLRenderer, ProjectionViewModel
from src.gpio import GPIOHandler
from src.display_messager import DisplayMessager
from src.file_based_event import FileBasedEvent

from tracker.fast_mosse_tracker import FastMosseTracker
from tracker.xor_tracker import XORTracker


if os.getenv("DISPLAY") is None:
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''
    os.environ['QT_QPA_PLATFORM'] = 'eglfs'

base_path = Path(__file__).resolve().parent / "debug"
Path.mkdir(base_path, parents=True, exist_ok=True)
base_path = str(base_path)

with open(Path(__file__).resolve().parent / "camera_params.json", "r") as f:
    camera_params = json.load(f)

class CameraResolution:
    LORES = "lores"
    MAIN = "main"

class Tracker:
    MOSSE = "MOSSE"
    MK = "MK"

net_interface = "wlan0"

KEEP_ASPECT_RATIO = "keep_aspect_ratio"
FREE_ASPECT_RATIO = "free_aspect_ratio"