import sys
import signal
import json
import time
import numpy as np
from src.picamera import Picamera2, init_camera_defaults
from src.frame_memory_share_handler import FrameMemoryShareHandler
from src.file_based_event import FileBasedEvent
from pathlib import Path

"""

    Picam service. Add it in systemd services.

"""

with open(Path(__file__).resolve().parent / "camera_params.json", "r") as f:
    camera_params = json.load(f)

camera_size_main = camera_params["size_main"]
camera_size_lores = camera_params["size_lores"]
camera_resolution = camera_params["res"]
camera_frame_rate = camera_params["frame_rate"]

if camera_resolution == "main":
    camera_size = camera_size_main
    frame_shape = (camera_size[1], camera_size[0], 3)
else:
    camera_size = camera_size_lores
    frame_shape = (int(camera_size[1] * 1.5), camera_size[0])
frame_dtype = np.uint8

sm_handler = FrameMemoryShareHandler(frame_shape, frame_dtype, camera_params["shared_name"])
target_frametime = 1 / camera_frame_rate
empty_frame = np.zeros(frame_shape, dtype=np.uint8)


camera = None
try:
    if not Picamera2.global_camera_info():
        raise RuntimeError("No camera detected")
    camera = init_camera_defaults(
        size_main=camera_size_main, size_lores=camera_size_lores, frame_rate=camera_frame_rate
    )
    camera.start()
except RuntimeError as e:
    print(e)
    sys.exit(1)


controls = camera.capture_metadata()
print(f"Current exposition time: {controls['ExposureTime']} мкс")
print(f"Current analogue gain: {controls['AnalogueGain']}")
print(f"Current digital gain: {controls['DigitalGain']}")

FileBasedEvent.cleanup_all()
camera_closed_event = FileBasedEvent("camera_closed_event")
server_closed_event = FileBasedEvent("server_closed_event")

def sleep(start_time, time_s):
    try:
        elapsed_time = time.time() - start_time
        sleep_time = max(0, time_s - elapsed_time)
        time.sleep(sleep_time) 
    except KeyboardInterrupt:
        raise
    finally:
        return time.time()

def clean_up():
    camera_closed_event.set()
    camera.close()
    try:
        sm_handler.set_frame(empty_frame)
        sm_handler.close()
        sm_handler.unlink()
    except FileNotFoundError:
        pass

def signal_handler(signum, frame):
    print("Capture timeout! Exiting.")
    clean_up()
    sys.exit(1)

def capture_frame(resolution="lores", timeout=3):
    signal.alarm(timeout)
    frame = camera.capture_array(resolution)
    signal.alarm(0)
    return frame

if __name__ == "__main__":
    FileBasedEvent("preview_started_event").wait()
    signal.signal(signal.SIGALRM, signal_handler)
    start_time = time.time()
    print("Camera is working")
    try:  
        while True:
            frame = capture_frame(resolution=camera_resolution)
            if frame is None:
                time.sleep(0.01)
                continue

            if server_closed_event.is_set():
                clean_up()

            sm_handler.set_frame(frame)
            start_time = sleep(start_time, target_frametime)

            if camera_closed_event.is_set():
                camera_closed_event.clear()
    except Exception as e:
        print(f"Exception occured in camera module: {e}")
    finally:
        clean_up()
