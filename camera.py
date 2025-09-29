import sys
import signal
import json
import time
import numpy as np
from src.picamera import init_camera_defaults
from src.frame_memory_share_handler import FrameMemoryShareHandler
from src.file_based_event import FileBasedEvent

with open("camera_params.json", "r") as f:
    camera_params = json.load(f)

camera_size_main = camera_params["size_main"]
camera_size_lores = camera_params["size_lores"]
camera_resolution = camera_params["res"]
camera_frame_rate = camera_params["frame_rate"]
camera_restart_timeout = 2

if camera_resolution == "main":
    camera_size = camera_size_main
    frame_shape = (camera_size[1], camera_size[0], 3)
else:
    camera_size = camera_size_lores
    frame_shape = (int(camera_size[1] * 1.5), camera_size[0])
frame_dtype = np.uint8
sm_handler = FrameMemoryShareHandler(frame_shape, frame_dtype, camera_params["shared_name"])
target_frametime = 1 / camera_frame_rate

try:
    camera = init_camera_defaults(size_main=camera_size_main, size_lores=camera_size_lores, frame_rate=camera_frame_rate)
    camera.start()
    FileBasedEvent("preview_started_event").wait()
except RuntimeError as e:
    print(e)
    sys.exit(1)

sm_creation_timeout = 1

controls = camera.capture_metadata()
print(f"Current exposition time: {controls['ExposureTime']} мкс")
print(f"Current analogue gain: {controls['AnalogueGain']}")
print(f"Current digital gain: {controls['DigitalGain']}")

FileBasedEvent.cleanup_all()
is_camera_closed_event = FileBasedEvent(camera_params["shared_name"])
is_server_closed_event = FileBasedEvent("is_server_closed_event")

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
    is_camera_closed_event.set()
    camera.close()
    try:
        sm_handler.close()
        sm_handler.unlink()
    except FileNotFoundError:
        pass

def signal_handler(signum, frame):
    print("Capture timeout! Exiting.")
    clean_up()
    time.sleep(camera_restart_timeout)
    sys.exit(1)

def capture_frame(resolution="lores", timeout=2):
    signal.alarm(timeout)
    frame = camera.capture_array(resolution)
    signal.alarm(0)
    return frame

if __name__ == "__main__":
    signal.signal(signal.SIGALRM, signal_handler)
    start_time = time.time()
    print("Camera is working")
    try:  
        while True:
            frame = capture_frame(resolution=camera_resolution)
            if frame is None:
                time.sleep(0.01)
                continue
                    
            if is_server_closed_event.is_set():
                is_server_closed_event.clear()
                time.sleep(sm_creation_timeout)
                sm_handler = FrameMemoryShareHandler(frame_shape, frame_dtype, camera_params["shared_name"])

            sm_handler.set_frame(frame)
            start_time = sleep(start_time, target_frametime)
    except Exception as e:
        print(f"Exception occured in camera module: {e}")
    finally:
        clean_up()