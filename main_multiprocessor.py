import os
import cv2
import time
import threading
import numpy as np
import struct
import queue
import multiprocessing as mp
from src.frame_memory_share_handler import FrameMemoryShareHandler, FrameMemoryShareClient

from src.event import subscribe
from src.event_types import (
    UPDATE_TRACKING, 
    STOP_TRACKING,
    START_STREAM_FOR_CLIENT,
    STOP_STREAM_FOR_CLIENT,
    SEND_CFS,
    REQUEST_TRACKING,
    SHOW_CAMERA_PREVIEW,
    STOP_CAMERA_PREVIEW,
    TOGGLE_ROI,
    TOGGLE_CROSSHAIR,
    CHANGE_FRAME_BORDERS )
from src.command import Command
from src.zeroconf import register_zeroconf
from src.picamera import setup_camera
from src.server import Server
from src.ffmpeg import FFmpeg, StreamProtocol
from src.uart_transmition import serial_transmit_binary, serial_receive_loop, open_serial, close_serial
from src.fps_counter import FPSCounter
from src.screen_stream import start_ffmpeg_screen_stream, stop_ffmpeg_procces, stream_height, stream_width, stream_lock
from src.pipeline import WrapperPipeline
from src.ui_draw import draw_crosshair, draw_roi
from tracker.fast_mosse_tracker import FastMosseTracker

os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''
os.environ['QT_QPA_PLATFORM'] = 'eglfs'

class Tracker:
    MOSSE = "MOSSE"
    MK = "MK"

net_interface = "wlan0"

window_name = 'Fullscreen PiCamera2 Feed'
camera_preview_thread = None
camera_preview_lock = threading.Lock()
camera_preview_clients = {}
frame_top_border, frame_bottom_border, frame_left_border, frame_right_border = (0, 0, 0, 0)
message_queue = mp.Queue()
#event = threading.Event() 

KEEP_ASPECT_RATIO = "keep_aspect_ratio"
FREE_ASPECT_RATIO = "free_aspect_ratio"

enable_uart = False

enable_preview = True

tracker_requested = False

is_streaming_screen = False
if is_streaming_screen:
    start_ffmpeg_screen_stream()
    
latest_frame_lock = threading.Lock()
latest_frame = None

camera_framerate = 60
camera_frametime = 1/camera_framerate
camera_size_main = (640, 480)
camera_size_lores = (448, 360)

camera_preview_framerate = 15


### Tracker default parameters ###
#--------------------------------#

manager = mp.Manager()
data_dict = manager.dict()
tracker_dict = manager.dict()

#№tracker = None
data_dict["tracker_initialized"] = False
tracker_size = camera_size_lores
tracking_frame_size = camera_size_lores
defualt_roi_size = 64
current_roi_size = defualt_roi_size
data_dict["current_roi"] = (int(tracker_size[0] // 2 - defualt_roi_size // 2), int(tracker_size[1] // 2 - defualt_roi_size // 2), defualt_roi_size, defualt_roi_size)
data_dict["max_skipped_frames"] = 1
data_dict["using_kalman"] = False 
data_dict["training_images_count"] = 9 
data_dict["alpha_smoothing"] = 0.9
data_dict["max_corr"] = 0.3
data_dict["sigma_factor"] = 0.05
# ------------------------------#

ROI_ZEROS = (0, 0, 0, 0)

uart_coefs = [0, 0, 0]
uart_lock = threading.Lock()

stream_protocol = StreamProtocol.UDP
server_port = 8000
ffmpeg_port = 8001
server = Server(net_interface, server_port)

pipeline = WrapperPipeline()
pipeline.register_operation(draw_crosshair, draw_crosshair.__name__, default_enabled=True)
pipeline.register_operation(draw_roi, draw_roi.__name__, default_enabled=True)

roi_lock = mp.Lock()

cam = setup_camera(main={"format": "BGR888", "size": camera_size_main}, lores={"format": "YUV420", "size": camera_size_lores}, fps=camera_framerate)
cam.set_controls({"AeEnable": True})
cam.start()
controls = cam.capture_metadata()
print(f"Current exposition time: {controls['ExposureTime']} мкс")
print(f"Current analogue gain: {controls['AnalogueGain']}")
print(f"Current digital gain: {controls['DigitalGain']}")

frame_shape = (int(camera_size_lores[1] * 1.5), camera_size_lores[0])
frame_dtype = np.uint8

stop_event = mp.Event()
wait_first_frame_event = mp.Event()


current_tracker = Tracker.MK

def reset_roi():
    data_dict["current_roi"] = (int(tracker_size[0] // 2 - current_roi_size // 2), int(tracker_size[1] // 2 - current_roi_size // 2), current_roi_size, current_roi_size)


# https://docs.python.org/3/library/struct.html
def prepare_uart_data(data_to_send):
    global current_roi
    data = data_to_send.copy()
    if current_roi is not None:
        dx = tracker_size[0] // 2 - (current_roi[0] + current_roi[2] // 2)
        dy = tracker_size[1] // 2 - (current_roi[1] + current_roi[3] // 2)  
    else:
        dx = 0
        dy = 0
    data.append(dx)
    data.append(dy)
    data = np.clip(data, -128, 127)
    format_string = '<' + 'b' * len(data)
    return struct.pack(format_string, *data)   


def set_uart_coefs(coefs):
    global uart_coefs
    with uart_lock:
        uart_coefs = coefs


def update_tracking(data):
    global current_roi_size
    with roi_lock:       
        data_dict["tracker_requested"] = False
        data_dict["tracker_initialized"] = False
        if "stream_size" in data:
            client_stream_size = data["stream_size"]
            width_offset = tracking_frame_size[0] / client_stream_size[0]
            height_offset = tracking_frame_size[1] / client_stream_size[1]
            scaled_roi = (int(data["roi"][0] * width_offset), int(data["roi"][1] * height_offset), int(data["roi"][2] * width_offset), int(data["roi"][3] * height_offset))
            data_dict["current_roi"] = scaled_roi
        else:
            data_dict["current_roi"] = data["roi"]
        current_roi_size = (data_dict["current_roi"][2] + data_dict["current_roi"][3]) // 2 
        tracker_dict["using_kalman"] = data["kalman"]
        tracker_dict["max_skipped_frames"] = int(data["skip_frames"])
        tracker_dict["training_images_count"] = int(data["training_images_count"])
        tracker_dict["alpha_smoothing"] = float(data["alpha_smoothing"])
        tracker_dict["max_corr"] = float(data["max_corr"])
        tracker_dict["sigma_factor"] = float(data["sigma_factor"])
        data_dict["tracker_requested"] = True
    print(f'Update tracking with ROI: {data_dict["current_roi"]}, Kalman: {data["kalman"]}, Skip frames: {data["skip_frames"]}, Training images count: {data["training_images_count"]}, Alpha smoothing: {data["alpha_smoothing"]}, Max corr: {data["max_corr"]}, Sigma factor: {data["sigma_factor"]}')
    

        
def start_stream_for_client(params):
    global stream_protocol, ffmpeg_port
    if server.check_client(params["ip"]):
        print(f"Client {params['ip']} is already getting stream. Recreating pipe...")
        stop_stream_for_client(params["ip"])
    ffmpeg_hanlder = FFmpeg(stream_protocol, port=ffmpeg_port, framerate=params["frame_rate"])
    pipe = ffmpeg_hanlder.create_stream(ip=params["ip"], stream_size=params["stream_size"], bitrate=params["bitrate"])   
    server.add_pipe_to_client(params["ip"], pipe)
    threading.Thread(target=stream, args=(ffmpeg_hanlder, pipe, params["stream_size"],), daemon=True).start()
    print(f"Stream for {params['ip']} is started")
    
    
def stop_stream_for_client(ip):
    if ip in server.clients and server.clients[ip] is None:
        print(f"Cannot close stop the stream for {ip}, because it is not opened.")
        return
    server.close_client_pipe(ip)
    server.clear_client(ip)
    print(f"Stream for {ip} is stoped")
            
               
def show_camera_preview_by_client(data, stop_event):
    global camera_preview_thread, camera_preview_clients
    camera_preview_clients[data["ip"]] = True
    if stop_event.is_set():
        print("Camera preview is already running")
        return
    stop_event.set()
    if camera_preview_thread is not None:
        print("Camera preview thread is already running")
        return
  
    frame_time_ms = int(1000 / data["frame_rate"])
    camera_preview_thread = mp.Process(target=_camera_preview_loop, args=(frame_time_ms, stop_event, sm_name, frame_shape, frame_dtype))
    print(f"Camera preview is starting by {data['ip']}")
    camera_preview_thread.start()


def show_camera_preview(frame_rate, stop_event):
    global camera_preview_thread
    if stop_event.is_set():
        print("Camera preview is already running")
        return
    stop_event.set()
    if camera_preview_thread is not None:
        print("Camera preview was resumed")
        return
  
    frame_time_ms = int(1000 / frame_rate)
    camera_preview_thread = mp.Process(target=_camera_preview_loop, args=(frame_time_ms, stop_event, sm_name, frame_shape, frame_dtype))
    camera_preview_thread.start()
    print("Camera preview was started")
    
    
def stop_camera_preview_by_client(ip, stop_event):
    global camera_preview_thread, camera_preview_clients
    if ip in camera_preview_clients:
        del camera_preview_clients[ip]
    if not camera_preview_clients:
        stop_event.clear()
        print(f"Camera preview is stopped by client ({ip})")


def stop_camera_preview(stop_event):
    stop_event.clear()
    print("Camera preview was stopped")


def process_message_queue():
    try:
        message = message_queue.get_nowait()
        if message == KEEP_ASPECT_RATIO:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_AUTOSIZE, 0)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_ASPECT_RATIO, 1)
        elif message == FREE_ASPECT_RATIO:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_AUTOSIZE, 1)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_ASPECT_RATIO, 0)
    except queue.Empty:
        pass


def change_frame_borders(data):
    global frame_top_border, frame_bottom_border, frame_left_border, frame_right_border
    frame_top_border = data["top"]
    frame_bottom_border = data["bottom"]
    frame_left_border = data["left"]
    frame_right_border = data["right"]
    if data["keep_aspect_ratio"]:
        message_queue.put(KEEP_ASPECT_RATIO)
    else:
        message_queue.put(FREE_ASPECT_RATIO)


def _camera_preview_loop(frame_time_ms, stop_event, shared_memory_name, frame_shape, frame_dtype):
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        print("Camera preview loop is started")
        sm_client = FrameMemoryShareClient(shared_memory_name, frame_shape, frame_dtype)
        while True:
            stop_event.wait()

            frame = sm_client.get_frame()
            if frame is None:
                print("No frames available for camera preview, skipping...")
                time.sleep(0.1)
                continue

            if len(pipeline.active_pipeline_steps) > 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
                processed_data = pipeline.process(rgb_frame, data_dict["current_roi"])
                frame = processed_data[0]
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
            frame = cv2.copyMakeBorder(frame, frame_top_border, frame_bottom_border, frame_left_border, frame_right_border, None, value = 0)
            try:
                cv2.imshow(window_name, frame)
                cv2.waitKey(frame_time_ms)
            except Exception as e:
                print("cv2.imshow error:", e)
                break
            process_message_queue()
    except Exception as e:
        print(f"Camera preview loop error: {e}")
    finally:
        sm_client.close()
        cv2.destroyAllWindows()
 
    
def stream(ffmpeg_hanlder, pipe, stream_size):
    try:
        while True:
            start = time.time()
            
            with latest_frame_lock:
                frame = latest_frame
                
            if frame is None:
                continue

            if stream_size != [frame.shape[1], int(frame.shape[0] / 1.5)]:
                frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
                frame = cv2.resize(frame, stream_size, cv2.INTER_LINEAR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
       
            exit_code = pipe.poll()
            if pipe.stdin.closed:
                print(f"Exit code {exit_code}: stdin pipe is closed. Exiting...")
                break
            ffmpeg_hanlder.write_frame(frame, pipe)
            ffmpeg_hanlder.wait_for_frametime(start)
    except BrokenPipeError as e:
        print(f"BrokenPipeError: {e}")
   
   
def toggle_roi(enabled):
    if enabled:
        print("Enabling ROI drawing")
        pipeline.enable_operation(draw_roi.__name__)
    else:
        print("Disabling ROI drawing")
        pipeline.disable_operation(draw_roi.__name__)


def toggle_crosshair(enabled):
    if enabled:
        print("Enabling crosshair drawing")
        pipeline.enable_operation(draw_crosshair.__name__)
    else:
        print("Disabling crosshair drawing")
        pipeline.disable_operation(draw_crosshair.__name__)


def stop_tracking(from_uart=False):
    tracker_initialized = data_dict.get("tracker_initialized", False)
    if not tracker_initialized:
        return
    with roi_lock:
        data_dict["tracker_initialized"] = False
        data_dict["tracker_requested"] = False
        reset_roi()
        if not from_uart:
            server.send_command_to_clients(Command.STOP_TRACKING)
        
        
def request_tracking_client():
    tracker_initialized = data_dict.get("tracker_initialized", False)
    if tracker_initialized:
        return
    server.send_command_to_clients(Command.REQUEST_TRACKING)
    
    
def request_tracking_server():
    tracker_initialized = data_dict.get("tracker_initialized", False)
    tracker_requested = data_dict.get("tracker_requested", False)
    if tracker_initialized or tracker_requested:
        return
    data_dict["tracker_requested"] = True
    print("Requesting tracking from UART")
    with roi_lock:
        data_dict["tracker_initialized"] = False
        reset_roi()
    request_tracking_client()
    
    
def get_tracker():
    if current_tracker == Tracker.MOSSE:
        return cv2.legacy.TrackerMOSSE_create()
    if current_tracker == Tracker.MK: 
        return FastMosseTracker(skip_frames=tracker_dict["using_kalman"], 
                                max_skipped_frames=tracker_dict["max_skipped_frames"], 
                                training_images_count=tracker_dict["training_images_count"], 
                                alpha_smoothing=tracker_dict["alpha_smoothing"], 
                                correlation_target=tracker_dict["max_corr"], 
                                output_sigma_factor=tracker_dict["sigma_factor"])


def capture_frame(resolution="lores"):
    frame = cam.capture_array(resolution)
    return frame


def reset_server_state():
    data_dict["tracker_initialized"] = False
    data_dict["tracker_requested"] = False
    reset_roi()


def main(frame_shared_memory_handler, shared_memory_name):
    print("Server is started.\n")
    try:
        while True:
            reset_server_state()
            start_main_loop(shared_memory_name)
    except KeyboardInterrupt:
        print('\nClosing server...')
    except Exception as e:
        print(e)
    finally:
        server.send_command_to_clients(Command.DISCONNECT)
        server.close_clients_pipes()
        print('\nClosing ffmpeg proccesses if were enabled...')
        stop_ffmpeg_procces()
        print('\nClosing serial port...')
        close_serial()
        print('\nClosing camera preview if was enabled...')
        cv2.destroyAllWindows()
        frame_shared_memory_handler.close()
        manager.shutdown()
    print('\nServer closed.')
        
        
def start_main_loop(shared_memory_name):
    global latest_frame
    try:
        sm_client = FrameMemoryShareClient(shared_memory_name, frame_shape, frame_dtype)
        wait_in_seconds_to_send_tracker_data = 0.3
        start_timer_tracker_data = time.time()
        print("Main loop is started.\n")
        while True: 
            frame = capture_frame()
            sm_client.set_frame(frame)
            with latest_frame_lock:
                latest_frame = frame
                
            success = data_dict.get("success", False)
            tracker_data = data_dict.get("tracker_data", {})
            tracker_initialized = data_dict.get("tracker_initialized", False)
            if tracker_initialized:
                if success:
                    if time.time() - start_timer_tracker_data > wait_in_seconds_to_send_tracker_data:
                        start_timer_tracker_data = time.time() 
                        server.send_command_to_clients(Command.TRACKER_DATA, tracker_data)
                else:        
                    tracker_data["roi"] = ROI_ZEROS  
                    server.send_command_to_clients(Command.TRACKER_DATA, tracker_data)
                    
            if enable_uart:   
                packet = prepare_uart_data(uart_coefs)
                with uart_lock:   
                    serial_transmit_binary(packet)
            if not wait_first_frame_event.is_set():
                wait_first_frame_event.set()
            #time.sleep(0.01)
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Exiting main loop...")
        raise
    except Exception as e:
        print(f"An error occurred in the main loop: {e}")
        restart_s = 5
        print(f"Attempting to restart main loop in {restart_s} seconds...")
        time.sleep(restart_s)
    finally:
        sm_client.close()


def tracking(shared_memory_name, frame_shape, frame_dtype, data_dict, target_frametime, wait_first_frame_event):
    wait_first_frame_event.wait()
    sm_client = FrameMemoryShareClient(shared_memory_name, frame_shape, frame_dtype)
    fps_counter = FPSCounter()      

    try:
        start_time = time.time()
        while True:
            tracker_requested = data_dict.get("tracker_requested", False)
            tracker_initialized = data_dict.get("tracker_initialized", False)
            current_roi = data_dict.get("current_roi", (0, 0, 0, 0))
            frame = sm_client.get_frame()
            
            with roi_lock:
                if tracker_requested:
                    if not tracker_initialized:
                        
                        tracker = get_tracker()  
                        try:
                            tracker.init(frame, current_roi)
                            data_dict["tracker_initialized"] = True
                            data_dict["tracker_requested"] = False
                            print("Tracker reinitialized with ROI:", current_roi)
                        except Exception as e:
                            print(f"Error initializing tracker with ROI {current_roi}: {e}")
                            data_dict["tracker_initialized"] = False
                    #print(data_dict["tracker_initialized"])
                if data_dict["tracker_initialized"]:   
                    #fps_timer = cv2.getTickCount()
                    success, bbox, _ = tracker.update(frame)
                    #tracker_current_fps = cv2.getTickFrequency() / (cv2.getTickCount() - fps_timer)
                    fps = fps_counter.update()

                    if success:
                        data_dict["current_roi"] = tuple(map(int, bbox))
                    data_dict["success"] = success
                    td = tracker.get_tracker_data()
                    td["fps"] = int(fps)
                    data_dict["tracker_data"] = td
            elapsed_time = time.time() - start_time
            sleep_time = max(0, target_frametime - elapsed_time)
            time.sleep(sleep_time)
            start_time = time.time()
    except Exception as e:
        print(f"Exception occured in tracking process: {e}")
    finally:
        sm_client.close()
                

   
if __name__ == "__main__": 
    print("Starting tracking server...\n")
    
    frame_shared_memory_handler = FrameMemoryShareHandler(frame_shape, frame_dtype)
    sm_name = frame_shared_memory_handler.get_name()
    
    subscribe(UPDATE_TRACKING, update_tracking)
    subscribe(STOP_TRACKING, stop_tracking)
    subscribe(START_STREAM_FOR_CLIENT, start_stream_for_client)
    subscribe(STOP_STREAM_FOR_CLIENT, stop_stream_for_client)
    subscribe(SEND_CFS, set_uart_coefs)
    subscribe(REQUEST_TRACKING, request_tracking_server)
    subscribe(SHOW_CAMERA_PREVIEW, lambda data: show_camera_preview_by_client(data, stop_event))
    subscribe(STOP_CAMERA_PREVIEW, lambda ip: stop_camera_preview_by_client(ip, stop_event))
    subscribe(TOGGLE_ROI, toggle_roi)
    subscribe(TOGGLE_CROSSHAIR, toggle_crosshair)
    subscribe(CHANGE_FRAME_BORDERS, change_frame_borders)
    
    register_zeroconf(server.ip, server_port, stream_protocol.name, ffmpeg_port, tracking_frame_size)
    
    if enable_uart:
        open_serial()
        print("Starting UART receiving...")
        threading.Thread(target=serial_receive_loop, daemon=True).start()
        
    if enable_preview:
        print("Starting camera preview...")
        show_camera_preview(frame_rate=camera_preview_framerate, stop_event=stop_event)

    tracking_process = mp.Process(target=tracking, args=(sm_name, frame_shape, frame_dtype, data_dict, camera_frametime, wait_first_frame_event,))
    tracking_process.start()
    main(frame_shared_memory_handler, sm_name)

