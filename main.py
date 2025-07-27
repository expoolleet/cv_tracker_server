import os
import cv2
import time
import threading
import subprocess
import numpy as np
import struct
import multiprocessing as mp

from src.event import subscribe
from src.event_types import UPDATE_TRACKING, STOP_TRACKING, START_STREAM_FOR_CLIENT, STOP_STREAM_FOR_CLIENT, SEND_CFS, REQUEST_TRACKING, SHOW_CAMERA_PREVIEW, STOP_CAMERA_PREVIEW
from src.command import Command
from src.zeroconf import register_zeroconf
from src.picamera import setup_camera
from src.server import Server
from src.ffmpeg import FFmpeg, StreamProtocol
from src.uart_transmition import serial_transmit_binary, serial_receive_loop
from src.fps_counter import FPSCounter
from src.screen_stream import start_ffmpeg_screen_stream, stop_ffmpeg_procces, stream_height, stream_width, stream_lock
from tracker.fast_mosse_tracker import FastMosseTracker

os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''
os.environ['QT_QPA_PLATFORM'] = 'eglfs'

class Tracker:
    MOSSE = "MOSSE"
    MK = "MK"

net_interface = "wlan0"

window_name = 'Fullscreen PiCamera2 Feed'
camera_preview_thread = None
is_camera_preview_running = False
camera_preview_lock = threading.Lock()
camera_preview_clients = {}

enable_uart = False

is_streaming_screen = False
if is_streaming_screen:
    start_ffmpeg_screen_stream()


tracker = None
current_roi = None
using_kalman = False
max_skipped_frames = 1
tracker_initialized = False

latest_frame_lock = threading.Lock()
latest_highres_frame = None
latest_lowres_frame = None

camera_framerate = 60
camera_size_main = (640, 480)
camera_size_lores = (448, 360)

tracker_size = camera_size_lores
tracking_frame_size = camera_size_lores

ROI_ZEROS = (0, 0, 0, 0)

uart_coefs = [0, 0, 0]
uart_lock = threading.Lock()

stream_protocol = StreamProtocol.UDP
server_port = 8000
ffmpeg_port = 8001
server = Server(net_interface, server_port) 

roi_lock = threading.Lock()

cam = setup_camera(main={"format": "BGR888", "size": camera_size_main}, lores={"format": "YUV420", "size": camera_size_lores}, fps=camera_framerate)
cam.set_controls({"AeEnable": False}) 
cam.set_controls({"ExposureTime": 6000}) 
cam.set_controls({"AnalogueGain": 8}) 
cam.start()
controls = cam.capture_metadata()
print(f"Current exposition time: {controls['ExposureTime']} мкс")
print(f"Current analogue gain: {controls['AnalogueGain']}")
print(f"Current digital gain: {controls['DigitalGain']}")

current_tracker = Tracker.MK

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
    global tracker_initialized, current_roi, max_skipped_frames, using_kalman, training_images_count, alpha_smoothing, max_corr, sigma_factor, tracking_frame_size
    with roi_lock:       
        tracker_initialized = False
        client_stream_size = data["stream_size"]
        width_offset = tracking_frame_size[0] / client_stream_size[0]
        height_offset = tracking_frame_size[1] / client_stream_size[1]
        scaled_roi = (int(data["roi"][0] * width_offset), int(data["roi"][1] * height_offset), int(data["roi"][2] * width_offset), int(data["roi"][3] * height_offset))
        current_roi = scaled_roi
        using_kalman = data["kalman"]
        max_skipped_frames = int(data["skip_frames"])
        training_images_count = int(data["training_images_count"])
        alpha_smoothing = float(data["alpha_smoothing"])
        max_corr = float(data["max_corr"])
        sigma_factor = float(data["sigma_factor"])
    print(f"Update tracking with ROI: {current_roi}, Kalman: {using_kalman}, Skip frames: {max_skipped_frames}, Training images count: {training_images_count}, Alpha smoothing: {alpha_smoothing}, Max corr: {max_corr}, Sigma factor: {sigma_factor}")
    

def stop_tracking(from_uart=False):
    global tracker_initialized, current_roi, server
    if not tracker_initialized:
        return
    with roi_lock:
        tracker_initialized = False
        current_roi = None
        server.send_command_to_clients(Command.STOP_TRACKING)
     
        
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
            
                
def show_camera_preview(data):
    global camera_preview_lock, is_camera_preview_running, camera_preview_thread, camera_preview_clients
    with camera_preview_lock:
         is_camera_preview_running = True
    camera_preview_clients[data["ip"]] = True
    if camera_preview_thread is not None:
        print(f"Camera preview thread is already running ({data['ip']} is initiator), skipping...")
        return
    
    frame_time_ms = int(1000 / data["frame_rate"])
    camera_preview_thread = threading.Thread(target=_camera_preview_loop, args=(frame_time_ms,), daemon=True)
    #camera_preview_thread = mp.Process(target=_camera_preview_loop, args=(frame_time_ms,))
    print(f"Camera preview is starting by {data['ip']}")
    camera_preview_thread.start()
    
    
def stop_camera_preview(ip):
    global is_camera_preview_running, camera_preview_thread, camera_preview_clients
    if ip in camera_preview_clients:
        del camera_preview_clients[ip]
    if not camera_preview_clients:
        with camera_preview_lock:
            is_camera_preview_running = False
        print("Camera preview is stopped")


def _camera_preview_loop(frame_time_ms):
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_FULLSCREEN | cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_KEEPRATIO)
        print("Camera preview loop is started")
        while True:
            with camera_preview_lock:
                 is_running = is_camera_preview_running

            with latest_frame_lock:
                frame = latest_lowres_frame.copy()

            if frame is None:
                print("No frames available for camera preview, skipping...")
                time.sleep(0.1)
                continue
                
            if is_running:
                cv2.imshow(window_name, cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420))
                cv2.waitKey(frame_time_ms)
            else:
                time.sleep(0.1)
    except Exception as e:
        print(f"Camera preview loop error: {e}")
    finally:
        cv2.destroyWindow(window_name)
    
    
def stream(ffmpeg_hanlder, pipe, stream_size):
    try:
        while True:
            start = time.time()
            
            with latest_frame_lock:
                frame_low = latest_lowres_frame
                frame_high = latest_highres_frame
                
            if frame_low is None:
                continue

            if stream_size == [frame_low.shape[1], int(frame_low.shape[0] / 1.5)]:
                frame = frame_low
            else:
                if frame_high is None:
                    frame = frame_low
                    frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
                    frame = cv2.resize(frame, stream_size, cv2.INTER_LINEAR)
                else:
                    frame = frame_high
                    if (frame.shape[1], frame.shape[0]) != stream_size:
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
        
        
def request_tracking():
    global server, tracker_initialized
    if tracker_initialized:
        return
    print("Requesting tracking from UART")
    server.send_command_to_clients(Command.REQUEST_TRACKING)


def get_tracker():
    if current_tracker == Tracker.MOSSE:
        return cv2.legacy.TrackerMOSSE_create()
    if current_tracker == Tracker.MK: 
        return FastMosseTracker(skip_frames=using_kalman, 
                                max_skipped_frames=max_skipped_frames, 
                                training_images_count=training_images_count, 
                                alpha_smoothing=alpha_smoothing, 
                                correlation_target=max_corr, 
                                output_sigma_factor=sigma_factor)


def main():
    global latest_highres_frame, latest_lowres_frame, current_roi, tracker_initialized, uart_lock
    
    subscribe(UPDATE_TRACKING, update_tracking)
    subscribe(STOP_TRACKING, stop_tracking)
    subscribe(START_STREAM_FOR_CLIENT, start_stream_for_client)
    subscribe(STOP_STREAM_FOR_CLIENT, stop_stream_for_client)
    subscribe(SEND_CFS, set_uart_coefs)
    subscribe(REQUEST_TRACKING, request_tracking)
    subscribe(SHOW_CAMERA_PREVIEW, show_camera_preview)
    subscribe(STOP_CAMERA_PREVIEW, stop_camera_preview)
    
    register_zeroconf(server.ip, server_port, stream_protocol.name, ffmpeg_port, tracking_frame_size)
    
    fps_counter = FPSCounter()
    
    if enable_uart:
        print("Starting UART receiving...")
        threading.Thread(target=serial_receive_loop, daemon=True).start()

    unsuccessful_tracking_frames = 0
    max_unsuccessful_tracking_frames = 30
    wait_frames_to_send_tracker_data = 10
    waited_frames = 0
    
    global current_frame
    print("Server is started\n")
    try:
        while True:
            if is_streaming_screen:
                with stream_lock:
                    if current_frame is None:
                        current_frame = np.zeros((int(stream_height * 1.5), stream_width), dtype=np.uint8)
                    frame_low = current_frame
            else:
                frame_low = cam.capture_array("lores")
            with roi_lock:
                if current_roi is not None:
                    if not tracker_initialized:
                        
                        tracker = get_tracker()  
                        try:
                            tracker.init(frame_low, current_roi)
                            tracker_initialized = True
                            print("Tracker reinitialized with ROI:", current_roi)
                        except Exception as e:
                            print(f"Error initializing tracker with ROI {current_roi}: {e}")
                            current_roi = None 
                            tracker_initialized = False
                    
                    if tracker_initialized:   
                        if current_tracker != Tracker.MOSSE:
                            frame = frame_low
                        else:
                            frame = cv2.cvtColor(frame_low, cv2.COLOR_YUV2GRAY_I420)
                        #fps_timer = cv2.getTickCount()
                        data = tracker.update(frame)
                        #current_fps = cv2.getTickFrequency() / (cv2.getTickCount() - fps_timer)

                        fps = fps_counter.update()

                        if len(data) == 3:
                            success, bbox, _ = data
                        else:
                            success, bbox = data
                        
                        if success:
                            waited_frames += 1
                            if waited_frames == wait_frames_to_send_tracker_data:
                                waited_frames = 0
                                if hasattr(tracker, "get_tracker_data"):
                                    tracker_data = tracker.get_tracker_data()
                                    tracker_data["fps"] = int(fps)
                                    server.send_command_to_clients(Command.TRACKER_DATA, tracker_data)
                                else:
                                    current_roi = tuple(map(int, bbox))
                                    server.send_new_roi_to_clients(current_roi)
                                unsuccessful_tracking_frames = 0
                        else:
                            unsuccessful_tracking_frames += 1
                            if unsuccessful_tracking_frames == max_unsuccessful_tracking_frames:          
                                if hasattr(tracker, "get_tracker_data"):  
                                    tracker_data = tracker.get_tracker_data()
                                    tracker_data["roi"] = ROI_ZEROS  
                                    server.send_command_to_clients(Command.TRACKER_DATA, tracker_data)
                                else:
                                    server.send_new_roi_to_clients(ROI_ZEROS)
 
            with latest_frame_lock:
                latest_lowres_frame = frame_low
             
            if enable_uart:   
                packet = prepare_uart_data(uart_coefs)
                with uart_lock:   
                    serial_transmit_binary(packet)

    except KeyboardInterrupt:
        print('\nClosing server...')
    finally:
        server.send_command_to_clients(Command.DISCONNECT)
        server.close_clients_pipes()
        print('\nClosing ffmpeg proccesses if were enabled...')
        stop_ffmpeg_procces()
        print('\nClosing camera preview if was enabled...')
        cv2.destroyAllWindows()
        print('\nServer closed.')
   
if __name__ == "__main__": 
    print("Starting tracking server...\n")
    main()
