from config import *

class CameraResolution:
    LORES = "lores"
    MAIN = "main"

class Tracker:
    MOSSE = "MOSSE"
    MK = "MK"

net_interface = "wlan0"

window_name = 'preview'
camera_preview_process = None
camera_preview_clients = {}
frame_top_border, frame_bottom_border, frame_left_border, frame_right_border = (0, 0, 0, 0)
message_queue = mp.Queue()

KEEP_ASPECT_RATIO = "keep_aspect_ratio"
FREE_ASPECT_RATIO = "free_aspect_ratio"

### functionality
enable_uart = True
enable_preview = True
enable_debug = False
enable_streaming_screen = False
###

if enable_streaming_screen:
    start_ffmpeg_screen_stream()
    
latest_frame_lock = threading.Lock()
latest_frame = None
preview_frame = None

camera_frame_rate = 60
camera_frame_time = 1 / camera_frame_rate
camera_size_main = (640, 480)#(960, 720)
camera_size_lores = camera_size_main#(448, 360)
camera_resolution = CameraResolution.LORES
camera_preview_frame_rate = 24
video_writing_frame_rate = camera_preview_frame_rate
camera_preview_frame_time_ms = mp.Value('i', int(1000 / camera_preview_frame_rate))
camera_restart_timeout = 5

if camera_resolution == CameraResolution.MAIN:
    camera_size = camera_size_main
    frame_shape = (camera_size[1], camera_size[0], 3)
else:
    camera_size = camera_size_lores
    frame_shape = (int(camera_size[1] * 1.5), camera_size[0])
frame_dtype = np.uint8


frame_shared_memory_handler = None
manager = mp.Manager()
data_dict = manager.dict()
tracker_dict = manager.dict()
data_dict["tracker_initialized"] = False
data_dict["tracker_requested"] = False
tracker_size = camera_size_lores
tracking_frame_size = camera_size_lores
tracking_process = None
### Tracker default parameters ###
#--------------------------------#
defualt_roi_size = 64
current_roi_size = defualt_roi_size
data_dict["current_roi"] = (int(tracker_size[0] // 2 - defualt_roi_size // 2), int(tracker_size[1] // 2 - defualt_roi_size // 2), defualt_roi_size, defualt_roi_size)
tracker_dict["max_skipped_frames"] = 1
tracker_dict["using_kalman"] = False 
tracker_dict["training_images_count"] = 9 
tracker_dict["alpha_smoothing"] = 0.9
tracker_dict["max_corr"] = 0.5
tracker_dict["sigma_factor"] = 0.05

# Searching 
xort_x_shift = 20
xort_y_shift = 15
xort_corel_target_modificator = 0.2
update_xor_tracker_every_n_frames = 5
xort_mask_shift = 4
xort_rescale = 2
search_strategies = [
    (6,  lambda im: (im.shape[1] // 2, im.shape[0] // 2)),
    (0,  lambda im: (im.shape[1] // 3, im.shape[0] // 3)),
]
tracking_lost_max_attempts = 12
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
pipeline.register_operation(draw_fps, draw_fps.__name__, default_enabled=True)

roi_lock = mp.Lock()

camera = None
def reset_camera():
    global camera
    if camera is not None:
        camera.close()
    camera = setup_camera(main={"format": "RGB888", "size": camera_size_main}, lores={"format": "YUV420", "size": camera_size_lores}, fps=camera_frame_rate)
    print(camera.sensor_modes)
    camera.set_controls({"AeEnable": True})
    camera.start()
    

reset_camera()
controls = camera.capture_metadata()
print(f"Current exposition time: {controls['ExposureTime']} мкс")
print(f"Current analogue gain: {controls['AnalogueGain']}")
print(f"Current digital gain: {controls['DigitalGain']}")


play_preview_event = mp.Event()
exit_event = mp.Event()
wait_first_frame_event = mp.Event()
is_camera_closed_event = mp.Event()
main_loop_event = mp.Event()
got_frame_event = threading.Event()

current_tracker = Tracker.MK


def reset_roi():
    data_dict["current_roi"] = (int(tracker_size[0] // 2 - current_roi_size // 2), int(tracker_size[1] // 2 - current_roi_size // 2), current_roi_size, current_roi_size)


def sleep(start_time, time_s):
    try:
        elapsed_time = time.time() - start_time
        sleep_time = max(0, time_s - elapsed_time)
        time.sleep(sleep_time) 
    except KeyboardInterrupt:
        raise
    finally:
        return time.time()


# https://docs.python.org/3/library/struct.html
def prepare_uart_data(data):
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
    data_dict["tracker_data"] = {"roi": data_dict["current_roi"], "correlation": 0, "template_scale": 0, "learning_rate": 0, "correlation_target": 0, "fps": 0}
    server.send_command_to_clients(Command.TRACKER_DATA, data_dict["tracker_data"])
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
            
               
def show_camera_preview_by_client(data, play_preview_event):
    global camera_preview_process, camera_preview_clients, camera_preview_frame_rate
    if not isinstance(data["frame_rate"], int):
        print(f"Wrong data for frame_rate! ({data['frame_rate']})")
        return
    
    camera_preview_frame_rate = data["frame_rate"]
    frame_time_ms = int(1000 / camera_preview_frame_rate)
    camera_preview_frame_time_ms.value = frame_time_ms
    camera_preview_clients[data["ip"]] = True
    
    if play_preview_event.is_set():
        print("Camera preview is already running")
        return
    play_preview_event.set()
    if camera_preview_process is not None:
        print("Camera preview thread is already running")
        return
    
    camera_preview_process = mp.Process(target=_camera_preview_loop, args=(play_preview_event, exit_event, is_camera_closed_event, sm_name, frame_shape, frame_dtype))
    print(f"Camera preview is starting by {data['ip']}")
    camera_preview_process.start()


def show_camera_preview(play_preview_event):
    global camera_preview_process
    if play_preview_event.is_set():
        print("Camera preview is already running")
        return
    play_preview_event.set()
    if camera_preview_process is not None:
        print("Camera preview was resumed")
        return
    
    camera_preview_process = mp.Process(target=_camera_preview_loop, args=(play_preview_event, exit_event, is_camera_closed_event, sm_name, frame_shape, frame_dtype))
    camera_preview_process.start()
    #os.setpgid(camera_preview_process.pid, camera_preview_process.pid) # change process group
    print("Camera preview was started")
    
    
def stop_camera_preview_by_client(ip, play_preview_event):
    global camera_preview_process, camera_preview_clients
    if ip in camera_preview_clients:
        del camera_preview_clients[ip]
    if not camera_preview_clients:
        play_preview_event.clear()
        print(f"Camera preview is stopped by client ({ip})")


def stop_camera_preview(play_preview_event):
    play_preview_event.clear()
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
     
        
def change_roi_from_uart(value):
    global current_roi_size
    a = 0#182
    b = 255#1811
    min_size = 32
    max_size = 128
    step = 4
    norm_value = (np.clip(value, a, b) - a) / (b - a)
    steps = round(norm_value * (max_size - min_size) / step)
    current_roi_size = min_size + steps * step
    print(current_roi_size)
    reset_roi()
    
    
def write_video(exit_event, fps):
    frame_time = 1 / fps
    with VideoWriter(path=base_path, fps=fps) as vw:
        while data_dict["tracker_initialized"] and not exit_event.is_set():
            vw.write(preview_frame)
            time.sleep(frame_time)     


def _camera_preview_loop(play_preview_event, exit_event, is_camera_closed_event, shared_memory_name, frame_shape, frame_dtype):
    global preview_frame
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        sm_client = FrameMemoryShareClient(shared_memory_name, frame_shape, frame_dtype)
        no_frame_image = cv2.putText(np.zeros((480, 640)), "No video", (90, 240), 2, 3, (255, 255, 255), 3)
        print("Camera preview loop is started")
        
        video_writer = None
        while not exit_event.is_set():
           
            if not is_camera_closed_event.is_set():
                frame = sm_client.get_frame(False)
            else:
                preview_frame = no_frame_image
            
            if not is_camera_closed_event.is_set():
                if len(pipeline.active_pipeline_steps) > 0:
                    if camera_resolution == CameraResolution.LORES:
                        preview_frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
                    else:
                        preview_frame = frame.copy()
                    current_roi = data_dict.get("current_roi", None)
                    tracker_data = data_dict.get("tracker_data", {})
                    
                    fps_data = [tracker_data["fps"], (50, 50)] if "fps" in tracker_data else None
                    processed_data = pipeline.process(preview_frame, current_roi, fps_data)
                    preview_frame = processed_data[0]
                else:
                    if camera_resolution == CameraResolution.LORES:
                        preview_frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
                    else:
                        preview_frame = frame.copy()
                    
            if enable_debug:
                tracker_initialized = data_dict.get("tracker_initialized", False)
                if video_writer is None and tracker_initialized:
                    video_writer = threading.Thread(target=write_video, args=(exit_event, video_writing_frame_rate,))
                    video_writer.start()
                elif video_writer is not None and not tracker_initialized:
                    video_writer = None
            
            if play_preview_event.is_set():
                canvas = cv2.copyMakeBorder(preview_frame, frame_top_border, frame_bottom_border, frame_left_border, frame_right_border, None, value = 0)
                try:
                    cv2.imshow(window_name, cv2.resize(canvas, (320, 240), cv2.INTER_LINEAR))
                except Exception as e:
                    print("cv2.imshow error:", e)
                    break
            process_message_queue()
            cv2.waitKey(camera_preview_frame_time_ms.value)
    except KeyboardInterrupt:
        close_server()
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

    print("Requesting tracking from UART")
    with roi_lock:
        data_dict["tracker_requested"] = True
        data_dict["tracker_initialized"] = False
        reset_roi()
    request_tracking_client()
    
    
def get_tracker():
    if current_tracker == Tracker.MOSSE:
        return cv2.legacy.TrackerMOSSE_create()
    if current_tracker == Tracker.MK:
        xort = XORTracker(x_shift=xort_x_shift, y_shift=xort_y_shift, corel_target_modificator=xort_corel_target_modificator, mask_shift=xort_mask_shift) 
        return FastMosseTracker(xort = xort,
                                skip_frames=tracker_dict["using_kalman"], 
                                max_skipped_frames=tracker_dict["max_skipped_frames"], 
                                training_images_count=tracker_dict["training_images_count"], 
                                alpha_smoothing=tracker_dict["alpha_smoothing"], 
                                correlation_target=tracker_dict["max_corr"], 
                                output_sigma_factor=tracker_dict["sigma_factor"],
                                update_xor_tracker_every_n_frames=update_xor_tracker_every_n_frames,
                                xort_rescale=xort_rescale,
                                search_strategies=search_strategies,
                                tracking_lost_max_attempts=tracking_lost_max_attempts)


def capture_frame(resolution="lores", timeout=1):
    global camera
    job = camera.capture_request(wait=False)
    try:
        request = camera.wait(job, timeout=timeout)
        frame = request.make_array(resolution)
        request.release()
        if is_camera_closed_event.is_set():
            is_camera_closed_event.clear()
        return frame
    except Exception as e:
        print("Timeout or error:", e)
        try:
            if not is_camera_closed_event.is_set():
                is_camera_closed_event.set()
            reset_camera()
        except RuntimeError:
            time.sleep(camera_restart_timeout)
    return None


def convert_frame_to_gray(frame):
    if np.ndim(frame) == 2:
        return cv2.cvtColor(frame, cv2.COLOR_YUV2GRAY_I420)
    elif np.ndim(frame) == 3:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        return None


def reset_server_state():
    data_dict["tracker_initialized"] = False
    data_dict["tracker_requested"] = False
    main_loop_event.clear()
    reset_roi()


def get_xy_deviations():
    dx = tracker_size[0] // 2 - (data_dict["current_roi"][0] + data_dict["current_roi"][2] // 2)
    dy = tracker_size[1] // 2 - (data_dict["current_roi"][1] + data_dict["current_roi"][3] // 2)
    return dx, dy  
        
        
def start_main_loop(shared_memory_name):
    global latest_frame
    try:
        sm_client = FrameMemoryShareClient(shared_memory_name, frame_shape, frame_dtype)
        wait_in_seconds_to_send_tracker_data = 0.3
        start_timer_tracker_data = time.time()   
        data_handler = None         
        xy_deviations = []
        target_frametime = camera_frame_time
        start_time = time.time()
        print("Main loop is started.\n")
        while not main_loop_event.is_set(): 
            frame = capture_frame(resolution=camera_resolution)
            if frame is None:
                continue
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
            dx, dy = get_xy_deviations()
            if enable_debug:
                if tracker_initialized:
                    xy_deviations.append([dx, dy])
                    if data_handler is None:
                        data_handler = CSVHandler(base_path)
                if not tracker_initialized and data_handler is not None:
                    data_handler.save({"fields": ["dx", "dy"], "rows": np.clip(xy_deviations, -128, 127)})
                    xy_deviations.clear()
                    data_handler = None
            if enable_uart:
                packet = prepare_uart_data([*uart_coefs, dx, dy])
                with uart_lock:   
                    serial_transmit_binary(packet)
            if not wait_first_frame_event.is_set():
                wait_first_frame_event.set()
                
            start_time = sleep(start_time, target_frametime)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected. Exiting main loop...")
        raise
    except Exception as e:
        print(f"An error occurred in the main loop: {e}")
        restart_s = 5
        print(f"Attempting to restart main loop in {restart_s} seconds...")
        time.sleep(restart_s)
    finally:
        sm_client.close()


def tracking(shared_memory_name, frame_shape, frame_dtype, data_dict, target_frametime, wait_first_frame_event, exit_event):
    wait_first_frame_event.wait()
    sm_client = FrameMemoryShareClient(shared_memory_name, frame_shape, frame_dtype)
    fps_counter = FPSCounter()      
    try:
        start_time = time.time()
        while not exit_event.is_set():
            tracker_requested = data_dict.get("tracker_requested", False)
            tracker_initialized = data_dict.get("tracker_initialized", False)
            current_roi = data_dict.get("current_roi", (0, 0, 0, 0))
            frame = convert_frame_to_gray(sm_client.get_frame())
            
            with roi_lock:
                if tracker_requested:
                    if not tracker_initialized:
                        
                        tracker = get_tracker()  
                        try:
                            print(frame.shape)
                            tracker.init(frame, current_roi)
                            data_dict["tracker_initialized"] = True
                            data_dict["tracker_requested"] = False
                            print("Tracker reinitialized with ROI:", current_roi)
                        except Exception as e:
                            print(f"Error initializing tracker with ROI {current_roi}: {e}")
                            data_dict["tracker_initialized"] = False
                if data_dict["tracker_initialized"]:   
                    #fps_timer = cv2.getTickCount()
                    success, bbox, _ = tracker.update(frame)
                    #tracker_current_fps = cv2.getTickFrequency() / (cv2.getTickCount() - fps_timer)
                    fps = fps_counter.update()

                    if success:
                        data_dict["current_roi"] = tuple(map(int, bbox))
                    elif data_dict["success"]:
                        reset_roi()
                    data_dict["success"] = success
                    td = tracker.get_tracker_data()
                    td["fps"] = int(fps)
                    data_dict["tracker_data"] = td
            start_time = sleep(start_time, target_frametime)
    except KeyboardInterrupt:
        close_server()
    except Exception as e:
        print(f"Exception occured in tracking process: {e}")
    finally:
        sm_client.close()


def close_server():
    server.send_command_to_clients(Command.DISCONNECT)
    print('1. Disconnecting all clients if any...')
    server.close_clients_pipes()
    print('2. Closing all ffmpeg proccesses if any...')
    stop_ffmpeg_procces()
    print('3. Closing serial port...')
    close_serial()
    print('4. Closing camera preview...')
    
    play_preview_event.clear()
    main_loop_event.set()
    exit_event.set()
    
    try:
        if tracking_process is not None:
            tracking_process.terminate()
    except Exception:
        pass

    try:
        if camera_preview_process is not None:
            camera_preview_process.terminate()
    except Exception:
        pass

    try:
        if frame_shared_memory_handler:
            frame_shared_memory_handler.close()
    except (FileNotFoundError, AttributeError):
        pass
    
    manager.shutdown()
    cv2.destroyAllWindows()
    print('Server closed.')               

def main(shared_memory_name):
    print("Server is started.\n")
    try:
        while not exit_event.is_set():
            reset_server_state()
            start_main_loop(shared_memory_name)
    except KeyboardInterrupt:
        print('\nClosing server...')
        close_server()
    except Exception as e:
        print(e)


if __name__ == "__main__": 
    print("Starting tracking server...\n")
    
    frame_shared_memory_handler = FrameMemoryShareHandler(frame_shape, frame_dtype)
    sm_name = frame_shared_memory_handler.get_name()
    
    subscribe(UPDATE_TRACKING_EVENT, update_tracking)
    subscribe(STOP_TRACKING_EVENT, stop_tracking)
    subscribe(START_STREAM_FOR_CLIENT_EVENT, start_stream_for_client)
    subscribe(STOP_STREAM_FOR_CLIENT_EVENT, stop_stream_for_client)
    subscribe(SEND_CFS_EVENT, set_uart_coefs)
    subscribe(REQUEST_TRACKING_EVENT, request_tracking_server)
    subscribe(SHOW_CAMERA_PREVIEW_EVENT, lambda data: show_camera_preview_by_client(data, play_preview_event))
    subscribe(STOP_CAMERA_PREVIEW_EVENT, lambda ip: stop_camera_preview_by_client(ip, play_preview_event))
    subscribe(TOGGLE_ROI_EVENT, toggle_roi)
    subscribe(TOGGLE_CROSSHAIR_EVENT, toggle_crosshair)
    subscribe(CHANGE_FRAME_BORDERS_EVENT, change_frame_borders)
    subscribe(CHANGE_ROI_FROM_UART_EVENT, change_roi_from_uart)
    
    register_zeroconf(server.ip, server_port, stream_protocol.name, ffmpeg_port, tracking_frame_size)
    
    if enable_uart:
        open_serial()
        print("Starting UART receiving...")
        threading.Thread(target=serial_receive_loop, daemon=True).start()
        
    if enable_preview:
        print("Starting camera preview...")
        show_camera_preview(play_preview_event=play_preview_event)
        
    tracking_process = mp.Process(target=tracking, args=(sm_name, frame_shape, frame_dtype, data_dict, camera_frame_time, wait_first_frame_event, exit_event,))
    tracking_process.start()
    main(sm_name)