import threading
import subprocess
import time
import numpy as np
from src.server import get_ip_for_interface

stream_lock = threading.Lock()
stream_height = 720
stream_width = 960
current_frame = None
ffmpeg_process = None
stream_thread = None
err_thread = None


def start_ffmpeg_screen_stream() -> None:
    global ffmpeg_process, stream_thread, err_thread
    port = 8002
    ip = get_ip_for_interface("wlan0")
    print(f"Starting ffmpeg screen stream on {ip}:{port}")
    args = [
        "ffmpeg",
        "-loglevel", "info",
        "-fflags", "nobuffer",
        "-fflags", "discardcorrupt",
        "-flags", "low_delay",
        "-pkt_size", "1316",
        "-probesize", "32",
        "-i", f"udp://{ip}:{port}",
        "-f", "rawvideo",
        "-pix_fmt", "yuv420p",
        "pipe:"
    ]
    
    ffmpeg_process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stream_thread = threading.Thread(target=read_stream, daemon=True)
    stream_thread.start()
    err_thread = threading.Thread(target=monitor_stderr, daemon=True)
    err_thread.start()


def read_stream() -> None:
    global current_frame, ffmpeg_process, stream_width, stream_height, stream_lock
    if stream_width == 0 or stream_height == 0:
        print("Stream size was not set, exiting...")
        kill_ffmpeg_procces()

    while True:
        try:
            if ffmpeg_process:
                data_size = int(stream_width * stream_height * 1.5)
                raw_frame = ffmpeg_process.stdout.read(data_size)
                ffmpeg_process.stdout.flush()

                if raw_frame is None or len(raw_frame) != data_size:
                    print("Raw frame is empty or broken, exiting...")
                    kill_ffmpeg_procces()
                    break

                with stream_lock:
                    current_frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(
                        [int(stream_height * 1.5), stream_width])
            else:
                time.sleep(0.1)
        except subprocess.SubprocessError as e:
            print(f"Subprocess error: {e}")
            time.sleep(1)
        except ValueError as e:
            print(f"ValueError error: {e}")
        except Exception as e:
            print(f"Exiting read stream due to an error: {e}")
            break
                
  
def kill_ffmpeg_procces() -> None:
    global ffmpeg_process
    if ffmpeg_process:
        ffmpeg_process.kill()
        ffmpeg_process = None              
 
   
def stop_ffmpeg_procces() -> None:
    if ffmpeg_process:
        ffmpeg_process.stdin.close()
        ffmpeg_process.wait()
  
                
def monitor_stderr() -> None:
    try:
        while ffmpeg_process is not None:
            line = ffmpeg_process.stderr.readline()
            if not line:
                print("stderr: EOF reached")
                break
            print(f"stderr: {line.decode().strip()}")
        print("stderr thread stopped")
    except Exception as e:
        print(f"Monitoring stderr was failed: {e}")