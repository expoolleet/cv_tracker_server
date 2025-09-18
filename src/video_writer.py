from pathlib import Path
import datetime
import subprocess

class VideoWriter:
    def __init__(self, path:str=None , file_name:str=None, fps:int=30, size:tuple[int, int]=(640, 480), bitrate:str="2M"):
        folder_name = "video"
        base_path = Path(path).resolve() / folder_name if path else Path(__file__).resolve().parent / folder_name
        Path(base_path).mkdir(parents=True, exist_ok=True)
        name = file_name if file_name else datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.file_path = base_path / f"{name}.mp4"
        
        cmd = [
            "ffmpeg",
            '-loglevel','error',
            "-hide_banner",
            "-y",
            "-f","rawvideo",
            "-pix_fmt","yuv420p",
            "-s",f"{size[0]}x{size[1]}",
            "-r",str(fps),
            "-i","-",
            "-an",
            "-c:v","h264_v4l2m2m",
            "-b:v",bitrate,
            "-g",str(fps),
            "-num_output_buffers","16",
            "-num_capture_buffers","8",
            "-f","mp4",
            f"{self.file_path}"   
        ]
        self.writer = subprocess.Popen(cmd, stdin=subprocess.PIPE)

        print(f"VideoWriter is started writing into {self.file_path} with parameters: fps={fps}, video size={size}")
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            print(f"Exception ocurred in VideoWriter: {exc_type}: {exc_val}")
        self.stop()
        
    def __del__(self):
        if self.writer and self.writer.poll() is not None:
             self.stop()
    
    def write(self, frame) -> None:
        self.writer.stdin.write(frame.tobytes())
        
    def stop(self) -> None:
        if self.writer:
            try:
                self.writer.stdin.close()    
            except Exception:
                pass
            self.writer.wait()
            self.writer = None
            
    def _monitor_stderr(self, pipe) -> None:
        while True:
            line = pipe.stderr.readline()
            if not line:
                print("stderr: EOF reached")
                break
            print("FFmpeg VideoWriter:", line.decode().strip())
            