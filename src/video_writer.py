import cv2
from pathlib import Path
import datetime
import subprocess
import threading

class VideoWriter:
    
    def __init__(self, path=None, file_name=None, fps=30, size=(640, 480)):
        folder_name = "video"
        base_path = Path(path).resolve() / folder_name if path else Path(__file__).resolve().parent / folder_name
        Path(base_path).mkdir(parents=True, exist_ok=True)
        name = file_name if file_name else datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.file_path = base_path / f"{name}.avi"
        
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self.writer = cv2.VideoWriter(str(self.file_path), fourcc, fps, size)
        
        # cmd = [
        #     "ffmpeg",
        #     '-loglevel','error',
        #     "-hide_banner",
        #     "-y",
        #     "-f","rawvideo",
        #     "-s",f"{size[0]}x{size[1]}",
        #     "-pix_fmt","bgr24",
        #     "-r",str(fps),
        #     "-i","-",
        #     "-an",
        #     "-c:v","libx264",
        #     "-preset","ultrafast",
        #     "-crf","23",
        #     "-pix_fmt","yuv420p",
        #     f"{self.file_path}"   
        # ]
        # self.writer = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        
        #threading.Thread(target=self._monitor_stderr, args=(self.writer,), daemon=True).start()
        
        #if self.writer.poll() is None:
        if not self.writer.isOpened():
            raise RuntimeError(f"Can not open VideoWriter for {self.file_path}")

        print(f"VideoWriter is started writing into {self.file_path} with parameters: fps={fps}, video size={size}")
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            print(f"Exception ocurred in VideoWriter: {exc_type}: {exc_val}")
        self.writer.release()
        #self.stop()
        
    def __del__(self):
        if self.writer.isOpened():
            self.writer.release()
        # if self.writer and self.writer.poll() is not None:
        #     self.stop()
    
    def write(self, frame):
        self.writer.write(frame)
        #self.writer.stdin.write(frame.tobytes())
        
    def stop(self):
        self.writer.release()
        # if self.writer:
        #     try:
        #         self.writer.stdin.close()    
        #     except Exception:
        #         pass
        #     self.writer.wait()
        #     self.writer = None
            
    def _monitor_stderr(self, pipe):
        while True:
            line = pipe.stderr.readline()
            if not line:
                print("stderr: EOF reached")
                break
            print("FFmpeg VideoWriter:", line.decode().strip())
            