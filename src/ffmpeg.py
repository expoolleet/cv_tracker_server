import threading
import time
import subprocess
import numpy as np
from enum import Enum

class StreamProtocol(Enum):
    HTTP = 0,
    UDP = 1
    
class FFmpeg:   
    def __init__(self, stream_protocol: StreamProtocol, port, framerate):
        self.stream_protocol = stream_protocol
        self.framerate = framerate
        self.frametime = 1.0/framerate
        self.port = port
        self.pipes = []   
        
    def create_stream(self, ip, stream_size, bitrate):
        pipe = self.get_pipe(ip, stream_size, bitrate)
        self.pipes.append(pipe)
        threading.Thread(target=self.monitor_stderr, args=(pipe, (ip, self.port)), daemon=True).start()
        return pipe     

    def get_pipe(self, ip, stream_size, bitrate):
        if self.stream_protocol == StreamProtocol.UDP:
            return subprocess.Popen([
            'ffmpeg',
            '-loglevel','error',
            '-y',
            '-threads','1',
            '-f','rawvideo',
            '-pix_fmt','yuv420p',
            '-s',f'{stream_size[0]}x{stream_size[1]}',
            '-r',f'{self.framerate}',
            '-i','-',
            '-an',
            '-c:v','h264_v4l2m2m',
            '-b:v',f'{bitrate}k',
            '-maxrate',f'{bitrate}k',
            '-bufsize',f'{bitrate * 2}k',
            '-fflags','nobuffer',
            '-flags','low_delay',
            '-strict','experimental',
            '-avioflags','direct',
            '-f','mpegts',
            '-muxdelay','0',
            '-muxpreload','0',
            '-flush_packets','1',
            '-timeout','10000000',
            f'udp://{ip}:{self.port}?pkt_size=1316',
            ], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        if self.stream_protocol == StreamProtocol.HTTP:
            return subprocess.Popen(['ffmpeg'
            '-f','rawvideo',
            '-pix_fmt','yuv420p',
            '-i','-',
            '-c:v','libx264',
            '-preset','ultrafast',
            '-tune','zerolatency',
            '-f','mpegts',
            f'http://{ip}:{self.port}'], stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def monitor_stderr(self, pipe, client):
        while True:
            line = pipe.stderr.readline()
            if not line:
                print("stderr: EOF reached")
                break
            print(f"FFmpeg [pipe to {client[0]}]:", line.decode().strip())     
            
    def close_pipes(self):
        for pipe in self.pipes:
            pipe.stdin.close()
            pipe.wait()
           
    def wait_for_frametime(self, start):
        elapsed_time = time.time() - start
        if elapsed_time < self.frametime:
            time.sleep(self.frametime - elapsed_time)

    def write_frame(self, frame, pipe):
        try:
            pipe.stdin.write(frame.astype(np.uint8).tobytes())
            pipe.stdin.flush()
        except Exception as e:
            print(f"An error occured while trying to write a frame: {e}")