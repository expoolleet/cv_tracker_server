from multiprocessing import shared_memory, Lock
import numpy as np

class FrameMemoryShareHandler:
    def __init__(self, shape, dtype):
        self.nbytes = int(np.prod(shape) * np.dtype(dtype).itemsize)
        self.frame_sm = shared_memory.SharedMemory(create=True, size=self.nbytes)
        self.frame_buffer = np.ndarray(shape, dtype=dtype,buffer=self.frame_sm.buf)
        self.memory_name = self.frame_sm.name
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            print(f"Exception in FrameMemoryShareHandler occured: {exc_type}, {exc_val}")
        self.close()
        
    def get_frame(self, copying=True):
        if copying:
            return self.frame_buffer.copy()
        return self.frame_buffer
    
    def set_frame(self, frame):
        np.copyto(self.frame_buffer, frame)
        
    def get_name(self):
        return self.memory_name
        
    def close(self):
        self.frame_sm.close()
        self.frame_sm.unlink()
        
        
class FrameMemoryShareClient:
    def __init__(self, memory_share_name, shape, dtype): 
        self.nbytes = int(np.prod(shape) * np.dtype(dtype).itemsize)
        self.frame_sm = shared_memory.SharedMemory(name=memory_share_name)
        self.frame_buffer = np.ndarray(shape, dtype=dtype,buffer=self.frame_sm.buf)
        self.lock = Lock()
        
    def get_frame(self, copying=True):
        if copying:
            return self.frame_buffer.copy()
        return self.frame_buffer    
    
    def set_frame(self, frame):
        np.copyto(self.frame_buffer, frame)
        
    def close(self):
        self.frame_sm.close()
    
        