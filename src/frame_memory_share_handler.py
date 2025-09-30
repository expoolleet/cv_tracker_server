from multiprocessing import shared_memory, Lock
import numpy as np

class FrameMemoryShareHandler:
    def __init__(self, shape: tuple, dtype: np.dtype, name: str = None):
        try:
            self.frame_sm = shared_memory.SharedMemory(name=name, create=False)
            print(f"Shared memory is already created with name: {self.frame_sm.name}. Returning")         
        except FileNotFoundError:
            self.nbytes = int(np.prod(shape) * np.dtype(dtype).itemsize)
            self.frame_sm = shared_memory.SharedMemory(create=True, size=self.nbytes, name=name)
            print(f"Shared memory is created with name: {self.frame_sm.name}")
            
        self.frame_buffer = np.ndarray(shape, dtype=dtype, buffer=self.frame_sm.buf)
        self.memory_name = self.frame_sm.name
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            print(f"Exception in FrameMemoryShareHandler occured: {exc_type}, {exc_val.__cause__}")
            return False
        self.close()
        raise True
        
    def get_frame(self, copying: bool = True) -> np.ndarray:
        return self.frame_buffer.copy() if copying else self.frame_buffer
    
    def set_frame(self, frame: np.ndarray) -> None:
        np.copyto(self.frame_buffer, frame)
        
    def get_name(self) -> str:
        return self.memory_name
        
    def close(self) -> None:
        try:
            self.frame_sm.close()
        except FileNotFoundError:
            pass

    def unlink(self) -> None:
        self.frame_sm.unlink()
        
        
class FrameMemoryShareClient:
    def __init__(self, memory_share_name: str, shape: tuple, dtype: np.dtype): 
        self.frame_sm = shared_memory.SharedMemory(name=memory_share_name, create=False)
        self.frame_buffer = np.ndarray(shape, dtype=dtype,buffer=self.frame_sm.buf)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            print(f"Exception in FrameMemoryShareClient occured: {exc_type}, {exc_val.__cause__}")
            return False
        self.close()
        return True
    
    def get_frame(self, copying: bool = True) -> np.ndarray:
        return self.frame_buffer.copy() if copying else self.frame_buffer    
    
    def set_frame(self, frame: np.ndarray) -> None:
        np.copyto(self.frame_buffer, frame)
        
    def close(self) -> None:
        try:
            self.frame_sm.close()
        except FileNotFoundError:
            pass
    
        