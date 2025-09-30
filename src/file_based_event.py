import os
import glob
import tempfile
import time

class FileBasedEvent:
    def __init__(self, name: str):
        self.event_name = f"/tmp/{name}.event"
           
    def set(self) -> None:
        open(self.event_name, "w").close()
        
    def is_set(self) -> bool:
        return os.path.exists(self.event_name)
    
    def wait(self) -> None:
        while not self.is_set():
            time.sleep(0.01)
    
    def clear(self) -> None:
        if os.path.exists(self.event_name):
            os.remove(self.event_name)
            
    @staticmethod
    def cleanup_all() -> None:
        temp_dir = tempfile.gettempdir()
        pattern = os.path.join(temp_dir, "*.event")
        event_filters = glob.glob(pattern)
        for file in event_filters:
            try:
                os.remove(file)
            except FileNotFoundError:
                pass
            
    
    