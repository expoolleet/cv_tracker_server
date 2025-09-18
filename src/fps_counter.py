import time

class FPSCounter:
    def __init__(self):
        self.frame_count = 0
        self.start_time = time.perf_counter()
        self.fps = 0
    
    def update(self, print_fps=False):
        self.frame_count += 1
        current_time = time.perf_counter()
        elapsed = current_time - self.start_time
        
        if elapsed >= 1.0:
            fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = current_time
            self.fps = fps
            if print_fps:
                print(f"FPS: {fps:.1f}")
        return self.fps