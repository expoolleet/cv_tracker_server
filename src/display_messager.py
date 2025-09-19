import cv2
import numpy as np
import time
from dataclasses import dataclass
from collections import deque
import queue

MESSAGES_MACOUNT = 10

@dataclass
class Message:
    message: str
    timeout: float
    is_displaying: bool
    start_time: float

class DisplayMessager:
    def __init__(self, message_position: tuple[int, int] = (50, 50), multiprocessing_shared_queue: object = None):
        if multiprocessing_shared_queue is not None:
            self.is_multiprocessing: bool = True
            self.messages_mp: object = multiprocessing_shared_queue
            self.messages_local_deque: deque[Message] = deque(maxlen=MESSAGES_MACOUNT)
        else:
            self.messages_mp = None
            self.messages_local_deque = None
        self.messages: deque[Message] = deque(maxlen=MESSAGES_MACOUNT)
        self.message_position: tuple[int, int] = message_position
        self.font_face: int = 2
        self.font_scale: float = 1
        self.color: tuple[int, int, int] = (255, 255, 255)
        self.thickness: float = 2
        self.short_timeout: float = 0.25
           
    def add_message(self, message_text: str, timeout: float = 2) -> None:
        message = Message(message_text, timeout, False, 0)
        self._put_message(message)
        
    def show_message(self, frame: np.ndarray) -> np.ndarray:        
        message: Message = self._get_message()
        if message is None:
            return frame
        
        if message.is_displaying and time.time() - message.start_time > message.timeout:
            return frame

        self._put_message_back(message)
        
        if not message.is_displaying:
            message.is_displaying = True
            message.start_time = time.time()

        return cv2.putText(frame.copy(), message.message, self.message_position, self.font_face, self.font_scale, self.color, self.thickness)
    
    def clear_messages(self) -> None:
        if self.is_multiprocessing:
            while not self.messages_mp.empty():
                self.messages_mp.get()
            self.messages_local_deque.clear()
        self.messages.clear()
    
    def _get_message(self) -> Message:
        if self.is_multiprocessing:
            try:
                message: Message = self.messages_mp.get(block=False)
                self.messages_local_deque.appendleft(message)
            except queue.Empty:
                pass
            if len(self.messages_local_deque) > 0:
                message = self.messages_local_deque.pop()
                if len(self.messages_local_deque) > 0:
                    message.timeout = self.short_timeout
                return message
            return None
        if len(self.messages) > 0:
            message = self.messages.pop()
            if len(self.messages) > 0:
                message.timeout = self.short_timeout
            return message
        return None
    
    def _put_message(self, message: Message) -> None:
        if self.is_multiprocessing:
            self.messages_mp.put(message)
            return
        self.messages.appendleft(message)
            
    def _put_message_back(self, message: Message) -> None:
        if self.is_multiprocessing:
            self.messages_local_deque.append(message)
            return
        self.messages.append(message)
        
        
        
        