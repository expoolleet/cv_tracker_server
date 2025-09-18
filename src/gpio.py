import RPi.GPIO as GPIO
import time
from dataclasses import dataclass
from typing import Callable

BUTTON_1_PIN = 17
PIN_OUT_1 = 27

BUTTON_2_PIN = 23
PIN_OUT_2 = 24

BUTTON_3_PIN = 5
PIN_OUT_3 = 6

PIN_OUT_4 = 22
PIN_OUT_5 = 25

BUTTON_PRESSED = GPIO.LOW
BUTTON_RELEASED = GPIO.HIGH

loop_sleep_time = 0.2

used_pins = [ BUTTON_1_PIN, BUTTON_2_PIN, BUTTON_3_PIN, PIN_OUT_1, PIN_OUT_2, PIN_OUT_3, PIN_OUT_4, PIN_OUT_5 ]

@dataclass
class GPIOButton:
    pin: int
    is_pressed: bool
    on_press_callback: Callable | None
    on_release_callback: Callable | None
    high_pins_when_pressed: list[int]
    high_pins_when_released: list[int]

def disable_pins(pins: list[int]) -> None:
    for pin in pins:
        GPIO.output(pin, False)
        
def enable_pins(pins: list[int]) -> None:
    for pin in pins:
        GPIO.output(pin, True)
        
def is_button_pressed(button: GPIOButton) -> bool:
    if GPIO.input(button.pin) == BUTTON_PRESSED and not button.is_pressed:
        time.sleep(0.05)
        return GPIO.input(button.pin) == BUTTON_PRESSED
    return False
    
def is_button_released(button: GPIOButton) -> bool:
        return GPIO.input(button.pin) == BUTTON_RELEASED and button.is_pressed
    
class GPIOHandler:
    def __init__(self):
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(PIN_OUT_1, GPIO.OUT)
        GPIO.setup(PIN_OUT_2, GPIO.OUT)
        GPIO.setup(PIN_OUT_3, GPIO.OUT)
        GPIO.setup(PIN_OUT_4, GPIO.OUT)
        GPIO.setup(PIN_OUT_5, GPIO.OUT)
        self.buttons: list[GPIOButton] = []
    
    def init_button_1(self, on_press_callback: Callable | None, on_release_callback: Callable | None) -> None:
        self.button_1 = GPIOButton(BUTTON_1_PIN, False, on_press_callback, on_release_callback, [PIN_OUT_1], [PIN_OUT_4])
        GPIO.setup(self.button_1.pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        self.buttons.append(self.button_1)
        
    def init_button_2(self, on_press_callback: Callable | None, on_release_callback: Callable | None) -> None:
        self.button_2 = GPIOButton(BUTTON_2_PIN, False, on_press_callback, on_release_callback, [PIN_OUT_2], [])
        GPIO.setup(self.button_2.pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        self.buttons.append(self.button_2)    
        
    def init_button_3(self, on_press_callback: Callable | None, on_release_callback: Callable | None) -> None:
        self.button_3 = GPIOButton(BUTTON_3_PIN, False, on_press_callback, on_release_callback, [PIN_OUT_3], [])
        GPIO.setup(self.button_3.pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        self.buttons.append(self.button_3)  
    
    def handle_buttons(self, exit_event: object) -> None:
        try:
            while not exit_event.is_set():   
                for button in self.buttons:
                    if is_button_pressed(button):
                        if button.on_press_callback:
                            button.on_press_callback()
                        button.is_pressed = True
                        enable_pins(button.high_pins_when_pressed)
                        disable_pins(button.high_pins_when_released)
                    elif is_button_released(button):
                        if button.on_release_callback:
                            button.on_release_callback()
                        button.is_pressed = False
                        enable_pins(button.high_pins_when_released)
                        disable_pins(button.high_pins_when_pressed)
                    #time.sleep(0.1)
                time.sleep(loop_sleep_time)
        except KeyboardInterrupt:
            pass
        finally:
            gpio_cleanup()

def gpio_cleanup():
    print("GPIO is cleaning...")
    GPIO.cleanup(used_pins)
    print("GPIO cleanup done.")

