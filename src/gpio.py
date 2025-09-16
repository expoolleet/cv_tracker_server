import RPi.GPIO as GPIO
import time

BUTTON = 27
LED_RED = 23
LED_GREEN = 24

BUTTON_PRESSED = GPIO.LOW
BUTTON_RELEASED = GPIO.HIGH
IS_BUTTON_PRESSED = False

def gpio_init():
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)

    GPIO.setup(BUTTON, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(LED_RED, GPIO.OUT)
    GPIO.setup(LED_GREEN, GPIO.OUT)

def handle_gpio27_pin_state_loop(exit_event, on_pin_low, on_pin_high):
    print("GPIO loop started")
    global IS_BUTTON_PRESSED
    try:
        while not exit_event.is_set():
            if GPIO.input(BUTTON) == BUTTON_PRESSED and not IS_BUTTON_PRESSED:
                on_pin_low()
                GPIO.output(LED_GREEN, True)
                GPIO.output(LED_RED, False)
                IS_BUTTON_PRESSED = True

            elif GPIO.input(BUTTON) == BUTTON_RELEASED and IS_BUTTON_PRESSED:
                on_pin_high()
                GPIO.output(LED_GREEN, False)
                GPIO.output(LED_RED, True)
                IS_BUTTON_PRESSED = False

            time.sleep(0.15)
    except KeyboardInterrupt:
        pass
    finally:
        gpio_cleanup()

def gpio_cleanup():
    print("GPIO is cleaning...")
    GPIO.cleanup([BUTTON, LED_RED, LED_GREEN])
    print("GPIO cleanup done.")

